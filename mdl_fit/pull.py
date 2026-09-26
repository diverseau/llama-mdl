"""mdl pull - a GGUF from the Hugging Face Hub, checked, with a preset.

  mdl pull org/repo[:quant] [--name NAME] [--run]

1. the repo pinned at its current commit: every shard of the quant, and
   its vision projector when it ships one
2. each file taken from the Hugging Face cache when a copy there
   matches the Hub's size and sha256, else downloaded, resuming a
   .part left by an earlier try, and kept only once it matches
3. a preset fitted to this machine (mdl fit --write), on a port no
   other preset uses, with --metrics
With --run the model starts once it is in.

Files go to $MDL_MODELS, else ~/models, in a folder per repo, with a
.mdl-pull.json saying which repo and commit they are. While it runs,
STATE_DIR/pull/<name>.json says how far it is - what the web UI's card
shows. Stopping it keeps the .part to resume from.

This is omarchy-local-ai's fetch (0xSero, MIT), in Python: pinned
revision, the cache adopted, every file checked against the Hub.
"""

import hashlib
import http.client
import io
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import progress, remote

CHUNK = 1 << 20
EVERY_S = 1.0                   # the status file at most this often
GiB = 1 << 30

USAGE = """\
usage: mdl pull org/repo[:quant] [--name NAME] [--run]

Download a GGUF from the Hugging Face Hub, check it, and add a preset
for it fitted to this machine. :quant picks one of several quants
(Q4_K_M, or part of a file name). --run starts it once it is in.
Files go to $MDL_MODELS, else ~/models.
"""


class PullError(Exception):
    pass


# -------------------------------------------------------------- where --

def models_dir():
    return Path(os.environ.get("MDL_MODELS") or Path.home() / "models")


def hub_cache():
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" /
                "huggingface") / "hub"


def cached(repo, rev, path):
    return (hub_cache() / ("models--" + repo.replace("/", "--")) /
            "snapshots" / rev / path)


def pull_dir():
    import mdl
    return mdl.STATE_DIR / "pull"


def status_path(name):
    import mdl
    return pull_dir() / ("%s.json" % mdl.check_name(name))


def default_name(repo):
    """unsloth/Qwen3-8B-GGUF -> qwen3-8b"""
    stem = re.sub(r"[-_.]gguf$", "", repo.split("/")[-1], flags=re.I)
    return re.sub(r"[^a-z0-9_-]+", "-", stem.lower()).strip("-")[:64] or "model"


# -------------------------------------------------------------- the hub --

def revision(repo, rev="main"):
    """The commit a branch or tag names now: what gets pinned."""
    url = "%s/api/models/%s/revision/%s" % (
        remote.endpoint(), repo, urllib.parse.quote(rev, safe=""))
    _, _, body = remote._request(url)
    try:
        sha = json.loads(body)["sha"]
    except (ValueError, KeyError, TypeError):
        raise PullError("the Hub did not say which commit %s is" % repo) from None
    if not re.fullmatch(r"[0-9a-f]{40}", str(sha)):
        raise PullError("the Hub gave %s an odd commit id" % repo)
    return sha


def plan(repo, selector=None, rev="main", choose=None):
    """(commit, quant label, [file dicts]): the model's shards, then its
    projector if the repo has one. `choose(repo, groups, commit)` picks
    one of several quants; without it, several is an error naming them."""
    sha = revision(repo, rev)
    files = remote.list_files(repo, sha)
    groups = remote.gguf_groups(files)
    if not groups:
        raise PullError("%s has no GGUF files" % repo)
    try:
        hit = remote.select(groups, selector)
    except remote.RemoteError as e:
        raise PullError(str(e)) from None
    if len(hit) > 1 and choose is not None:
        chosen = choose(repo, hit, sha)
        hit = {chosen: hit[chosen]}
    if len(hit) > 1:
        raise PullError("%s has %d quants; name one, e.g. mdl pull %s:%s "
                        "(have: %s)" % (repo, len(hit), repo,
                                        _quant(example(hit)),
                                        ", ".join(_quant(k) for k in sorted(hit))))
    key, shards = next(iter(hit.items()))
    need = list(shards)
    mm = remote.mmproj_files(files)
    if mm:
        # one projector: full precision when there is a choice, as it is
        # small and its quants cost the model its eyes first
        mm.sort(key=lambda f: (not re.search(r"f16|bf16|f32", f["path"], re.I),
                               f["path"]))
        need.append(mm[0])
    for f in need:
        p = f["path"]
        if p.startswith(("/", "\\")) or ".." in Path(p).parts:
            raise PullError("the Hub listed an unsafe path: %r" % p)
    return sha, key, need


def _quant(key):
    from . import catalog
    q = catalog.quant_of(key)
    return q if q != "?" else Path(key).name


def example(groups):
    """The quant to suggest by name: Q4_K_M, the usual first choice, when
    the repo has it, else the middle one by size. Not the first
    alphabetically - that is BF16, the one nobody should start with."""
    for k in groups:
        if _quant(k).upper() == "Q4_K_M":
            return k
    by_size = sorted(groups, key=lambda k: sum(s["size"] for s in groups[k]))
    return by_size[len(by_size) // 2]


def chooser(binary, status):
    """plan()'s `choose`: the quant find would pick for this machine,
    said in one line through the status."""
    def choose(repo, groups, sha):
        from . import find, gguf
        status.spin("choosing among %d quants for this machine (reading "
                    "one header)" % len(groups))
        try:
            key, why, _ = find.choose_quant(repo, groups, binary,
                                            revision=sha)
        except (remote.RemoteError, gguf.NotGGUF, gguf.Truncated,
                ValueError, OSError) as e:
            raise PullError("%s has %d quants and none could be picked for "
                            "this machine (%s); name one, e.g. mdl pull "
                            "%s:%s" % (repo, len(groups), e, repo,
                                       _quant(example(groups)))) from None
        size = sum(s["size"] for s in groups[key])
        status.say("picked %s (%s): %s; another: mdl pull %s:QUANT"
                   % (_quant(key), _size(size), why, repo))
        return key
    return choose


def _size(n):
    return progress.size_words(n)


def fetched_words(status, need):
    """The line a finished fetch leaves: what came down, how fast, and
    that it was checked."""
    total = sum(f["size"] for f in need)
    files = "%d files" % len(need) if len(need) > 1 else Path(
        need[0]["path"]).name
    if status.downloaded:
        secs = max(0.1, time.monotonic() - status.fetch_started)
        return ("downloaded %s in %s (%s/s), checked against the Hub's "
                "sha256" % (progress.size_words(status.downloaded),
                            progress.time_words(secs),
                            progress.size_words(status.downloaded / secs)))
    return "already here: %s (%s), checked against the Hub's sha256" % (
        files, progress.size_words(total))


def added_words(name):
    """'added [x] to ~/.config/mdl/models.toml: 40k context, q8_0 KV, all
    layers on the GPU'."""
    import mdl
    cfg = mdl.load_config(missing_ok=True)[0].get(name, {})
    bits = []
    if cfg.get("ctx"):
        bits.append("%dk context" % (cfg["ctx"] // 1024))
    if cfg.get("kv_type"):
        bits.append("%s KV" % cfg["kv_type"])
    if cfg.get("n_cpu_moe"):
        bits.append("the experts of %d layers on the CPU" % cfg["n_cpu_moe"])
    elif cfg.get("ngl", 99) >= 99:
        bits.append("all layers on the GPU")
    return "added [%s] to %s%s" % (name, mdl.short(mdl.CONFIG),
                                   ": " + ", ".join(bits) if bits else "")


# ------------------------------------------------------------ the files --

def sha256(path, seen=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(CHUNK * 8)
            if not b:
                break
            h.update(b)
            if seen:
                seen(len(b))
    return h.hexdigest()


def good(path, f, seen=None):
    """The file is whole and, when the Hub gave a hash, matches it."""
    try:
        if Path(path).stat().st_size != f["size"]:
            return False
    except OSError:
        return False
    oid = f.get("oid") or ""
    # the Hub's oid is the sha256 for an LFS file; a small file's is git's
    return len(oid) != 64 or sha256(path, seen) == oid


def download(repo, rev, f, dest, seen):
    """One file to dest, resuming dest.part, checked before it is kept."""
    part = Path(str(dest) + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if have > f["size"]:
        have = 0
    h = hashlib.sha256()
    if have:
        with open(part, "rb") as fh:
            for b in iter(lambda: fh.read(CHUNK * 8), b""):
                h.update(b)
        seen(have, "resumed")          # on disk already, not downloaded now
    if have < f["size"]:
        url = "%s/%s/resolve/%s/%s" % (remote.endpoint(), repo, rev,
                                       urllib.parse.quote(f["path"]))
        headers = {"User-Agent": remote.UA}
        tok = remote.token()
        if tok:
            headers["Authorization"] = "Bearer " + tok
        if have:
            headers["Range"] = "bytes=%d-" % have
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                url, headers=headers), timeout=60)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise PullError("%s is gated: accept its terms on the Hub "
                                "and set HF_TOKEN" % repo) from None
            raise PullError("%s: HTTP %d" % (f["path"], e.code)) from None
        except (urllib.error.URLError, OSError) as e:
            raise PullError("%s: %s; run again to resume" % (
                f["path"], getattr(e, "reason", e))) from None
        with r:
            if have and r.status != 206:     # the Range was ignored
                seen(-have, "resumed")
                have, h = 0, hashlib.sha256()
            with open(part, "ab" if have else "wb") as fh:
                try:
                    for b in iter(lambda: r.read(CHUNK), b""):
                        fh.write(b)
                        h.update(b)
                        seen(len(b))
                except (OSError, http.client.HTTPException) as e:
                    raise PullError("%s: %s; run again to resume" % (
                        f["path"], e.__class__.__name__ if not str(e)
                        else e)) from None
    got = part.stat().st_size
    if got < f["size"]:                 # the connection ended early
        raise PullError("%s stopped at %d of %d MB; run again to resume" % (
            f["path"], got >> 20, f["size"] >> 20))
    oid = f.get("oid") or ""
    if got != f["size"] or (len(oid) == 64 and h.hexdigest() != oid):
        part.unlink()
        raise PullError("%s does not match the Hub's copy; run again"
                        % f["path"])
    os.replace(part, dest)


# ------------------------------------------------------------ progress --

class Status:
    """How far a pull is: STATE_DIR/pull/<name>.json for the web UI's
    card, and bars and a spinner on the terminal (none when quiet, the
    web UI's pulls having no terminal)."""

    def __init__(self, name, repo, keys=(), quiet=False, spec=None):
        self.path = status_path(name) if name else None
        self.base = {"name": name, "repo": repo, "spec": spec,
                     "keys": list(keys),
                     "pid": os.getpid(), "started": time.time()}
        self.quiet, self.last = quiet, 0.0
        # quiet still keeps a bar, off screen, for the card's speed and ETA
        self.line = progress.Line(io.StringIO() if quiet else None)
        self.bar = self.spinner = None
        self.downloaded, self.started = 0, time.monotonic()

    def __call__(self, state, detail="", percent=0, error="", force=True):
        """The card's state: download (with how far) or error (with why)."""
        now = time.monotonic()
        if not force and now - self.last < EVERY_S:
            return
        self.last = now
        if state == "error":
            self.clear()            # the error is printed on a line of its own
        if not self.path:
            return
        import mdl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mdl.write_atomic(self.path, json.dumps(dict(
            self.base, state=state, detail=detail, percent=int(percent),
            error=error, at=time.time())))

    def spin(self, label):
        """A wait with no size, on the terminal: asking the Hub, fitting."""
        self._stop_spin()
        if not self.quiet:
            self.spinner = progress.Spinner(label, line=self.line).start()

    def _stop_spin(self):
        if self.spinner:
            self.spinner.stop()
            self.spinner = None

    def progress(self, what, label, done, total):
        """Bytes moving, `what` being downloading or checking; a new bar
        when either changes."""
        self._stop_spin()
        if self.bar is None or self.bar.what != what:
            if self.bar:
                self.bar.finish()
            self.bar = progress.Bar(label, total, line=self.line)
            self.bar.what = what
        self.bar.update(done, label)
        detail = self.bar.words() if what == "downloading" else \
            "checking the files"
        self("download", detail, self.bar.percent(), force=False)

    def clear(self):
        """Take the progress line off the terminal, for what is printed next."""
        self._stop_spin()
        if self.bar:
            self.bar.finish()
            self.bar = None
        self.line.clear()

    def say(self, line):
        """A line of its own, above the progress line rather than over it."""
        if self.quiet:
            return
        if self.bar:
            self.bar.finish()
            self.bar = None
        # on stdout, where results go; a spinner carries on under it, and
        # holding its line keeps it from drawing between the two
        with self.line.lock:
            self.line.clear()
            print(line, flush=True)

    def done(self):
        self.clear()
        if self.path:
            try:
                self.path.unlink()
            except OSError:
                pass


def read_all():
    """{name: status} for every pull, marking one whose process is gone."""
    import mdl
    out = {}
    try:
        files = sorted(pull_dir().glob("*.json"))
    except OSError:
        return out
    for f in files:
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(s, dict):
            continue
        if s.get("state") != "error" and not mdl.alive(s.get("pid") or 0):
            s["state"] = "error"
            s["error"] = "stopped before it finished; run again to resume"
        out[f.stem] = s
    return out


def stop(name):
    """Stop a pull, keeping what it has for the next try; or forget a
    failed one."""
    import mdl
    s = read_all().get(name)
    if s is None:
        return False
    if s.get("state") != "error" and s.get("pid"):
        import signal
        mdl.terminate(s["pid"], signal.SIGTERM)
    try:
        status_path(name).unlink()
    except OSError:
        pass
    return True


# --------------------------------------------------------------- pull --

def pull(repo, selector=None, name=None, keys=(), run=False, quiet=False):
    """Fetch, check, add. Returns the preset's name."""
    import mdl
    name = mdl.check_name(name or default_name(repo))
    mdl.ensure_config()
    models, binary = mdl.load_config()
    spec = "hf:%s" % repo + (":" + selector if selector else "")
    status = Status(name, repo, keys, quiet, spec)
    status("download", "checking the files", 0)
    status.spin("asking the Hub about %s" % repo)
    try:
        sha, key, need = plan(repo, selector,
                              choose=chooser(binary, status))
        total = sum(f["size"] for f in need) or 1
        if name in models and not _same(models[name], repo, sha, need):
            raise PullError("%s is already in %s; pick another name with "
                            "--name" % (name, mdl.CONFIG))
        where = _fetch(repo, sha, need, total, status)
        status.say(fetched_words(status, need))
        if name not in models:
            model = where[0]
            mmproj = where[-1] if _is_mm(need[-1]) else None
            status.spin("fitting a preset to this machine")
            _add(name, model, mmproj, models)
            status.say(added_words(name))
        status.clear()
    except (PullError, remote.RemoteError, OSError) as e:
        status("error", "", 0, str(e))
        raise PullError(str(e)) from None
    if run:
        models, binary = mdl.load_config()
        try:
            proc, log, port = mdl.spawn(name, models, binary)
        except mdl.MdlError as e:
            status("error", "", 0, str(e))
            raise PullError(str(e)) from None
        if not quiet:
            status.done()
            mdl.tail_until_ready(proc, log, name, port)
            print("ready: %s on http://127.0.0.1:%d (pid %d)"
                  % (name, port, proc.pid))
            if sys.stdout.isatty():
                print(mdl.next_steps(name, port))
            return name
        # the web UI's: its state file shows the load; a start that
        # fails leaves why here, as a start from the page does
        deadline = time.monotonic() + mdl.ready_timeout()
        while time.monotonic() < deadline:
            if mdl.server_ready(port):
                status.done()
                return name
            if proc.poll() is not None:
                status("error", "", 0, last_words(log) or "exited with "
                       "status %s while loading" % proc.returncode)
                raise PullError("%s did not start" % name)
            time.sleep(0.5)
        status("error", "", 0, "not ready after %d s" % mdl.ready_timeout())
        raise PullError("%s did not start" % name)
    else:
        status.done()
    return name


def last_words(log):
    """The line of a log that says what went wrong, else its last."""
    try:
        lines = Path(log).read_text(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        low = line.lower()
        if "error" in low or "failed" in low or "unable" in low:
            return line.strip()[:300]
    return lines[-1].strip()[:300] if lines else None


def _is_mm(f):
    return Path(f["path"]).name.lower().startswith("mmproj")


def _same(cfg, repo, sha, need):
    """An existing preset for these very files: pulling again is a no-op."""
    model = Path(str(cfg.get("model") or ""))
    if model.name != Path(need[0]["path"]).name:
        return False
    try:
        rec = json.loads((model.parent / ".mdl-pull.json").read_text())
        if rec.get("repository") == repo and rec.get("revision") == sha:
            return True
    except (OSError, ValueError, AttributeError):
        pass
    try:
        return model.resolve() == cached(repo, sha, need[0]["path"]).resolve()
    except OSError:
        return False


def _fetch(repo, sha, need, total, status):
    """Every file checked, from the cache or downloaded. Returns their
    paths, in `need`'s order."""
    done = [0]
    label = [""]
    status.fetch_started = time.monotonic()

    def seen(n, what="downloading"):
        done[0] += n
        if what == "downloading":
            status.downloaded += n
        elif what == "resumed":         # a .part from before: progress,
            what = "downloading"        # but not speed
            status.fetch_started = time.monotonic()
        status.progress(what, ("checking " if what != "downloading" else "")
                        + label[0], done[0], total)

    def naming(i, f):
        # the count first: a long name is clipped at its end
        label[0] = ("%d/%d " % (i + 1, len(need)) if len(need) > 1
                    else "") + Path(f["path"]).name

    # everything already in the cache: use it where it is
    hits, checked = [], {}              # cache path -> whether it was good
    for i, f in enumerate(need):
        naming(i, f)
        src = cached(repo, sha, f["path"])
        checked[src] = good(src, f, lambda n: seen(n, "checking"))
        if not checked[src]:
            break
        hits.append(src)
    if len(hits) == len(need):
        return [Path(os.path.abspath(p)) for p in hits]
    done[0] = 0
    folder = models_dir() / repo.replace("/", "--")
    out = []
    for i, f in enumerate(need):
        naming(i, f)
        dest = folder / f["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not good(dest, f, lambda n: seen(n, "checking")):
            # a cache copy is hashed once: above, or here if it was not
            src = cached(repo, sha, f["path"])
            ok = checked.get(src)
            if ok is None and src.is_file():
                ok = good(src, f, lambda n: seen(n, "checking"))
            if ok:
                _adopt(src, dest)
            else:
                download(repo, sha, f, dest, seen)
        else:
            seen(0)
        out.append(dest)
    import mdl
    mdl.write_atomic(folder / ".mdl-pull.json", json.dumps(
        {"repository": repo, "revision": sha,
         "files": [f["path"] for f in need]}, indent=1))
    return out


def _adopt(src, dest):
    """The cache's copy, linked when it can be, else copied."""
    try:
        if dest.exists():
            dest.unlink()
        os.link(os.path.realpath(src), dest)
    except OSError:
        shutil.copyfile(os.path.realpath(src), dest)


def _add(name, model, mmproj, models):
    """A preset fitted to this machine, on a free port, with --metrics;
    mdl add's plain one when the fit cannot run."""
    import contextlib

    import mdl
    port = free_port(models)
    args = [str(model), "--write", name]
    if mmproj:
        args += ["--mmproj", str(mmproj)]
    sink = io.StringIO()
    try:
        from . import cli
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            cli.main(args, out=sink)
    except (SystemExit, mdl.MdlError, Exception):     # noqa: BLE001
        pass
    if name not in mdl.load_config()[0]:
        with contextlib.redirect_stdout(sink):
            mdl.cmd_add([str(model), name, str(port)])
    keys = {"port": port}
    if mmproj:                  # whichever wrote it, the projector pulled
        keys["mmproj"] = str(mmproj).replace(chr(92), "/")
    mdl.patch_params(name, keys)


def free_port(models):
    import mdl
    taken = {cfg.get("port", mdl.DEFAULT_PORT) for cfg in models.values()}
    port = mdl.DEFAULT_PORT
    while port in taken or mdl.port_busy(port):
        port += 1
    return port


# ----------------------------------------------------------------- cli --

def main(args):
    import mdl
    if not args or args[0] in ("-h", "--help"):
        sys.stdout.write(USAGE)
        return None
    spec, name, run, keys, quiet = None, None, False, (), False
    rest = list(args)
    while rest:
        a = rest.pop(0)
        if a == "--name" and rest:
            name = rest.pop(0)
        elif a == "--run":
            run = True
        elif a == "--quiet":                  # the web UI's: no terminal
            quiet = True
        elif a == "--keys" and rest:          # the web UI's cards, to show
            keys = tuple(k for k in rest.pop(0).split(",") if k)
        elif not a.startswith("-") and spec is None:
            spec = a
        else:
            mdl.die(USAGE.splitlines()[0])
    if spec is None:
        mdl.die(USAGE.splitlines()[0])
    try:
        repo, selector = remote.parse_spec(spec)
    except remote.RemoteError as e:
        mdl.die(str(e))
    try:
        got = pull(repo, selector, name, keys, run, quiet)
    except PullError as e:
        mdl.die(str(e))
    if not run:
        print("run it with: mdl run %s" % got)
    return got
