"""mdl update - move this install of mdl to the newest release.

Asks PyPI which release is newest, works out how this copy was installed
(pipx, uv tool, or pip into some environment), and has that same tool
upgrade it, so a pipx install stays a pipx install. A copy run from a
source checkout is refused: updating that is `git pull`, and pulling
someone's working tree is not ours to do.

Standard library only, like mdl.py, and reached lazily: by `mdl update`,
`mdl doctor` and the dashboard, never by the everyday commands. The
dashboard's check is cached for a day, so PyPI is asked at most once a
day, and it is off with MDL_NO_UPDATE_CHECK=1 or `update_check = false`
at the top of models.toml. Nothing is sent but the request itself.
"""

import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import site
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import mdl

PACKAGE = "llama-mdl"
PYPI = "https://pypi.org/pypi/%s/json" % PACKAGE
CHANGES = "https://github.com/diverseau/llama-mdl/blob/main/CHANGELOG.md"
CHECK_EVERY = 24 * 3600          # seconds between the dashboard's checks
UA = "mdl/%s (+https://github.com/diverseau/llama-mdl)"

# The result a dashboard exits with when it has updated mdl and wants the
# new one started in its place.
RESTART = "restart-after-update"


def index_url():
    """PyPI's JSON for the package; MDL_PYPI_URL points the tests at a fake."""
    return os.environ.get("MDL_PYPI_URL") or PYPI


def cache_path():
    return mdl._base("XDG_CACHE_HOME", ".cache") / "mdl" / "update.json"


# ------------------------------------------------------------ versions --

def parse(version):
    """'0.10.0' -> (0, 10, 0). None for anything else: a pre-release, a
    dev or post build, a local tag. Those are never offered, so there is
    no need for the whole of PEP 440 to order them."""
    if not isinstance(version, str) or not re.fullmatch(r"\d+(\.\d+)*",
                                                        version):
        return None
    return tuple(int(p) for p in version.split("."))


def _padded(v, n=3):
    return tuple(v) + (0,) * (n - len(v))


def python_ok(spec, have=None):
    """Whether this Python satisfies a release's requires_python.

    Only the operators PyPI metadata uses in practice; a clause this
    cannot read admits the release, because pip is the one who decides
    and says so in words, whereas a wrong refusal here would hide an
    update for good.
    """
    have = _padded(have or sys.version_info[:3])
    for clause in (spec or "").split(","):
        m = re.fullmatch(r"\s*(~=|==|!=|>=|<=|>|<)\s*([\d.]+)(\.\*)?\s*",
                         clause)
        if not m:
            continue
        op, raw, star = m.groups()
        want = tuple(int(p) for p in raw.strip(".").split("."))
        if star or op == "~=":
            # a prefix: ==3.11.* is any 3.11, ~=3.11 is >=3.11 and 3.*
            head = have[:len(want) - (1 if op == "~=" else 0)]
            same = head == want[:len(head)]
            if op == "~=":
                ok = same and have >= _padded(want)
            else:
                ok = same if op == "==" else not same
        else:
            w = _padded(want)
            ok = {"==": have == w, "!=": have != w, ">=": have >= w,
                  "<=": have <= w, ">": have > w, "<": have < w}[op]
        if not ok:
            return False
    return True


def newest(data, have=None):
    """The newest release in PyPI's JSON that this machine can take: a
    plain X.Y.Z, with a file that is not yanked, whose requires_python
    admits this Python. None if nothing qualifies."""
    best = None
    releases = data.get("releases") if isinstance(data, dict) else None
    for version, files in (releases or {}).items():
        v = parse(version)
        if v is None or not isinstance(files, list):
            continue
        usable = [f for f in files
                  if isinstance(f, dict) and not f.get("yanked")]
        if not usable:
            continue
        if not python_ok(usable[0].get("requires_python"), have):
            continue
        if best is None or _padded(v) > _padded(best[0]):
            best = (v, version)
    return best[1] if best else None


def is_newer(candidate, current=None):
    a, b = parse(candidate), parse(current or mdl.VERSION)
    return a is not None and b is not None and _padded(a) > _padded(b)


def fetch(timeout=5):
    """The newest release PyPI has for us. Raises MdlError, in one line."""
    url = index_url()
    req = urllib.request.Request(url, headers={
        "User-Agent": UA % mdl.VERSION, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        mdl.die("cannot ask %s which mdl is newest: HTTP %d" % (url, e.code))
    except (urllib.error.URLError, OSError) as e:
        reason = str(getattr(e, "reason", e) or type(e).__name__)
        if "CERTIFICATE_VERIFY_FAILED" in reason and sys.platform == "darwin":
            # python.org's macOS installer ships no CA certificates until
            # its Install Certificates.command has been run
            reason += ("; if this Python came from python.org, run 'Install "
                       "Certificates.command' in its Applications folder")
        mdl.die("cannot ask %s which mdl is newest: %s" % (url, reason))
    except (ValueError, UnicodeError):
        mdl.die("%s did not answer with JSON" % url)
    version = newest(data)
    if version is None:
        mdl.die("%s lists no release this Python can install" % url)
    return version


# --------------------------------------------------------------- cache --

def _load():
    try:
        data = json.loads(cache_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(data):
    """Best effort: a check that cannot be remembered is just asked again."""
    try:
        cache_path().parent.mkdir(parents=True, exist_ok=True)
        mdl.write_atomic(cache_path(), json.dumps(data))
    except OSError:
        pass


def check(max_age=CHECK_EVERY, timeout=3):
    """The newest release, from the cache if it is under a day old. None
    if it cannot be had; a check that failed is not retried for a day
    either, so being offline costs one timeout a day, not one a launch."""
    data = _load()
    checked = data.get("checked")
    if isinstance(checked, (int, float)) and 0 <= time.time() - checked < max_age:
        return data.get("latest")
    try:
        latest = fetch(timeout)
    except mdl.MdlError:
        latest = data.get("latest")
    data.update(checked=time.time(), latest=latest)
    _save(data)
    return latest


def offer():
    """The version to offer the dashboard's user, or None: newer than this
    one, and not the one they said to skip."""
    latest = check()
    if latest and is_newer(latest) and latest != _load().get("skip"):
        return latest
    return None


def skip(version):
    """Stop offering `version`. A newer one is offered again."""
    data = _load()
    data["skip"] = version
    _save(data)


def disabled():
    """Why the dashboard and doctor should not check, or None."""
    if os.environ.get("MDL_NO_UPDATE_CHECK", "").strip() not in ("", "0"):
        return "MDL_NO_UPDATE_CHECK is set"
    try:
        mdl.load_config()
    except mdl.MdlError:
        pass
    if (mdl.CONFIG_DATA or {}).get("update_check") is False:
        return "update_check = false in %s" % mdl.CONFIG
    if detect().kind == "source":
        # a checkout is updated with git; offering a pip install over the
        # top of someone's working tree would be worse than no offer
        return "running from a source checkout"
    return None


# ------------------------------------------------------------- install --

class Install:
    """How the running copy of mdl was installed: `kind` is pipx, uv, pip
    or source; `where` is the directory that holds it."""

    def __init__(self, kind, where, user=False):
        self.kind, self.where, self.user = kind, Path(where), user

    def describe(self):
        if self.kind == "source":
            return "a source checkout at %s" % self.where
        return "%s%s, in %s" % (self.kind, " --user" if self.user else "",
                                self.where)


def _installed(here):
    """The installed distribution the running mdl.py belongs to, or None.

    Every one on sys.path is looked at, not the first: `pip install .`
    leaves a llama_mdl.egg-info in the checkout it built, which points
    back at the checkout's own mdl.py and, the checkout being first on
    sys.path, is found before the real install. An installed copy is a
    .dist-info with a RECORD of its files - pip, pipx and uv all write
    one - and a build leftover has none. That egg-info passed for an
    install in CI and would have had pip install over a clone.
    """
    # Directories only. sys.path holds zips too, and on Windows the first
    # entry is the running mdl.exe - pip's launcher is a zip archive with
    # the script inside - which importlib.metadata then holds open for
    # the life of the process. It could not be moved aside, pip failed on
    # it halfway through its uninstall, and the install was left without
    # mdl.py (measured). An installed mdl never lives in a zip.
    dirs = [p for p in sys.path if p and os.path.isdir(p)]
    try:
        dists = list(importlib.metadata.distributions(name=PACKAGE,
                                                      path=dirs))
    except (OSError, ValueError):
        return None
    for dist in dists:
        if dist.read_text("RECORD") is None:
            continue
        try:
            installed = Path(dist.locate_file("mdl.py")).resolve()
        except (OSError, TypeError, ValueError):
            continue
        if installed == here:
            return dist
    return None


def detect():
    here = Path(mdl.__file__).resolve()
    dist = _installed(here)
    if dist is None:
        # a clone run as `python mdl.py`, whether or not a copy is also
        # installed: the one an upgrade would touch is not the one running
        return Install("source", here.parent)
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
    except ValueError:
        direct = {}
    if (direct.get("dir_info") or {}).get("editable"):
        return Install("source", here.parent)
    prefix = Path(sys.prefix)
    if (prefix / "pipx_metadata.json").is_file():
        return Install("pipx", prefix)
    if (prefix / "uv-receipt.toml").is_file():
        return Install("uv", prefix)
    user = False
    try:
        user = bool(site.ENABLE_USER_SITE) and here.is_relative_to(
            Path(site.getusersitepackages()).resolve())
    except (AttributeError, OSError, ValueError):
        pass
    return Install("pip", here.parent, user=user)


def command(inst, version):
    """The argv that upgrades `inst` to `version`."""
    if inst.kind == "pipx":
        # pipx remembers the [ui] extra it was installed with. It takes
        # no version here; the check afterwards says what arrived.
        return ["pipx", "upgrade", PACKAGE]
    if inst.kind == "uv":
        return ["uv", "tool", "upgrade", PACKAGE]
    if inst.kind == "pip":
        # pinned, so what is installed is what was offered; the extra is
        # kept when the dashboard's dependency is here to be kept
        extra = "[ui]" if importlib.util.find_spec("textual") else ""
        argv = [sys.executable, "-m", "pip", "install", "--upgrade",
                "--disable-pip-version-check",
                "%s%s==%s" % (PACKAGE, extra, version)]
        return argv + (["--user"] if inst.user else [])
    mdl.die("running from %s; update it with git pull" % inst.describe())


def busy():
    """Other mdl processes that an upgrade underneath would break, as
    'what (pid N)'. An eval reaches into mdl_fit as it goes; replacing
    those files mid-run mixes two versions in one process. A launch in
    progress is the same. Found by the locks they hold, so `mdl lab`,
    which takes none, is not seen."""
    from mdl_fit import hw
    found = []
    for pattern, what in ((mdl.run_dir(), "a server is starting"),
                          (hw.config_dir() / "eval-runs", "mdl eval")):
        try:
            locks = sorted(pattern.glob("*.lock"))
        except OSError:
            continue
        for lock in locks:
            try:
                pid = int(lock.read_text() or 0)
            except (OSError, ValueError):
                continue
            if pid > 0 and pid != os.getpid() and mdl.alive(pid):
                found.append("%s (pid %d)" % (what, pid))
    return found


def _launcher():
    """The .exe that started this process on Windows, if it is mdl's.

    uv copies the tool's exe into its bin directory and cannot while that
    exe is running (os error 32, measured with uv 0.12); pip and pipx
    manage, but Windows lets anyone rename a running exe, so it is moved
    aside for every installer and the new one lands at the free name.
    """
    if os.name != "nt" or not sys.argv or not sys.argv[0]:
        return None
    path = Path(sys.argv[0])
    if path.suffix.lower() != ".exe":
        path = path.with_name(path.name + ".exe")
    if path.stem.lower() != "mdl" or not path.is_file():
        return None                       # python mdl.py, python -m mdl
    return path


def sweep():
    """Delete the launchers earlier updates moved aside. One still in use
    (its update restarted the dashboard, which is still running) stays
    until a later sweep."""
    exe = _launcher()
    if exe is None:
        return
    for old in exe.parent.glob(exe.name + ".old-*"):
        try:
            old.unlink()
        except OSError:
            pass


def installed_version():
    """The version a fresh interpreter now imports. -P: never a stray
    mdl.py in the current directory."""
    try:
        r = subprocess.run([sys.executable, "-P", "-c",
                            "import mdl; print(mdl.VERSION)"],
                           capture_output=True, text=True, timeout=60,
                           stdin=subprocess.DEVNULL,
                           creationflags=getattr(subprocess,
                                                 "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def install(version, out=print, force=False):
    """Upgrade this install to `version`, streaming the installer's lines
    to `out`. Returns the version now installed; raises MdlError, with
    this copy left as it was, when anything short of that happened."""
    inst = detect()
    argv = command(inst, version)
    if not force:
        others = busy()
        if others:
            mdl.die("%s would be running two versions of mdl at once; wait "
                    "for it, or pass --force" % ", ".join(others))
    exe = _launcher()
    moved = None
    if exe is not None:
        moved = exe.with_name("%s.old-%d" % (exe.name, os.getpid()))
        try:
            os.replace(exe, moved)
        except OSError as e:
            # the installer needs this same file out of the way, and pip
            # failing on it midway leaves the install half removed: not
            # starting is the only safe answer
            mdl.die("cannot move %s aside to replace it (%s); nothing was "
                    "changed - close other programs using it and try again"
                    % (exe, getattr(e, "strerror", None) or e))
    lines = []
    try:
        out("running: %s" % " ".join(
            a if " " not in a else '"%s"' % a for a in argv))
        try:
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as e:
            where = ("; is %s on your PATH?" % argv[0]
                     if inst.kind in ("pipx", "uv") else "")
            mdl.die("cannot run %s: %s%s" % (argv[0], e, where))
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                out(line)
                lines.append(line)
        code = proc.wait()
    finally:
        if moved is not None and not exe.exists():
            # the installer failed before writing a new launcher; put
            # the old one back rather than leave no `mdl` at all
            try:
                os.replace(moved, exe)
            except OSError:
                pass
    if code:
        mdl.die("%s; still on %s" % (_why_failed(argv, code, lines, inst),
                                     mdl.VERSION))
    now = installed_version()
    if now is None:
        mdl.die("the installer finished, but mdl no longer imports; "
                "reinstall with: %s" % " ".join(argv))
    if not is_newer(now):
        why = ""
        if inst.kind in ("pipx", "uv"):
            why = ("; the install may be pinned to %s (reinstall it without "
                   "a version), or its index does not have %s yet"
                   % (now, version))
        mdl.die("the installer finished, but mdl is still %s%s" % (now, why))
    return now


def _why_failed(argv, code, lines, inst):
    """One line on why the installer failed, from all it said. pip's
    reason is its first ERROR line, and it goes on for a dozen lines of
    notes and hints after it - the last line alone said nothing."""
    text = "\n".join(lines)
    name = os.path.basename(argv[0])
    if "externally-managed-environment" in text:
        # Debian and Ubuntu's own Python, Homebrew's: pip will not touch
        # it, --user included, and forcing it past is not ours to do
        return ("pip will not install into this Python, which your OS or "
                "package manager owns (PEP 668); install mdl with pipx "
                "instead: pipx install \"llama-mdl[ui]\"")
    first = next((x for x in lines
                  if x.lower().startswith(("error:", "error "))), None)
    first = first or (lines[-1] if lines else "")
    why = "%s exited with status %d%s" % (
        name, code, ": " + " ".join(first.split()) if first else "")
    if inst.kind == "pip" and ("Permission denied" in text
                               or "[Errno 13]" in text):
        why += ("; mdl is installed where you cannot write (%s), so update it "
                "as whoever installed it, or move to pipx" % inst.where)
    return why


def path_note(version):
    """A line if the `mdl` found first on PATH is not the one updated."""
    found = shutil.which("mdl")
    if not found:
        return None
    try:
        r = subprocess.run([found, "--version"], capture_output=True,
                           text=True, timeout=20, stdin=subprocess.DEVNULL,
                           creationflags=getattr(subprocess,
                                                 "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return None
    got = r.stdout.strip().rpartition(" ")[2]
    if r.returncode == 0 and got and got != version:
        return ("note: the mdl first on your PATH (%s) is %s; it is another "
                "install, and this did not update it" % (found, got))
    return None


def restart(args):
    """Start mdl again as `mdl <args>` from the code now installed, in
    this terminal, and never return.

    POSIX replaces this process. Windows has no such thing - os.exec*
    starts a child and exits, and the console takes its prompt back while
    the child is still drawing - so there the new one runs as a child and
    this one waits for it and passes its exit status on.
    """
    env = dict(os.environ, MDL_UPDATED_FROM=mdl.VERSION)
    argv = [sys.executable, "-P", "-m", "mdl", *args]
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        try:
            sys.exit(subprocess.call(argv, env=env))
        except KeyboardInterrupt:
            sys.exit(130)
    try:
        os.execve(sys.executable, argv, env)
    except OSError as e:
        # the update itself stands; only starting it again failed
        mdl.die("mdl is updated, but could not start again (%s): run "
                "mdl tui" % e)


# ----------------------------------------------------------------- cli --

USAGE = "usage: mdl update [--check] [--force]"


def main(args):
    if any(a not in ("--check", "--force") for a in args):
        mdl.die(USAGE)
    sweep()
    latest = fetch()
    # the explicit command asks PyPI every time; the dashboard's cache
    # learns from it too
    data = _load()
    data.update(checked=time.time(), latest=latest)
    _save(data)
    if not is_newer(latest):
        if latest == mdl.VERSION:
            print("mdl %s is the newest release" % mdl.VERSION)
        else:
            print("mdl %s here is newer than the newest release (%s)"
                  % (mdl.VERSION, latest))
        return
    if "--check" in args:
        print("mdl %s is out (you have %s); run: mdl update"
              % (latest, mdl.VERSION))
        return
    inst = detect()
    if inst.kind == "source":
        mdl.die("mdl %s is out, but this is running from %s; update it with "
                "git pull" % (latest, inst.describe()))
    print("mdl %s -> %s (%s)" % (mdl.VERSION, latest, inst.describe()))
    now = install(latest, force="--force" in args)
    print("updated mdl %s -> %s" % (mdl.VERSION, now))
    print("what changed: %s" % CHANGES)
    note = path_note(now)
    if note:
        print(note)
