"""mdl eval - runs the suites against llama-server and keeps the score.

The model runs as models.toml has it - whatever `mdl fit` picked, at the
quant, KV type and build you actually use. A model that is already
running is used as it is; one that is not is started for the run and
stopped after. Results go to ~/.config/mdl/evals.jsonl, keyed by a hash
of the file, the quant, the KV type and the build, with a bootstrap 95%
interval per suite and per domain.
"""

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import calib, evalsuite, hw, model, perf

MAX_TURNS = 8
TIMEOUT = 1800
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
IMAGE = "python:3.12-slim"

USAGE = """\
usage: mdl eval <name> [--suite code,tools,longctx,instruct,reason,custom]
                       [--limit N] [--sandbox] [--estimate] [--json]
       mdl eval --results [name]

Runs a private, auto-graded suite against <name> the way models.toml
runs it - starting it if it is not running - and stores the scores:

  code      40  functions, graded by hidden unit tests that are executed
  tools     30  tool calls, single and multi-step against mock worlds,
                one of them paged and a dozen calls long
  longctx   24  retrieval, multi-hop, counting, finding every match, and
                one question the document does not answer, at 32k/64k/128k
  instruct  20  checkable format rules
  reason    20  exact-answer maths and logic
  custom        your own tasks, from ~/.config/mdl/evals/*.toml

Three of every five items are the harder tier, scored separately, and
code and format items are marked in parts rather than all or nothing.

The items are generated from a seed kept in ~/.config/mdl/eval-seed, so
they exist on this machine only. Model-written code runs in a
subprocess in a temp dir with a timeout; --sandbox runs it in a
throwaway podman or docker container with no network instead.

  --limit N     only the first N items of each suite
  --estimate    say how long it would take, and stop
  --results     past runs
  --compare A B  two models on the same items, scored item by item
"""


def die(msg):
    import mdl
    mdl.die(msg)


# -------------------------------------------------------------- client --

class Call:
    __slots__ = ("id", "name", "args", "raw")

    def __init__(self, cid, name, args, raw):
        self.id, self.name, self.args, self.raw = cid, name, args, raw


class Reply:
    def __init__(self, content="", reasoning="", calls=(), finish="",
                 prompt_tokens=0, completion_tokens=0, error=None):
        self.content, self.reasoning = content, reasoning
        self.calls, self.finish = list(calls), finish
        self.prompt_tokens, self.completion_tokens = (prompt_tokens,
                                                      completion_tokens)
        self.error = error


_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S)


def split_thinking(content):
    """(answer, thinking) from a reply that may carry <think> inline."""
    thinking = "".join(m.group(0) for m in _THINK.finditer(content or ""))
    return _THINK.sub("", content or "").strip(), thinking


class Client:
    """llama-server's OpenAI-style API, not streamed: an eval wants the
    whole reply and its token counts, not the tokens as they come."""

    def __init__(self, port, host="127.0.0.1", timeout=TIMEOUT):
        self.base = "http://%s:%d" % (host, port)
        self.timeout = timeout

    def _call(self, path, body=None, timeout=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + path, data=data, method="POST" if data else "GET",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    def chat(self, messages, tools=None, max_tokens=1024, seed=None):
        body = {"messages": messages, "max_tokens": max_tokens,
                "cache_prompt": True, "stream": False}
        if seed is not None:
            body["seed"] = seed
        if tools:
            body["tools"] = tools
        try:
            data = self._call("/v1/chat/completions", body)
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", "replace")
            return Reply(error="HTTP %d %s" % (e.code, detail))
        except (urllib.error.URLError, OSError, ValueError) as e:
            return Reply(error=str(e))
        try:
            choice = data["choices"][0]
            msg = choice.get("message") or {}
        except (KeyError, IndexError, TypeError, AttributeError):
            return Reply(error="unexpected reply: %.200s" % data)
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content
                              if isinstance(p, dict))
        content, inline = split_thinking(content)
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw = fn.get("arguments")
            if isinstance(raw, str):
                try:
                    args = json.loads(raw) if raw.strip() else {}
                except ValueError:
                    args = None
            else:
                args, raw = raw, json.dumps(raw)
            calls.append(Call(tc.get("id") or "call_%d" % i,
                              fn.get("name", ""), args, raw))
        usage = data.get("usage") or {}
        return Reply(content, (msg.get("reasoning_content") or "") + inline,
                     calls, choice.get("finish_reason") or "",
                     usage.get("prompt_tokens", 0),
                     usage.get("completion_tokens", 0))

    def tokens(self, text):
        try:
            return len(self._call("/tokenize", {"content": text}, 60)
                       .get("tokens", []))
        except (urllib.error.URLError, OSError, ValueError, AttributeError):
            return None

    def props(self):
        try:
            got = self._call("/props", None, 10)
            return got if isinstance(got, dict) else {}
        except (urllib.error.URLError, OSError, ValueError):
            return {}


# ------------------------------------------------------------- sandbox --

def find_runtime():
    return shutil.which("podman") or shutil.which("docker")


class Env:
    """Where model-written code runs: a subprocess in a temp dir with a
    timeout and a scrubbed environment, or - with a runtime - a
    throwaway container with no network, 512 MB and 128 processes."""

    def __init__(self, runtime=None, timeout=10, image=IMAGE):
        self.runtime, self.timeout, self.image = runtime, timeout, image

    def run_python(self, files, entry="main.py", timeout=None, read=None):
        """Run `entry` among `files`; (passed, the tail of its output).

        With `read`, also the text of that file as the run left it, or
        None - a third item, so a grader can take its answer from a file
        rather than from output the code under test also writes to.

        The output kept is the last TAIL characters however much is
        printed, and a run that outlives its timeout is killed with
        everything it started, not just the process we launched.
        """
        timeout = timeout or self.timeout
        with tempfile.TemporaryDirectory(prefix="mdl-eval-",
                                         ignore_cleanup_errors=True) as tmp:
            for name, text in files.items():
                Path(tmp, name).write_text(text, encoding="utf-8")
            box = None
            if self.runtime:
                box = "mdl-eval-%s" % os.urandom(6).hex()
                argv = [self.runtime, "run", "--rm", "--name", box,
                        "--network", "none",
                        "--memory", "512m", "--pids-limit", "128",
                        "-v", "%s:/w" % tmp, "-w", "/w", self.image,
                        "python", "-E", "-s", entry]
                timeout += 60                  # container start
            else:
                argv = [sys.executable, "-E", "-s", entry]
            env = {k: v for k, v in (
                ("PATH", os.environ.get("PATH", "")),
                ("SYSTEMROOT", os.environ.get("SYSTEMROOT", "")),
                ("PYTHONIOENCODING", "utf-8"),
                ("PYTHONDONTWRITEBYTECODE", "1"),
                ("TEMP", tmp), ("TMP", tmp), ("TMPDIR", tmp)) if v}
            ok, out = _bounded(argv, tmp, env, timeout, box, self.runtime)
            if read is None:
                return ok, out
            try:
                got = Path(tmp, read).read_text(encoding="utf-8",
                                                errors="replace")
            except OSError:
                got = None
            return ok, out, got


TAIL = 4000


def _kill_tree(p):
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=NO_WINDOW)
    else:
        try:
            os.killpg(p.pid, 9)          # its own session: the group is the tree
        except OSError:
            pass
    try:
        p.kill()
    except OSError:
        pass


def _bounded(argv, cwd, env, timeout, box=None, runtime=None):
    """Run argv with stdout and stderr merged into a pipe that is drained
    as it fills, keeping only the tail - so a flood of output can neither
    fill memory nor block the child on a full pipe."""
    try:
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             creationflags=NO_WINDOW,
                             start_new_session=os.name != "nt")
    except OSError as e:
        return False, str(e)
    tail = bytearray()

    def pump():
        for chunk in iter(lambda: p.stdout.read(8192), b""):
            tail.extend(chunk)
            if len(tail) > 2 * TAIL:
                del tail[:-TAIL]

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        code = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(p)
        if box:
            subprocess.run([runtime, "rm", "-f", box],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=NO_WINDOW)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        reader.join(1)
        return False, "timed out after %ds" % timeout
    reader.join(5)
    text = bytes(tail[-TAIL:]).decode("utf-8", errors="replace")
    return code == 0, text


# ----------------------------------------------------------------- run --

def run_item(client, item, env, cpt=4.0):
    t0 = time.time()
    msgs = item.messages(cpt)
    seed = int(hashlib.sha256(item.id.encode()).hexdigest()[:8], 16)
    world = item.world() if item.world else None
    prompt = completion = 0
    capped = thought = False
    budget = item.meta.get("turns", MAX_TURNS) if world else 1
    for _ in range(budget):
        r = client.chat(msgs, item.tools, item.max_tokens, seed)
        prompt += r.prompt_tokens
        completion += r.completion_tokens
        capped = capped or r.finish == "length"
        thought = thought or bool(r.reasoning)
        if r.error or world is None or not r.calls:
            break
        msgs.append({"role": "assistant", "content": r.content or "",
                     "tool_calls": [{"id": c.id, "type": "function",
                                     "function": {"name": c.name,
                                                  "arguments": c.raw or "{}"}}
                                    for c in r.calls]})
        for c in r.calls:
            msgs.append({"role": "tool", "tool_call_id": c.id,
                         "content": json.dumps(world.call(c.name, c.args))})
    if r.error:
        score, why = 0.0, "error: " + r.error[:200]
    else:
        try:
            score, why = item.grade(r, env, world)
        except Exception as e:                   # noqa: BLE001 - scored, not raised
            score, why = 0.0, "grader failed: %s" % e
    return {"id": item.id, "suite": item.suite, "domain": item.domain,
            "tier": item.meta.get("tier", "base"),
            "score": float(score), "why": why, "capped": capped,
            "error": bool(r.error), "thinking": thought,
            "reply": (r.content or "")[-400:],
            "prompt_tokens": prompt, "completion_tokens": completion,
            "seconds": round(time.time() - t0, 2)}


def run_items(client, items, env, cpt=4.0, n_ctx=None, sink=None,
              progress=None):
    """Every item in order, into `sink` as they finish (so an interrupt
    keeps what was done). A long document that does not fit n_ctx is
    skipped, not failed."""
    sink = [] if sink is None else sink
    for it in items:
        need = it.meta.get("tokens")
        if need and n_ctx and need + evalsuite.ROOM > n_ctx:
            res = {"id": it.id, "suite": it.suite, "domain": it.domain,
                   "tier": it.meta.get("tier", "base"),
                   "skipped": "needs %dk of context, has %dk" % (
                       (need + evalsuite.ROOM) // 1000, n_ctx // 1024)}
        else:
            res = run_item(client, it, env, cpt)
        sink.append(res)
        if progress:
            progress(it, res)
    return sink


def bootstrap(scores, n=1000, seed=0):
    """(mean, lo, hi): a 95% percentile interval over n resamples."""
    k = len(scores)
    mean = sum(scores) / k
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(scores) for _ in range(k)) / k
                   for _ in range(n))
    return mean, means[int(0.025 * n)], means[int(0.975 * n) - 1]


def summarize(results, key):
    groups = {}
    for r in results:
        if "score" in r:
            groups.setdefault(r[key], []).append(r)
    out = {}
    for k, rs in sorted(groups.items()):
        mean, lo, hi = bootstrap([r["score"] for r in rs], seed=len(rs))
        spend = sum(r.get("completion_tokens", 0) for r in rs)
        earned = sum(r["score"] for r in rs)
        out[k] = {"n": len(rs), "score": round(mean, 4), "lo": round(lo, 4),
                  "hi": round(hi, 4),
                  "capped": sum(1 for r in rs if r["capped"]),
                  "errors": sum(1 for r in rs if r["error"]),
                  "tokens": spend,
                  "seconds": round(sum(r.get("seconds", 0) for r in rs), 1),
                  # what a right answer cost: a model that thinks four
                  # times as long for the same score is not as good
                  "per_point": round(spend / earned) if earned else None}
    return out


# ------------------------------------------------------------ estimate --

SAMPLING = ("--temp", "--top-p", "--top-k", "--min-p", "--typical",
            "--repeat-penalty", "--presence-penalty", "--frequency-penalty",
            "--reasoning", "--reasoning-budget", "--chat-template-kwargs")


def sampling_of(argv):
    """The flags that change what a model says, for the record.

    A score belongs to a model at a temperature, not to a model. Two
    runs of one file at different sampling settings are two results.
    """
    out = {}
    for i, a in enumerate(argv[:-1]):
        if a in SAMPLING:
            out[a.lstrip("-")] = argv[i + 1]
    return out


def thinking_flag(argv):
    """True or False when the command line says; None when it is left
    to the chat template."""
    for i, a in enumerate(argv[:-1]):
        if a == "--reasoning":
            return argv[i + 1] != "off"
        if a == "--reasoning-budget":
            return argv[i + 1] != "0"
    return None


def spent(name_hash, records=None, version=None):
    """{suite: mean reply tokens} from this model's last finished run.

    How much a model thinks is a property of the model and its sampling
    flags, not something a constant can carry: a reasoning model told to
    keep it short spends a quarter of what one left to ramble does. Once
    it has run here, stop guessing.
    """
    best = None
    for want in ([version] if version else []) + [None]:
        # a run of a different generation of the suite answered different
        # questions, so only fall back to one when there is nothing newer
        for rec in (records if records is not None else load()):
            if rec.get("hash") != name_hash or rec.get("partial"):
                continue
            if want is not None and rec.get("suite_version") != want:
                continue
            if best is None or rec.get("at", "") > best.get("at", ""):
                best = rec
        if best is not None:
            break
    if best is None:
        return {}
    total, n = {}, {}
    for r in best.get("items", []):
        if "completion_tokens" in r and "suite" in r:
            total[r["suite"]] = total.get(r["suite"], 0) + r["completion_tokens"]
            n[r["suite"]] = n.get(r["suite"], 0) + 1
    return {k: total[k] / n[k] for k in total if n[k]}


def estimate(items, shape, flags, mach, eff=None, thinking=False, cpt=4.0,
             seen=None):
    """Seconds the items should take at this config, from the perf model:
    each prompt prefilled, each expected reply decoded at its depth. A
    long document is paid for once; the questions after it hit the
    prompt cache. `seen` is what this model actually spent last time."""
    eff = eff or {}
    seen = seen or {}
    p = perf.params(mach)
    pl = perf.Placement(shape, flags)
    e_tg, e_pp = eff.get("tg", 1.0), eff.get("pp", 1.0)
    total, docs = 0.0, set()
    for it in items:
        doc = it.meta.get("doc")
        if doc:
            if it.meta["tokens"] + evalsuite.ROOM > flags.ctx:
                continue
            depth = it.meta["tokens"] if doc in docs else 0
            n_in = 80 if doc in docs else it.meta["tokens"]
            docs.add(doc)
        else:
            depth = 0
            n_in = (len(it.text(cpt)) + len(json.dumps(it.tools or ""))) / cpt
        n_out = seen.get(it.suite) or evalsuite.EXPECT.get(it.suite, 400) * (
            evalsuite.THINK_FACTOR if thinking else 1)
        n_out = min(n_out, it.max_tokens)
        turns = (it.meta["turns"] // 2 if it.meta.get("turns")
                 else 3) if it.world else 1
        total += turns * (
            perf.prefill_time(pl, p, n_in + 150, flags.ub, depth, e_pp)
            + n_out * perf.decode_time(pl, p, depth + n_in, e_tg))
    return total


def minutes(seconds):
    if seconds < 90:
        return "%d s" % max(1, round(seconds))
    if seconds < 5400:
        return "%d min" % round(seconds / 60)
    return "%.1f h" % (seconds / 3600)


# --------------------------------------------------------------- store --

def results_path():
    return hw.config_dir() / "evals.jsonl"


def custom_dir():
    return hw.config_dir() / "evals"


def file_hash(path, window=8 << 20):
    """Size plus the first and last 8 MiB: tells two fine-tunes of one
    base apart without reading 20 GB."""
    h = hashlib.sha256()
    size = Path(path).stat().st_size
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(window))
        if size > window:
            f.seek(max(window, size - window))
            h.update(f.read(window))
    return h.hexdigest()[:16]


def save(rec):
    path = results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def load(name=None):
    try:
        lines = results_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if name is None or rec.get("name") == name:
            out.append(rec)
    return out


# ---------------------------------------------------------------- show --

class Progress:
    """A row of marks per suite: . pass  x fail  ! error  - skipped."""

    def __init__(self, out):
        self.out, self.suite, self.row, self.t = out, None, [], time.time()

    def __call__(self, item, res):
        if item.suite != self.suite:
            self.close()
            self.suite, self.row, self.t = item.suite, [], time.time()
            self.out.write("%-9s" % item.suite)
        self.row.append(res)
        self.out.write("-" if "skipped" in res else "!" if res["error"]
                       else "." if res["score"] >= 1 else "x")
        self.out.flush()

    def close(self):
        if self.suite:
            done = [r for r in self.row if "score" in r]
            self.out.write("  %d/%d  %s\n" % (
                sum(1 for r in done if r["score"] >= 1), len(done),
                minutes(time.time() - self.t)))
            self.suite = None


def _float(x, default=1.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def report(rec, w):
    w("\nresults  suite v%d · %d items · %s%s\n" % (
        rec["suite_version"], sum(v["n"] for v in rec["suites"].values()),
        minutes(rec["minutes"] * 60), " · PARTIAL" if rec.get("partial")
        else ""))
    for title, part in (("suite", rec["suites"]), ("domain", rec["domains"]),
                        ("tier", rec.get("tiers") or {})):
        w("  %-13s score  95%% CI       items  capped  errors\n" % title)
        for k, v in part.items():
            w("  %-13s %.2f   %.2f–%.2f    %-6d %-7d %d\n" % (
                k, v["score"], v["lo"], v["hi"], v["n"], v["capped"],
                v["errors"]))
    # one number, and it weights the five abilities equally: weighting by
    # item count would move the headline whenever a suite changes size,
    # which says nothing about the model
    doms = rec.get("domains") or {}
    if doms:
        w("  %-13s %.2f   the five domains, equally weighted\n" % (
            "overall", sum(v.get("score", 0) for v in doms.values())
            / len(doms)))
    whole = rec.get("suites") or {}
    spend = sum(v.get("tokens", 0) for v in whole.values())
    earned = sum(v.get("score", 0) * v.get("n", 0) for v in whole.values())
    if spend:
        w("cost     %s reply tokens · %s per right answer · %s\n" % (
            "{:,}".format(spend),
            "{:,}".format(round(spend / earned)) if earned else "-",
            minutes(rec.get("minutes", 0) * 60)))
    items = rec.get("items", [])
    skipped = [r for r in items if "skipped" in r]
    if skipped:
        w("note     %d long-context items skipped: %s\n" % (
            len(skipped), skipped[0]["skipped"]))
    errs = [r for r in items if r.get("error")]
    whys = [r.get("why", "").lower() for r in errs]
    if any("jinja" in y or "tools" in y for y in whys):
        w("note     tool calls need --jinja in the config's args\n")
    elif errs:
        w("note     %d errors, e.g. %s\n" % (len(errs), whys[0][:120]))
    # the interval is sampling error over items. It says nothing about
    # the same model answering differently next time, which is what
    # temperature buys, and two runs of one model at 0.8 can differ by
    # more than the interval printed above
    temp = (rec.get("sampling") or {}).get("temp")
    if temp is None or _float(temp) > 0.3:
        w("note     sampled at %s; the intervals above are over items, "
          "not over runs\n" % ("the server default" if temp is None
                               else "temp " + str(temp)))
    if rec.get("thinking"):
        capped = sum(1 for r in items if r.get("capped"))
        w("note     it thinks; replies are capped per suite (%d hit the "
          "cap)\n" % capped)
    misses = [r for r in items if r.get("score") == 0 and not r.get("error")]
    for r in misses[:5]:
        # a reply cut off mid-answer failed for a different reason than
        # one that ran and got it wrong, and the miss line is what
        # anyone actually reads
        w("miss     %-26s %s%s\n" % (
            r["id"], r.get("why", "")[:90],
            "  (reply hit the cap)" if r.get("capped") else ""))
    if len(misses) > 5:
        w("         ... and %d more\n" % (len(misses) - 5))


def latest(name, records):
    """The newest finished run of one model."""
    best = None
    for rec in records:
        if rec.get("name") != name or rec.get("partial"):
            continue
        if best is None or rec.get("at", "") > best.get("at", ""):
            best = rec
    return best


def paired(a, b):
    """[(score a, score b, domain)] over the items both runs answered.

    Both runs answer the same generated items, so the scores can be
    paired item by item. That removes the variance of the item set
    itself, which is most of it: two models differ by far less than two
    questions do.
    """
    left = {r["id"]: r for r in a.get("items", []) if "score" in r}
    out = []
    for r in b.get("items", []):
        if "score" in r and r["id"] in left:
            out.append((left[r["id"]]["score"], r["score"],
                        r.get("domain", "general")))
    return out


def diff_ci(pairs, n=2000, seed=7):
    """(mean difference, lo, hi) by resampling the pairs, not the models."""
    if not pairs:
        return 0.0, 0.0, 0.0
    diffs = [x - y for x, y, _ in pairs]
    mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs)
                   for _ in range(n))
    return mean, means[int(0.025 * n)], means[int(0.975 * n) - 1]


def verdict(lo, hi, a, b):
    if lo > 0:
        return "%s ahead" % a
    if hi < 0:
        return "%s ahead" % b
    return "too close to call"


def compare(a, b, w):
    """Two runs, item by item, with a verdict that survives the noise."""
    if a.get("items_hash") != b.get("items_hash"):
        w("note     different item sets; the two runs are not comparable\n")
        return None
    pairs = paired(a, b)
    if not pairs:
        w("note     no items in common\n")
        return None
    na, nb = a["name"], b["name"]
    w("compare  %s vs %s   (%d items, suite v%s, item set %s)\n" % (
        na, nb, len(pairs), a.get("suite_version", "?"),
        a.get("items_hash", "?")))
    w("  %-13s %-6s %-6s %-7s %-18s %s\n" % (
        "domain", na[:6], nb[:6], "diff", "95% CI", "verdict"))
    rows = {}
    for x, y, dom in pairs:
        rows.setdefault(dom, []).append((x, y, dom))
    for dom in sorted(rows) + ["overall"]:
        got = pairs if dom == "overall" else rows[dom]
        mean, lo, hi = diff_ci(got)
        w("  %-13s %-6.2f %-6.2f %+-7.2f %+.2f to %+-10.2f %s\n" % (
            dom, sum(x for x, _, _ in got) / len(got),
            sum(y for _, y, _ in got) / len(got), mean, lo, hi,
            verdict(lo, hi, na, nb)))
    for rec in (a, b):
        whole = rec.get("suites") or {}
        spend = sum(v.get("tokens", 0) for v in whole.values())
        if spend:
            w("cost     %-10s %s reply tokens · %s\n" % (
                rec["name"], "{:,}".format(spend),
                minutes(rec.get("minutes", 0) * 60)))
    return pairs


def show_results(name, w):
    recs = load(name)
    if not recs:
        w("no eval results%s yet\n" % (" for %s" % name if name else ""))
        return
    doms = ["coding", "agentic", "long-context", "general", "reasoning"]
    w("%-16s %-18s %-8s %-10s %-6s %s\n" % (
        "when", "name", "quant", "kv", "build",
        "  ".join("%-12s" % d for d in doms)))
    for r in recs:
        cells = []
        for d in doms:
            v = r.get("domains", {}).get(d)
            cells.append("%-12s" % ("%.2f ±%.2f" % (
                v["score"], (v["hi"] - v["lo"]) / 2) if v else "-"))
        w("%-16s %-18s %-8s %-10s %-6s %s%s\n" % (
            r["at"], r["name"][:18], r.get("quant", "?")[:8],
            r.get("kv", "?"), r.get("build") or "?", "  ".join(cells),
            "  partial" if r.get("partial") else ""))


# ---------------------------------------------------------------- main --

def parse(args):
    o, pos, i = {}, [], 0
    while i < len(args):
        a = args[i]
        if a in ("--suite", "--limit", "--port"):
            if i + 1 >= len(args):
                die("%s needs a value" % a)
            o[a[2:]] = args[i + 1]
            i += 2
        elif a in ("--sandbox", "--estimate", "--json", "--results",
                   "--compare"):
            o[a[2:]] = True
            i += 1
        elif a.startswith("-"):
            die("unknown option %s\n%s" % (a, USAGE))
        else:
            pos.append(a)
            i += 1
    return o, pos


def pick_suites(text):
    if not text:
        have = any(custom_dir().glob("*.toml")) if custom_dir().is_dir() \
            else False
        return list(evalsuite.SUITES) + (["custom"] if have else [])
    got = [s.strip() for s in text.split(",") if s.strip()]
    bad = [s for s in got if s not in evalsuite.SUITES + ("custom",)]
    if bad:
        die("unknown suite %s (have: %s, custom)" % (
            ", ".join(bad), ", ".join(evalsuite.SUITES)))
    return got


def chars_per_token(client):
    sample = evalsuite.filler(random.Random(1), 4000)
    n = client.tokens(sample)
    return len(sample) / n if n else 4.0


def wait_ready(proc, port, name, log):
    import mdl
    deadline = time.monotonic() + mdl.ready_timeout()
    while not mdl.server_ready(port):
        if proc.poll() is not None:
            mdl.state_path(name).unlink(missing_ok=True)
            die("%s exited with status %s while loading; see %s" % (
                name, proc.returncode, log))
        if time.monotonic() > deadline:
            state = mdl.read_state(name)
            if state:
                mdl.stop_one(name, state)
            die("%s was not ready after %ds; see %s" % (
                name, mdl.ready_timeout(), log))
        time.sleep(0.3)


def main(args, out=None):
    out = out or sys.stdout
    w = out.write
    if not args or args[0] in ("-h", "--help"):
        w(USAGE)
        return None
    o, pos = parse(args)
    if o.get("results"):
        return show_results(pos[0] if pos else None, w)
    if o.get("compare"):
        if len(pos) != 2:
            die("--compare takes two model names")
        records = load()
        runs = [latest(n, records) for n in pos]
        for n, rec in zip(pos, runs, strict=True):
            if rec is None:
                die("no finished run for %r; run: mdl eval %s" % (n, n))
        return compare(runs[0], runs[1], w)
    if len(pos) != 1:
        die(USAGE.rstrip())
    import mdl

    from . import cli
    name = pos[0]
    models, binary = mdl.load_config()
    if name not in models:
        die("no model named %r in %s" % (name, mdl.CONFIG))
    target = cli.resolve(name)
    flags = target.flags
    seed = evalsuite.secret()
    try:
        items = evalsuite.build(pick_suites(o.get("suite")), seed,
                                limit=int(o["limit"]) if "limit" in o else None,
                                custom_dir=custom_dir())
    except ValueError as e:
        die(str(e))
    if not items:
        die("nothing to run")
    argv = mdl.build_argv(name, models[name], binary)
    binary = argv[0]                    # the build this model runs on
    thinking = thinking_flag(argv)
    mach = hw.probe(binary, quick=True, now=True)
    shape = model.Shape(target.inv)
    eff = calib.efficiency(target.inv.arch)
    build = (mach.build or {}).get("build")
    counts = {}
    for it in items:
        counts[it.suite] = counts.get(it.suite, 0) + 1
    w("model    %s · %s · kv %s · ctx %s · build %s\n" % (
        name, target.model_path.name, flags.kv_label, cli.kctx(flags.ctx),
        build or "?"))
    w("suites   %s   (suite v%d, item set %s/%s)\n" % (
        " · ".join("%s %d" % kv for kv in counts.items()),
        evalsuite.SUITE_VERSION, evalsuite.seed_id(seed),
        evalsuite.fingerprint(items)))
    seen = spent(file_hash(target.model_path),
                 version=evalsuite.SUITE_VERSION)
    fast = estimate(items, shape, flags, mach, eff, False, seen=seen)
    slow = estimate(items, shape, flags, mach, eff, True, seen=seen)
    if seen:
        est = "~%s, from what it spent on its last run here" % minutes(fast)
    else:
        est = ("~%s; it thinks, and replies are capped per suite"
               % minutes(slow) if thinking
               else "~%s" % minutes(fast) if thinking is False
               else "~%s, or ~%s if it thinks" % (minutes(fast),
                                                  minutes(slow)))
    w("estimate %s%s\n" % (est, "" if mach.calibrated
                           else " (speeds uncalibrated: mdl fit hw)"))
    if o.get("estimate"):
        return None
    env = Env(find_runtime() if o.get("sandbox") else None)
    if o.get("sandbox") and not env.runtime:
        die("--sandbox needs podman or docker on PATH")
    state = mdl.read_state(name)
    started = False
    if state:
        port = state["port"]
        w("server   %s is already running on port %d; using it\n" % (
            name, port))
    else:
        proc, log, port = mdl.spawn(name, models, binary,
                                    int(o["port"]) if "port" in o else None)
        w("server   starting %s on port %d...\n" % (name, port))
        out.flush()
        wait_ready(proc, port, name, log)
        started = True
    w("\n")
    t0, done, partial = time.time(), [], False
    progress = Progress(out)
    try:
        client = Client(port)
        cpt = chars_per_token(client)
        props = client.props()
        n_ctx = (props.get("default_generation_settings") or {}).get(
            "n_ctx") or flags.ctx
        run_items(client, items, env, cpt, n_ctx, done, progress)
        progress.close()
    except KeyboardInterrupt:
        progress.close()
        partial = True
        w("\ninterrupted; keeping the %d items done\n" % len(done))
    finally:
        if started:
            state = mdl.read_state(name)
            if state:
                mdl.stop_one(name, state)
    if not any("score" in r for r in done):
        die("no items finished; nothing saved")
    rec = {"at": time.strftime("%Y-%m-%d %H:%M"), "name": name,
           "model": str(target.model_path), "file": target.model_path.name,
           "hash": file_hash(target.model_path),
           "arch": target.inv.arch, "quant": target.inv.quant_label,
           "bpw": round(target.inv.bpw, 3), "params": target.inv.n_params,
           "kv": flags.kv_label, "ctx": flags.ctx, "build": build,
           "sampling": sampling_of(mdl.build_argv(name, models[name],
                                                  binary)),
           "suite_version": evalsuite.SUITE_VERSION,
           "grader_version": evalsuite.GRADER_VERSION,
           "seed_id": evalsuite.seed_id(seed),
           "items_hash": evalsuite.fingerprint(items),
           "suites": summarize(done, "suite"),
           "tiers": summarize(done, "tier"),
           "domains": summarize(done, "domain"),
           "thinking": any(r.get("thinking") for r in done),
           "minutes": round((time.time() - t0) / 60, 1),
           "partial": partial, "items": done}
    save(rec)
    if o.get("json"):
        w(json.dumps(rec, indent=1) + "\n")
        return rec
    report(rec, w)
    w("saved    %s\n" % results_path())
    return rec
