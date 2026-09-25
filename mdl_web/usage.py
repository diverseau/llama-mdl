"""What every model has served, kept while any server mdl started is up.

One recorder per state directory, started by `mdl run` (and by `mdl ui`)
and gone half a minute after the last server stops. It reads each running
server's /metrics and /slots every POLL_S seconds and books what moved
since the last reading, per model and per hour, in usage/<model>.json.
Deltas, not the counters: a server that restarts starts its counters at
zero, and a count that went down is a new server's first reading, taken
whole. Nothing here needs mdl ui to be open.

Tokens and seconds are llama-server's own counters, and exact. Requests,
and how fast each one ran at the depth it ran at, come from the server's
log, which mdl keeps: llama-server logs every request's start, its
prompt as it is processed (a line per batch: how far, and how long so
far), its generation every few seconds (tokens so far, and the speed
over the last three seconds), its totals, and its release with how full
the context was then. That gives prefill and decode speed against
context depth from real use, and every request counted once. A server
with no log mdl can read has its requests counted from /slots instead:
a slot whose task changed between two readings, which is a floor.

A file per model holds

    hours  {"<unix hour>": [prompt, predicted, prompt_s, predicted_s,
                            requests]}
    first, last   the first and latest hour anything moved
    load_s        how long its last load took, to estimate the next
    speed  {"<config>": {"n_ctx": N,
                         "decode"|"prefill": {"<depth // 1024>":
                             [tokens, seconds, depth x seconds]}}}
           per config (config_key), since speed is the config's as much
           as the model's
    cursor where the recorder got to with the running server - its
           counters and how far into its log - so a recorder that
           restarts carries on rather than counting it all again
"""

import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

POLL_S = 2.0
LINGER_S = 30.0         # no server for this long: the recorder goes
HOUR = 3600
BUCKET = 1024                   # tokens of depth a speed bucket spans
LOG_CHUNK = 4 << 20             # at most this much log a reading
FIELDS = ("llamacpp:prompt_tokens_total", "llamacpp:tokens_predicted_total",
          "llamacpp:prompt_seconds_total",
          "llamacpp:tokens_predicted_seconds_total")


def usage_dir():
    import mdl
    return mdl.STATE_DIR / "usage"


def path_for(name):
    import mdl
    return usage_dir() / ("%s.json" % mdl.check_name(name))


def load(name):
    try:
        data = json.loads(path_for(name).read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("hours"), dict):
            return data
    except (OSError, ValueError):
        pass
    return {"hours": {}, "first": None, "last": None}


def save(name, data):
    import mdl
    usage_dir().mkdir(parents=True, exist_ok=True)
    mdl.write_atomic(path_for(name), json.dumps(data, separators=(",", ":")))


def load_all():
    """{model: usage} for every model with any."""
    out = {}
    try:
        files = sorted(usage_dir().glob("*.json"))
    except OSError:
        return out
    for f in files:
        out[f.stem] = load(f.stem)
    return out


def note_load(name, seconds):
    """How long a load took, for the next one's progress."""
    data = load(name)
    data["load_s"] = round(seconds, 1)
    save(name, data)


# ----------------------------------------------------------- the reading --

class Cursor:
    """The last reading of one server, so the next books only what moved."""

    def __init__(self):
        self.counters = None     # FIELDS values at the last reading
        self.tasks = {}          # slot id -> its task at the last reading
        self.log = LogReader()
        self.baseline = False    # the first reading sets the counters only

    def resume(self, saved, ident, booked_since=False):
        """Carry on from a file's cursor for this very server. With none,
        but history booked since this server started (by a recorder
        that kept no cursor), its first reading is a baseline: counting
        it would count this server's work twice."""
        if isinstance(saved, dict) and saved.get("id") == ident:
            c = saved.get("counters")
            if isinstance(c, list) and len(c) == len(FIELDS):
                self.counters = c
            self.log.offset = int(saved.get("log") or 0)
        elif booked_since:
            self.baseline = True

    def saved(self, ident):
        return {"id": ident, "counters": self.counters,
                "log": self.log.offset}


# ------------------------------------------------------------- the log --

TASK = re.compile(r"\| task (-?\d+) \|")
PROGRESS = re.compile(r"prompt processing, n_tokens =\s*(\d+),.*?"
                      r"t =\s*([\d.]+) s")
GEN = re.compile(r"n_gen =\s*(\d+),.*?tg_3s =\s*([\d.]+) t/s")
PROMPT_EVAL = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
EVAL = re.compile(r"\|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
RELEASE = re.compile(r"stop processing: n_tokens =\s*(\d+)")


class LogReader:
    """A server's log, read as it grows, one request at a time."""

    def __init__(self):
        self.offset = 0
        self.rest = b""
        self.open = {}           # task id -> what its lines said so far

    def read(self, path):
        """The requests that finished since the last read, as
        (prefill, decode) samples of (depth, tokens, seconds)."""
        try:
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                if fh.tell() < self.offset:          # a new log: from the top
                    self.offset, self.rest, self.open = 0, b"", {}
                fh.seek(self.offset)
                raw = fh.read(LOG_CHUNK)
        except OSError:
            return []
        self.offset += len(raw)
        lines = (self.rest + raw).split(b"\n")
        self.rest = lines.pop()
        done = []
        for line in lines:
            got = self.line(line.decode("utf-8", "replace"))
            if got:
                done.append(got)
        return done

    def line(self, text):
        m = TASK.search(text)
        if not m or m.group(1) == "-1":
            return None
        task = m.group(1)
        if "processing task" in text:
            self.open[task] = {"progress": [], "gen": []}
            return None
        t = self.open.get(task)
        if t is None:
            return None
        for rx, key in ((PROGRESS, "progress"), (GEN, "gen")):
            m = rx.search(text)
            if m:
                t[key].append((int(m.group(1)), float(m.group(2))))
                return None
        m = PROMPT_EVAL.search(text)
        if m:
            t["prompt"] = (float(m.group(1)) / 1000, int(m.group(2)))
            return None
        m = EVAL.search(text)
        if m:
            t["eval"] = (float(m.group(1)) / 1000, int(m.group(2)))
            return None
        m = RELEASE.search(text)
        if m:
            del self.open[task]
            return samples(t, int(m.group(1)))
        return None


def samples(t, final):
    """One finished request as (prefill, decode) samples of (depth,
    tokens, seconds): prefill batch by batch, decode three seconds at a
    time, each at the depth the context had reached."""
    p_s, p_n = t.get("prompt") or (0.0, 0)
    g_s, g_n = t.get("eval") or (0.0, 0)
    cached = max(0, final - p_n - g_n)
    prog = t["progress"]
    # a batch line counts from the start of the context on some builds
    # and from the end of the cached prefix on others
    shift = cached if prog and prog[-1][0] <= p_n + 16 else 0
    prefill, decode = [], []
    at, was = cached, 0.0
    for n, secs in prog:
        n += shift
        # a last few tokens on their own are the request's overhead,
        # not a speed at that depth
        if n - at >= 64 and secs > was:
            prefill.append(((at + n) / 2, n - at, secs - was))
        at, was = max(at, n), max(was, secs)
    end = cached + p_n
    if end - at >= 64 and p_s > was:    # the rest after the last line
        prefill.append(((at + end) / 2, end - at, p_s - was))
    for n, rate in t["gen"]:
        if rate > 0:
            decode.append((end + n, rate * 3.0, 3.0))
    if not t["gen"] and g_n and g_s > 0:
        decode.append((end + g_n / 2, g_n, g_s))
    return prefill, decode


def config_key(argv):
    """The config a server runs, as a short id: its command without the
    port and key, which do not change how fast it is."""
    argv, out, skip = list(argv or []), [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in ("--port", "--api-key"):
            skip = True
            continue
        if a.startswith(("--port=", "--api-key=")):
            continue
        out.append(str(a).replace(chr(92), "/"))
    return hashlib.sha1("\0".join(out).encode()).hexdigest()[:12]


def ctx_of(argv):
    argv = list(argv or [])
    for i, a in enumerate(argv[:-1]):
        if a in ("-c", "--ctx-size"):
            try:
                return int(argv[i + 1])
            except ValueError:
                return None
    return None


def book_speed(data, key, n_ctx, done):
    """Finished requests' samples into the config's depth buckets."""
    if not done:
        return False
    sp = data.setdefault("speed", {}).setdefault(key, {})
    if n_ctx:
        sp["n_ctx"] = n_ctx
    for pair in done:
        for kind, rows in zip(("prefill", "decode"), pair, strict=True):
            buckets = sp.setdefault(kind, {})
            for depth, tokens, secs in rows:
                b = buckets.setdefault(str(int(depth // BUCKET)), [0, 0.0, 0.0])
                b[0] = int(b[0] + tokens)
                b[1] = round(b[1] + secs, 3)
                b[2] = round(b[2] + depth * secs, 1)
    return True


def delta(before, now):
    """What a counter moved: all of it when it went down (a new server)."""
    if before is None or now < before:
        return now
    return now - before


def read(state, cursor, get=None, slots=True):
    """[prompt, predicted, prompt_s, predicted_s, requests] since the
    cursor, or None when the server has no metrics to read. Without
    `slots`, requests are left at 0 for the log to count."""
    from . import snapshot
    get = get or snapshot.http_get
    key = snapshot.api_key(state.get("argv"))
    status, raw = get(state["port"], "/metrics", 1.5, key)
    if status != 200:
        return None
    m = snapshot.parse_metrics(raw)
    now = [m.get(f) for f in FIELDS]
    if any(v is None for v in now):
        return None
    before = cursor.counters or [None] * len(FIELDS)
    moved = [delta(b, n) for b, n in zip(before, now, strict=True)]
    cursor.counters = now
    requests = 0
    status, raw = get(state["port"], "/slots", 1.5, key) if slots else (0, None)
    if status == 200:
        try:
            slots = json.loads(raw)
        except ValueError:
            slots = []
        for s in slots if isinstance(slots, list) else []:
            if not isinstance(s, dict) or s.get("id_task") in (None, -1):
                continue
            if cursor.tasks.get(s.get("id")) != s["id_task"]:
                requests += 1
            cursor.tasks[s.get("id")] = s["id_task"]
    return moved + [requests]


def book(data, at, moved):
    """Add a reading to the hour it fell in."""
    if not any(moved):
        return False
    hour = str(int(at // HOUR * HOUR))
    row = data["hours"].setdefault(hour, [0, 0, 0.0, 0.0, 0])
    for i, v in enumerate(moved):
        row[i] = round(row[i] + v, 3) if i in (2, 3) else int(row[i] + v)
    data["first"] = data.get("first") or int(hour)
    data["last"] = int(at)
    return True


def tick(cursors, now=None):
    """One pass over the running servers. Returns how many are up."""
    import mdl
    now = time.time() if now is None else now
    states = mdl.read_states(read_only=True)
    for name, state in states.items():
        key = (state.get("pid"), state.get("born"))
        ident = "%s:%s" % key
        mine = cursors.setdefault(name, {})
        data = load(name)
        if key not in mine:
            mine[key] = Cursor()
            mine[key].resume(data.get("cursor"), ident,
                             (data.get("last") or 0) >= (state.get("started")
                                                         or now))
        cur = mine[key]
        log = state.get("log")
        was = cur.saved(ident)
        try:
            moved = read(state, cur, slots=not log)
        except OSError:
            moved = None
        done = cur.log.read(log) if log else []
        changed = False
        if moved is not None:
            if log:
                moved[4] = len(done)          # every request, from the log
            if cur.baseline:
                moved = [0] * len(moved)      # its speeds still count
                cur.baseline = False
            changed = book(data, now, moved)
        changed |= book_speed(data, config_key(state.get("argv")),
                              ctx_of(state.get("argv")), done)
        if changed or cur.saved(ident) != was:
            data["cursor"] = cur.saved(ident)
            save(name, data)
    for name in [n for n in cursors if n not in states]:
        del cursors[name]
    return len(states)


# ----------------------------------------------------------- the process --

def main():
    """The recorder: one at a time, until the servers have been gone a
    while. Quiet: it has no terminal, and a failure costs only a reading."""
    import mdl
    lock = mdl.file_lock(usage_dir() / "recorder.lock", "busy", tries=1)
    try:
        lock.__enter__()
    except mdl.MdlError:
        return 0                 # one is running already
    cursors, idle_since = {}, None
    try:
        while True:
            try:
                up = tick(cursors)
            except Exception:    # noqa: BLE001 - keep recording the rest
                up = 1
            if up:
                idle_since = None
            else:
                idle_since = idle_since or time.monotonic()
                if time.monotonic() - idle_since > LINGER_S:
                    return 0
            if not mdl.STATE_DIR.is_dir():
                return 0         # its state went away (a test's, say)
            time.sleep(POLL_S)
    finally:
        lock.__exit__(None, None, None)


def ensure():
    """Start the recorder if it is not running. $MDL_RECORD=off stops it
    starting (the tests set it); it never fails a launch."""
    if os.environ.get("MDL_RECORD") == "off":
        return
    try:
        import subprocess

        import mdl
        lock = usage_dir() / "recorder.lock"
        try:
            holder = int(lock.read_text() or 0)
        except (OSError, ValueError):
            holder = 0
        if holder and mdl.alive(holder):
            return
        root = str(Path(__file__).resolve().parent.parent)
        code = ("import sys; sys.path.insert(0, %r); import mdl; "
                "mdl.STATE_DIR = __import__('pathlib').Path(%r); "
                "from mdl_web import usage; sys.exit(usage.main())"
                % (root, str(mdl.STATE_DIR)))
        flags = {}
        if os.name == "nt":
            flags["creationflags"] = (subprocess.CREATE_NO_WINDOW
                                      | subprocess.DETACHED_PROCESS
                                      | subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            flags["start_new_session"] = True
        subprocess.Popen([sys.executable, "-c", code],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, close_fds=True, **flags)
    except Exception:            # noqa: BLE001 - a launch never fails for it
        pass


# ------------------------------------------------------------ summaries --

def summarize(data, now=None, points=48):
    """One model's usage as the page shows it: totals, averages, and its
    tokens so far as a line from its first hour to now."""
    now = time.time() if now is None else now
    hours = sorted((int(h), row) for h, row in data.get("hours", {}).items())
    tot = [0, 0, 0.0, 0.0, 0]
    for _, row in hours:
        for i in range(5):
            tot[i] += row[i]
    out = {"tokens": int(tot[0] + tot[1]), "requests": int(tot[4]),
           "decode": round(tot[1] / tot[3]) if tot[3] > 0 else None,
           "prefill": round(tot[0] / tot[2]) if tot[2] > 0 else None,
           # the prompt's time is the wait before the first token
           "ttft": round(tot[2] / tot[4] * 1000) if tot[4] and tot[2] else None,
           "first": hours[0][0] if hours else None,
           "last": data.get("last"), "line": []}
    if hours:
        # an hour's tokens spread across it, the latest only up to the
        # last reading, so the line rises from nothing rather than
        # standing at its total from the first point
        last = max(data.get("last") or now, hours[-1][0] + 1)
        start, end = hours[0][0], max(now, last)
        step = (end - start) / (points - 1)
        line = []
        for p in range(points):
            t = start + p * step
            acc = 0.0
            for h, row in hours:
                span = max(min(HOUR, last - h), 1)
                acc += (row[0] + row[1]) * min(max((t - h) / span, 0), 1)
            line.append(int(acc))
        line[-1] = out["tokens"]
        out["line"] = line
    return out


def speed(data, key, points=24):
    """One config's speed against context depth as the page draws it:
    {n_ctx, decode, prefill}, each [[depth, tokens a second], ...] in
    about `points` steps across the context, a step shown once it has
    enough time behind it to mean something."""
    sp = (data.get("speed") or {}).get(key)
    if not sp:
        return None
    n_ctx = sp.get("n_ctx") or 0
    top = max([int(b) for k in ("decode", "prefill")
               for b in sp.get(k, {})] + [0]) + 1
    width = max(1, math.ceil(max(n_ctx // BUCKET, top) / points))
    out = {"n_ctx": n_ctx}
    for kind, floor in (("decode", 2.0), ("prefill", 0.5)):
        steps = {}
        for b, (tokens, secs, weighted) in sp.get(kind, {}).items():
            acc = steps.setdefault(int(b) // width, [0, 0.0, 0.0])
            acc[0] += tokens
            acc[1] += secs
            acc[2] += weighted
        out[kind] = [[int(w / sec), round(tok / sec, 1)]
                     for _, (tok, sec, w) in sorted(steps.items())
                     if sec >= floor]
    return out


_LAB = {"key": None, "value": {}}


def lab_points(key):
    """What `mdl lab` measured for this very config: its median decode
    and prefill at each depth it ran, to set beside what use shows."""
    try:
        from mdl_fit import lab
        path = lab.lab_dir() / "records.jsonl"
        stamp = (path.stat().st_mtime, path.stat().st_size)
    except (OSError, ImportError, AttributeError):
        return None
    if _LAB["key"] != stamp:
        by = {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                r = json.loads(line)
                if r.get("warmup"):
                    continue
                k = config_key(r["argv"])
                depth = int(r["workload"]["prompt_tokens"])
                m = r["metrics"]
                row = by.setdefault(k, {}).setdefault(depth, [[], []])
                if m.get("decode", {}).get("p50"):
                    row[0].append((m["decode"]["p50"], m.get("tokens") or 0))
                if m.get("prefill", {}).get("tps"):
                    row[1].append(m["prefill"]["tps"])
            except (ValueError, KeyError, TypeError):
                continue
        _LAB["key"], _LAB["value"] = stamp, by
    mine = _LAB["value"].get(key)
    if not mine:
        return None

    def mid(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    return {"decode": [[d + int(mid([t for _, t in dec]) / 2),
                        round(mid([v for v, _ in dec]), 1)]
                       for d, (dec, _) in sorted(mine.items()) if dec],
            "prefill": [[d // 2, round(mid(pre), 1)]
                        for d, (_, pre) in sorted(mine.items()) if pre]}


def days(all_usage, now=None, weeks=20):
    """Tokens a day across every model: a column a week, Monday first,
    `weeks` of them ending with this one. (start, today's index, days)."""
    now = time.time() if now is None else now
    local = time.localtime(now)
    midnight = time.mktime((local.tm_year, local.tm_mon, local.tm_mday,
                            0, 0, 0, 0, 0, -1))
    back = local.tm_wday                    # days since Monday
    start = _shift_days(midnight, -(back + 7 * (weeks - 1)))
    cells = [0] * (7 * weeks)
    today = back + 7 * (weeks - 1)
    for data in all_usage.values():
        for h, row in data.get("hours", {}).items():
            i = _day_index(start, int(h))
            if 0 <= i < len(cells):
                cells[i] += int(row[0] + row[1])
    return start, today, cells


def _shift_days(t, n):
    lt = time.localtime(t)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday + n, 0, 0, 0, 0, 0,
                        -1))


def _day_index(start, t):
    """Days from local midnight `start` to the day `t` falls in, by the
    calendar, so a daylight-saving change does not move an hour's day."""
    a, b = time.localtime(start), time.localtime(t)
    da = time.mktime((a.tm_year, a.tm_mon, a.tm_mday, 12, 0, 0, 0, 0, -1))
    db = time.mktime((b.tm_year, b.tm_mon, b.tm_mday, 12, 0, 0, 0, 0, -1))
    return int(round((db - da) / 86400))


if __name__ == "__main__":
    sys.exit(main())
