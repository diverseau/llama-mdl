"""What every model has served, kept while any server mdl started is up.

One recorder per state directory, started by `mdl run` (and by `mdl ui`)
and gone half a minute after the last server stops. It reads each running
server's /metrics and /slots every POLL_S seconds and books what moved
since the last reading, per model and per hour, in usage/<model>.json.
Deltas, not the counters: a server that restarts starts its counters at
zero, and a count that went down is a new server's first reading, taken
whole. Nothing here needs mdl ui to be open.

A request is a slot whose task changed between two readings. llama-server
counts no requests itself, and its task ids are shared with its own
bookkeeping (a /metrics read takes one), so this is a floor: two replies
on one slot inside one poll are one request here. Tokens and seconds are
llama-server's own and exact.

A file per model holds

    hours  {"<unix hour>": [prompt, predicted, prompt_s, predicted_s,
                            requests]}
    first, last   the first and latest hour anything moved
    load_s        how long its last load took, to estimate the next
"""

import json
import os
import sys
import time
from pathlib import Path

POLL_S = 2.0
LINGER_S = 30.0         # no server for this long: the recorder goes
HOUR = 3600
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


def delta(before, now):
    """What a counter moved: all of it when it went down (a new server)."""
    if before is None or now < before:
        return now
    return now - before


def read(state, cursor, get=None):
    """[prompt, predicted, prompt_s, predicted_s, requests] since the
    cursor, or None when the server has no metrics to read."""
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
    status, raw = get(state["port"], "/slots", 1.5, key)
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
        cur = cursors.setdefault(name, {}).setdefault(key, Cursor())
        try:
            moved = read(state, cur)
        except OSError:
            moved = None
        if moved is None:
            continue
        data = load(name)
        if book(data, now, moved):
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
