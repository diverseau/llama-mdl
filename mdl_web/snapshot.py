"""The one document the web UI draws: every model in the config, which of
them run, how fast and on what, and the GPUs - as `mdl snapshot` prints it.

A snapshot is a point in time. Speeds that need two points - tokens a
second while a reply streams, the tokens-over-time line on a model's card
- come from a Tracker, which the web server keeps between snapshots; a
one-shot `mdl snapshot` has none, and gives the server's own averages.
"""

import collections
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request

SCHEMA = 1
SERIES_POINTS = 720             # a model card's line: 12 minutes at 1 s
GPU_TTL = 2.0                   # nvidia-smi is ~100 ms; not every snapshot


# ---------------------------------------------------------------- http --

def http_get(port, path, timeout=1.5, key=None):
    """(status, body) from a server on 127.0.0.1, or (None, None) when it
    does not answer. `key` is its --api-key: llama-server asks for it on
    /metrics and /props, though not on /health."""
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path))
    if key:
        req.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, OSError, ValueError):
        return None, None


def http_json(port, path, timeout=1.5, key=None):
    status, body = http_get(port, path, timeout, key)
    if status != 200 or body is None:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


METRIC_LINE = re.compile(r"^([a-zA-Z_:][\w:]*)\s+([0-9.eE+-]+)$")


def parse_metrics(raw):
    """Prometheus text exposition -> {name: float}. Comments and labels
    ignored."""
    out = {}
    if not raw:
        return out
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = METRIC_LINE.match(line)
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def api_key(argv):
    """The --api-key a server was started with, if any."""
    argv = list(argv or [])
    for i, a in enumerate(argv):
        if a == "--api-key" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--api-key="):
            return a.split("=", 1)[1]
    return None


# ----------------------------------------------------------------- GPUs --

_GPU = {"at": -1e9, "value": []}


def gpus():
    """Every NVIDIA card: name, memory used and total in bytes, temperature
    and load. Other vendors are not read yet: an empty list, and the page
    draws no card line rather than a wrong one."""
    now = time.monotonic()
    if now - _GPU["at"] < GPU_TTL:
        return _GPU["value"]
    _GPU["at"] = now
    exe = shutil.which("nvidia-smi")
    out = []
    if exe:
        try:
            raw = subprocess.run(
                [exe, "--query-gpu=name,memory.used,memory.total,"
                 "temperature.gpu,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            ).stdout
            for line in raw.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 5:
                    continue

                def num(x):
                    try:
                        return float(x)
                    except ValueError:
                        return None
                used, total = num(parts[1]), num(parts[2])
                out.append({"name": parts[0],
                            "used": int(used * 2**20) if used else None,
                            "total": int(total * 2**20) if total else None,
                            "temp": num(parts[3]), "load": num(parts[4])})
        except (OSError, subprocess.SubprocessError):
            out = []
    _GPU["value"] = out
    return out


# -------------------------------------------------------------- tracker --

class Tracker:
    """What a live speed needs: the last reading of each server's counters,
    and the line of its tokens over time. Keyed by pid, so a restart under
    the same name starts a new line rather than drawing a cliff."""

    def __init__(self, points=SERIES_POINTS):
        self.points = points
        self.last = {}          # pid -> (time, decoded, prompt)
        self.series = {}        # pid -> deque of [time, tokens]
        self.props = {}         # pid -> /props, read once

    def observe(self, pid, now, metrics):
        """Tokens a second since the last reading, from n_decode_total -
        the one counter that moves while a reply streams (the others
        publish once the request ends). None the first time."""
        decoded = metrics.get("llamacpp:n_decode_total")
        prompt = metrics.get("llamacpp:prompt_tokens_total")
        rate = None
        prev = self.last.get(pid)
        if prev and decoded is not None and prev[1] is not None:
            gap = max(now - prev[0], 1e-3)
            rate = max(0.0, (decoded - prev[1]) / gap)
        self.last[pid] = (now, decoded, prompt)
        total = session_tokens(metrics)
        if total is not None:
            line = self.series.setdefault(
                pid, collections.deque(maxlen=self.points))
            line.append([round(now, 1), int(total)])
        return rate

    def forget(self, alive):
        """Drop what is kept for servers no longer running."""
        for table in (self.last, self.series, self.props):
            for pid in [p for p in table if p not in alive]:
                del table[pid]


def session_tokens(metrics):
    """Prompt and generated tokens since the server started."""
    p = metrics.get("llamacpp:prompt_tokens_total")
    g = metrics.get("llamacpp:tokens_predicted_total")
    if p is None and g is None:
        return None
    return (p or 0) + (g or 0)


def _avg(metrics, tokens, seconds):
    t, s = metrics.get(tokens), metrics.get(seconds)
    return round(t / s, 1) if t and s else None


# ------------------------------------------------------------- snapshot --

def _size(path):
    try:
        return os.path.getsize(path)
    except (OSError, TypeError, ValueError):
        return None


def _quant(path):
    from mdl_fit import catalog
    q = catalog.quant_of(str(path or ""))
    return None if q == "?" else q


def live(state, tracker):
    """A running server as the page shows it: loading or ready, and, once
    ready, what its counters say."""
    port, pid = state.get("port"), state.get("pid")
    key = api_key(state.get("argv"))
    status, _ = http_get(port, "/health", timeout=1)
    now = time.time()
    out = {"pid": pid, "port": port, "started": state.get("started"),
           "up": round(now - (state.get("started") or now)),
           "log": state.get("log"), "api_key": bool(key),
           "url": "http://127.0.0.1:%d/v1" % port if port else None}
    if status != 200:
        # 503 while the weights load; nothing at all before it listens
        out["state"] = "loading"
        return out
    out["state"] = "ready"
    if tracker is not None and pid not in tracker.props:
        tracker.props[pid] = http_json(port, "/props", key=key) or {}
    props = (tracker.props.get(pid) if tracker is not None
             else http_json(port, "/props", key=key)) or {}
    out["n_ctx"] = (props.get("default_generation_settings") or {}).get(
        "n_ctx")
    mstatus, raw = http_get(port, "/metrics", key=key)
    if mstatus != 200:
        # no --metrics in its args: running, but nothing to count with
        out["metrics"] = None
        return out
    m = parse_metrics(raw)
    rate = tracker.observe(pid, time.monotonic(), m) if tracker else None
    if rate is None:
        rate = m.get("llamacpp:predicted_tokens_seconds")
    out["metrics"] = {
        "tps": round(rate, 1) if rate is not None else None,
        "decode_avg": _avg(m, "llamacpp:tokens_predicted_total",
                           "llamacpp:tokens_predicted_seconds_total"),
        "prefill_avg": _avg(m, "llamacpp:prompt_tokens_total",
                            "llamacpp:prompt_seconds_total"),
        "tokens": session_tokens(m),
        "generated": m.get("llamacpp:tokens_predicted_total"),
        "busy": m.get("llamacpp:requests_processing"),
        "waiting": m.get("llamacpp:requests_deferred"),
        "kv": m.get("llamacpp:kv_cache_usage_ratio"),
    }
    if tracker is not None and pid in tracker.series:
        out["series"] = list(tracker.series[pid])
    return out


CFG_KEYS = ("ctx", "ngl", "n_cpu_moe", "kv_type", "flash_attn", "parallel",
            "port", "group")


def build(tracker=None):
    """The snapshot. A config mdl cannot read is an `error`, not a raise:
    the page says what is wrong rather than going blank."""
    import mdl
    from mdl_fit import hw
    total, avail = hw.ram()
    snap = {"schema": SCHEMA, "version": mdl.VERSION, "at": round(time.time()),
            "config": str(mdl.CONFIG), "gpus": gpus(),
            "ram": {"total": total, "free": avail}, "models": [],
            "error": None}
    try:
        models, _ = mdl.load_config()
    except mdl.MdlError as e:
        models, snap["error"] = {}, str(e)
    states = mdl.read_states()
    for name, cfg in models.items():
        path = cfg.get("model")
        m = {"name": name, "file": str(path) if path else None,
             "quant": _quant(path), "size": _size(path),
             "vision": bool(cfg.get("mmproj")), "state": "stopped",
             "own_server": bool(cfg.get("llama_server"))}
        m.update({k: cfg.get(k) for k in CFG_KEYS if cfg.get(k) is not None})
        m.setdefault("port", mdl.DEFAULT_PORT)
        if name in states:
            m["run"] = live(states[name], tracker)
            m["state"] = m["run"]["state"]
        snap["models"].append(m)
    for name, state in states.items():
        if name not in models:
            # running, but gone from the config since it started
            run = live(state, tracker)
            snap["models"].append({"name": name, "state": run["state"],
                                   "run": run, "unconfigured": True})
    if tracker is not None:
        tracker.forget({s.get("pid") for s in states.values()})
    return snap
