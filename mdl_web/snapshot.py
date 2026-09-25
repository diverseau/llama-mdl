"""The one document the web UI draws, as `mdl snapshot` prints it.

Its shape is the one omarchy-local-ai's panel reads, so the page's view
model can be his, line for line: `gpus` (the cards), `kinds` (the cards
grouped by model, each with the models it can run, best first - here,
every model in the config), `deployments` (what runs: loading, ready,
stopping, or a start that failed), `life` (tokens a day for 20 weeks),
`total` and `week`, and what an agent opens with.

Figures that need history - all-time averages, the token line, the
activity grid - come from usage.py's recorder. Everything else is read
now: the state files, each server's /health and /metrics, nvidia-smi.
"""

import datetime
import http.client
import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

SCHEMA = 2
GPU_TTL = 2.0                   # nvidia-smi is ~100 ms; not every snapshot
AGENTS_TTL = 30.0               # which agents are installed changes rarely
PICKS_TTL = 12 * 3600           # mdl find again after this, or a config change
PICKS = 6                       # find's picks offered beside your own models
TAKEN = 0.6                     # a card this full with nothing of ours on it


# ---------------------------------------------------------------- http --

def http_get(port, path, timeout=1.5, key=None):
    """(status, body) from a server on 127.0.0.1, or (None, None) when it
    does not answer. `key` is its --api-key: llama-server asks for it on
    /metrics, /props and /slots, though not on /health."""
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path))
    if key:
        req.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, OSError, ValueError,
            http.client.HTTPException):
        # HTTPException: a server stopped mid-reply, IncompleteRead
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
    """Every NVIDIA card as the panel reads one: key, name, memory in use
    (MiB) and in all (GB), temperature. Other vendors are not read yet."""
    now = time.monotonic()
    if now - _GPU["at"] < GPU_TTL:
        return _GPU["value"]
    _GPU["at"] = now
    exe = shutil.which("nvidia-smi")
    out = []
    if exe:
        try:
            raw = subprocess.run(
                [exe, "--query-gpu=index,name,memory.used,memory.total,"
                 "temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            ).stdout
            for line in raw.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 5:
                    continue
                used, total, temp = (_num(parts[2]), _num(parts[3]),
                                     _num(parts[4]))
                out.append({"key": parts[0], "name": _short(parts[1]),
                            "usedMiB": used,
                            "vramGb": round(total / 1024) if total else None,
                            "tempC": int(temp) if temp is not None else None})
        except (OSError, subprocess.SubprocessError):
            out = []
    _GPU["value"] = out
    return out


def _num(x):
    try:
        return float(x)
    except ValueError:
        return None


def _short(name):
    return re.sub(r"^(NVIDIA )?(GeForce )?", "", name).strip() or name


def machine():
    """The cards, or - with none mdl can read - the CPU and its memory as
    one, so a machine without a GPU still has somewhere to run."""
    found = gpus()
    if found:
        return found
    from mdl_fit import hw
    total, avail = hw.ram()
    used = (total - avail) / 2**20 if total and avail is not None else None
    return [{"key": "cpu", "name": "CPU", "usedMiB": used,
             "vramGb": round(total / 2**30) if total else None,
             "tempC": None, "cpu": True}]


# --------------------------------------------------------------- models --

def _files(path):
    from mdl_fit.manifest import shards
    try:
        return shards(str(path))
    except Exception:           # noqa: BLE001 - a missing file: no size
        return [str(path)]


def size_gb(cfg):
    total = 0
    for key in ("model", "mmproj"):
        if cfg.get(key):
            for f in _files(cfg[key]):
                try:
                    total += Path(f).stat().st_size
                except OSError:
                    pass
    return round(total / 2**30, 1) if total else 0


def quant(path):
    from mdl_fit import catalog
    q = catalog.quant_of(str(path or ""))
    return None if q == "?" else q


def family(name, path):
    """Which logo: only the ones the page ships."""
    text = "%s %s" % (name, Path(str(path or "")).name)
    if re.search(r"qwen|qwq", text, re.I):
        return "qwen"
    if re.search(r"lfm|liquid", text, re.I):
        return "lfm"
    return ""


HF_CACHE = re.compile(r"models--([^/\\]+)--([^/\\]+)[/\\]snapshots[/\\]"
                      r"([0-9a-f]{7,40})")


def weights(path):
    """The Hugging Face repository a file came from, when its path says
    so (the hub's cache, or `mdl pull`'s record beside it)."""
    p = str(path or "")
    m = HF_CACHE.search(p)
    if m:
        return [{"repository": "%s/%s" % (m.group(1), m.group(2)),
                 "revision": m.group(3)}]
    try:
        rec = json.loads(Path(p).with_name(".mdl-pull.json").read_text())
        if rec.get("repository"):
            return [{"repository": rec["repository"],
                     "revision": rec.get("revision") or "main"}]
    except (OSError, ValueError, AttributeError):
        pass
    return []


def recipe(name, cfg):
    """A configured model as the panel's recipe."""
    q = quant(cfg.get("model"))
    return {"id": name, "name": name, "family": family(name, cfg.get("model")),
            "format": "GGUF" + (" · " + q if q else ""),
            "ctx": cfg.get("ctx") or 0, "sizeGb": size_gb(cfg),
            "caps": {"vision": bool(cfg.get("mmproj"))},
            "weights": weights(cfg.get("model")), "cards": 1,
            "port": cfg.get("port"), "group": cfg.get("group")}


# ---------------------------------------------------------------- picks --

def picks_path():
    from . import agents
    return agents.ui_dir() / "picks.json"


def _config_stamp():
    import mdl
    try:
        return mdl.CONFIG.stat().st_mtime
    except OSError:
        return None


def refresh_picks(force=False):
    """Run `mdl find` for the page when its last run is stale: slow (it
    may fetch GGUF headers), so a thread's job, never a snapshot's."""
    import mdl
    try:
        old = json.loads(picks_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    stamp = _config_stamp()
    if (not force and old.get("config") == stamp
            and time.time() - (old.get("at") or 0) < PICKS_TTL):
        return False
    from . import proc
    raw = proc.output(["find", "--json"], timeout=900)
    try:
        found = json.loads(raw)
        rows = found.get("main", []) + found.get("explore", [])
    except (TypeError, ValueError, AttributeError):
        rows = old.get("rows", [])      # keep the last good ones
    picks_path().parent.mkdir(parents=True, exist_ok=True)
    mdl.write_atomic(picks_path(), json.dumps(
        {"at": time.time(), "config": stamp, "rows": rows}))
    return True


def picks(models):
    """find's picks on the Hub as the panel's recipes, leaving out any
    already pulled: Run on one downloads it first."""
    try:
        rows = json.loads(picks_path().read_text(encoding="utf-8"))["rows"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    have = {w["repository"] for cfg in models.values()
            for w in weights(cfg.get("model"))}
    out, seen = [], set()
    for r in rows:
        repo, f = r.get("repo"), r.get("file")
        if (not str(r.get("spec", "")).startswith("hf:") or not repo
                or not f or repo in have or repo in seen):
            continue
        seen.add(repo)
        # picks saved before the name reader knew a quant say "?"
        q = r.get("quant") if r.get("quant") not in (None, "?") else quant(f)
        # named for what lands on disk: the repo, not the model find
        # ranked it as (a fine-tune's parent, say)
        name = re.sub(r"[-_.]gguf", "", repo.split("/")[-1], flags=re.I)
        out.append({"id": "hf:%s:%s" % (repo, f), "name": name,
                    "family": family(name, f),
                    "format": "GGUF" + (" · " + q if q and q != "?" else ""),
                    "ctx": r.get("ctx") or 0,
                    "sizeGb": round((r.get("size") or 0) / 2**30, 1),
                    "caps": {"vision": False},
                    "weights": [{"repository": repo, "revision": "main"}],
                    "cards": 1, "port": None, "group": None, "pull": True})
    return out[:PICKS]


# ---------------------------------------------------------------- usage --

def _day(t):
    d = datetime.datetime.fromtimestamp(t)
    return "%s %d" % (d.strftime("%b"), d.day)


def _usage(names, now):
    from . import usage
    everything = usage.load_all()
    per = {n: usage.summarize(everything.get(n, {}), now) for n in names}
    for s in per.values():
        s["since"] = _day(s["first"]) if s["first"] else ""
    total = sum(usage.summarize(u, now)["tokens"] for u in everything.values())
    week_from = now - 7 * 86400
    week = sum(int(r[0] + r[1]) for u in everything.values()
               for h, r in u.get("hours", {}).items() if int(h) >= week_from)
    requests = sum(int(r[4]) for u in everything.values()
                   for r in u.get("hours", {}).values())
    firsts = [u.get("first") for u in everything.values() if u.get("first")]
    start, today, days = usage.days(everything, now)
    life = {"requests": requests, "days": days, "start": int(start),
            "today": today, "since": _day(min(firsts)) if firsts else ""}
    loads = {n: u.get("load_s") for n, u in everything.items()}
    return per, total, week, life, loads, everything


# ---------------------------------------------------------- deployments --

def deployment(name, state, cfg, keys, per, loads, own, now, history=None):
    """A running server as the panel's deployment."""
    port, key = state.get("port"), api_key(state.get("argv"))
    started = state.get("started") or now
    d = {"id": name, "name": name, "family": family(name, cfg.get("model")),
         "keys": keys, "port": port, "api_key": bool(key),
         "startedAt": datetime.datetime.fromtimestamp(
             started, datetime.timezone.utc).isoformat(),
         "agent": own["agent"], "folder": own["folder"],
         "shared": own.get("shared") if key else None,
         "format": "GGUF" + (" · " + quant(cfg.get("model"))
                             if quant(cfg.get("model")) else ""),
         "ctx": cfg.get("ctx") or 0, "caps": {"vision": bool(cfg.get("mmproj"))},
         "weights": weights(cfg.get("model")), "log": state.get("log"),
         "session": {"tokens": 0, "all": per.get(name) or {}}}
    # how fast this config runs as its context fills, from use and lab
    from . import usage
    ck = usage.config_key(state.get("argv"))
    d["session"]["speed"] = usage.speed(history or {}, ck)
    d["session"]["lab"] = usage.lab_points(ck)
    d["ctxMax"] = usage.ctx_of(state.get("argv")) or cfg.get("ctx") or 0
    status, _ = http_get(port, "/health", timeout=1)
    if status != 200:
        # 503 while the weights load; nothing at all before it listens
        d["state"], d["detail"] = "starting", "loading"
        took = loads.get(name)
        d["percent"] = (min(95, int((now - started) / took * 100))
                        if took else -1)
        return d
    d["state"] = "ready"
    status, raw = http_get(port, "/metrics", key=key)
    if status == 200:
        m = parse_metrics(raw)
        d["session"]["tokens"] = int(
            (m.get("llamacpp:prompt_tokens_total") or 0)
            + (m.get("llamacpp:tokens_predicted_total") or 0))
    else:
        d["metrics"] = False     # no --metrics: nothing to count with
    return d


def build(failed=None, stopping=None):
    """The snapshot. `failed` is {name: why} for starts from the page that
    did not come up; `stopping` the names being stopped. A config mdl
    cannot read is an `error`, not a raise: the page says what is wrong
    rather than going blank."""
    import mdl

    from . import agents
    failed, stopping = failed or {}, stopping or set()
    now = time.time()
    snap = {"schema": SCHEMA, "version": mdl.VERSION, "at": round(now),
            "config": str(mdl.CONFIG), "error": None}
    try:
        models, _ = mdl.load_config()
    except mdl.MdlError as e:
        models, snap["error"] = {}, str(e)
    states = mdl.read_states()
    cards = machine()
    everything = sorted(set(models) | set(states))
    per, total, week, life, loads, booked = _usage(everything, now)

    on_gpu = [g["key"] for g in cards if not g.get("cpu")]
    deps = []
    for name in everything:
        cfg = models.get(name, {})
        if name in states:
            ngl = cfg.get("ngl", 99)
            keys = on_gpu if on_gpu and ngl != 0 else [cards[0]["key"]]
            d = deployment(name, states[name], cfg, keys, per, loads,
                           agents.run_config(name, states[name]), now,
                           booked.get(name))
            if name in stopping:
                d["state"], d["detail"] = "stopping", "stopping"
            deps.append(d)
        elif name in failed:
            deps.append({"id": name, "name": name,
                         "family": family(name, cfg.get("model")),
                         "keys": on_gpu or [cards[0]["key"]], "state": "error",
                         "error": failed[name], "port": cfg.get("port"),
                         "session": {"tokens": 0, "all": per.get(name) or {}}})
    # downloads the page started: how far, or why one stopped
    from mdl_fit import pull
    for name, st in pull.read_all().items():
        if name in states or any(d["id"] == name for d in deps):
            continue
        known = {g["key"] for g in cards}
        keys = ([k for k in st.get("keys") or [] if k in known]
                or on_gpu or [cards[0]["key"]])
        d = {"id": name, "name": name,
             "family": family(name, st.get("repo")), "keys": keys,
             "port": None, "session": {"tokens": 0, "all": {}}}
        if st.get("state") == "error":
            d.update(state="error", error=st.get("error") or "stopped")
        else:
            d.update(state="download", detail=st.get("detail") or "starting",
                     percent=st.get("percent") or 0)
        deps.append(d)
    held = {k for d in deps if d["state"] != "error" for k in d["keys"]}

    # the configured models, best first: the one last run, then the rest
    # in the config's order
    last = {n: (per.get(n) or {}).get("last") or 0 for n in models}
    order = sorted(models, key=lambda n: -last[n])
    if not any(last.values()):
        order = list(models)
    recipes = [recipe(n, models[n]) for n in order] + picks(models)
    kinds = []
    for g in cards:
        kd = next((k for k in kinds if k["hw"] == g["name"]), None)
        if not kd:
            kd = {"hw": g["name"], "keys": [], "free": [], "taken": [],
                  "models": recipes, "groups": []}
            kinds.append(kd)
        kd["keys"].append(g["key"])
        if g["key"] in held:
            continue
        full = (g.get("usedMiB") or 0) / 1024 / (g.get("vramGb") or 1)
        if not g.get("cpu") and full > TAKEN:
            kd["taken"].append(g["key"])
        else:
            kd["free"].append(g["key"])
    for g in cards:
        g["busy"] = g["key"] in held
    if not recipes:
        kinds = []

    d = agents.defaults()
    snap.update({"gpus": cards, "kinds": kinds, "deployments": deps,
                 "total": total, "week": week, "life": life,
                 "agents": _installed(), "defaults": {
                     "agent": d["agent"], "folder": d["folder"]},
                 "folders": d["folders"], "tailnet": _tailnet(),
                 "home": str(Path.home())})
    return snap


_AGENTS = {"at": -1e9, "value": []}


def _installed():
    from . import agents
    now = time.monotonic()
    if now - _AGENTS["at"] > AGENTS_TTL:
        _AGENTS["at"], _AGENTS["value"] = now, agents.installed()
    return _AGENTS["value"]


def _tailnet():
    from . import tailnet
    return tailnet.available()
