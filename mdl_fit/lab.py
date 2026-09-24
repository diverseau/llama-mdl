"""mdl lab - the same prompt through N configs, measured, kept as records.

A parameter sweep used to live in the names of models.toml backups; the
numbers lived nowhere. This runs one workload - a prompt, a context
depth, a reply length, fixed sampling - against each variant of a
model's config in turn, and keeps what each did: time to first token,
decode speed and how steady it was, prefill, and what the machine paid
for it in VRAM, RAM and CPU, sampled against the token count so "usage
500 tokens in" is a lookup, not a guess.

What makes the numbers comparable (docs/mdl-lab.md, section 8):

- Each variant runs from a config written for it alone, in a temp dir,
  through `mdl run` and `mdl stop` with XDG_CONFIG_HOME and
  XDG_STATE_HOME pointed there. models.toml, the state dir and the logs
  are never touched, and a server of yours on the port is never stopped:
  the run refuses to start instead.
- Serial, always: one server, one port, one request at a time.
- Sampling is the workload's, sent with every request, so presets that
  pin --temp in their args are measured on the same terms. Replies run
  to max_tokens (ignore_eos), and the prompt cache is off, so every
  repetition prefills.
- A warmup repetition is run and not counted; the rest are reported as
  a median with their spread, and two variants whose spreads overlap are
  called indistinguishable rather than ranked.
- The build (number and commit) and backend go in every record, and a
  comparison across builds says it is one.

Standard library only, like the rest of mdl_fit: VRAM, GPU load, clocks
and temperature come from nvidia-smi (a second or so per sample, not
NVML's millisecond), RAM from the process table, CPU from the OS. With
no nvidia-smi, the only VRAM figure is what the server's own load log
claims, and the record says so.

Records are one JSON line each in <state>/lab/records.jsonl; the samples
behind one go beside it in samples/<id>.jsonl.
"""

import hashlib
import http.client
import itertools
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from pathlib import Path

from . import calib, evalsuite, hw, usage

SCHEMA = 1
MiB, GiB = 1 << 20, 1 << 30
RAM_SPARE = int(2.5 * GiB)      # free RAM a variant must leave, per the design
VRAM_SLACK = 512 * MiB          # what "back to baseline" allows after a stop
WINDOW = 1.0                    # seconds: the sliding window a rate is over
SKIP = 0.02                     # share of tokens left out of floor and peak
BASELINE_S = 3.0                # seconds of idle sampled before a variant
DEPTHS = [0, "100%"]            # empty, and full: where configs part ways
TREND = 0.10                    # the last third of a reply against the first
DRIFT = 0.05                    # rep over rep, first to last
# tokens left free at the top of a "full" context for the chat template's
# wrapping, which /tokenize does not count: a choice, with room to spare
TEMPLATE_SLACK = 64

DEFAULT_PROMPT = (
    "Write a practical, detailed guide to keeping a small vegetable garden "
    "through its first year: choosing the site, preparing the soil, what to "
    "plant in each season and why, watering, pests, and what to do with the "
    "harvest. Use headings and explain your reasoning as you go.")

USAGE = """usage: mdl lab run <name>... [options]     run and record
       mdl lab run --suite <name|file.toml>
       mdl lab report [run] [--format table|md|csv|json] [--at N]
       mdl lab compare <a> <b> [--run R]     two variants, deltas, verdict
       mdl lab ls                            runs recorded
       mdl lab export <record|run>           as JSON
       mdl lab apply <variant|record>        its models.toml table, printed
       mdl lab baseline [set [run] | diff [run] [--fail]]
                                             pin a run; what moved since

run options:
  --set KEY=V[,V...]   a config override; a list sweeps it (repeatable,
                       every combination is a variant)
  --server PATH[,...]  llama-server builds to run each variant on
  --prompt FILE|-      the workload (default: a long-form writing prompt)
  --depth 0,50%,100%   context filled before the prompt: tokens (8k), or
                       a share of each variant's own context, where 100%
                       (or full) leaves just room for the reply
                       (default 0,100%: empty, and full)
  --max-tokens N       reply length, run to the end (default 512)
  --reps N             measured repetitions (default 3)
  --warmup N           repetitions run first and not counted (default 1)
  --at N               the token count usage is snapshotted at (default 500)
  --seed N  --temp T   sampling, the same for every variant (42, 0)
  --cooldown S         seconds between variants (default 10)
  --interval S         seconds between system samples (default 1)
  --dry-run            the matrix and a time estimate; runs nothing

A suite file (~/.config/mdl/lab/<name>.toml) holds the same as a
[suite] table of options and [[variant]] tables of base, label, set
and server. Records: <state>/lab/records.jsonl.
"""


def die(msg):
    import mdl
    mdl.die(msg)


def lab_dir():
    import mdl
    return mdl.STATE_DIR / "lab"


def records_path():
    return lab_dir() / "records.jsonl"


# --------------------------------------------------------------- parse --

def number(text, what):
    """8k, 32K, 4096 -> an int; one line if it is not one."""
    t = str(text).strip().lower()
    mult = 1024 if t.endswith("k") else 1
    try:
        n = int(float(t[:-1] if mult > 1 else t) * mult)
    except ValueError:
        die("%s takes a number, not %r" % (what, text))
    if n < 0:
        die("%s cannot be negative" % what)
    return n


def depth_spec(text, what="--depth"):
    """A depth: tokens (8k is 8192), or a share of the variant's context,
    kept as text ("50%") and resolved when the server says how much it
    has. "full" is 100%."""
    t = str(text).strip().lower()
    if t == "full":
        return "100%"
    if t.endswith("%"):
        try:
            p = float(t[:-1])
        except ValueError:
            die("%s takes tokens or a share like 50%%, not %r" % (what, text))
        if not 0 <= p <= 100:
            die("%s takes a share from 0%% to 100%%, not %r" % (what, text))
        return "%g%%" % p if p else 0
    return number(text, what)


def resolve_depth(spec, ctx, w, prompt_tokens):
    """Tokens of filler for a depth: a share is of the room left once the
    prompt, the reply and the template's wrapping are counted, so 100%
    ends the reply at the top of the context."""
    if not isinstance(spec, str):
        return spec
    room = ctx - w["max_tokens"] - prompt_tokens - TEMPLATE_SLACK
    return max(0, int(room * float(spec[:-1]) / 100))


def _dkey(d):
    """Order depths: token counts, then shares."""
    return (1, float(d[:-1]), "") if isinstance(d, str) else (0, d or 0, "")


def value(text):
    """A --set value as TOML reads it: 45, true, "q8_0", ["--metrics"],
    and a size as --depth does: 8k is 8192. A bare word that is not TOML
    is taken as a string."""
    if re.fullmatch(r"\d+(\.\d+)?[kK]", text.strip()):
        return number(text, "--set")
    try:
        return tomllib.loads("v = " + text)["v"]
    except tomllib.TOMLDecodeError:
        return text


def split_values(text):
    """V1,V2 at the top level only: a list value keeps its commas."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [v.strip() for v in out if v.strip()]


INT_OPTS = {"--max-tokens": "max_tokens", "--reps": "reps",
            "--warmup": "warmup", "--at": "at", "--seed": "seed"}
FLOAT_OPTS = {"--temp": "temp", "--cooldown": "cooldown",
              "--interval": "interval"}


def parse(args):
    o = {"names": [], "set": [], "servers": []}
    i = 0
    while i < len(args):
        a = args[i]
        takes = a in ("--set", "--server", "--prompt", "--depth", "--suite",
                      "--format", "--run") or a in INT_OPTS or a in FLOAT_OPTS
        if takes:
            if i + 1 >= len(args):
                die("%s needs a value" % a)
            v = args[i + 1]
            i += 2
            if a == "--set":
                key, eq, raw = v.partition("=")
                if not eq or not key.strip():
                    die("--set takes KEY=VALUE, e.g. --set ngl=45,38")
                o["set"].append((key.strip(), [value(x)
                                               for x in split_values(raw)]))
            elif a == "--server":
                o["servers"] += split_values(v)
            elif a == "--depth":
                o["depths"] = [depth_spec(x) for x in split_values(v)]
            elif a in INT_OPTS:
                o[INT_OPTS[a]] = number(v, a)
            elif a in FLOAT_OPTS:
                try:
                    o[FLOAT_OPTS[a]] = float(v)
                except ValueError:
                    die("%s takes a number, not %r" % (a, v))
                if o[FLOAT_OPTS[a]] < 0:
                    die("%s cannot be negative" % a)
            else:
                o[a[2:]] = v
        elif a in ("--dry-run", "--all", "--fail"):
            o[a[2:].replace("-", "_")] = True
            i += 1
        elif a.startswith("-") and a != "-":
            die("unknown option %s; mdl lab --help lists them" % a)
        else:
            o["names"].append(a)
            i += 1
    return o


# --------------------------------------------------------------- suite --

class Variant:
    def __init__(self, base, overrides, server=None, label=None):
        self.base, self.set, self.server = base, dict(overrides), server
        self.label = label or self.auto_label()

    def auto_label(self):
        bits = [self.base] + ["%s%s" % (k, _short(v))
                              for k, v in self.set.items()]
        if self.server:
            bits.append(Path(self.server).parent.name or Path(self.server).name)
        return "/".join(bits)

    def config(self, models):
        cfg = dict(models[self.base])
        cfg.update(self.set)
        if self.server:
            cfg["llama_server"] = self.server
        return cfg

    def as_dict(self):
        return {"label": self.label, "base": self.base, "set": self.set,
                "server": self.server}


def _short(v):
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, list):
        return "[%d]" % len(v)
    return str(v)


def suite_path(name):
    p = Path(name)
    if p.suffix == ".toml" and p.is_file():
        return p
    q = hw.config_dir() / "lab" / (name + ".toml")
    if q.is_file():
        return q
    die("no suite %r: not a .toml file, nor %s" % (name, q))


def load_suite(name, o):
    """Options and variants from a suite file; the command line wins."""
    path = suite_path(name)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        die("%s: %s" % (path, e))
    head = data.get("suite", {})
    for key in ("prompt", "prompt_file", "max_tokens", "reps", "warmup", "at",
                "seed", "temp", "cooldown", "interval"):
        if key in head and key not in o:
            o[key] = head[key]
    if "depth" in head and "depths" not in o:
        o["depths"] = [depth_spec(d, "depth") for d in head["depth"]] \
            if isinstance(head["depth"], list) else [depth_spec(head["depth"],
                                                                "depth")]
    variants = []
    for i, v in enumerate(data.get("variant", [])):
        if not isinstance(v, dict) or not v.get("base"):
            die("%s: variant %d needs a base (a name in models.toml)"
                % (path, i + 1))
        if not isinstance(v.get("set", {}), dict):
            die("%s: variant %d: set is a table of keys" % (path, i + 1))
        variants.append(Variant(v["base"], v.get("set", {}), v.get("server"),
                                v.get("label")))
    if not variants:
        die("%s has no [[variant]] tables" % path)
    o["suite"] = head.get("name", path.stem)
    return variants


def matrix(o):
    """Every combination of the names, the --set values and the servers."""
    keys = [k for k, _ in o["set"]]
    combos = list(itertools.product(*[vals for _, vals in o["set"]])) or [()]
    servers = o["servers"] or [None]
    out = []
    for name in o["names"]:
        for combo in combos:
            for server in servers:
                out.append(Variant(name, dict(zip(keys, combo, strict=True)),
                                   server))
    return out


def workload(o):
    if o.get("prompt_file"):
        o["prompt"] = "@" + str(o["prompt_file"])
    raw = o.get("prompt")
    if raw == "-":
        text = sys.stdin.read()
    elif raw and raw.startswith("@"):
        text = _read(raw[1:])
    elif raw and Path(raw).is_file():
        text = _read(raw)
    else:
        text = raw or DEFAULT_PROMPT
    if not text.strip():
        die("the prompt is empty")
    w = {"prompt": text, "prompt_sha": hashlib.sha256(
             text.encode()).hexdigest()[:12],
         "max_tokens": o.get("max_tokens", 512), "seed": o.get("seed", 42),
         "temp": o.get("temp", 0.0), "depths": o.get("depths", DEPTHS),
         "reps": o.get("reps", 3), "warmup": o.get("warmup", 1),
         "at": o.get("at", 500), "cooldown": o.get("cooldown", 10.0),
         "interval": o.get("interval", 1.0), "ignore_eos": True}
    if w["reps"] < 1:
        die("--reps must be 1 or more")
    if w["max_tokens"] < 1:
        die("--max-tokens must be 1 or more")
    return w


def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as e:
        die("cannot read the prompt: %s" % e)


# ------------------------------------------------------------- machine --

def gpu_now():
    """(VRAM used, GPU %, °C, SM MHz) of the first NVIDIA card, or Nones.
    Total-device only: on Windows the driver keeps no per-process count."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None, None, None
    out = hw._run([exe, "--query-gpu=memory.used,utilization.gpu,"
                   "temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"],
                  timeout=8)
    line = (out.strip().splitlines() or [""])[0]
    vals = []
    for x in line.split(","):
        try:
            vals.append(float(x))
        except ValueError:
            vals.append(None)
    vals += [None] * (4 - len(vals))
    used = int(vals[0] * MiB) if vals[0] is not None else None
    return used, vals[1], vals[2], vals[3]


def tree_rss(pid):
    """RAM held by the server and everything it started: a wrapper script
    leaves the real server as its child."""
    procs, _ = usage.processes(quick=True)
    kids = {}
    for p in procs:
        kids.setdefault(p.ppid, []).append(p)
    total, todo, seen = 0, [pid], set()
    by_pid = {p.pid: p for p in procs}
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        if n in by_pid:
            total += by_pid[n].ram
        todo += [k.pid for k in kids.get(n, [])]
    return total if seen & set(by_pid) else None


class Sampler(threading.Thread):
    """The machine every `interval` seconds while a server runs, each
    sample tagged with the phase and the tokens generated so far - and
    one more whenever `poke` asks, so a decode shorter than an interval
    (a fast card, a short reply) still has samples of its own."""

    def __init__(self, pid, interval, clock=time.perf_counter):
        super().__init__(daemon=True)
        self.pid, self.interval, self.clock = pid, interval, clock
        self.t0 = clock()
        self.phase, self.tok, self.rep, self.depth = "ready", 0, None, None
        self.samples, self.halt = [], threading.Event()
        self.cond = threading.Condition()
        self.pending, self.busy = [], False

    def tag(self):
        return {"phase": self.phase, "rep": self.rep, "depth": self.depth,
                "tok": self.tok}

    def take(self, tag=None):
        tag = tag or self.tag()
        vram, util, temp, clock_mhz = gpu_now()
        total, avail = hw.ram()
        cpu = usage.cpu_load(0.2)
        return dict(tag, t=round(self.clock() - self.t0, 3),
                    vram=vram, gpu_util=util, temp=temp, clock=clock_mhz,
                    rss=tree_rss(self.pid),
                    ram_used=(total - avail) if total and avail else None,
                    cpu=round(cpu * 100, 1) if cpu is not None else None)

    def poke(self):
        """A sample as things stand now, taken on this thread rather than
        the caller's: the stream reader must not wait on nvidia-smi, or
        the gap it leaves is read as the model being slow. The tag is
        the moment asked for; the reading follows it within a sample's
        time, while the server still holds what it held."""
        with self.cond:
            self.pending.append(self.tag())
            self.cond.notify_all()

    def flush(self, timeout=30):
        """Wait for every poked sample to be taken."""
        with self.cond:
            self.cond.wait_for(lambda: not self.pending and not self.busy,
                               timeout)

    def run(self):
        due = self.clock()
        while not self.halt.is_set():
            with self.cond:
                self.cond.wait_for(
                    lambda d=due: self.pending or self.halt.is_set()
                    or self.clock() >= d, max(0.0, due - self.clock()))
                if self.halt.is_set():
                    break
                tag = self.pending.pop(0) if self.pending else None
                self.busy = True
            try:
                self.samples.append(self.take(tag))
            except Exception:           # noqa: BLE001 - never kill a run
                pass
            finally:
                with self.cond:
                    self.busy = False
                    self.cond.notify_all()
            if tag is None:
                due = self.clock() + self.interval

    def stop(self):
        self.halt.set()
        with self.cond:
            self.cond.notify_all()
        self.join(timeout=30)


def baseline(seconds=None, interval=1.0):
    """VRAM and RAM in use with nothing of ours running: what the deltas
    are against."""
    vram, ram = [], []
    end = time.monotonic() + (BASELINE_S if seconds is None else seconds)
    while True:
        v = gpu_now()[0]
        total, avail = hw.ram()
        if v is not None:
            vram.append(v)
        if total and avail:
            ram.append(total - avail)
        if time.monotonic() >= end:
            break
        time.sleep(interval)
    return {"vram": statistics.median(vram) if vram else None,
            "ram": statistics.median(ram) if ram else None}


# -------------------------------------------------------------- client --

def stream(port, text, w, seed, on_token, timeout=900):
    """One streamed reply: (arrival times from the request, in seconds,
    one per chunk that carried text; the server's timings, or {})."""
    body = {"messages": [{"role": "user", "content": text}], "stream": True,
            "max_tokens": w["max_tokens"], "seed": seed,
            "temperature": w["temp"], "ignore_eos": w["ignore_eos"],
            # every repetition prefills: a cached prompt would make the
            # second one's time to first token a cache hit, not a measure
            "cache_prompt": False}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    arrivals, timings = [], {}
    try:
        t0 = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", json.dumps(body),
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            detail = resp.read()[:200].decode("utf-8", "replace")
            raise OSError("HTTP %d %s" % (resp.status, detail))
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            now = time.perf_counter() - t0
            choice = (chunk.get("choices") or [{}])[0] or {}
            delta = choice.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                arrivals.append(now)
                on_token(len(arrivals))
            if isinstance(chunk.get("timings"), dict):
                timings = chunk["timings"]
    finally:
        conn.close()
    return arrivals, timings


# ------------------------------------------------------------- metrics --

def windowed(arrivals, window=WINDOW, skip=SKIP):
    """Tokens per second over a sliding `window`, one value per token
    once a whole window has passed, the first `skip` share left out: the
    fastest or slowest single gap is a scheduler hiccup, not a speed."""
    if len(arrivals) < 3:
        return []
    start = arrivals[int(len(arrivals) * skip)]
    rates, lo = [], 0
    for i, t in enumerate(arrivals):
        if t - window < start:
            continue
        while arrivals[lo] < t - window:
            lo += 1
        # gaps over the time they spanned: a count of tokens in a fixed
        # window is off by one whenever a token sits on its edge
        if t > arrivals[lo]:
            rates.append((i - lo) / (t - arrivals[lo]))
    return rates


def quantile(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _r(x, n=2):
    return round(x, n) if isinstance(x, (int, float)) else None


def _rate(seg):
    """Tokens per second across a run of arrivals."""
    if len(seg) < 2 or seg[-1] <= seg[0]:
        return None
    return (len(seg) - 1) / (seg[-1] - seg[0])


def rep_metrics(arrivals, timings, samples, at, base):
    """What one repetition did, from its token arrivals, the server's own
    timings, and the samples taken while it ran."""
    m, flags = {}, []
    n = len(arrivals)
    if n:
        m["ttft"] = _r(arrivals[0], 3)
        m["tokens"] = n
    wall = arrivals[-1] - arrivals[0] if n > 1 else 0
    avg = (n - 1) / wall if wall > 0 else None
    rates = windowed(arrivals)
    m["decode"] = {"avg": _r(avg), "p05": _r(quantile(rates, 0.05)),
                   "p50": _r(quantile(rates, 0.5)),
                   "p95": _r(quantile(rates, 0.95)),
                   "floor": _r(min(rates)) if rates else None,
                   "peak": _r(max(rates)) if rates else None,
                   "cv": _r(statistics.pstdev(rates) / statistics.mean(rates),
                            3) if len(rates) > 1 and statistics.mean(rates)
                   else None}
    m["prefill"] = {"tps": _r(timings.get("prompt_per_second"), 1),
                    "tokens": timings.get("prompt_n")}
    m["time_to_1k"] = _r(arrivals[999], 2) if n >= 1000 else None
    served = timings.get("predicted_per_second")
    m["server_decode"] = _r(served)
    if served and avg and abs(served / avg - 1) > 0.05:
        # the client's clock and the server's disagree: both are kept,
        # and the table says so rather than picking one
        flags.append("timings_disagree")
    if rates and len(rates) > 1 and m["decode"]["cv"] and \
            m["decode"]["cv"] > 0.15:
        flags.append("high_variance")
    # a model that gathers speed as it goes: the last third of the reply
    # against the first, the first tokens left out as the floor's are
    k = int(n * SKIP)
    if n - k >= 30:
        third = (n - k) // 3
        ra = _rate(arrivals[k:k + third])
        rb = _rate(arrivals[n - third:])
        if ra and rb:
            m["decode"]["trend"] = _r(rb / ra - 1, 3)
            if rb / ra - 1 > TREND:
                flags.append("warming_up")
            elif rb / ra - 1 < -TREND:
                flags.append("slowing")
    dec = [s for s in samples if s["phase"] == "decode"]
    vram = [s["vram"] for s in dec if s["vram"] is not None]
    rss = [s["rss"] for s in dec if s["rss"] is not None]
    ram = [s["ram_used"] for s in dec if s["ram_used"] is not None]
    cpu = [s["cpu"] for s in dec if s["cpu"] is not None]
    util = [s["gpu_util"] for s in dec if s["gpu_util"] is not None]
    temp = [s["temp"] for s in dec if s["temp"] is not None]
    clocks = [s["clock"] for s in dec if s["clock"] is not None]
    m["vram"] = {"peak": max(vram) if vram else None,
                 "steady": int(statistics.median(vram)) if vram else None,
                 "delta": (max(vram) - base["vram"]) if vram
                 and base.get("vram") is not None else None}
    m["ram"] = {"peak_rss": max(rss) if rss else None,
                "delta": (max(ram) - base["ram"]) if ram
                and base.get("ram") is not None else None}
    m["cpu"] = {"mean": _r(statistics.mean(cpu), 1) if cpu else None,
                "peak": max(cpu) if cpu else None}
    m["gpu"] = {"util_mean": _r(statistics.mean(util), 1) if util else None,
                "temp_peak": max(temp) if temp else None,
                "clock_min": min(clocks) if clocks else None}
    top = clocks.index(max(clocks)) if clocks else 0
    if clocks and min(clocks[top:]) < 0.85 * clocks[top]:
        # a card that throttled is a slow run, not a slow config. Only a
        # fall from its peak: the first decode sample can catch the clock
        # still climbing out of idle, and that is not a throttle
        flags.append("clock_drop")
    m["at"] = {str(at): snapshot_at(samples, at)}
    return m, flags


def snapshot_at(samples, tok):
    """The first decode sample taken at or after `tok` tokens."""
    for s in samples:
        if s["phase"] == "decode" and s["tok"] >= tok:
            return {"tok": s["tok"], "vram": s["vram"], "rss": s["rss"],
                    "cpu": s["cpu"]}
    return None


# --------------------------------------------------------------- store --

def new_id(prefix=""):
    return prefix + time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(2).hex()


def append(rec, samples=None):
    lab_dir().mkdir(parents=True, exist_ok=True)
    with open(records_path(), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    if samples:
        d = lab_dir() / "samples"
        d.mkdir(exist_ok=True)
        (d / (rec["id"] + ".jsonl")).write_text(
            "".join(json.dumps(s) + "\n" for s in samples), encoding="utf-8")


def load():
    try:
        lines = records_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue                    # a line a crash cut off
        if isinstance(r, dict) and r.get("schema") == SCHEMA:
            out.append(r)
    return out


def samples_of(rec):
    try:
        text = (lab_dir() / "samples" / (rec["id"] + ".jsonl")).read_text(
            encoding="utf-8")
    except OSError:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# -------------------------------------------------------------- runner --

class Env:
    """A config and state dir of the variant's own, and `mdl run` / `mdl
    stop` pointed at them."""

    def __init__(self, variant, models, binary, top):
        import mdl
        self.variant = variant
        self.cfg = variant.config(models)
        mdl.check_cfg(variant.base, self.cfg)
        self.root = Path(tempfile.mkdtemp(prefix="mdl-lab-"))
        conf = self.root / "config" / "mdl"
        conf.mkdir(parents=True)
        lines = ["llama_server = %s" % mdl.toml_value(binary)]
        for key in ("ready_timeout",):
            if key in top:
                lines.append("%s = %s" % (key, mdl.toml_value(top[key])))
        lines += ["", "[%s]" % mdl.toml_key(variant.base)]
        lines += ["%s = %s" % (k, mdl.toml_value(v))
                  for k, v in self.cfg.items()]
        (conf / "models.toml").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")
        self.state = self.root / "state" / "mdl"
        self.env = dict(os.environ, XDG_CONFIG_HOME=str(self.root / "config"),
                        XDG_STATE_HOME=str(self.root / "state"),
                        PYTHONIOENCODING="utf-8")
        # the variant's own llama_server, or the top-level one above: the
        # environment's would beat the first and is already in the second
        self.env.pop("MDL_LLAMA_SERVER", None)
        self.env.pop("MDL_FIT_HOME", None)

    def mdl(self, *args, timeout=None):
        import mdl
        return subprocess.run([sys.executable, str(Path(mdl.__file__))]
                              + list(args), env=self.env, capture_output=True,
                              text=True, errors="replace", timeout=timeout,
                              creationflags=hw.NO_WINDOW)

    def start(self):
        import mdl
        p = self.mdl("run", self.variant.base,
                     timeout=mdl.ready_timeout() + 120)
        if p.returncode != 0:
            why = (p.stderr.strip().splitlines() or ["exit %d" % p.returncode])
            raise RuntimeError(why[-1].removeprefix("mdl: "))
        return json.loads((self.state / "run" / (
            self.variant.base + ".json")).read_text(encoding="utf-8"))

    def stop(self):
        try:
            self.mdl("stop", "--all", timeout=120)
        except (OSError, subprocess.SubprocessError):
            pass

    def log_text(self):
        try:
            return (self.state / (self.variant.base + ".log")).read_text(
                errors="replace")
        except OSError:
            return ""

    def remove(self):
        shutil.rmtree(self.root, ignore_errors=True)


_BUILDS = {}


def build_of(binary):
    if binary not in _BUILDS:
        info = hw.llama_build(binary) if (shutil.which(binary)
                                          or Path(binary).is_file()) else {}
        _BUILDS[binary] = {"build": info.get("build"),
                           "commit": info.get("commit"),
                           "backend": hw.backend_for(binary)}
    return _BUILDS[binary]


def claimed_vram(log_text):
    """What the load log says it put on the GPUs, in bytes: the honest
    attribution where nvidia-smi can only give a device total."""
    found = calib.scrape_log(log_text)
    total = 0
    for dev, kinds in found["buffers"].items():
        if dev.upper().startswith(("CPU", "HOST")):
            continue
        total += sum(kinds.values())
    return total or None


def prompt_for(w, depth, cpt, rng):
    """The workload's prompt behind `depth` tokens of filler, so decode is
    measured at that depth - where configs part ways."""
    if not depth:
        return w["prompt"]
    fill = evalsuite.filler(rng, int(depth * cpt))
    return ("Here is a document for context:\n\n%s\n\nNow, setting it "
            "aside: %s" % (fill, w["prompt"]))


def fit_prompt(w, spec, depth, cpt, seed, client, ctx):
    """The prompt behind `depth` tokens of filler, counted by the server's
    own tokenizer: (text, tokens), or (None, tokens) when an absolute depth
    does not fit the context. A share that comes out over - the filler is
    sized from an average - is trimmed until it fits."""
    limit = ctx - w["max_tokens"] - TEMPLATE_SLACK if ctx else None
    for _ in range(5):
        text = prompt_for(w, depth, cpt, random.Random(seed))
        n = client.tokens(text)
        if n is None:
            n = int(len(text) / cpt)
        if limit is None or n <= limit:
            return text, n
        if not isinstance(spec, str) or depth <= 0:
            return None, n
        depth = max(0, depth - (n - limit) - 16)
    return None, n


def host_needed(variant, cfg, binary):
    """RAM the variant's weights and cache will hold off the GPU, from
    mdl fit's placement, or None when the model cannot be read."""
    import mdl

    from . import gguf, model
    try:
        argv = mdl.build_argv(variant.base, cfg, binary)
        flags, _, path, _ = model.parse_argv(argv)
        return model.memory(model.Shape(gguf.load(path)), flags).host
    except (mdl.MdlError, OSError, ValueError, KeyError, IndexError,
            gguf.NotGGUF, gguf.Truncated):
        return None


def clean_machine(binary):
    """What the machine holds at idle - the OS and what starts with it,
    not the browser and the rest opened since boot - as mdl fit plans
    for it, so a report can say what a config leaves free on a machine
    you sat down to. None when it cannot be told."""
    try:
        mach = hw.probe(binary, quick=True)
    except Exception:       # noqa: BLE001 - a report column, not a stop
        return None
    idle = mach.idle
    if idle is None:
        return None
    return {"ram_total": mach.ram_total or None, "ram_idle": idle.ram,
            "ram_how": idle.ram_how, "vram_total": mach.vram_total or None,
            "vram_idle": idle.vram, "vram_how": idle.vram_how}


def run(variants, w, out, suite=None):
    import mdl
    wr = out.write
    models, binary = mdl.load_config()
    top = dict(mdl.CONFIG_DATA)
    for v in variants:
        if v.base not in models:
            die("no model named %r in %s" % (v.base, mdl.CONFIG))
        cfg = v.config(models)
        mdl.check_cfg(v.base, cfg)
        port = cfg.get("port", mdl.DEFAULT_PORT)
        if mdl.port_busy(port):
            owner = next((s["name"] for s in mdl.read_states().values()
                          if s.get("port") == port), None)
            die("port %d is in use%s; mdl lab never stops a server of yours "
                "- stop it, then run again" % (port, " by '%s'" % owner
                                               if owner else ""))
    run_id = new_id("run-")
    clean = clean_machine(binary)
    per = len(w["depths"]) * (w["warmup"] + w["reps"])
    progress = {"n": 0, "of": len(variants) * per,
                "t0": time.monotonic()}
    wr("lab      %s: %d variant%s x depth %s x %d rep%s (+%d warmup)\n" % (
        run_id, len(variants), "" if len(variants) == 1 else "s",
        ",".join(_k(d) for d in w["depths"]), w["reps"],
        "" if w["reps"] == 1 else "s", w["warmup"]))
    out.flush()
    done = skipped = 0
    for vi, v in enumerate(variants):
        if vi:
            time.sleep(w["cooldown"])
        cfg = v.config(models)
        need = host_needed(v, cfg, binary)
        _, avail = hw.ram()
        if need is not None and avail and avail < need + RAM_SPARE:
            wr("skip     %s: needs ~%.1f G of RAM free, has %.1f G\n" % (
                v.label, (need + RAM_SPARE) / GiB, avail / GiB))
            append(_skip_record(run_id, suite, v, cfg, w, "insufficient RAM"))
            skipped += 1
            progress["n"] = (vi + 1) * per
            continue
        got = run_variant(run_id, suite, v, models, binary, top, w, out,
                          clean, progress)
        # a variant or depth skipped counts as done, so what is left is
        # what will actually run
        progress["n"] = (vi + 1) * per
        done += got
        if got == 0:
            skipped += 1
    wr("\nlab      %d repetitions recorded, %d variant%s skipped; "
       "mdl lab report %s\n" % (done, skipped, "" if skipped == 1 else "s",
                                run_id))
    return run_id


def _k(n):
    if isinstance(n, str):
        return n
    return "%dk" % (n // 1024) if n and n % 1024 == 0 else str(n)


def _kt(n):
    """A token count to read, not to type back: 31.2k."""
    if not isinstance(n, (int, float)):
        return "-"
    return "%.1fk" % (n / 1024) if n >= 1024 else str(int(n))


def _skip_record(run_id, suite, v, cfg, w, why, build=None):
    return {"schema": SCHEMA, "id": new_id(), "run": run_id, "suite": suite,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "variant": v.as_dict(), "config": cfg, "build": build,
            "workload": _work_record(w, None), "skipped": why}


def _work_record(w, depth, tokens=None, ctx=None):
    """`depth` is as asked - 8192, or "100%" - so a share compares across
    variants whose contexts differ; `prompt_tokens` is what it came to."""
    return {k: w[k] for k in ("prompt_sha", "max_tokens", "seed", "temp",
                              "ignore_eos", "at")} | {
        "depth": depth, "prompt_tokens": tokens, "ctx": ctx}


def run_variant(run_id, suite, v, models, binary, top, w, out, clean=None,
                progress=None):
    """Start, measure every depth and repetition, stop. The number of
    repetitions recorded (0 if the variant could not run)."""
    from . import evalrun
    wr = out.write
    env = Env(v, models, binary, top)
    base = baseline()
    sampler, recorded, stopped = None, 0, False
    try:
        wr("\n%s\n" % v.label)
        out.flush()
        try:
            state = env.start()
        except (RuntimeError, OSError, subprocess.SubprocessError,
                ValueError) as e:
            wr("  failed to start: %s\n" % e)
            append(_skip_record(run_id, suite, v, env.cfg, w,
                                "did not start: %s" % e))
            return 0
        argv = state.get("argv") or []
        build = build_of(argv[0]) if argv else None
        log = env.log_text()
        port = state["port"]
        sampler = Sampler(state["pid"], w["interval"])
        sampler.start()
        client = evalrun.Client(port, timeout=60)
        cpt = evalrun.chars_per_token(client)
        # per slot: with parallel > 1 the server splits its context, and a
        # request can fill only its own share of it
        ctx = (client.props().get("default_generation_settings")
               or {}).get("n_ctx") or (env.cfg.get("ctx") or 0) // max(
                   1, env.cfg.get("parallel") or 1) or None
        bare = client.tokens(w["prompt"]) or int(len(w["prompt"]) / cpt)
        for i, spec in enumerate(w["depths"]):
            if isinstance(spec, str) and not ctx:
                wr("  depth %s: the server did not say its context; "
                   "skipped\n" % spec)
                continue
            depth = resolve_depth(spec, ctx, w, bare)
            text, used = fit_prompt(w, spec, depth, cpt, w["seed"] + i,
                                    client, ctx)
            if text is None:
                wr("  depth %s: %s tokens and a %d-token reply do not fit "
                   "its %s context; skipped\n" % (
                       _k(spec), _kt(used), w["max_tokens"], _k(ctx)))
                continue
            if isinstance(spec, str):
                wr("  depth %s: a %s-token prompt, the reply ending at "
                   "%s of %s\n" % (spec, _kt(used), _kt(
                       used + w["max_tokens"]), _kt(ctx)))
            for rep in range(w["warmup"] + w["reps"]):
                warm = rep < w["warmup"]
                seed = w["seed"] + max(0, rep - w["warmup"])
                first = len(sampler.samples)
                sampler.rep, sampler.depth, sampler.tok = rep, spec, 0
                sampler.phase = "prefill"

                def on_token(n, s=sampler, at=w["at"]):
                    first_tok = s.phase != "decode"
                    s.phase, s.tok = "decode", n
                    if first_tok or n == at:
                        s.poke()        # decode's start, and the @N token
                try:
                    arrivals, timings = stream(port, text, w, seed, on_token)
                except (OSError, http.client.HTTPException) as e:
                    wr("  depth %s rep %d: the request failed: %s\n"
                       % (_k(spec), rep + 1, e))
                    continue
                finally:
                    if sampler.phase == "decode":
                        sampler.poke()  # its end, with the server still full
                    sampler.phase = "idle"
                    sampler.flush()
                taken = sampler.samples[first:]
                m, flags = rep_metrics(arrivals, timings, taken, w["at"], base)
                rec = {"schema": SCHEMA, "id": new_id(), "run": run_id,
                       "suite": suite, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "variant": v.as_dict(), "config": env.cfg,
                       "argv": argv, "build": build,
                       "workload": _work_record(w, spec, used, ctx),
                       "rep": rep,
                       "warmup": warm, "metrics": m, "flags": flags,
                       "vram_claimed": claimed_vram(log),
                       "baseline": base, "clean": clean,
                       "env": {"ram_free": hw.ram()[1],
                               "os": sys.platform}}
                append(rec, taken)
                recorded += 0 if warm else 1
                wr("  %sdepth %-5s %s  %4d tok  %6s t/s  ttft %5ss  VRAM %s  "
                   "RAM %s  CPU %s%%\n" % (
                       _progress(progress), _k(spec),
                       "warmup  " if warm else "rep %d/%d" % (
                           rep - w["warmup"] + 1, w["reps"]),
                       m.get("tokens", 0), _fmt(m["decode"]["avg"], 1),
                       _fmt(m.get("ttft"), 2), _gb(m["vram"]["peak"]),
                       _ram(m), _fmt(m["cpu"]["mean"], 0)))
                out.flush()
        sampler.phase = "settle"
        env.stop()
        stopped = True
        if base.get("vram") is not None:
            # the next variant is measured against this one's baseline;
            # memory not handed back would be counted to it
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                now = gpu_now()[0]
                if now is None or now <= base["vram"] + VRAM_SLACK:
                    break
                time.sleep(1)
            else:
                die("VRAM did not return to its baseline after %s stopped "
                    "(%s G in use, %s G before); something else is using "
                    "the card, so the rest would not be comparable"
                    % (v.label, _gb(gpu_now()[0]), _gb(base["vram"])))
        return recorded
    except KeyboardInterrupt:
        wr("\ninterrupted; what was measured is kept\n")
        raise
    finally:
        if sampler:
            sampler.stop()
        if not stopped:
            env.stop()
        env.remove()


def _progress(p):
    """[3/8, ~2 min left]: requests done, and the rest at the pace so far."""
    if not p:
        return ""
    p["n"] += 1
    spent = time.monotonic() - p["t0"]
    left = spent / p["n"] * (p["of"] - p["n"])
    return "[%d/%d%s] " % (p["n"], p["of"], ", ~%s left" % (
        "%d min" % round(left / 60) if left >= 90 else "%d s" % left)
        if p["n"] < p["of"] else "")


def _ram(m):
    """The machine's RAM above its level before the load - what a model
    with experts in RAM costs - else the server's resident size."""
    ram = m.get("ram") or {}
    if ram.get("delta") is not None:
        return "+" + _gb(ram["delta"])
    return _gb(ram.get("peak_rss"))


def _fmt(x, n):
    return ("%%.%df" % n) % x if isinstance(x, (int, float)) else "-"


def _gb(b):
    return "%.1fG" % (b / GiB) if isinstance(b, (int, float)) else "-"


# ------------------------------------------------------------- dry run --

def dry_run(variants, w, out):
    import mdl
    wr = out.write
    models, binary = mdl.load_config()
    total_s, known = 0.0, True
    wr("variant%s  %s\n" % (" " * 25, "config"))
    for v in variants:
        if v.base not in models:
            die("no model named %r in %s" % (v.base, mdl.CONFIG))
        cfg = v.config(models)
        mdl.check_cfg(v.base, cfg)
        est = estimate(v, cfg, binary, w)
        if est is None:
            known = False
        else:
            total_s += est
        wr("%-32s %s%s\n" % (v.label, " ".join(
            "%s=%s" % (k, mdl.toml_value(x)) for k, x in v.set.items())
            or "(as configured)", "  on " + v.server if v.server else ""))
    reqs = len(variants) * len(w["depths"]) * (w["reps"] + w["warmup"])
    wr("\n%d load%s, %d request%s (%d depth%s x %d reps + %d warmup, "
       "%d tokens each)\n" % (
           len(variants), "" if len(variants) == 1 else "s", reqs,
           "" if reqs == 1 else "s", len(w["depths"]),
           "" if len(w["depths"]) == 1 else "s", w["reps"], w["warmup"],
           w["max_tokens"]))
    total_s += w["cooldown"] * max(0, len(variants) - 1)
    wr("estimate %s\n" % (
        "~%d min, from mdl fit's predicted speeds, loads not counted"
        % max(1, round(total_s / 60)) if known else
        "none: a model mdl fit cannot read" + (
            " (~%d min for the rest)" % max(1, round(total_s / 60))
            if total_s else "")))
    wr("dry run: nothing started\n")


def estimate(v, cfg, binary, w):
    """Seconds the requests of one variant should take, from mdl fit."""
    import mdl

    from . import gguf, model, perf
    try:
        argv = mdl.build_argv(v.base, cfg, binary)
        flags, _, path, _ = model.parse_argv(argv)
        shape = model.Shape(gguf.load(path))
        mach = hw.probe(argv[0], quick=True)
    except (mdl.MdlError, OSError, ValueError, KeyError, IndexError,
            gguf.NotGGUF, gguf.Truncated):
        return None
    total = 0.0
    ctx = (cfg.get("ctx") or 4096) // max(1, cfg.get("parallel") or 1)
    for spec in w["depths"]:
        depth = resolve_depth(spec, ctx, w, int(len(w["prompt"]) / 4))
        sp = perf.speed(shape, flags, mach, depth=max(depth, 1))
        dec = sp.decode_d if depth else sp.decode0
        if not dec or not sp.prefill:
            return None
        per = (depth + len(w["prompt"]) / 4) / sp.prefill + w["max_tokens"] / dec
        total += per * (w["reps"] + w["warmup"])
    return total


# -------------------------------------------------------------- report --

def runs(records):
    out = {}
    for r in records:
        out.setdefault(r.get("run"), []).append(r)
    return out


def pick_run(records, run_id=None):
    by = runs(records)
    if not by:
        die("no lab runs recorded yet; mdl lab run <name>")
    if run_id is None:
        return max(by.items(), key=lambda kv: max(r.get("at", "")
                                                  for r in kv[1]))
    if run_id not in by:
        die("no lab run %r; mdl lab ls lists them" % run_id)
    return run_id, by[run_id]


def groups(recs):
    """(label, depth) -> the measured repetitions, in run order."""
    out = {}
    for r in recs:
        if r.get("skipped") or r.get("warmup"):
            continue
        key = (r["variant"]["label"], r["workload"]["depth"])
        out.setdefault(key, []).append(r)
    return out


def _med(recs, *path):
    vals = []
    for r in recs:
        x = r["metrics"]
        for p in path:
            x = x.get(p) if isinstance(x, dict) else None
        if isinstance(x, (int, float)):
            vals.append(x)
    if not vals:
        return None, None
    return statistics.median(vals), (statistics.stdev(vals)
                                     if len(vals) > 1 else 0.0)


def _idle_free(recs, what, cost):
    """What this config leaves free on the machine at idle: its total,
    less what the OS and startup programs hold, less what the model took.
    Below zero, it does not fit a machine you sat down to."""
    c = recs[0].get("clean") or {}
    total, idle = c.get(what + "_total"), c.get(what + "_idle")
    if not total or idle is None or not cost:
        return None
    return total - idle - cost


def row(label, depth, recs, at):
    dec, sd = _med(recs, "decode", "avg")
    # the efficiency column: throughput for each GB the model itself took,
    # the card's rise over its idle baseline, else what the load log claimed;
    # the card's peak would charge the model for the desktop's share too
    vram = _med(recs, "vram", "delta")[0]
    if not vram or vram <= 0:
        vram = recs[0].get("vram_claimed")
    ram_cost = _med(recs, "ram", "delta")[0]
    if ram_cost is None or ram_cost <= 0:
        ram_cost = _med(recs, "ram", "peak_rss")[0]
    at_vals = []
    for r in recs:
        snap = (r["metrics"].get("at") or {}).get(str(at))
        if snap is None:
            snap = snapshot_at(samples_of(r), at)
        at_vals.append(snap or {})
    at_vram = [s["vram"] for s in at_vals if s.get("vram") is not None]
    at_rss = [s["rss"] for s in at_vals if s.get("rss") is not None]
    b = recs[0].get("build") or {}
    flags = {f for r in recs for f in r.get("flags", [])}
    avgs = [x for x in _vals(recs, ("decode", "avg")) if x]
    if len(avgs) >= 3 and all(avgs[i + 1] > avgs[i]
                                  for i in range(len(avgs) - 1)) \
            and avgs[-1] / avgs[0] - 1 > DRIFT:
        # each repetition faster than the last: the warmup was not enough,
        # and the median is of a model still getting up to speed
        flags.add("rep_drift")
    flags = sorted(flags)
    toks = [(r.get("workload") or {}).get("prompt_tokens") for r in recs]
    toks = [t for t in toks if t]
    return {"variant": label, "depth": depth, "reps": len(recs),
            "prompt_tokens": statistics.median(toks) if toks else None,
            "build": "%s %s" % (b.get("backend") or "?", b.get("build") or "?"),
            "decode": dec, "decode_sd": sd,
            "p05": _med(recs, "decode", "p05")[0],
            "floor": _med(recs, "decode", "floor")[0],
            "ttft": _med(recs, "ttft")[0],
            "prefill": _med(recs, "prefill", "tps")[0],
            "vram_peak": _med(recs, "vram", "peak")[0],
            "vram_delta": _med(recs, "vram", "delta")[0],
            "vram_claimed": recs[0].get("vram_claimed"),
            "rss_peak": _med(recs, "ram", "peak_rss")[0],
            "cpu": _med(recs, "cpu", "mean")[0],
            "to_1k": _med(recs, "time_to_1k")[0],
            "per_gb": (dec / (vram / GiB)) if dec and vram else None,
            "ram_idle_free": _idle_free(recs, "ram", ram_cost),
            "vram_idle_free": _idle_free(recs, "vram", vram),
            "at_vram": statistics.median(at_vram) if at_vram else None,
            "at_rss": statistics.median(at_rss) if at_rss else None,
            "flags": flags}


COLUMNS = [("variant", "variant"), ("build", "build"), ("depth", "depth"),
           ("decode t/s", "decode"), ("p05", "p05"), ("floor", "floor"),
           ("ttft s", "ttft"), ("prefill t/s", "prefill"),
           ("VRAM peak", "vram_peak"), ("VRAM +", "vram_delta"),
           ("RSS", "rss_peak"), ("CPU %", "cpu"), ("to 1k s", "to_1k"),
           ("t/s per G", "per_gb"),
           ("RAM free idle", "ram_idle_free"),
           ("VRAM free idle", "vram_idle_free"),
           ("@N VRAM", "at_vram"), ("@N RSS", "at_rss"), ("flags", "flags")]


def cell(key, r):
    v = r[key]
    if key == "decode":
        return "-" if v is None else "%.1f ±%.1f" % (v, r["decode_sd"] or 0)
    if key == "depth":
        if isinstance(v, str) and r.get("prompt_tokens"):
            return "%s (%s)" % (v, _kt(r["prompt_tokens"]))
        return _k(v) if v is not None else "-"
    if key in ("vram_peak", "vram_delta", "rss_peak", "at_vram", "at_rss",
               "ram_idle_free", "vram_idle_free"):
        if v is None and key == "vram_peak" and r.get("vram_claimed"):
            return "%s (log)" % _gb(r["vram_claimed"])
        return _gb(v)
    if key == "flags":
        return ",".join(v) or "-"
    if key in ("ttft", "to_1k"):
        return _fmt(v, 2)
    if isinstance(v, float):
        return _fmt(v, 1)
    return "-" if v is None else str(v)


def report(o, out):
    wr = out.write
    records = load()
    run_id, recs = pick_run(records, o["names"][0] if o["names"] else
                            o.get("run"))
    at = o.get("at") or next((r["workload"].get("at") for r in recs
                              if r.get("workload")), 500)
    rows = [row(label, depth, rs, at) for (label, depth), rs in
            sorted(groups(recs).items(),
                   key=lambda kv: (kv[0][0], _dkey(kv[0][1])))]
    fmt = o.get("format", "table")
    if fmt == "json":
        wr(json.dumps({"run": run_id, "at": at, "rows": rows}, indent=1)
           + "\n")
        return rows
    heads = [h.replace("@N", "@%d" % at) for h, _ in COLUMNS]
    table = [[cell(k, r) for _, k in COLUMNS] for r in rows]
    if fmt == "csv":
        import csv
        wcsv = csv.writer(out, lineterminator="\n")
        wcsv.writerow(heads)
        wcsv.writerows(table)
        return rows
    if fmt not in ("table", "md"):
        die("--format is table, md, csv or json")
    wr("run      %s  (%s)\n\n" % (run_id, recs[0].get("suite") or "no suite"))
    if fmt == "md":
        wr("| " + " | ".join(heads) + " |\n")
        wr("|" + "---|" * len(heads) + "\n")
        for t in table:
            wr("| " + " | ".join(t) + " |\n")
    else:
        widths = [max(len(h), *(len(t[i]) for t in table)) if table
                  else len(h) for i, h in enumerate(heads)]
        wr("  ".join(h.ljust(widths[i]) for i, h in enumerate(heads)).rstrip()
           + "\n")
        for t in table:
            wr("  ".join(c.ljust(widths[i]) for i, c in enumerate(t)).rstrip()
               + "\n")
    for r in recs:
        if r.get("skipped"):
            wr("skipped  %s: %s\n" % (r["variant"]["label"], r["skipped"]))
    builds = {(b.get("build"), b.get("commit")) for b in
              (r.get("build") for r in recs) if b}
    if len(builds) > 1:
        wr("note     these ran on %d llama.cpp builds (%s): a difference "
           "between them is the build's as much as the config's\n" % (
               len(builds), ", ".join("b%s" % b for b, _ in sorted(
                   builds, key=str))))
    if any(r["vram_peak"] is None for r in rows):
        wr("note     no nvidia-smi: VRAM is what each server's load log "
           "claims, marked (log)\n")
    return rows


def compare(o, out):
    wr = out.write
    if len(o["names"]) != 2:
        die("mdl lab compare takes two variants (labels in a run, or "
            "record ids)")
    records = load()
    run_id, recs = pick_run(records, o.get("run"))
    by_id = {r["id"]: r for r in records}
    sides = []
    for name in o["names"]:
        if name in by_id:
            r = by_id[name]
            sides.append((name, [r]))
            continue
        got = [r for r in recs if r["variant"]["label"] == name
               and not r.get("warmup") and not r.get("skipped")]
        if not got:
            die("no variant %r in %s; its labels: %s" % (
                name, run_id, ", ".join(sorted({
                    r["variant"]["label"] for r in recs}))))
        sides.append((name, got))
    (a, ra), (b, rb) = sides
    wr("%s  vs  %s\n" % (a, b))

    def by_depth(recs):
        out = {}
        for r in recs:
            out.setdefault(r["workload"].get("depth"), []).append(r)
        return out
    da, db = by_depth(ra), by_depth(rb)
    both = sorted(set(da) & set(db), key=_dkey)
    if not both and len(da) == 1 and len(db) == 1:
        both = [None]           # two records named: set side by side as asked
    if not both:
        die("%s and %s share no depth: %s against %s" % (
            a, b, ",".join(_k(d) for d in sorted(da, key=_dkey)),
            ",".join(_k(d) for d in sorted(db, key=_dkey))))
    for depth in both:
        # an empty context and a full one are different workloads: pooled,
        # their spread would call any two configs indistinguishable
        xa = da[depth] if depth is not None else ra
        xb = db[depth] if depth is not None else rb
        wr("\ndepth %s\n" % (_k(depth) if depth is not None else "as run"))
        for what, path, better in (("decode t/s", ("decode", "avg"), max),
                                   ("floor t/s", ("decode", "floor"), max),
                                   ("ttft s", ("ttft",), min),
                                   ("prefill t/s", ("prefill", "tps"), max),
                                   ("VRAM peak", ("vram", "peak"), min),
                                   ("RSS peak", ("ram", "peak_rss"), min)):
            va, vb = _vals(xa, path), _vals(xb, path)
            if not va or not vb:
                continue
            ma, mb = statistics.mean(va), statistics.mean(vb)
            big = what.startswith(("VRAM", "RSS"))
            show = _gb if big else (lambda x: _fmt(x, 2))
            wr("%-12s %10s  %10s  %+6.1f%%  %s\n" % (
                what, show(ma), show(mb), (mb / ma - 1) * 100 if ma else 0,
                verdict(va, vb, a, b, better)))
    ba = (ra[0].get("build") or {}).get("build")
    bb = (rb[0].get("build") or {}).get("build")
    if ba != bb:
        wr("\nnote     %s ran on build %s, %s on %s: the difference is the "
           "build's as much as the config's\n" % (a, ba, b, bb))


def _vals(recs, path):
    out = []
    for r in recs:
        x = r["metrics"]
        for p in path:
            x = x.get(p) if isinstance(x, dict) else None
        if isinstance(x, (int, float)):
            out.append(x)
    return out


def verdict(va, vb, a, b, better):
    """Which side wins, or "indistinguishable" when the two intervals of
    mean ± 2 standard errors overlap - a gap the noise could make."""
    def interval(v):
        m = statistics.mean(v)
        se = statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0
        return m - 2 * se, m + 2 * se
    (la, ha), (lb, hb) = interval(va), interval(vb)
    if len(va) < 2 or len(vb) < 2:
        return "one run each: not enough to call"
    if la <= hb and lb <= ha:
        return "indistinguishable"
    win = a if better(statistics.mean(va), statistics.mean(vb)) == \
        statistics.mean(va) else b
    return "%s better" % win


def ls(out):
    wr = out.write
    by = runs(load())
    if not by:
        wr("no lab runs recorded yet; mdl lab run <name>\n")
        return
    for run_id, recs in sorted(by.items(), key=lambda kv: min(
            r.get("at", "") for r in kv[1])):
        labels = sorted({r["variant"]["label"] for r in recs})
        measured = sum(1 for r in recs if not r.get("warmup")
                       and not r.get("skipped"))
        wr("%s  %s  %-12s %d variant%s, %d rep%s  %s\n" % (
            run_id, min(r.get("at", "?") for r in recs)[:16],
            recs[0].get("suite") or "-", len(labels),
            "" if len(labels) == 1 else "s", measured,
            "" if measured == 1 else "s", ", ".join(labels)[:60]))


def export(o, out):
    if len(o["names"]) != 1:
        die("mdl lab export takes a record id or a run id")
    key = o["names"][0]
    got = [r for r in load() if key in (r["id"], r.get("run"))]
    if not got:
        die("no lab record or run %r" % key)
    out.write(json.dumps(got[0] if len(got) == 1 and got[0]["id"] == key
                         else got, indent=1) + "\n")


def apply(o, out):
    """Print the variant's table for models.toml. Printing only: editing
    the config stays a decision made with it in front of you."""
    import mdl
    if len(o["names"]) != 1:
        die("mdl lab apply takes a variant label or a record id")
    key = o["names"][0]
    records = load()
    got = [r for r in records if r["id"] == key] or [
        r for r in reversed(records) if r["variant"]["label"] == key]
    if not got:
        die("no lab record or variant %r" % key)
    r = got[0]
    out.write("# from mdl lab %s, %s\n[%s]\n" % (r.get("run"), r["variant"][
        "label"], mdl.toml_key(r["variant"]["base"])))
    for k, v in r["config"].items():
        out.write("%s = %s\n" % (k, mdl.toml_value(v)))


# ------------------------------------------------------------ baseline --

def baseline_path():
    return lab_dir() / "baseline.json"


def pinned():
    try:
        return json.loads(baseline_path().read_text(encoding="utf-8"))["run"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def baseline_cmd(o, out):
    """Pin a run, and say what moved in a later one: the regression check
    a new llama.cpp build or a config change should pass before it stays."""
    import mdl
    wr = out.write
    records = load()
    what = o["names"][0] if o["names"] else None
    if what is None:
        base = pinned()
        wr("baseline %s\n" % (base or "none pinned; mdl lab baseline set"))
        return base
    if what == "set":
        run_id, _ = pick_run(records, o["names"][1] if len(o["names"]) > 1
                             else None)
        lab_dir().mkdir(parents=True, exist_ok=True)
        mdl.write_atomic(baseline_path(), json.dumps({"run": run_id}))
        wr("baseline %s pinned\n" % run_id)
        return run_id
    if what != "diff":
        die("mdl lab baseline takes set or diff")
    base = pinned()
    if base is None:
        die("no baseline pinned; mdl lab baseline set [run]")
    _, base_recs = pick_run(records, base)
    run_id, recs = pick_run(records, o["names"][1] if len(o["names"]) > 1
                            else None)
    if run_id == base:
        die("the latest run is the baseline; run again, or name a run")
    now, then = groups(recs), groups(base_recs)
    wr("%s  against baseline %s\n\n" % (run_id, base))
    worse = []
    for key in sorted(now, key=lambda k: (k[0], _dkey(k[1]))):
        if key not in then:
            wr("%-32s depth %-5s new: not in the baseline\n" % (
                key[0], _k(key[1])))
            continue
        for what_, path, better in (("decode t/s", ("decode", "avg"), max),
                                    ("ttft s", ("ttft",), min),
                                    ("VRAM peak", ("vram", "peak"), min)):
            va, vb = _vals(then[key], path), _vals(now[key], path)
            if not va or not vb:
                continue
            call = verdict(va, vb, "baseline", "now", better)
            ma, mb = statistics.mean(va), statistics.mean(vb)
            show = _gb if what_ == "VRAM peak" else (lambda x: _fmt(x, 2))
            wr("%-32s depth %-5s %-11s %9s -> %-9s %+6.1f%%  %s\n" % (
                key[0], _k(key[1]), what_, show(ma), show(mb),
                (mb / ma - 1) * 100 if ma else 0, call))
            if call == "baseline better":
                worse.append("%s %s" % (key[0], what_))
    gone = sorted(set(then) - set(now), key=lambda k: (k[0], _dkey(k[1])))
    for key in gone:
        wr("%-32s depth %-5s not run this time\n" % (key[0], _k(key[1])))
    wr("\n%s\n" % ("regressed: " + "; ".join(worse) if worse
                   else "no regression the spread can tell from noise"))
    if worse and o.get("fail"):
        die("%d regression%s against the baseline" % (
            len(worse), "" if len(worse) == 1 else "s"))
    return worse


# ---------------------------------------------------------------- main --

def main(args, out=None):
    out = out or sys.stdout
    if not args or args[0] in ("-h", "--help"):
        out.write(USAGE)
        return None
    sub, o = args[0], parse(args[1:])
    if sub == "run":
        if o.get("suite"):
            if o["names"] or o["set"] or o["servers"]:
                die("--suite takes its variants from the file; leave out "
                    "names, --set and --server")
            variants = load_suite(o["suite"], o)
        else:
            if not o["names"]:
                die("mdl lab run takes a model name from models.toml, or "
                    "--suite")
            variants = matrix(o)
        labels = [v.label for v in variants]
        dupes = sorted({x for x in labels if labels.count(x) > 1})
        if dupes:
            die("two variants share the label %s; give them labels in a "
                "suite file" % dupes[0])
        w = workload(o)
        if o.get("dry_run"):
            return dry_run(variants, w, out)
        return run(variants, w, out, o.get("suite"))
    if sub == "report":
        return report(o, out)
    if sub == "compare":
        return compare(o, out)
    if sub == "ls":
        return ls(out)
    if sub == "export":
        return export(o, out)
    if sub == "apply":
        return apply(o, out)
    if sub == "baseline":
        return baseline_cmd(o, out)
    die("unknown lab command %r; it is run, report, compare, ls, export, "
        "apply or baseline" % sub)
