"""Calibration: the oracle, the benchmark, and what they taught us.

llama.cpp is the source of truth. `llama-fit-params -fitp on` prints the
model, context and compute bytes per device for a set of flags without
allocating anything, in well under a second, so it is cheap enough to
run in the loop: the search proposes, the oracle checks, the difference
is stored against this exact file and build, and the next prediction
starts from it. llama-bench does the same job for speed.

Everything lands in calib.jsonl, one line per observation.
"""

import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

from . import hw, model

MiB = 1 << 20


def calib_path():
    return hw.config_dir() / "calib.jsonl"


def signature(inv):
    """Identifies one GGUF across renames of its folder: name + bytes."""
    return "%s|%d" % (Path(str(inv.source)).name, inv.file_size)


# Past this the file is compacted: every mdl run books a load log and
# every eval a profile, and nothing ever took one out.
MAX_BYTES = 2 << 20
LOGS_KEPT = 20           # load logs per model file; failures are kept too


def append(entry):
    """Book one observation. Never raises: a fit, a run or an eval must
    not fail because its bookkeeping could not be written."""
    import mdl
    entry = dict(entry, at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    path = calib_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # the lock keeps an append from landing while compact() has read
        # the file and not yet replaced it, where it would be lost
        with mdl.file_lock(path.with_name(path.name + ".lock"),
                           "calib.jsonl is busy", tries=20):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            if path.stat().st_size > MAX_BYTES:
                compact(path)
    except (OSError, mdl.MdlError):
        pass


def _keep_key(e):
    """Entries with one key say the same thing; only the latest is read.
    None: kept whatever else is there."""
    kind = e.get("kind")
    if kind == "profile":
        return ("profile", e.get("key"))
    if kind in ("oracle", "bench"):
        return (kind, e.get("sig"), e.get("build"),
                json.dumps(e.get("flags"), sort_keys=True))
    return None


def compact(path=None):
    """Drop what nothing reads: an oracle or bench entry, or a profile,
    superseded by a later one for the same configuration, and load logs
    beyond the last LOGS_KEPT per model file (failures kept). Order is
    kept, so the latest still wins. Called under append()'s lock."""
    import mdl
    path = path or calib_path()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    entries = []
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue                    # a line a crash cut off
        if isinstance(e, dict):
            entries.append((line, e))
    last, logs = {}, {}
    for i, (_, e) in enumerate(entries):
        key = _keep_key(e)
        if key is not None:
            last[key] = i
        elif e.get("kind") == "log" and not e.get("failed"):
            logs.setdefault(e.get("model"), []).append(i)
    drop = {i for i, (_, e) in enumerate(entries)
            if _keep_key(e) is not None and last[_keep_key(e)] != i}
    for idx in logs.values():
        drop.update(idx[:-LOGS_KEPT])
    kept = [line for i, (line, _) in enumerate(entries) if i not in drop]
    mdl.write_atomic(path, "".join(line + "\n" for line in kept))
    return len(lines) - len(kept)


def load(kind=None):
    try:
        lines = calib_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if kind is None or e.get("kind") == kind:
            out.append(e)
    return out


# --------------------------------------------------------------- oracle --

def oracle_argv(fit_bin, model_path, flags):
    argv = [fit_bin, "-m", str(model_path), "-c", str(flags.ctx),
            "-ngl", str(flags.ngl), "-ncmoe", str(flags.ncmoe),
            "-fa", "on" if flags.fa else "off", "-ctk", flags.ctk,
            "-ctv", flags.ctv, "-b", str(max(flags.b, flags.ub)),
            "-ub", str(flags.ub), "-np", str(flags.np),
            "-fitp", "on", "-fit", "off"]
    if flags.swa_full:
        argv.append("--swa-full")
    if flags.kvu:
        argv.append("-kvu")
    return argv


def parse_oracle(stdout):
    """{'Vulkan0': [model, context, compute], 'Host': [...]} in bytes."""
    out = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 4 and all(p.lstrip("-").isdigit() for p in parts[1:]):
            out[parts[0]] = [int(p) * MiB for p in parts[1:]]
    return out


def run_oracle(fit_bin, model_path, flags, timeout=180):
    """The oracle's verdict for one config, or None if it would not say."""
    try:
        p = subprocess.run(oracle_argv(fit_bin, model_path, flags),
                           capture_output=True, text=True, timeout=timeout,
                           creationflags=hw.NO_WINDOW, errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    got = parse_oracle(p.stdout)
    gpu = next((v for k, v in got.items() if k != "Host"), None)
    if not gpu:
        return None
    return {"gpu": gpu, "host": got.get("Host", [0, 0, 0]),
            "device": next(k for k in got if k != "Host")}


def check(fit_bin, inv, shape, flags, build, coef=None, max_alloc=None):
    """Run the oracle on one config and record how far off we were."""
    got = run_oracle(fit_bin, inv.source, flags)
    if not got:
        return None
    pred = model.memory(shape, flags, coef, max_alloc or model.BIG_ALLOC)
    entry = {"kind": "oracle", "sig": signature(inv), "arch": inv.arch,
             "build": build, "flags": flags.as_dict(),
             "pred": [pred.gpu_weights, pred.gpu_context, pred.gpu_compute],
             "actual": got["gpu"], "host": got["host"],
             "device": got["device"]}
    append(entry)
    return entry


# ------------------------------------------------------------ residuals --

class Residuals:
    """What calibration has learned about the compute buffer.

    Weights and cache sizes come out of the model exact, so every oracle
    miss is booked against the compute term. For a file with its own
    observations the residual is interpolated over context at the same
    ubatch (it is linear in context, which is the mask). A file with none
    borrows the median from files of the same arch at that ubatch, with
    a wider band.
    """

    def __init__(self, entries=None, build=None):
        entries = load("oracle") if entries is None else entries
        self.by_sig, self.by_arch = {}, {}
        for e in entries:
            if build and e.get("build") not in (None, build):
                continue
            f = e.get("flags", {})
            pred, act = e.get("pred"), e.get("actual")
            if not pred or not act:
                continue
            miss = (act[0] - pred[0]) + (act[1] - pred[1]) + (act[2] - pred[2])
            key = (f.get("ub"), bool(f.get("ncmoe")))
            point = (f.get("ctx", 0), miss)
            self.by_sig.setdefault(e["sig"], {}).setdefault(key, []).append(point)
            self.by_arch.setdefault(e.get("arch"), {}).setdefault(
                f.get("ub"), []).append(miss)

    def level(self, sig, arch):
        if sig in self.by_sig:
            return "oracle"
        if arch in self.by_arch:
            return "arch"
        return "none"

    def lookup(self, sig, arch, flags):
        """Bytes to add to the predicted GPU compute buffer."""
        table = self.by_sig.get(sig)
        if table:
            pts = (table.get((flags.ub, bool(flags.ncmoe)))
                   or table.get((flags.ub, not flags.ncmoe)))
            if pts:
                return _interp(sorted(pts), flags.ctx)
        misses = self.by_arch.get(arch, {}).get(flags.ub)
        if misses:
            misses = sorted(misses)
            return max(0, misses[len(misses) // 2])   # never borrow optimism
        return 0

    def band(self, sig, arch, compute_bytes):
        """Memory uncertainty in bytes for this level of calibration."""
        lvl = self.level(sig, arch)
        if lvl == "oracle":
            return 96 * MiB
        if lvl == "arch":
            return 256 * MiB
        return max(512 * MiB, compute_bytes // 4)


def _interp(pts, x):
    """Linear through the two nearest context points; flat past the ends."""
    if len(pts) == 1 or x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        if x0 <= x <= x1:
            if x1 == x0:
                return max(y0, y1)
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


# ---------------------------------------------------------------- bench --

def bench_argv(bench_bin, model_path, flags, n_prompt=512, n_gen=128,
               depth=0, reps=2, threads=None):
    argv = [bench_bin, "-m", str(model_path), "-ngl", str(flags.ngl),
            "-ncmoe", str(flags.ncmoe), "-fa", "on" if flags.fa else "off",
            "-ctk", flags.ctk, "-ctv", flags.ctv,
            "-b", str(max(flags.b, flags.ub)), "-ub", str(flags.ub),
            "-p", str(n_prompt), "-n", str(n_gen), "-d", str(depth),
            "-r", str(reps), "-o", "json"]
    if not flags.mmap:
        argv += ["-mmp", "0"]
    threads = threads or flags.threads     # the config's, unless overridden
    if threads:
        argv += ["-t", str(threads)]
    return argv


OOM = re.compile(r"out of memory|OutOfDeviceMemory|failed to allocate|"
                 r"cudaMalloc failed|unable to allocate|alloc.*failed|"
                 r"not enough memory|std::bad_alloc", re.I)
UNSUPPORTED = re.compile(r"unknown argument|invalid argument|"
                         r"invalid parameter|unrecognized|error: unknown",
                         re.I)


def bench(bench_bin, model_path, flags, timeout=1800, **kw):
    """(result or None, why): why is "ok", "oom", "timeout",
    "unsupported" or "error", so a caller can tell a config that did not
    fit from one this build does not take - only the first is evidence
    about memory."""
    try:
        p = subprocess.run(bench_argv(bench_bin, model_path, flags, **kw),
                           capture_output=True, text=True, timeout=timeout,
                           creationflags=hw.NO_WINDOW, errors="replace")
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except (OSError, subprocess.SubprocessError):
        return None, "error"
    got = parse_bench(p.stdout)
    if got:
        return got, "ok"
    said = p.stderr or ""
    if OOM.search(said):
        return None, "oom"
    if UNSUPPORTED.search(said):
        return None, "unsupported"
    return None, "error"


def run_bench(bench_bin, model_path, flags, timeout=1800, **kw):
    """{'pp': t/s, 'tg': {depth: t/s}} from llama-bench, or None."""
    return bench(bench_bin, model_path, flags, timeout, **kw)[0]


def parse_bench(stdout):
    start = stdout.find("[")
    if start < 0:
        return None
    try:
        rows = json.loads(stdout[start:])
    except ValueError:
        return None
    out = {"tg": {}}
    for r in rows:
        tps = r.get("avg_ts")
        if not tps:
            continue
        if r.get("n_gen", 0) > 0 and not r.get("n_prompt"):
            out["tg"][int(r.get("n_depth", 0) or 0)] = tps
        elif r.get("n_prompt", 0) > 0 and not r.get("n_gen"):
            out["pp"] = tps
    return out if (out["tg"] or out.get("pp")) else None


def efficiency(arch):
    """Per-arch speed multipliers from verify runs: measured / predicted."""
    tg, pp = [], []
    for e in load("bench"):
        if e.get("arch") != arch:
            continue
        for depth, tps in (e.get("tg") or {}).items():
            pred = (e.get("pred_tg") or {}).get(depth)
            if tps and pred:
                tg.append(tps / pred)
        if e.get("pp") and e.get("pred_pp"):
            pp.append(e["pp"] / e["pred_pp"])
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 1.0   # noqa: E731
    return {"tg": med(tg), "pp": med(pp), "runs": max(len(tg), len(pp))}


# ------------------------------------------------------------ load logs --

_BUF = re.compile(r"(\w+?\d*|CPU|Host)[_ ]\w*?\s*(model|KV|compute|RS) "
                  r"buffer size\s*=\s*([\d.]+) MiB")
_FAIL = re.compile(r"failed to allocate (\w+) buffer of size (\d+)")
_BREAKDOWN = re.compile(r"\|\s+-\s+(\S+).*?\|\s*(\d+)\s*=\s*(-?\d+)\s*\+\s*"
                        r"\(\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*\)")


def scrape_log(text):
    """Buffer sizes a llama-server load log gives away for free.

    {'buffers': {device: {'model': b, 'KV': b, ...}}, 'failed': (device,
    bytes) or None, 'breakdown': {device: [model, context, compute]}}.
    """
    buffers = {}
    for m in _BUF.finditer(text):
        dev = m.group(1)
        kind = m.group(2)
        buffers.setdefault(dev, {})
        buffers[dev][kind] = buffers[dev].get(kind, 0) + int(
            float(m.group(3)) * MiB)
    fail = _FAIL.search(text)
    breakdown = {}
    for m in _BREAKDOWN.finditer(text):
        breakdown[m.group(1)] = [int(m.group(i)) * MiB for i in (5, 6, 7)]
    return {"buffers": buffers,
            "failed": (fail.group(1), int(fail.group(2))) if fail else None,
            "breakdown": breakdown}


def passive(name, argv, log_path, build=None):
    """Book whatever a finished load log says, against its flags. Called
    by mdl run; never raises, because a launch must not fail on this."""
    try:
        text = Path(log_path).read_text(errors="replace")
        found = scrape_log(text)
        if not (found["breakdown"] or found["failed"] or found["buffers"]):
            return None
        flags, _, model_path, _ = model.parse_argv(argv)
        if build is None:
            # the load log names its own build; nothing passed one, so
            # these entries could never be told apart by build
            from . import manifest
            m = manifest.BUILD.search(text)
            build = int(m.group(1)) if m else None
        entry = {"kind": "log", "name": name, "model": model_path,
                 "flags": flags.as_dict(), "build": build}
        entry.update(found)
        append(entry)
        return entry
    except Exception:          # noqa: BLE001 - passive by definition
        return None


# ------------------------------------------------------------ profiles --
#
# A measured profile is what one exact configuration did on this
# machine: decode speed at the context depths it was measured at, and
# prefill speed. Exact means the same model bytes, the same llama.cpp
# build from the same binary, and the same flags - change any of them and
# the numbers are another profile's. They are kept apart from the
# per-arch efficiency factors above: those nudge every prediction for an
# architecture, these only ever describe the configuration they were
# measured at, and say nothing about quality.

DEPTHS = (0, 4096, 16384, 32768, 65536, 131072)
MIN_DECODE = 16          # tokens: fewer is timer noise, not a speed
MIN_PREFILL = 256


def _build_no(build):
    return build.get("build") if isinstance(build, dict) else build


def _build_id(build):
    """The build number and its commit: two commits can share a number."""
    if isinstance(build, dict):
        return "%s-%s" % (build.get("build"), build.get("commit"))
    return build


def _binary_id(binary):
    """A binary as stat names it, so a rebuild at the same path - another
    backend, other compiler flags - is another configuration."""
    if isinstance(binary, dict):
        return [binary.get("path"), binary.get("size"),
                binary.get("mtime_ns") or binary.get("mtime")]
    return str(binary or "")


# llama-server flags llama-bench reproduces (bench_argv), or that do not
# change single-stream speed. A preset with anything else was not what
# llama-bench measured, so the measurement is not booked to its command.
BENCHED = {"-ngl", "--n-gpu-layers", "--gpu-layers", "-ncmoe", "--n-cpu-moe",
           "-cmoe", "--cpu-moe", "-fa", "--flash-attn", "-ctk",
           "--cache-type-k", "-ctv", "--cache-type-v", "-b", "--batch-size",
           "-ub", "--ubatch-size", "-t", "--threads", "--no-mmap", "--mmap",
           "-c", "--ctx-size", "-np", "--parallel", "-kvu", "--kv-unified",
           "--no-kv-unified", "--jinja", "--no-jinja", "-a", "--alias",
           "--host", "--api-key", "--api-key-file", "--chat-template",
           "--chat-template-file", "--reasoning-format", "--reasoning-budget",
           "--temp", "--top-k", "--top-p", "--min-p", "--repeat-penalty",
           "--presence-penalty", "--frequency-penalty", "-s", "--seed",
           "--metrics", "--slots", "--no-webui", "-to", "--timeout",
           "--log-file", "--log-disable", "--log-verbosity", "-v",
           "--verbose", "--no-mmproj-offload", "--mmproj-offload"}
FLAG = re.compile(r"^--?[A-Za-z]")


def unbenched(argv):
    """The flags in argv llama-bench did not run with."""
    return sorted({a.split("=", 1)[0] for a in speed_argv(argv)
                   if FLAG.match(a)} - BENCHED)


def speed_argv(argv):
    """A command as far as speed goes: without the binary, the port, and
    the model paths (the model is named by its bytes instead)."""
    out, skip = [], False
    for a in list(argv or [])[1:]:
        if skip:
            skip = False
        elif a in ("--port", "-m", "--model", "-mm", "--mmproj"):
            skip = True
        elif a.split("=", 1)[0] not in ("--port", "--model", "--mmproj"):
            out.append(a)
    return out


def profile_key(model_hash, build, binary, flags, argv=None):
    """What must match for a measurement to describe a configuration.

    The modelled flags are not enough: -ot, a split mode or a draft model
    change the speed and are not among them. So the whole command goes
    in when there is one - a preset's, or the running server's."""
    body = {"model": model_hash, "build": _build_id(build),
            "binary": _binary_id(binary), "flags": flags.as_dict()
            if hasattr(flags, "as_dict") else dict(flags),
            "argv": speed_argv(argv) if argv else None}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()
                          ).hexdigest()[:16]


def bucket(depth):
    """The measured depth a sample is filed under: the largest of DEPTHS
    at or below it, so a 20k-token prompt counts toward 16k."""
    return max(d for d in DEPTHS if d <= max(0, depth))


def from_samples(samples):
    """{"tg": {depth: t/s}, "pp": t/s, "n": {depth: count}} from
    (depth, decode t/s, decode tokens, prefill t/s, prefill tokens)
    samples, taking the median per depth so one slow reply - a cold
    cache, a busy GPU - does not become the profile."""
    tg, pp = {}, []
    for depth, dec, n_dec, pre, n_pre in samples:
        if dec and n_dec >= MIN_DECODE:
            tg.setdefault(bucket(depth), []).append(dec)
        if pre and n_pre >= MIN_PREFILL:
            pp.append(pre)
    if not tg and not pp:
        return None
    return {"tg": {d: statistics.median(v) for d, v in sorted(tg.items())},
            "n": {d: len(v) for d, v in sorted(tg.items())},
            "pp": statistics.median(pp) if pp else None}


def record_profile(source, model_path, model_hash, build, binary, flags,
                   measured, argv=None):
    """Book a measurement against its exact configuration."""
    if not measured:
        return None
    entry = {"kind": "profile", "source": source, "model": str(model_path),
             "model_hash": model_hash, "build": _build_no(build),
             "binary": binary.get("path") if isinstance(binary, dict)
             else str(binary or ""),
             "flags": flags.as_dict() if hasattr(flags, "as_dict")
             else dict(flags),
             "key": profile_key(model_hash, build, binary, flags, argv),
             "argv": speed_argv(argv) if argv else None,
             "tg": {str(d): round(v, 2) for d, v in measured["tg"].items()},
             "n": {str(d): v for d, v in (measured.get("n") or {}).items()},
             "pp": round(measured["pp"], 1) if measured.get("pp") else None}
    append(entry)
    return entry


def profiles(model_hash):
    """The latest profile per configuration for one model's bytes,
    newest first."""
    latest = {}
    for i, e in enumerate(load("profile")):
        if e.get("model_hash") == model_hash:
            latest[e.get("key")] = (i, e)       # later lines win
    # file order, not the timestamp: two measured in one second still
    # come out in the order they were made
    return [e for _, e in sorted(latest.values(), key=lambda p: -p[0])]


def seen_failures(model_path):
    return [e for e in load("log") if e.get("model") == str(model_path)
            and e.get("failed")]


def env_fit_bin(binary):
    return os.environ.get("MDL_FIT_PARAMS") or hw.sibling(binary,
                                                          "llama-fit-params")


def env_bench_bin(binary):
    return os.environ.get("MDL_BENCH") or hw.sibling(binary, "llama-bench")
