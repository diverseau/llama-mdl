"""Calibration: the oracle, the benchmark, and what they taught us.

llama.cpp is the source of truth. `llama-fit-params -fitp on` prints the
model, context and compute bytes per device for a set of flags without
allocating anything, in well under a second, so it is cheap enough to
run in the loop: the search proposes, the oracle checks, the difference
is stored against this exact file and build, and the next prediction
starts from it. llama-bench does the same job for speed.

Everything lands in calib.jsonl, one line per observation.
"""

import json
import os
import re
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


def append(entry):
    entry = dict(entry, at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    path = calib_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass


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
    if threads:
        argv += ["-t", str(threads)]
    return argv


def run_bench(bench_bin, model_path, flags, timeout=1800, **kw):
    """{'pp': t/s, 'tg': {depth: t/s}} from llama-bench, or None (which
    is also what an OOM looks like)."""
    try:
        p = subprocess.run(bench_argv(bench_bin, model_path, flags, **kw),
                           capture_output=True, text=True, timeout=timeout,
                           creationflags=hw.NO_WINDOW, errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_bench(p.stdout)


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
        entry = {"kind": "log", "name": name, "model": model_path,
                 "flags": flags.as_dict(), "build": build}
        entry.update(found)
        append(entry)
        return entry
    except Exception:          # noqa: BLE001 - passive by definition
        return None


def seen_failures(model_path):
    return [e for e in load("log") if e.get("model") == str(model_path)
            and e.get("failed")]


def env_fit_bin(binary):
    return os.environ.get("MDL_FIT_PARAMS") or hw.sibling(binary,
                                                          "llama-fit-params")


def env_bench_bin(binary):
    return os.environ.get("MDL_BENCH") or hw.sibling(binary, "llama-bench")
