"""What this machine has right now, and what it has been measured doing.

Free VRAM is read live at fit time, not taken off the box: the desktop and
a browser eat a few hundred MB, and it moves. llama.cpp's own view of the
device (--list-devices) wins over nvidia-smi when both answer, because
llama.cpp is the one that will be allocating - and under Vulkan on Windows
it is the right one: the driver's budget counts what idle desktop apps
will give back, nvidia-smi counts it as taken, and the gap is a gigabyte.

RAM has two numbers. What is free right now is not the limit: a load
that needs more makes the OS page idle programs out, slowly but fine. The
limit is the total less what the OS and desktop cannot do without.

Both are planned for the machine at idle, not as the scan finds it: a
game or a browser full of tabs open during the scan is taken back off
(see usage.py). --now plans for the machine as it is this minute.
"""

import copy
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import usage

MiB = 1 << 20
GiB = 1 << 30
DEFAULT_MARGIN = 256 * MiB       # held back on the card, learned per machine
OS_HEADROOM = 1024 * MiB         # RAM left alone when counting 'free now'
RAM_RESERVE = 3 * GiB            # RAM the OS and desktop cannot give up
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def config_dir():
    """$MDL_FIT_HOME, else the same place mdl keeps models.toml."""
    if os.environ.get("MDL_FIT_HOME"):
        return Path(os.environ["MDL_FIT_HOME"])
    root = os.environ.get("XDG_CONFIG_HOME")
    return (Path(root) if root else Path.home() / ".config") / "mdl"


def cache_dir():
    if os.environ.get("MDL_FIT_HOME"):
        return Path(os.environ["MDL_FIT_HOME"]) / "cache"
    root = os.environ.get("XDG_CACHE_HOME")
    return (Path(root) if root else Path.home() / ".cache") / "mdl"


def _run(argv, timeout=20):
    try:
        p = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout, creationflags=NO_WINDOW,
                           errors="replace")
        return p.stdout + p.stderr
    except (OSError, subprocess.SubprocessError):
        return ""


# ------------------------------------------------------------------ GPU --

def nvidia():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    fields = ("name,memory.total,memory.free,memory.used,driver_version,"
              "pcie.link.gen.current,pcie.link.gen.max,"
              "pcie.link.width.current,pcie.link.width.max")
    out = _run([exe, "--query-gpu=" + fields,
                "--format=csv,noheader,nounits"], timeout=8)
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 9:
            continue

        def num(x):
            try:
                return int(float(x))
            except ValueError:
                return None
        gpus.append({"name": parts[0], "total": (num(parts[1]) or 0) * MiB,
                     "free": (num(parts[2]) or 0) * MiB,
                     "used": (num(parts[3]) or 0) * MiB,
                     "driver": parts[4],
                     "pcie": [num(parts[5]), num(parts[6]),
                              num(parts[7]), num(parts[8])]})
    return gpus


_DEVICE = re.compile(r"^\s*(\w+?)(\d+):\s*(.+?)\s*\((\d+) MiB,\s*(\d+) MiB free\)",
                     re.M)


def llama_devices(binary):
    """[(backend, index, name, total, free)] as llama.cpp sees them."""
    out = _run([binary, "--list-devices"], timeout=30)
    return [(m.group(1), int(m.group(2)), m.group(3),
             int(m.group(4)) * MiB, int(m.group(5)) * MiB)
            for m in _DEVICE.finditer(out)]


# ------------------------------------------------------------------ RAM --

class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def ram():
    """(total, available) bytes, or (None, None)."""
    if os.name == "nt":
        st = _MemStatus()
        st.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullTotalPhys, st.ullAvailPhys
        return None, None
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            info[key] = int(rest.split()[0]) * 1024
        return info.get("MemTotal"), info.get("MemAvailable",
                                              info.get("MemFree"))
    except (OSError, ValueError, IndexError):
        pass
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        try:
            total = int(out.strip())
            return total, total // 2          # no cheap honest number; halve
        except ValueError:
            pass
    return None, None


# ------------------------------------------------------------------ CPU --

def cpu_cores():
    """(logical, physical, performance) core counts; None where unknown.

    On a hybrid Intel chip the performance-core count is the one that
    matters: decode threads parked on E-cores drag the rest down.
    """
    logical = os.cpu_count()
    if os.name == "nt":
        try:
            return (logical,) + _win_cores()
        except (OSError, ValueError, AttributeError):
            return logical, None, None
    try:
        cores, classes = set(), {}
        block = {}
        for line in Path("/proc/cpuinfo").read_text().splitlines() + [""]:
            if not line.strip():
                if block:
                    cores.add((block.get("physical id"), block.get("core id")))
                block = {}
                continue
            k, _, v = line.partition(":")
            block[k.strip()] = v.strip()
        physical = len(cores) or None
        base = Path("/sys/devices/system/cpu")
        for d in base.glob("cpu[0-9]*/cpufreq/cpuinfo_max_freq"):
            classes[d.read_text().strip()] = classes.get(
                d.read_text().strip(), 0) + 1
        perf = None
        if len(classes) > 1 and physical:
            top = max(classes, key=int)
            perf = sum(1 for _ in range(classes[top]))
            perf = max(1, perf // max(1, (logical or 1) // physical))
        return logical, physical, perf
    except OSError:
        return logical, None, None


def _win_cores():
    k32 = ctypes.windll.kernel32
    size = ctypes.c_ulong(0)
    k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
        raise OSError("GetLogicalProcessorInformationEx failed")
    raw, off, classes = buf.raw, 0, []
    while off < size.value:
        rel = int.from_bytes(raw[off:off + 4], "little")
        rec = int.from_bytes(raw[off + 4:off + 8], "little")
        if rel == 0 and rec:              # RelationProcessorCore
            classes.append(raw[off + 9])  # EfficiencyClass
        off += rec or size.value
    physical = len(classes) or None
    perf = None
    if classes and len(set(classes)) > 1:
        perf = classes.count(max(classes))
    return physical, perf


# ---------------------------------------------------------- llama.cpp --

_BUILD = re.compile(r"build (\d+)(?:.*?commit ([0-9a-f]+))?", re.I)


def llama_build(binary):
    """{'build': 10424, 'commit': ..., 'backend': 'Vulkan', ...} or {}."""
    out = _run([binary, "--version"])
    m = _BUILD.search(out)
    info = {"build": int(m.group(1)) if m else None,
            "commit": m.group(2) if m else None, "version_text":
            out.strip().splitlines()[0] if out.strip() else ""}
    helptext = _run([binary, "--help"], timeout=20)
    info["load_mode"] = "--load-mode" in helptext
    info["fit_flag"] = re.search(r"-fit,\s+--fit", helptext) is not None
    info["no_mmproj_offload"] = "--no-mmproj-offload" in helptext
    return info


_LIBS = {}
_ARCHES = {}


def arch_supported(binary, arch):
    """True if this llama.cpp build names `arch`, False if it does not,
    None when its library cannot be read. Every architecture llama.cpp
    loads is a C string in libllama; one it has never heard of is not.
    Asked once per model in the catalog, so the answer is kept: a PATH
    search and a scan of the library each time cost `find` seconds."""
    if (binary, arch) in _ARCHES:
        return _ARCHES[binary, arch]
    exe = Path(shutil.which(binary) or binary)
    if exe not in _LIBS:
        blob = b""
        for p in [exe.parent / n for n in ("llama.dll", "libllama.so",
                                           "libllama.dylib")] + [
                exe.parent.parent / "lib" / "libllama.so", exe]:
            try:
                if p.is_file() and (p.suffix in (".dll", ".so", ".dylib")
                                    or not blob):
                    blob += p.read_bytes()
            except OSError:
                pass
        _LIBS[exe] = blob
    blob = _LIBS[exe]
    got = None if not blob or not arch else arch.encode() + b"\x00" in blob
    _ARCHES[binary, arch] = got
    return got


def sibling(binary, name):
    """llama-bench or llama-fit-params next to llama-server, else PATH."""
    exe = name + (".exe" if os.name == "nt" else "")
    found = shutil.which(binary) or binary
    here = Path(found).parent / exe
    if here.is_file():
        return str(here)
    return shutil.which(name)


# ---------------------------------------------------------------- probe --

class Machine:
    """Everything the fit needs to know about the box, in one place."""

    def __init__(self, **kw):
        self.gpu_name = kw.get("gpu_name", "no GPU")
        self.backend = kw.get("backend", "CPU")
        self.vram_total = kw.get("vram_total", 0)
        self.vram_free = kw.get("vram_free", 0)
        self.margin = kw.get("margin", DEFAULT_MARGIN)
        # {arch: bytes}, raised by `mdl fit --verify` when a config it
        # predicted to fit ran out of memory; see for_arch()
        self.margin_arch = dict(kw.get("margin_arch") or {})
        self.ram_total = kw.get("ram_total", 0)
        self.ram_avail = kw.get("ram_avail", 0)
        self.os_headroom = kw.get("os_headroom", OS_HEADROOM)
        self.ram_reserve = kw.get("ram_reserve", RAM_RESERVE)
        self.cores = kw.get("cores", (os.cpu_count(), None, None))
        self.pcie = kw.get("pcie")
        self.driver = kw.get("driver")
        self.build = kw.get("build", {})
        self.bench = kw.get("bench", {})         # measured bandwidths
        self.binary = kw.get("binary")
        self.notes = kw.get("notes", [])
        # vram_free and ram_avail are what the plan is for: the machine at
        # idle, or as it is now (plan "now"). The _now figures are always
        # this minute's.
        self.plan = kw.get("plan", "now")
        self.vram_free_now = kw.get("vram_free_now", self.vram_free)
        self.ram_avail_now = kw.get("ram_avail_now", self.ram_avail)
        self.idle = kw.get("idle")            # usage.Baseline, when probed
        self.snap = kw.get("snap")            # usage.Snapshot, when probed
        self.cpu_load = kw.get("cpu_load")

    @property
    def vram_usable(self):
        return max(0, self.vram_free - self.margin)

    def for_arch(self, arch):
        """This machine as a fit of `arch` sees it: with the margin that
        `mdl fit --verify` raised for that arch after llama-bench ran out
        of memory. It used to book the margin and say so, and nothing
        read it back - the next fit made the same prediction."""
        margin = self.margin_arch.get(arch)
        if not isinstance(margin, int) or margin <= self.margin:
            return self
        out = copy.copy(self)
        out.margin = margin
        return out

    @property
    def ram_usable(self):
        """The hard limit: past this the machine thrashes."""
        if self.ram_total:
            return max(0, self.ram_total - self.ram_reserve)
        return self.ram_free_idle

    @property
    def ram_free_idle(self):
        """Past this, loading pages other programs out first."""
        return max(0, self.ram_avail - self.os_headroom)

    @property
    def ram_free_now(self):
        return max(0, self.ram_avail_now - self.os_headroom)

    def held(self, what):
        """[(app, bytes)] of 'vram' or 'ram' held by apps open now."""
        return self.snap.held(what) if self.snap else []

    @property
    def max_alloc(self):
        """Largest single buffer the backend will hand out. Vulkan drivers
        cap it (4 GiB on NVIDIA); a 262k-vocab logits tensor at ubatch
        4096 hits exactly that, and llama.cpp moves it to the CPU."""
        return 4 * GiB if self.backend == "Vulkan" else 1 << 62

    @property
    def calibrated(self):
        return bool(self.bench.get("bw_gpu"))

    def as_dict(self):
        return dict(self.__dict__)

    def summary(self):
        """RTX 3060 · 11.5 G free at idle (11.3 usable, 10.4 now) ·
        16 G RAM (10.3 free at idle, 7.7 now, 12.8 max)"""
        g = self.gpu_name.replace("NVIDIA GeForce ", "")
        when = "at idle" if self.plan == "idle" else "now"
        parts = []
        if self.vram_total:
            now = ""
            if self.vram_free - self.vram_free_now >= 64 * MiB:
                now = ", %.1f now" % (self.vram_free_now / GiB)
            parts.append("%s · %.1f G free %s (%.1f usable%s)"
                         % (g, self.vram_free / GiB, when,
                            self.vram_usable / GiB, now))
        else:
            parts.append("no GPU found")
        if self.ram_total:
            now = ""
            if self.ram_avail - self.ram_avail_now >= 256 * MiB:
                now = ", %.1f now" % (self.ram_avail_now / GiB)
            parts.append("%.0f G RAM (%.1f free %s%s, %.1f max)"
                         % (self.ram_total / GiB, self.ram_avail / GiB, when,
                            now, self.ram_usable / GiB))
        parts.append("calibrated ✓" if self.calibrated else "uncalibrated")
        return " · ".join(parts)


def hw_path():
    return config_dir() / "hw.json"


def load_saved():
    try:
        return json.loads(hw_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def update(change, busy_ok=False):
    """Apply change(saved) to hw.json and write it back: read, changed
    and written under one lock, so each writer adds only what it knows.

    Every fit, find and eval probes the machine and books its build, and
    a probe used to write back the whole file as it had read it at the
    start - over whatever a calibration measured in the meantime.
    True if written. With busy_ok, a file another mdl holds is left for
    it (False) rather than an error: a probe must not fail a fit on it.
    """
    import mdl
    path = hw_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with mdl.file_lock(path.with_name(path.name + ".lock"),
                           "another mdl is writing %s; try again" % path,
                           tries=10 if busy_ok else 50):
            saved = load_saved()
            change(saved)
            mdl.write_atomic(path, json.dumps(saved, indent=1))
    except mdl.MdlError:
        if not busy_ok:
            raise
        return False
    return True


def probe(binary="llama-server", quick=False, now=False):
    """A Machine, planned for at idle (now: as it is this minute). quick
    skips llama.cpp's own device query, which has to start the backend,
    and the per-app VRAM and CPU readings; a second or two each."""
    saved = load_saved()
    kw = {"binary": binary, "notes": []}
    gpus = nvidia()
    if gpus:
        g = gpus[0]
        kw.update(gpu_name=g["name"], backend="CUDA", vram_total=g["total"],
                  vram_free=g["free"], pcie=g["pcie"], driver=g["driver"])
        if len(gpus) > 1:
            kw["notes"].append("%d GPUs found; mdl fit models the first "
                               "only" % len(gpus))
    bkey = _binary_key(binary)
    build = saved.get("builds", {}).get(bkey)
    if not build and shutil.which(binary) or not build and Path(binary).is_file():
        build = llama_build(binary)
    kw["build"] = build or {}
    devs = [] if quick else llama_devices(binary)
    if devs:
        backend, _, name, total, free = devs[0]
        kw.update(backend=backend, gpu_name=name, vram_total=total,
                  vram_free=free)
    elif gpus:
        kw["backend"] = backend_for(binary, saved)
    total, avail = ram()
    kw.update(ram_total=total or 0, ram_avail=avail or 0,
              cores=cpu_cores())
    used_vram = gpus[0]["used"] if gpus else (
        devs[0][3] - devs[0][4] if devs else None)
    snap = usage.snapshot(used_vram, total - avail if total and avail
                          else None, quick)
    idle = saved.get("idle", {})
    base = usage.baseline(snap, idle)
    kw.update(vram_free_now=kw.get("vram_free", 0), ram_avail_now=avail or 0,
              idle=base, snap=snap, cpu_load=snap.cpu,
              ram_reserve=max(RAM_RESERVE, usage.TYPICAL.get(
                  snap.os_key, (0, 0))[0]))
    if not now:
        kw["plan"] = "idle"
        kw["vram_free"] = usage.plan_free(kw.get("vram_total", 0),
                                          kw.get("vram_free", 0), used_vram,
                                          base.vram)
        if total:
            kw["ram_avail"] = max(avail or 0, total - base.ram)
    booked = usage.record_boot(idle, snap, base)
    kw["margin"] = int(saved.get("margin", DEFAULT_MARGIN))
    kw["margin_arch"] = {str(k): int(v) for k, v in (
        saved.get("margin_arch") or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)}
    bench = saved.get("bench", {})
    if bench.get("build") and build and bench.get("build") != build.get("build"):
        kw["notes"].append("calibrated on build %s, running %s; speeds may "
                           "have moved" % (bench.get("build"), build.get("build")))
    kw["bench"] = bench
    if os.name == "nt" and kw.get("backend") == "CUDA":
        kw["notes"].append("set 'CUDA - Sysmem Fallback Policy' to 'Prefer No "
                           "Sysmem Fallback' for llama-server.exe, or an "
                           "overflow runs at 3 t/s instead of failing")
    m = Machine(**kw)
    # only what hw.json does not have yet: a build read from it, or found
    # again unchanged, used to rewrite the file on every fit, find and eval
    new_build = bool(build) and (
        saved.get("builds", {}).get(bkey) != build
        or saved.get("backends", {}).get(bkey) != m.backend)

    def book(now):
        # only what this probe learned, onto the file as it is now
        if new_build:
            now.setdefault("builds", {})[bkey] = build
            now.setdefault("backends", {})[bkey] = m.backend
        if booked:          # one sample per boot, however many probes race
            usage.record_boot(now.setdefault("idle", {}), snap, base)
    if new_build or booked:
        try:
            update(book, busy_ok=True)
        except OSError:
            pass
    return m


def set_idle(**values):
    """Book what the machine holds at idle, in bytes ('vram', 'ram');
    with no values, forget it and go back to measuring."""
    def change(saved):
        if not values:
            saved.pop("idle", None)
            return
        fixed = saved.setdefault("idle", {}).setdefault("set", {})
        fixed.update({k: int(v) for k, v in values.items() if v is not None})
        fixed["at"] = time.strftime("%Y-%m-%d %H:%M")
    update(change)


def _binary_key(binary):
    path = shutil.which(binary) or binary
    try:
        return "%s@%d" % (path, int(Path(path).stat().st_mtime))
    except OSError:
        return str(path)


def backend_for(binary, saved=None):
    """The backend this build runs on, without starting it: what a probe
    of this binary booked, else what the libraries beside it are built
    for, else Metal on a Mac and CUDA elsewhere."""
    saved = load_saved() if saved is None else saved
    got = saved.get("backends", {}).get(_binary_key(binary))
    if got:
        return got
    here = Path(shutil.which(binary) or binary).parent
    names = " ".join(p.name.lower() for p in here.glob("*ggml*"))
    for key, name in (("cuda", "CUDA"), ("vulkan", "Vulkan"),
                      ("hip", "ROCm"), ("metal", "Metal")):
        if key in names:
            return name
    if names:
        return "CPU"            # its ggml libraries, and none for a GPU
    # nothing beside it: a static build, as Metal ones usually are
    return "Metal" if sys.platform == "darwin" else "CUDA"


def _guess_backend(binary):
    here = Path(shutil.which(binary) or binary).parent
    names = " ".join(p.name.lower() for p in here.glob("*ggml*"))
    for key, name in (("cuda", "CUDA"), ("vulkan", "Vulkan"),
                      ("hip", "ROCm"), ("metal", "Metal")):
        if key in names:
            return name
    return "CUDA"


def record_bench(values):
    bench = dict(values, at=time.strftime("%Y-%m-%d %H:%M"))
    update(lambda saved: saved.update(bench=bench))


def raise_margin(arch, margin):
    """Book a bigger VRAM margin for one arch; see Machine.for_arch."""
    update(lambda saved: saved.setdefault("margin_arch", {}).update(
        {arch: int(margin)}))
