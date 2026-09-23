"""mdl fit --verify, mdl fit hw and the machine probe, against stand-ins
for llama-bench and llama-fit-params that answer the way the real ones
do - so the learning they drive (the oracle's misses, the measured
speeds, the margin an out-of-memory run books) is exercised without
llama.cpp or a GPU.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                   # noqa: E402
from support import llama, mdl, run, sandbox, teardown  # noqa: E402

from mdl_fit import calib, hw, model  # noqa: E402

t = support.Tally("test_fit_verify")
check = t.check
MiB, GiB = model.MiB, model.GiB
TMP = Path(tempfile.mkdtemp(prefix="mdl-fit-verify-test-"))
os.environ["MDL_FIT_HOME"] = str(TMP / "home")   # never the real ~/.config
REAL = {"probe": hw.probe, "sibling": hw.sibling, "nvidia": hw.nvidia,
        "devices": hw.llama_devices, "snapshot": hw.usage.snapshot,
        "build": hw.llama_build}


def tool(name, code):
    """A runnable stand-in: `code` run by this Python, with any args."""
    script = TMP / (name + ".py")
    script.write_text(code, encoding="utf-8")
    if os.name == "nt":
        path = TMP / (name + ".cmd")
        path.write_text('@echo off\r\n"%s" "%s" %%*\r\n' % (sys.executable,
                                                             script),
                        encoding="ascii")
    else:
        path = TMP / name
        path.write_text('#!/bin/sh\nexec "%s" "%s" "$@"\n' % (sys.executable,
                                                              script),
                        encoding="ascii")
        path.chmod(0o755)
    return str(path)


# llama-bench: rows as -o json writes them. $FAKE_BENCH picks how it
# fails; the speeds move with -ub and -t so calibration has something to
# solve for, and every command is logged for the checks below.
BENCH = tool("llama-bench", r'''
import json, os, sys
a = sys.argv[1:]
with open(os.environ["FAKE_BENCH_LOG"], "a") as fh:
    fh.write(" ".join(a) + "\n")
mode = os.environ.get("FAKE_BENCH", "ok")
if mode == "oom":
    sys.stderr.write("ggml_backend_cuda_buffer_type_alloc_buffer: "
                     "allocating 9000.00 MiB on device 0: cudaMalloc "
                     "failed: out of memory\n")
    sys.exit(1)
if mode == "unsupported":
    sys.stderr.write("error: unknown argument: -ncmoe\n")
    sys.exit(1)
def arg(flag):
    return a[a.index(flag) + 1] if flag in a else None
p, n, ub = int(arg("-p")), int(arg("-n")), int(arg("-ub"))
threads = int(arg("-t") or 0)
rows = []
if p:
    seconds = (p // ub) * 0.05 + 0.2
    rows.append({"n_prompt": p, "n_gen": 0, "n_depth": 0,
                 "avg_ts": p / seconds})
if n:
    for d in arg("-d").split(","):
        rows.append({"n_prompt": 0, "n_gen": n, "n_depth": int(d),
                     "avg_ts": 50.0 + threads - int(d) / 1000})
print(json.dumps(rows))
''')
# llama-fit-params: one row per device, MiB of model, context, compute
FIT = tool("llama-fit-params", "print('CUDA0 900 100 200')\n"
                               "print('Host 10 0 30')\n")
LOG = TMP / "bench.log"
os.environ["FAKE_BENCH_LOG"] = str(LOG)


def fixed(binary="llama-server", quick=False, now=False):
    """The same card every time, with the margins hw.json has booked."""
    return hw.Machine(gpu_name="NVIDIA GeForce RTX 3060", backend="CUDA",
                      vram_total=12 * GiB, vram_free=11 * GiB,
                      margin_arch=hw.load_saved().get("margin_arch"),
                      ram_total=32 * GiB, ram_avail=24 * GiB,
                      cores=(16, 8, None), binary=binary,
                      build={"build": 4242, "load_mode": True,
                             "fit_flag": True})


dense = llama(TMP / "Tiny-Dense-Q8_0.gguf")
moe = llama(TMP / "Tiny-MoE-Q4_K.gguf", arch="qwen3moe", experts=8, used=2)
hw.probe = fixed
hw.sibling = lambda binary, name: None     # not whatever is on this PATH
root, port = sandbox(model=dense, extra='\n[moe]\nmodel = "%s"\nport = %d\n'
                     % (str(moe).replace("\\", "/"), support.free_port()))

# ---------------------------------------------------------- --verify ----
out, err, code = run(mdl.cmd_fit, ["demo", "--verify"])
check("with neither tool, --verify says what it skipped",
      (code, "memory   no llama-fit-params" in out,
       "speed    no llama-bench found; skipped" in out), (0, True, True))

os.environ["MDL_FIT_PARAMS"], os.environ["MDL_BENCH"] = FIT, BENCH
out, err, code = run(mdl.cmd_fit, ["demo", "--verify"])
oracle = calib.load("oracle")
bench = calib.load("bench")
check("the oracle's answer is shown beside the prediction, and booked",
      (code, "oracle 1.2 G  (model -898  ctx -96  compute -183 MiB)" in out,
       [e["actual"] for e in oracle]),
      (0, True, [[900 * MiB, 100 * MiB, 200 * MiB]]))
check("llama-bench runs the preset's config: its cache types and -fa",
      all("-ctk q8_0 -ctv q8_0" in line and "-fa on" in line
          for line in LOG.read_text().splitlines()), True)
check("both speeds are measured, booked, and set against the prediction",
      (sorted(bench[-1]["tg"]), bench[-1]["pp"] > 0,
       "decode @0" in out, "prefill   predicted" in out,
       "recorded; the next fit for llama uses it" in out),
      (["0", "3840"], True, True, True, True))
check("and it is booked as the preset's measured profile",
      [e["kind"] for e in calib.load("profile")], ["profile"])

os.environ["FAKE_BENCH"] = "unsupported"
_, err, code = run(mdl.cmd_fit, ["demo", "--verify"])
check("a flag this build does not take is not a verdict on memory",
      (code, "did not complete (unsupported)" in err,
       "margin_arch" in hw.load_saved()), (1, True, False))

os.environ["FAKE_BENCH"] = "oom"
_, err, code = run(mdl.cmd_fit, ["demo", "--verify"])
first = hw.load_saved().get("margin_arch", {}).get("llama")
check("running out of memory books a bigger margin for the arch, "
      "in one line", (code, len(err.splitlines()), first),
      (1, 1, hw.DEFAULT_MARGIN + 256 * MiB))
check("and says what it is", "is now %d MiB" % (first // MiB) in err, True)
_, err, code = run(mdl.cmd_fit, ["demo", "--verify"])
check("a second one grows it from there, not from the default again",
      hw.load_saved()["margin_arch"]["llama"], first + 256 * MiB)
out, err, code = run(mdl.cmd_fit, ["demo", "--no-oracle"])
check("and the next fit holds it back",
      (code, "%d MiB held back on the card for llama" % (
          (first + 256 * MiB) // MiB) in out), (0, True))
os.environ.pop("FAKE_BENCH")

# ------------------------------------------------------------ fit hw ----
out, err, code = run(mdl.cmd_fit, ["hw", "--probe"])
check("fit hw --probe describes the machine and measures nothing",
      (code, "machine  RTX 3060" in out, "build 4242" in out,
       "cpu      16 logical · 8 physical" in out, "calibrated:" in out),
      (0, True, True, True, False))

LOG.write_text("")
out, err, code = run(mdl.cmd_fit, ["hw"])
saved = hw.load_saved()["bench"]
check("fit hw measures the GPU on the dense model and the CPU on the MoE",
      (code, "calibrated: bw_gpu" in out, saved.get("bw_gpu", 0) > 0,
       saved.get("bw_cpu", 0) > 0, saved.get("build")), (0, True, True,
                                                           True, 4242))
check("prefill at two ubatches gives the PCIe rate",
      (saved.get("bw_pcie", 0) > 0, "bw_pcie" in out), (True, True))
check("and the thread count is the fastest one tried",
      (saved.get("threads"), sorted(
          line.split("-t ")[1].split()[0] for line in
          LOG.read_text().splitlines() if " -t " in line)),
      (16, ["16", "8"]))

os.environ["FAKE_BENCH"] = "unsupported"
_, err, code = run(mdl.cmd_fit, ["hw"])
check("no benchmark completing is said, in one line",
      (code, "no benchmark completed" in err, len(err.splitlines())),
      (1, True, 1))
os.environ.pop("FAKE_BENCH")
os.environ.pop("MDL_BENCH")
_, err, code = run(mdl.cmd_fit, ["hw"])
check("with no llama-bench, fit hw says where it looked",
      (code, "llama-bench not found" in err), (1, True))
os.environ.pop("MDL_FIT_PARAMS")
teardown(root)

# -------------------------------------------------------------- probe ---
hw.probe = REAL["probe"]
hw.hw_path().write_text(json.dumps({
    "margin": 600 * MiB, "bench": {"build": 1000, "bw_gpu": 3e11},
    "margin_arch": {"llama": 900 * MiB, "odd": "x", "flag": True}}),
    encoding="utf-8")
card = {"name": "NVIDIA GeForce RTX 3060", "total": 12 * GiB,
        "free": 10 * GiB, "used": 2 * GiB, "driver": "580.1",
        "pcie": [4, 4, 16, 16]}
hw.nvidia = lambda: [card, dict(card, name="NVIDIA GeForce RTX 3090")]
hw.llama_devices = lambda binary: []
hw.llama_build = lambda binary: {"build": 4242}
hw.usage.snapshot = lambda *a, **k: hw.usage.Snapshot("linux", up=10 ** 6)
binary = TMP / "llama-server"
binary.write_text("")
try:
    m = hw.probe(str(binary), quick=True, now=True)
    check("the probe reads the first card, and says there are more",
          (m.gpu_name, m.vram_total, m.pcie, m.driver,
           any("2 GPUs found" in n for n in m.notes)),
          (card["name"], 12 * GiB, [4, 4, 16, 16], "580.1", True))
    check("it plans for this minute when asked to",
          (m.plan, m.vram_free), ("now", 10 * GiB))
    check("it reads the margins booked, and drops what is not a size",
          (m.margin, m.margin_arch), (600 * MiB, {"llama": 900 * MiB}))
    check("a calibration from another build is noted",
          any("calibrated on build 1000, running 4242" in n
              for n in m.notes), True)
    check("and the build it found is booked for the next probe",
          [b["build"] for b in hw.load_saved()["builds"].values()], [4242])
    hw.llama_devices = lambda binary: [("Vulkan", 0, "Radeon", 16 * GiB,
                                        15 * GiB)]
    m = hw.probe(str(binary), now=True)
    check("llama.cpp's own device list wins over nvidia-smi",
          (m.backend, m.gpu_name, m.vram_total), ("Vulkan", "Radeon",
                                                  16 * GiB))
    hw.nvidia = lambda: []
    hw.llama_devices = lambda binary: []
    m = hw.probe(str(binary), quick=True)
    check("with no GPU at all it is a CPU machine, planned at idle",
          (m.backend, m.vram_total, m.plan), ("CPU", 0, "idle"))
    writes, real_update = [], hw.update
    hw.update = lambda change, busy_ok=False: (
        writes.append(1), real_update(change, busy_ok))[1]
    try:
        hw.probe(str(binary), quick=True)
        before = len(writes)
        hw.probe(str(binary), quick=True)
    finally:
        hw.update = real_update
    check("a probe that learns nothing new leaves hw.json alone",
          len(writes) - before, 0)
    saved = {"backends": {hw._binary_key(str(binary)): "Vulkan"}}
    libs = TMP / "rocm"
    libs.mkdir()
    (libs / "llama-server").write_text("")
    (libs / "ggml-hip.dll").write_text("")
    bare = TMP / "bare"
    bare.mkdir()
    (bare / "llama-server").write_text("")
    check("a build's backend: booked, else its libraries, else the platform",
          [hw.backend_for(str(binary), saved),
           hw.backend_for(str(libs / "llama-server"), saved),
           hw.backend_for(str(bare / "llama-server"), saved)],
          ["Vulkan", "ROCm", "Metal" if sys.platform == "darwin" else "CUDA"])
finally:
    hw.nvidia, hw.llama_devices = REAL["nvidia"], REAL["devices"]
    hw.usage.snapshot, hw.llama_build = REAL["snapshot"], REAL["build"]
    hw.sibling = REAL["sibling"]

# --------------------------------------------------------- verify_picks --
# The picks go in front of the oracle before they are shown; what it
# says does not fit sends the search round again, at most `rounds` times.
from mdl_fit import cli, gguf, search  # noqa: E402

inv = gguf.load(dense)
card = fixed()
opts = cli.options({})
said = {"n": 0, "over": 0}         # oracle calls; how many say "too big"
real_check, real_solve = search.calib.check, search.solve
solves = []


def oracle(fit_bin, inv_, shape, flags, build, max_alloc=None):
    said["n"] += 1
    if said["mode"] == "silent":
        return None
    big = said["mode"] == "always" or (said["mode"] == "once"
                                       and len(solves) == 0)
    if big:
        said["over"] += 1
    return {"actual": [20 * GiB if big else MiB, 0, 0]}


def counted(ctx_obj, opts_):
    solves.append(1)
    return real_solve(ctx_obj, opts_)


def verify(mode, fit_bin="llama-fit-params", source=None):
    said.update(n=0, over=0, mode=mode)
    solves.clear()
    ctx_obj = search.Context(inv, card)
    first = real_solve(ctx_obj, opts)
    if source:
        ctx_obj.inv.source = source
    search.calib.check, search.solve = oracle, counted
    try:
        got = search.verify_picks(ctx_obj, opts, first, fit_bin)
    finally:
        search.calib.check, search.solve = real_check, real_solve
        ctx_obj.inv.source = str(dense)
    return first, got


first, got = verify("agree", fit_bin=None)
check("with no oracle binary the picks are shown as they are",
      (got is first, said["n"]), (True, 0))
first, got = verify("agree", source="hf:org/repo/x.gguf")
check("nor for a model read from the hub: there is no file to load",
      (got is first, said["n"]), (True, 0))
first, got = verify("silent")
check("a build with no oracle is asked once, and the picks kept",
      (got is first, said["n"], [p.oracle for p in got.picks]),
      (True, 1, [None] * len(got.picks)))
first, got = verify("agree")
check("picks the oracle agrees with are kept, each checked, no re-search",
      (got is first, len(solves), all(p.oracle for p in got.picks),
       said["n"] == len(got.picks) > 0), (True, 0, True, True))
first, got = verify("once")
check("picks it says do not fit send the search round again",
      (len(solves), said["over"] > 0, all(p.oracle for p in got.picks)),
      (1, True, True))
first, got = verify("always")
check("at most `rounds` times, with the last round's picks still checked",
      (len(solves), all(p.oracle for p in got.picks)), (2, True))

# ----------------------------------------------------- process listing --
# What decides "idle": every process, its RAM, and the VRAM it holds.
from types import SimpleNamespace as NS  # noqa: E402

from mdl_fit import usage  # noqa: E402

procs, _ = usage.processes(quick=True)
me = next((p for p in procs if p.pid == os.getpid()), None)
# (whether it counts as the system's depends on how the tests are run:
# a CI runner can be a service, so that is left to the parsers below)
check("this machine's listing has this process, its parent and its RAM",
      (me is not None, me and me.ppid == os.getppid(), me and me.ram > 0),
      (True, True, True))

real = (usage.os, usage.sys, usage.shutil, usage._run, usage._procs_linux,
        usage._procs_windows)
TP = ('"(PDH-CSV 4.0)","\\\\PC\\GPU Process Memory(pid_2_luid_0x0_0x1'
      '_phys_0)\\Dedicated Usage"\r\n"09/11/2026 20:34:31.203","2097152.0"\r\n')
NV = ("| Processes:                                         |\n"
      "|    0   N/A  N/A         1      C   python            100MiB |\n")
try:
    usage._procs_windows = lambda: [usage.Proc(1, 0, "a"), usage.Proc(2, 0, "b")]
    usage._procs_linux = lambda quick: [usage.Proc(1, 0, "a", vram=5),
                                        usage.Proc(2, 0, "b")]
    usage._run = lambda argv, timeout=10: TP if argv[0] == "typeperf" else NV
    usage.shutil = NS(which=lambda name: "/usr/bin/" + name)
    usage.os, usage.sys = NS(name="nt"), NS(platform="win32")
    got, have = usage.processes()
    check("Windows: VRAM per process from the GPU counters",
          ([p.vram for p in got], have), ([0, 2 << 20], True))
    got, have = usage.processes(quick=True)
    check("and a quick listing does not wait for them",
          ([p.vram for p in got], have), ([0, 0], False))
    usage.os, usage.sys = NS(name="posix"), NS(platform="linux")
    got, have = usage.processes()
    check("Linux: the DRM figure, plus nvidia-smi's for its own processes",
          ([p.vram for p in got], have), ([5 + 100 * MiB, 0], True))

    def broken(quick):
        raise OSError("/proc is gone")
    usage._procs_linux = broken
    check("a listing that fails is empty, not a traceback",
          usage.processes(), ([], False))
finally:
    (usage.os, usage.sys, usage.shutil, usage._run, usage._procs_linux,
     usage._procs_windows) = real

PS = ("  1     0     0   1024 /sbin/launchd\n"
      "  50    1   501   2048 /Applications/Steam.app/Contents/MacOS/steam\n"
      "  51    1   501    512 /System/Library/CoreServices/Dock\n"
      "  52    1     0    256 /usr/sbin/cfprefsd\n"
      "  garbled line\n")
had_uid = hasattr(os, "getuid")
real_uid, real_run = getattr(os, "getuid", None), usage._run
os.getuid, usage._run = (lambda: 501), (lambda argv, timeout=10: PS)
try:
    mac = usage._procs_mac()
finally:
    usage._run = real_run
    if had_uid:
        os.getuid = real_uid
    else:
        del os.getuid
check("macOS: ps rows, RAM in bytes, and the system's own marked as such",
      [(p.pid, p.ppid, p.ram, p.system) for p in mac],
      [(1, 0, 1024 * 1024, True), (50, 1, 2048 * 1024, False),
       (51, 1, 512 * 1024, True), (52, 1, 256 * 1024, True)])

proc_dir = TMP / "proc" / "4242"
(proc_dir / "fd").mkdir(parents=True)
(proc_dir / "fdinfo").mkdir()
for fd, text in (("3", "drm-client-id: 7\ndrm-memory-vram: 1024 KiB\n"),
                 ("4", "drm-client-id: 7\ndrm-memory-vram: 1024 KiB\n"),
                 ("5", "drm-client-id: 8\ndrm-total-vram0: 2 MiB\n"),
                 ("6", "not a gpu\n")):
    (proc_dir / "fd" / fd).write_text("")
    (proc_dir / "fdinfo" / fd).write_text(text)
real_readlink = usage.os.readlink
usage.os.readlink = lambda p: ("/dev/dri/renderD128" if Path(p).name != "6"
                               else "/dev/null")
try:
    held = usage._drm_vram(proc_dir)
finally:
    usage.os.readlink = real_readlink
check("DRM fdinfo: VRAM per client, a client's second fd not counted twice",
      held, 1024 * 1024 + 2 * MiB)

# ---------------------------------------------------------- calib.jsonl --
calib.calib_path().unlink(missing_ok=True)
flags = {"ctx": 4096, "ub": 512}
for n in range(3):
    calib.append({"kind": "profile", "key": "k1", "tg": {"0": n}})
calib.append({"kind": "profile", "key": "k2", "tg": {"0": 9}})
for n in range(2):
    calib.append({"kind": "oracle", "sig": "s", "build": 1, "flags": flags,
                  "actual": [n, 0, 0]})
calib.append({"kind": "oracle", "sig": "s", "build": 2, "flags": flags,
              "actual": [7, 0, 0]})
for n in range(calib.LOGS_KEPT + 5):
    calib.append({"kind": "log", "model": "a.gguf", "n": n})
calib.append({"kind": "log", "model": "a.gguf", "failed": ["CUDA0", 1]})
calib.append({"kind": "log", "model": "b.gguf", "n": 0})
calib.append({"kind": "something-newer", "n": 1})
before = calib.load()
dropped = calib.compact()
after = calib.load()
check("compaction drops only what nothing reads",
      (dropped, len(before) - len(after)), (2 + 1 + 5, 8))
check("the latest profile per configuration is kept, in order",
      [(e["key"], e["tg"]["0"]) for e in calib.load("profile")],
      [("k1", 2), ("k2", 9)])
check("an oracle per file, build and flags: the latest",
      [e["actual"][0] for e in calib.load("oracle")], [1, 7])
check("the last load logs per model, and every failure",
      ([e["n"] for e in calib.load("log") if e["model"] == "a.gguf"
        and "n" in e][:1], len(calib.seen_failures("a.gguf")),
       sum(e["model"] == "b.gguf" for e in calib.load("log"))),
      ([5], 1, 1))
check("a kind it does not know is left alone",
      [e["n"] for e in calib.load("something-newer")], [1])
calib.MAX_BYTES, real_max = 1, calib.MAX_BYTES
calib.append({"kind": "profile", "key": "k1", "tg": {"0": 3}})
calib.MAX_BYTES = real_max
check("an append past the size limit compacts",
      [e["tg"]["0"] for e in calib.load("profile") if e["key"] == "k1"], [3])
check("and leaves no temp file or lock behind",
      sorted(p.name for p in calib.calib_path().parent.iterdir()
             if p.name.startswith("calib.jsonl")), ["calib.jsonl"])

log = TMP / "load.log"
log.write_text("build: 6789 (abc1234) with cc for x86_64\n"
               "load_tensors:   CUDA0 model buffer size =  1000.00 MiB\n",
               encoding="utf-8")
got = calib.passive("demo", ["srv", "-m", "a.gguf", "-c", "4096"], log)
check("a load log's own build is booked with what it shows",
      (got["build"], got["buffers"]["CUDA0"]["model"]), (6789, 1000 * MiB))

sys.exit(t.done())
