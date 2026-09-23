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
finally:
    hw.nvidia, hw.llama_devices = REAL["nvidia"], REAL["devices"]
    hw.usage.snapshot, hw.llama_build = REAL["snapshot"], REAL["build"]
    hw.sibling = REAL["sibling"]

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
