"""mdl setup, and the scan behind it: GGUFs already on this machine,
found where people and other tools keep them, and added as presets.

Only folders made here are scanned: scan.places() is replaced, so a
developer's own LM Studio or Hugging Face cache never reaches a test.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                   # noqa: E402
from support import run, sandbox, teardown        # noqa: E402

import mdl                                        # noqa: E402
from mdl_fit import hw, scan, setup               # noqa: E402

t = support.Tally("test_setup")
check = t.check

TMP = Path(tempfile.mkdtemp(prefix="mdl-setup-"))
MODELS, HF, LMS, OWN = (TMP / d for d in ("models", "hf", "lms", "own"))
for d in (MODELS, HF, LMS, OWN):
    d.mkdir()
scan.MIN_BYTES = 1000                  # the test's models are small
hw.nvidia = lambda: []
hw.llama_devices = lambda binary: []
PLACES = [("models folder", MODELS), ("Hugging Face cache", HF),
          ("LM Studio", LMS), ("nowhere", TMP / "missing")]
scan.places = lambda: list(PLACES)

# -- names -----------------------------------------------------------------------
check("a name from the file, the quant left off",
      [scan.name_for(Path(n)) for n in (
          "Qwen3-8B-Q4_K_M.gguf", "Qwen3-8B-UD-Q4_K_XL.gguf",
          "gemma-3-4b-it-IQ4_XS.gguf", "Bonsai-27B-PQ2_0.gguf",
          "Big-Model-Q8_0-00001-of-00003.gguf", "model.f16.gguf",
          "!!!.gguf")],
      ["qwen3-8b", "qwen3-8b", "gemma-3-4b-it", "bonsai-27b", "big-model",
       "model", "model"])
check("and made unique against the names taken",
      scan.name_for(Path("Qwen3-8B-Q4_K_M.gguf"), {"qwen3-8b", "qwen3-8b-2"}),
      "qwen3-8b-3")

# -- which of the found an answer picks ---------------------------------------------
check("choose: yes, nothing, or all is all; no is none",
      [setup.choose(3, a) for a in ("", "Y", "all", "n", "none")],
      [[0, 1, 2]] * 3 + [[]] * 2)
check("choose: numbers, lists and ranges, each once, in order given",
      [setup.choose(4, a) for a in ("1 3", "3,1", "2-4", "1 1", " 2 ")],
      [[0, 2], [2, 0], [1, 2, 3], [0], [1]])
check("choose: anything else is asked again",
      [setup.choose(3, a) for a in ("4", "0", "3-1", "maybe", "1 x")],
      [None] * 5)

# -- what is on disk ---------------------------------------------------------------
for d in (MODELS / "someorg--One-GGUF", LMS / "pub" / "Two-GGUF"):
    d.mkdir(parents=True)
one = support.llama(MODELS / "someorg--One-GGUF" / "One-Q4_K_M.gguf")
(MODELS / "someorg--One-GGUF" / "mmproj-One-F16.gguf").write_bytes(b"x" * 2000)
# the HF cache: a snapshot entry that is the same file as the pulled copy
snap = HF / "models--someorg--One-GGUF" / "snapshots" / "abc"
snap.mkdir(parents=True)
os.link(one, snap / "One-Q4_K_M.gguf")
two = support.llama(LMS / "pub" / "Two-GGUF" / "Two-Q8_0.gguf")
# LM Studio names a projector after its model
(two.parent / "Two-mmproj-BF16.gguf").write_bytes(b"x" * 2000)
split = LMS / "pub" / "Big-GGUF"
split.mkdir(parents=True)
for i in (1, 2):
    (split / ("Big-Q8_0-%05d-of-00002.gguf" % i)).write_bytes(b"GGUF" + b"x" * 3000)
broken = LMS / "pub" / "Broken-GGUF"
broken.mkdir(parents=True)
(broken / "Broken-Q4_0-00001-of-00002.gguf").write_bytes(b"GGUF" + b"x" * 3000)
(LMS / "pub" / "vocab.gguf").write_bytes(b"GGUF")
(LMS / "pub" / "Half.gguf.part").write_bytes(b"x" * 5000)
(LMS / ".hidden").mkdir()
(LMS / ".hidden" / "Hidden-Q4_0.gguf").write_bytes(b"x" * 5000)
mine = support.llama(OWN / "Mine-Q4_0.gguf")

found = scan.found(known=[mine])
check("found: a model each, biggest first; one entry per file however many "
      "paths reach it; a split model once, its shards summed",
      sorted((f.path.name, f.place) for f in found),
      [("Big-Q8_0-00001-of-00002.gguf", "LM Studio"),
       ("One-Q4_K_M.gguf", "models folder"),
       ("Two-Q8_0.gguf", "LM Studio")])
big = next(f for f in found if f.path.name.startswith("Big"))
check("a split model's size is every shard's", big.size, 2 * 3004)
check("each with the projector beside it, whichever way it is named",
      {f.path.name: f.mmproj and f.mmproj.name for f in found},
      {"Big-Q8_0-00001-of-00002.gguf": None,
       "One-Q4_K_M.gguf": "mmproj-One-F16.gguf",
       "Two-Q8_0.gguf": "Two-mmproj-BF16.gguf"})
check("what the config already runs is left out",
      [f.path.name for f in scan.found([("own", OWN)], known=[mine])], [])
check("and found when it does not",
      [f.path.name for f in scan.found([("own", OWN)])], ["Mine-Q4_0.gguf"])

# -- mdl setup ------------------------------------------------------------------------
root, port = sandbox()
mdl.CONFIG.unlink()                     # a new machine: no config at all
os.environ["MDL_LLAMA_SERVER"] = str(TMP / "no-llama-server")
out, err, code = run(setup.main, [])
check("off a terminal, without --yes: it lists what it found, adds nothing, "
      "and says how", (code, out.count("\n  "), "mdl setup --yes" in out,
                       mdl.load_config(missing_ok=True)[0]),
      (0, 4, True, {}))
check("it says what llama.cpp is missing, how to get it, and makes the config",
      ("llama.cpp" in out and "install" in out, mdl.CONFIG.is_file(),
       "(new)" in out), (True, True, True))

out, err, code = run(setup.main, ["--yes"])
models = mdl.load_config()[0]
check("--yes adds every one found, under the names it listed",
      (code, sorted(models)), (0, ["big", "one", "two"]))
check("each on a port of its own, with its projector",
      (len({m["port"] for m in models.values()}), models["one"].get("mmproj", "")
       .endswith("mmproj-One-F16.gguf"),
       models["two"].get("mmproj", "").endswith("Two-mmproj-BF16.gguf")),
      (3, True, True))
check("and says each as it goes, then what to do next",
      (out.count("added "), "next" in out), (3, True))
out, err, code = run(setup.main, ["--yes"])
check("run again, it adds nothing twice",
      (code, "not already run" in out, sorted(mdl.load_config()[0])),
      (0, True, ["big", "one", "two"]))
out, err, code = run(setup.main, ["--yes", "--dir", str(OWN)])
check("--dir looks somewhere else too",
      (code, sorted(mdl.load_config()[0])), (0, ["big", "mine", "one", "two"]))

out, err, code = run(setup.main, ["--help"])
check("--help says where it looks", (code, "LM Studio" in out), (0, True))
out, err, code = run(setup.main, ["--nope"])
check("an unknown option is one line", (code, err.count("\n")), (1, 1))
teardown(root)

sys.exit(t.done())
