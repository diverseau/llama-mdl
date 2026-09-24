"""mdl lab: variants of a config run through the fake llama-server, one
at a time, from configs of their own - and the records, the report and
the comparison made from what they did.

The GPU is stood in for (gpu_now) so the VRAM figures are fixed and no
real card's other users can move them; RAM and CPU are this machine's.
"""
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                   # noqa: E402
from support import llama, mdl, run, sandbox, teardown  # noqa: E402

from mdl_fit import hw, lab  # noqa: E402

t = support.Tally("test_lab")
check = t.check
TMP = Path(tempfile.mkdtemp(prefix="mdl-lab-test-"))
os.environ["MDL_FIT_HOME"] = str(TMP / "home")   # never the real ~/.config
GiB, MiB = lab.GiB, lab.MiB
lab.BASELINE_S = 0.2
VRAM = {"used": 1 * GiB}
lab.gpu_now = lambda: (VRAM["used"], 50.0, 60.0, 1800.0)

# ------------------------------------------------------------- metrics --
steady = [0.5 + i * 0.02 for i in range(200)]          # 50 t/s, 4 s long
rates = lab.windowed(steady)
check("a steady stream's windowed rate is its speed, every window",
      (round(min(rates)), round(max(rates))), (50, 50))
stall = steady[:100] + [t + 0.5 for t in steady[100:]]  # one half-second stall
m, flags = lab.rep_metrics(stall, {"predicted_per_second": 49.0,
                                   "prompt_per_second": 800.0,
                                   "prompt_n": 120}, [], 500,
                           {"vram": None, "ram": None})
check("a stall shows as a floor under the median, not a lower average only",
      (m["decode"]["floor"] < m["decode"]["p50"] - 10,
       round(m["decode"]["peak"])), (True, 50))
check("time to first token, prefill and the server's own rate are kept",
      (m["ttft"], m["prefill"]["tps"], m["server_decode"]), (0.5, 800.0, 49.0))
check("a client and server that disagree by more than 5% are flagged",
      "timings_disagree" in flags, True)
_, flags = lab.rep_metrics(steady, {"predicted_per_second": 50.0}, [], 500,
                           {"vram": None, "ram": None})
check("and ones that agree are not", "timings_disagree" in flags, False)
samples = [{"phase": "prefill", "tok": 0, "vram": 5, "rss": 1, "cpu": 1,
            "ram_used": 1, "gpu_util": 1, "temp": 60, "clock": 1800},
           {"phase": "decode", "tok": 300, "vram": 8, "rss": 2, "cpu": 20,
            "ram_used": 3, "gpu_util": 90, "temp": 70, "clock": 1800},
           {"phase": "decode", "tok": 620, "vram": 9, "rss": 3, "cpu": 40,
            "ram_used": 4, "gpu_util": 95, "temp": 80, "clock": 1200}]
m, flags = lab.rep_metrics(steady, {}, samples, 500, {"vram": 4, "ram": 1})
check("usage at token N is the first decode sample at or past it",
      m["at"]["500"], {"tok": 620, "vram": 9, "rss": 3, "cpu": 40})
check("VRAM and RAM against the baseline, over decode only",
      (m["vram"], m["ram"]["delta"]),
      ({"peak": 9, "steady": 8, "delta": 5}, 3))
check("a clock that dropped is flagged: a throttled run, not a slow config",
      "clock_drop" in flags, True)
check("two spreads that overlap are indistinguishable",
      lab.verdict([40, 42, 41], [41, 43, 40], "a", "b", max),
      "indistinguishable")
check("two that do not are called",
      lab.verdict([40, 40.5, 40.2], [50, 50.3, 50.1], "a", "b", max),
      "b better")
check("--set values read as TOML, lists keep their commas",
      [lab.value(x) for x in lab.split_values('45,"q8_0",true,["-a","-b"]')],
      [45, "q8_0", True, ["-a", "-b"]])
o = lab.parse(["m", "n", "--set", "ngl=45,38", "--set", "ctx=8k"])
check("the matrix is every name by every combination, labelled by it",
      [v.label for v in lab.matrix(o)],
      ["m/ngl45/ctx8192", "m/ngl38/ctx8192", "n/ngl45/ctx8192",
       "n/ngl38/ctx8192"])
check("a depth is tokens, or a share of each variant's context",
      lab.parse(["m", "--depth", "0,8k,50%,full"])["depths"],
      [0, 8192, "50%", "100%"])
check("and by default it is both ends: empty, and full",
      lab.workload(lab.parse(["m"]))["depths"], [0, "100%"])
w = {"max_tokens": 512}
check("100% leaves just the prompt, the reply and the template's room",
      (lab.resolve_depth("100%", 8192, w, 100),
       lab.resolve_depth("50%", 8192, w, 100),
       lab.resolve_depth(4096, 8192, w, 100)),
      (8192 - 512 - 100 - lab.TEMPLATE_SLACK,
       (8192 - 512 - 100 - lab.TEMPLATE_SLACK) // 2, 4096))
for bad, words in ((["--depth", "lots"], "--depth takes a number"),
                   (["--depth", "150%"], "a share from 0% to 100%"),
                   (["--reps"], "--reps needs a value"),
                   (["--frobnicate"], "unknown option --frobnicate")):
    _, err, code = run(lab.parse, bad)
    check("%s is one line" % bad[0], (words in err, code, err.count("\n")),
          (True, 1, 1))

s = lab.Sampler(os.getpid(), 60)          # no tick falls in what follows
s.start()
s.phase, s.tok = "decode", 7
s.poke()
s.phase = "idle"                          # the stream ended at once
s.flush()
s.stop()
check("a poked sample is taken for the moment asked, however short",
      [(x["phase"], x["tok"]) for x in s.samples if x["phase"] != "ready"],
      [("decode", 7)])

# ------------------------------------------------------------ the runs --
model_file = llama(TMP / "Tiny-Q8_0.gguf")
root, port = sandbox(model=model_file)
config_before = mdl.CONFIG.read_bytes()
mine_before = {p.name for p in Path(tempfile.gettempdir()).glob("mdl-lab-*")}


def lab_main(*args):
    out = io.StringIO()
    got = {}
    _, err, code = run(lambda: got.update(r=lab.main(list(args), out)))
    return got.get("r"), out.getvalue(), err, code


ARGS = ["--reps", "2", "--warmup", "1", "--max-tokens", "64",
        "--cooldown", "0", "--interval", "0.2", "--at", "20", "--depth", "0"]
_, out, err, code = lab_main("run", "demo", "--set", "ctx=4096,8192",
                             "--dry-run", *ARGS)
check("a dry run shows the matrix, the loads and the requests, and starts "
      "nothing", ("demo/ctx4096" in out and "demo/ctx8192" in out,
                  "2 loads, 6 requests" in out, "dry run: nothing started" in
                  out, "estimate ~" in out, mdl.read_states(), code),
      (True, True, True, True, {}, 0))

run_id, out, err, code = lab_main("run", "demo", "--set", "ctx=4096,8192",
                                  *ARGS)
first = run_id
records = lab.load()
mine = [r for r in records if r["run"] == run_id]
check("a run records every repetition of every variant, warmups marked",
      (code, len(mine), sum(r["warmup"] for r in mine),
       sorted({r["variant"]["label"] for r in mine})),
      (0, 6, 2, ["demo/ctx4096", "demo/ctx8192"]))
rec = next(r for r in mine if not r["warmup"])
check("each is run to max_tokens, with its speed, first token and prefill",
      (rec["metrics"]["tokens"], rec["metrics"]["decode"]["avg"] > 0,
       rec["metrics"]["ttft"] > 0, rec["metrics"]["prefill"]["tps"]),
      (64, True, True, 900.0))
check("the config it ran is the variant's, and the build is recorded",
      (rec["config"]["ctx"] in (4096, 8192), "-c" in rec["argv"],
       set(rec["build"]) >= {"build", "commit", "backend"}), (True, True,
                                                              True))
check("the samples behind it are kept, tagged by phase and token",
      {s["phase"] for s in lab.samples_of(rec)} <= {"ready", "prefill",
                                                     "decode", "idle"}
      and any(s["phase"] == "decode" for s in lab.samples_of(rec)), True)
check("the fake's own rate disagrees with its stream, and that is flagged",
      "timings_disagree" in rec["flags"], True)
check("models.toml is untouched, and no server of the run is left",
      (mdl.CONFIG.read_bytes() == config_before, mdl.read_states(),
       mdl.port_busy(port)), (True, {}, False))
check("nor its temp config and state",
      {p.name for p in Path(tempfile.gettempdir()).glob("mdl-lab-*")}
      - mine_before - {TMP.name}, set())

rows, out, err, code = lab_main("report")
check("the report has a row per variant, medians and spread",
      (code, [r["variant"] for r in rows], "decode t/s" in out,
       all(r["reps"] == 2 for r in rows), "±" in out),
      (0, ["demo/ctx4096", "demo/ctx8192"], True, True, True))
check("with the usage at the token asked for",
      ("@20 VRAM" in out, rows[0]["at_vram"]), (True, 1 * GiB))
_, md, _, _ = lab_main("report", run_id, "--format", "md")
_, csv, _, _ = lab_main("report", "--format", "csv")
_, js, _, _ = lab_main("report", "--format", "json")
check("as markdown, CSV and JSON too",
      (md.count("| demo/ctx"), csv.splitlines()[0].startswith("variant,"),
       len(json.loads(js)["rows"])), (2, True, 2))
_, out, err, code = lab_main("compare", "demo/ctx4096", "demo/ctx8192")
check("compare sets two variants side by side with a verdict",
      (code, "decode t/s" in out, any(v in out for v in (
          "indistinguishable", "better"))), (0, True, True))
_, out, err, code = lab_main("compare", "demo/ctx4096", "nope")
check("a label not in the run is refused, naming the ones that are",
      ("no variant 'nope'" in err and "demo/ctx8192" in err, code), (True, 1))
_, out, err, code = lab_main("apply", "demo/ctx8192")
check("apply prints the winner's table, and writes nothing",
      ("[demo]" in out, "ctx = 8192" in out,
       mdl.CONFIG.read_bytes() == config_before), (True, True, True))
_, out, _, _ = lab_main("ls")
check("ls lists the run", run_id in out, True)
_, out, _, _ = lab_main("export", run_id)
check("export gives its records as JSON", len(json.loads(out)), 6)

run_id, out, err, code = lab_main("run", "demo", "--reps", "1", "--warmup",
                                  "0", "--max-tokens", "64", "--cooldown", "0",
                                  "--interval", "60", "--at", "20")
rec = next(r for r in lab.load() if r["run"] == run_id)
check("a decode shorter than the interval is sampled at its start, at the "
      "token asked for and at its end",
      (code, {1, 20, 64} <= {x["tok"] for x in lab.samples_of(rec)
                             if x["phase"] == "decode"},
       (rec["metrics"]["at"]["20"] or {}).get("vram"),
       rec["metrics"]["vram"]["peak"]), (0, True, 1 * GiB, 1 * GiB))

# ---------------------------------------------------- depth as a share --
run_id, out, err, code = lab_main("run", "demo", "--set", "ctx=2048,4096",
                                  "--depth", "0,100%", "--reps", "2",
                                  "--warmup", "0", "--max-tokens", "64",
                                  "--cooldown", "0", "--interval", "0.2")
mine = [r for r in lab.load() if r["run"] == run_id]
full = [r for r in mine if r["workload"]["depth"] == "100%"]
check("100% fills each variant's own context, not one number for all",
      (code, sorted({r["workload"]["ctx"] for r in full}),
       all(r["workload"]["prompt_tokens"] + 64 <= r["workload"]["ctx"]
           - lab.TEMPLATE_SLACK for r in full),
       all(r["workload"]["prompt_tokens"] > 0.9 * (r["workload"]["ctx"] - 64
                                                   - lab.TEMPLATE_SLACK)
           for r in full)),
      (0, [2048, 4096], True, True))
check("and says what it came to",
      "depth 100%: a" in out and "-token prompt, the reply ending at" in out,
      True)
check("the prefill is the prompt the share asked for",
      all(r["metrics"]["prefill"]["tokens"] >= r["workload"]["prompt_tokens"]
          for r in full), True)
rows, out, err, code = lab_main("report", run_id)
check("the report has a row per depth, a share showing its tokens",
      ([(r["variant"], r["depth"]) for r in rows], "100% (" in out),
      ([("demo/ctx2048", 0), ("demo/ctx2048", "100%"),
        ("demo/ctx4096", 0), ("demo/ctx4096", "100%")], True))
_, out, err, code = lab_main("compare", "demo/ctx2048", "demo/ctx4096",
                             "--run", run_id)
check("compare sets the two side by side depth by depth, never pooled",
      (code, out.count("\ndepth "), "depth 0\n" in out,
       "depth 100%\n" in out), (0, 2, True, True))
_, out, err, code = lab_main("run", "demo", "--depth", "8k", "--reps", "1",
                             "--warmup", "0", "--max-tokens", "64",
                             "--cooldown", "0", "--interval", "0.2")
check("an absolute depth the context cannot hold is skipped, and says so",
      ("do not fit its 4k context; skipped" in out, code), (True, 0))

# --------------------------------------------------------------- suites --
suites = hw.config_dir() / "lab"
suites.mkdir(parents=True, exist_ok=True)
(suites / "deep.toml").write_text(
    '[suite]\nmax_tokens = 32\nreps = 1\nwarmup = 0\ndepth = ["0", "1k"]\n'
    'cooldown = 0\ninterval = 0.2\n\n'
    '[[variant]]\nbase = "demo"\nlabel = "q4 cache"\n'
    'set = {kv_type = "q4_0"}\n', encoding="utf-8")
run_id, out, err, code = lab_main("run", "--suite", "deep")
mine = [r for r in lab.load() if r["run"] == run_id]
check("a suite file gives the variants, their labels and the depths",
      (code, sorted(r["workload"]["depth"] for r in mine),
       {r["variant"]["label"] for r in mine}, {r["suite"] for r in mine},
       mine[0]["config"]["kv_type"]), (0, [0, 1024], {"q4 cache"},
                                        {"deep"}, "q4_0"))
check("a deeper prompt is a longer prefill",
      sorted(r["metrics"]["prefill"]["tokens"] for r in mine)[1]
      > sorted(r["metrics"]["prefill"]["tokens"] for r in mine)[0] + 500,
      True)

# ------------------------------------------------------------- refusals --
run(mdl.cmd_run, ["demo"])                         # a server of yours
_, out, err, code = lab_main("run", "demo", *ARGS)
check("a server of yours on the port is never stopped: the run refuses",
      ("port %d is in use by 'demo'" % port in err, code,
       (mdl.read_state("demo") or {}).get("port")), (True, 1, port))
run(mdl.cmd_stop, ["--all"])

run_id, out, err, code = lab_main("run", "demo", "--set",
                                  'model="%s"' % (TMP / "gone.gguf").as_posix(),
                                  *ARGS)
mine = [r for r in lab.load() if r["run"] == run_id]
check("a variant that will not start is recorded as such, and the run "
      "goes on", (code, [bool(r.get("skipped")) for r in mine],
                  "model file not found" in (mine[0].get("skipped") or "")),
      (0, [True], True))
_, out, _, _ = lab_main("report", run_id)
check("and the report says why", "skipped" in out and "gone.gguf" in out,
      True)

real_ram = lab.hw.ram
lab.hw.ram = lambda: (4 * GiB, 1 * GiB)
try:
    run_id, out, err, code = lab_main("run", "demo", *ARGS)
finally:
    lab.hw.ram = real_ram
check("a variant the free RAM cannot hold is skipped, not paged to disk",
      ("needs ~" in out and "of RAM free" in out, code,
       [r.get("skipped") for r in lab.load() if r["run"] == run_id]),
      (True, 0, ["insufficient RAM"]))

VRAM["used"] = 8 * GiB                   # something else took the card
real_baseline = lab.baseline
lab.baseline = lambda seconds=None, interval=1.0: {"vram": 1 * GiB, "ram": 1}
try:
    _, out, err, code = lab_main("run", "demo", "--set", "ctx=4096,8192",
                                 *ARGS)
finally:
    lab.baseline = real_baseline
    VRAM["used"] = 1 * GiB
check("VRAM that does not come back stops the run: the rest would not "
      "compare", ("did not return to its baseline" in err, code,
                  mdl.read_states()), (True, 1, {}))

# ------------------------------------------------------------- baseline --
_, out, err, code = lab_main("baseline", "diff")
check("a diff with nothing pinned says how to pin one",
      ("no baseline pinned" in err, code), (True, 1))
_, out, err, code = lab_main("baseline", "set", first)
check("a run is pinned as the baseline",
      (code, "%s pinned" % first in out, lab.pinned()), (0, True, first))
again, out, err, code = lab_main("run", "demo", "--set", "ctx=4096,8192",
                                 *ARGS)
_, out, err, code = lab_main("baseline", "diff")
check("a later run is set against it, variant by variant",
      (code, "against baseline %s" % first in out,
       out.count("demo/ctx4096") >= 2, "decode t/s" in out),
      (0, True, True, True))
_, out, err, code = lab_main("baseline", "diff", first)
check("the baseline against itself is refused", ("is the baseline" in err,
                                                 code), (True, 1))


def fake_run(run_id, decode):
    for i, d in enumerate(decode):
        lab.append({"schema": lab.SCHEMA, "id": "%s-%d" % (run_id, i),
                    "run": run_id, "at": "2026-09-23T0%d" % i,
                    "warmup": False, "variant": {"label": "demo/fast"},
                    "workload": {"depth": 0}, "build": {}, "flags": [],
                    "metrics": {"decode": {"avg": d}, "ttft": 0.2}})


fake_run("fake-base", [50.0, 50.2, 49.9])
fake_run("fake-slow", [40.0, 40.1, 39.8])
lab_main("baseline", "set", "fake-base")
_, out, err, code = lab_main("baseline", "diff", "fake-slow", "--fail")
check("a speed the spread cannot explain is a regression, and --fail "
      "makes it the exit code", ("regressed: demo/fast decode t/s" in out,
                                 "1 regression against" in err, code),
      (True, True, 1))
_, out, err, code = lab_main("baseline", "diff", "fake-slow")
check("without --fail it is reported, not failed", code, 0)
fake_run("fake-same", [50.1, 49.8, 50.0])
_, out, err, code = lab_main("baseline", "diff", "fake-same", "--fail")
check("noise is not a regression", ("no regression" in out, code), (True, 0))

rec = {"metrics": {"decode": {"avg": 40.0}, "vram": {"delta": 4 * GiB,
                                                     "peak": 12 * GiB},
                   "at": {"20": {}}}, "build": {}, "flags": []}
check("t/s per GB is over the VRAM the model took, not the card's peak",
      lab.row("x", 0, [rec], 20)["per_gb"], 10.0)
rec["metrics"]["vram"] = {}
rec["vram_claimed"] = 8 * GiB
check("and over what the load log claimed when no card figure is there",
      lab.row("x", 0, [rec], 20)["per_gb"], 5.0)

teardown(root)
sys.exit(t.done())
