"""mdl eval end to end, around the items: the server it starts and stops,
one it finds running, the lock on a run, interrupt and resume.

The server is the fake llama-server and the model a GGUF written here.
Running the items is stubbed - the runners and graders have their own
tests in test_eval.py - so what is left is main()'s bookkeeping, which
nothing else drives.
"""
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                   # noqa: E402
from support import llama, mdl, run, sandbox, teardown  # noqa: E402

from mdl_fit import evalrun, hw  # noqa: E402

t = support.Tally("test_eval_main")
check = t.check
TMP = Path(tempfile.mkdtemp(prefix="mdl-eval-main-test-"))
os.environ["MDL_FIT_HOME"] = str(TMP / "home")   # never the real ~/.config
# a fixed machine: the probe would read this one's GPU, and time it
hw.probe = lambda binary="llama-server", quick=False, now=False: \
    hw.Machine(binary=binary)

model_file = llama(TMP / "Tiny-Q8_0.gguf")
ARGS = ["demo", "--suite", "reason", "--limit", "3", "--no-sandbox"]
seen = []                  # one entry per run_items call
stop_at = {"n": None}      # interrupt before the item at this index
inside = {"fn": None}      # run from inside a run, while it holds its lock


def fake_items(client, items, env, cpt=4.0, n_ctx=None, sink=None,
               progress=None, keep=None, done_ids=()):
    seen.append({"todo": [it.id for it in items if it.id not in done_ids],
                 "state": mdl.read_state("demo")})
    if inside["fn"]:
        seen[-1]["inside"] = inside["fn"]()
    for n, it in enumerate(items):
        if it.id in done_ids:
            continue
        if n == stop_at["n"]:
            raise KeyboardInterrupt
        res = {"id": it.id, "suite": it.suite, "domain": it.domain,
               "tier": "base", "score": 1.0, "capped": False, "error": False}
        sink.append(res)
        keep(res)
    return sink


evalrun.run_items = fake_items


def main(*extra):
    """(record, output, stderr, exit code) of one `mdl eval`."""
    out = io.StringIO()
    got = {}
    _, err, code = run(lambda: got.update(rec=evalrun.main(ARGS + list(extra),
                                                           out)))
    return got.get("rec"), out.getvalue(), err, code


def checkpoints():
    return sorted(p.name for p in evalrun.runs_dir().glob("*.jsonl"))


root, port = sandbox(model=model_file)

# ------------------------------------------- a server it starts itself --
rec, out, err, code = main("--json")
check("with nothing running, it starts the model's server for the run",
      ("starting demo on port %d" % port in out,
       seen[-1]["state"] is not None and seen[-1]["state"]["port"] == port),
      (True, True))
check("and stops it when the run is done",
      (mdl.read_state("demo"), mdl.port_busy(port)), (None, False))
check("the run is saved whole, and leaves no checkpoint behind",
      (code, rec and rec["partial"], rec and len(rec["items"]),
       rec and rec["runtime_checked"], checkpoints()),
      (0, False, 3, True, []))

# ------------------------------------------------ one already running --
_, err, code = run(mdl.cmd_run, ["demo"])
check("(a server started by mdl run)", code, 0)
pid = mdl.read_state("demo")["pid"]
rec, out, err, code = main("--json")
check("a server already running is used, not started again",
      ("already running on port %d; using it" % port in out, code),
      (True, 0))
check("and it is left running: it was not the eval's to stop",
      (mdl.read_state("demo") or {}).get("pid"), pid)


def second_eval():
    return main()


inside["fn"] = second_eval
rec, out, err, code = main("--json")
inside["fn"] = None
_, _, err2, code2 = seen[-1]["inside"]
check("a second eval of the same items on that server is refused, "
      "in one line", ("another mdl eval is running these items" in err2,
                      code2, err2.count("\n")), (True, 1, 1))
check("while the first one finishes and saves", (code, len(rec["items"])),
      (0, 3))

cfg = mdl.CONFIG.read_text(encoding="utf-8")
mdl.CONFIG.write_text(cfg.replace("ctx = 4096", "ctx = 8192"),
                      encoding="utf-8")
calls = len(seen)
rec, out, err, code = main()
check("a server running with other settings than the config is refused "
      "before a single item", ("other settings than" in err, code,
                               len(seen) - calls), (True, 1, 0))
check("and it is left running", (mdl.read_state("demo") or {}).get("pid"),
      pid)
mdl.CONFIG.write_text(cfg, encoding="utf-8")
run(mdl.cmd_stop, ["demo"])

# ----------------------------------------------- interrupt and resume --
stop_at["n"] = 2
rec, out, err, code = main("--json")
stop_at["n"] = None
check("an interrupt keeps the items done and says how to go on",
      ("interrupted; keeping the 2 items done; mdl eval demo --resume "
       "continues" in out, rec and rec["partial"], rec and len(rec["items"])),
      (True, True, 2))
check("the server it started is stopped all the same",
      (mdl.read_state("demo"), mdl.port_busy(port)), (None, False))
check("and the checkpoint is kept for --resume", len(checkpoints()), 1)
rec, out, err, code = main("--resume", "--json")
check("--resume runs only what is left",
      ("resume   2 of 3 items already done" in out, len(seen[-1]["todo"])),
      (True, 1))
check("and the result has every item, and is whole",
      (code, rec["partial"], len(rec["items"]), checkpoints()),
      (0, False, 3, []))
rec, out, err, code = main("--resume")
check("--resume with nothing to resume says so, and starts over",
      ("nothing to resume; starting over" in out, len(seen[-1]["todo"])),
      (True, 3))

stop_at["n"] = 1
main()
stop_at["n"] = None
rec, out, err, code = main("--json")
check("without --resume, an interrupted run is noted and started over",
      ("an interrupted run of these items has 1 done" in out,
       len(seen[-1]["todo"]), checkpoints()), (True, 3, []))

# ------------------------------------------- an interrupt while loading --
real_wait = evalrun.wait_ready


def interrupted(*a):
    raise KeyboardInterrupt


evalrun.wait_ready = interrupted
rec, out, err, code = main()
evalrun.wait_ready = real_wait
check("a Ctrl-C while the server loads stops it too, and saves nothing",
      ("no items finished" in err, code, mdl.read_state("demo"),
       mdl.port_busy(port)), (True, 1, None, False))

teardown(root)
sys.exit(t.done())
