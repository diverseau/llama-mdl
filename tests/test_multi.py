"""Several servers at once: ports, disambiguation, --all, migration.

The single-server rules have to keep working exactly as they did while
only one is up - that is the whole compatibility story - and only ask
which one when there is a genuine choice.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                        # noqa: E402
from support import mdl, run, sandbox, teardown       # noqa: E402

t = support.Tally("test_multi")
check = t.check


def second(root, name, port):
    """Append another model to the sandbox config."""
    with open(mdl.CONFIG, "a", encoding="utf-8") as fh:
        fh.write('\n[%s]\nmodel = "%s"\nngl = 99\nctx = 512\nport = %d\n'
                 % (name, str(support.FAKE).replace("\\", "/"), port))


# ------------------------------------------------- one behaves as before ----
root, port = sandbox()
second(root, "other", support.free_port())
out, _, code = run(mdl.cmd_ps, ["--json"])
check("ps --json is a list when idle", (out.strip(), code), ("[]", 0))

root, port = sandbox()
second(root, "other", support.free_port())
run(mdl.cmd_run, ["demo"])
state = mdl.read_state()
check("read_state with one running needs no name", state["name"], "demo")
out, _, code = run(mdl.cmd_stop, [])
check("bare stop stops the only one",
      (out, code), ("stopped demo (pid %d)\n" % state["pid"], 0))
check("and it is gone", mdl.read_states(), {})

# ------------------------------------------------------------ two at once ---
run(mdl.cmd_run, ["demo"])
out, err, code = run(mdl.cmd_run, ["other"])
check("a second server starts", code, 0)
states = mdl.read_states()
check("both are running", sorted(states), ["demo", "other"])
check("each kept its own port",
      states["demo"]["port"] != states["other"]["port"], True)
check("each has its own state file",
      (mdl.state_path("demo").is_file(), mdl.state_path("other").is_file()),
      (True, True))

out, _, _ = run(mdl.cmd_ps, [])
check("ps lists both", ("demo" in out and "other" in out, len(out.splitlines())),
      (True, 2))
out, _, _ = run(mdl.cmd_ps, ["--json"])
rows = json.loads(out)
check("ps --json lists both", [r["name"] for r in rows], ["demo", "other"])
check("every row carries uptime", all("uptime" in r for r in rows), True)

# --- now the single-server shortcuts must ask ------------------------------
_, err, code = run(mdl.cmd_stop, [])
check("bare stop asks which", ("several servers are running" in err, code),
      (True, 1))
check("and it names them", "demo, other" in err, True)
check("and stops neither", sorted(mdl.read_states()), ["demo", "other"])

_, err, code = run(mdl.cmd_logs, [])
check("bare logs asks which too", ("several servers are running" in err, code),
      (True, 1))
out, _, code = run(mdl.cmd_logs, ["demo"])
check("logs by name still works", ("offloaded 33/33" in out, code), (True, 0))

out, _, code = run(mdl.cmd_stop, ["other"])
check("stop by name stops that one", (out, code),
      ("stopped other (pid %d)\n" % states["other"]["pid"], 0))
check("and leaves the other alone", sorted(mdl.read_states()), ["demo"])
_, err, code = run(mdl.cmd_stop, ["other"])
check("stopping what is not running says so",
      ("'other' is not running" in err, code), (True, 1))
run(mdl.cmd_stop, [])

# ---------------------------------------------------------------- --all ----
run(mdl.cmd_run, ["demo"])
run(mdl.cmd_run, ["other"])
out, _, code = run(mdl.cmd_stop, ["--all"])
check("--all stops every one", (len(out.strip().splitlines()), code), (2, 0))
check("nothing is left", mdl.read_states(), {})
out, _, code = run(mdl.cmd_stop, ["--all"])
check("--all with nothing running is not an error",
      (out.strip(), code), ("nothing running", 0))

# ------------------------------------------------------------- the port ----
# The sandbox config shares a port on purpose here: legal, but only one
# of them can be up.
teardown(root)
root, port = sandbox()
second(root, "twin", port)                 # deliberately the same port
run(mdl.cmd_run, ["demo"])
_, err, code = run(mdl.cmd_run, ["twin"])
check("a port clash is refused", code, 1)
check("and it names who holds the port", "already serving 'demo'" in err, True)
check("and says what to do about it",
      ("give twin its own port" in err, "mdl stop demo" in err), (True, True))
check("the clash left no state behind", sorted(mdl.read_states()), ["demo"])

_, err, code = run(mdl.cmd_run, ["demo"])
check("the same model twice is refused",
      ("'demo' is already running" in err, code), (True, 1))

# --port overrides the config, and the state must record what we used
free = support.free_port()
out, _, code = run(mdl.cmd_run, ["twin", "--port", str(free)])
check("--port lets a sharing model run anyway", code, 0)
check("state records the port we actually used",
      mdl.read_state("twin")["port"], free)
out, _, _ = run(mdl.cmd_ps, ["--json"])
check("ps agrees with it",
      [r["port"] for r in json.loads(out) if r["name"] == "twin"], [free])
_, err, code = run(mdl.cmd_run, ["twin", "--port", "nonsense"])
check("--port is validated", ("port must be a number" in err, code), (True, 1))
run(mdl.cmd_stop, ["--all"])

# --- check reports the sharing, without calling it a failure ---------------
out, _, code = run(mdl.cmd_check, [])
check("check notes a shared port",
      "share port %d; only one at a time" % port in out, True)
check("but a shared port is not a problem", code, 0)
teardown(root)

# ------------------------------------------------------- state migration ----
# Someone upgrading with a server up keeps control of it.
root, port = sandbox()
run(mdl.cmd_run, ["demo"])
old = mdl.read_state("demo")
mdl.state_path("demo").unlink()
mdl.STATE.write_text(json.dumps(old), encoding="utf-8")   # the pre-0.3 layout
check("a pre-0.3 state file is still understood",
      mdl.read_state("demo")["pid"], old["pid"])
check("it was moved to the new place", mdl.state_path("demo").is_file(), True)
check("and the old file is gone", mdl.STATE.exists(), False)
out, _, code = run(mdl.cmd_stop, [])
check("and the migrated server can be stopped", code, 0)

mdl.STATE.write_text("not json at all", encoding="utf-8")
check("a corrupt one is discarded, not fatal", mdl.read_states(), {})
check("and cleared away", mdl.STATE.exists(), False)
teardown(root)

sys.exit(t.done())
