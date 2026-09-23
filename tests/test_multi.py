"""Several servers at once: ports, disambiguation, --all.

The single-server rules have to keep working exactly as they did while
only one is up - that is the whole compatibility story - and only ask
which one when there is a genuine choice.
"""
import json
import os
import socket
import subprocess
import sys
import time
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

# ------------------------------- a wrapper that leaves its server (B03) ----
# The wrapper exits at once; the server it started ignores SIGTERM and
# holds the port. Stop used to watch only the wrapper's pid, see it gone,
# and say "stopped" with the server still up and ps showing nothing.

root, port = sandbox()
kid_py = root / "server.py"
kid_py.write_text(
    "import signal, socket, sys, time\n"
    "if hasattr(signal, 'SIGTERM'):\n"
    "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "s = socket.socket()\n"
    "s.bind(('127.0.0.1', int(sys.argv[1])))\n"
    "s.listen()\n"
    "open(sys.argv[2], 'w').close()\n"
    "while True:\n"
    "    time.sleep(1)\n", encoding="utf-8")
up = root / "up"
wrap_py = root / "wrapper.py"
wrap_py.write_text(
    "import subprocess, sys\n"
    "subprocess.Popen([sys.executable] + sys.argv[1:],\n"
    "                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,\n"
    "                 stderr=subprocess.DEVNULL)\n", encoding="utf-8")
w = subprocess.Popen([sys.executable, str(wrap_py), str(kid_py), str(port),
                      str(up)], start_new_session=True)
mdl.run_dir().mkdir(parents=True, exist_ok=True)
mdl.write_atomic(mdl.state_path("wrapped"), json.dumps(
    {"name": "wrapped", "pid": w.pid, "port": port, "started": time.time(),
     "log": "", "born": mdl.proc_started(w.pid),
     "pgid": w.pid if os.name != "nt" else None}))
w.wait(10)
for _ in range(100):
    if up.exists():
        break
    time.sleep(0.1)
check("the wrapper is gone and its server is up",
      (mdl.alive(w.pid), mdl.port_busy(port)), (False, True))
check("ps still lists a server whose wrapper exited",
      sorted(mdl.read_states()), ["wrapped"])
out, err, code = run(mdl.cmd_stop, ["wrapped"])
check("stop takes the orphaned server down with it", code, 0)
check("the port can be bound again", not mdl.port_busy(port), True)
with socket.socket() as again:
    again.bind(("127.0.0.1", port))
check("and nothing is left listed", mdl.read_states(), {})
teardown(root)

# ------------------------------------------- launch transaction (B13) ----
root, port = sandbox()
real_write = mdl.write_atomic


def refuse(path, *a, **k):
    if str(path).endswith(".json"):
        raise OSError("disk full")
    return real_write(path, *a, **k)


mdl.write_atomic = refuse
try:
    _, err, code = run(mdl.cmd_run, ["demo"])
finally:
    mdl.write_atomic = real_write
check("a state that cannot be written stops the server, in one line",
      ("stopped it rather than leave it untracked" in err, code), (True, 1))
for _ in range(50):
    if not mdl.port_busy(port):
        break
    time.sleep(0.1)
check("and nothing is left on the port", mdl.port_busy(port), False)
check("nor a lock behind it", list(mdl.run_dir().glob("*.lock")), [])

lock = mdl.run_dir() / "demo.lock"
lock.write_text(str(os.getpid()))            # a live launcher holds it
t0 = time.time()
_, err, code = run(mdl.spawn, "demo", *mdl.load_config())
check("a launch already under way is waited for, then refused",
      ("being started by another mdl" in err, code, time.time() - t0 > 3),
      (True, 1, True))
dead = subprocess.Popen([sys.executable, "-c", "pass"])
dead.wait()
lock.write_text(str(dead.pid))               # its launcher died
proc, _, _ = mdl.spawn("demo", *mdl.load_config())
check("a lock left by a dead launcher is taken over",
      (mdl.read_state("demo")["pid"], lock.exists()), (proc.pid, False))
_, err, code = run(mdl.spawn, "demo", *mdl.load_config())
check("and the running check is made again inside the lock",
      ("already running" in err, code), (True, 1))
check("the state records what ran",
      mdl.read_state("demo")["argv"][:2],
      mdl.build_argv("demo", mdl.load_config()[0]["demo"],
                     mdl.load_config()[1])[:2])
run(mdl.cmd_stop, ["--all"])
teardown(root)

# ------------------------------------------- taking over a dead lock ----
root, port = sandbox()
lock = mdl.run_dir() / "race.lock"
guard = mdl.run_dir() / "race.lock.takeover"
held = mdl.file_lock(lock, "busy")
lock.parent.mkdir(parents=True, exist_ok=True)
lock.write_text(str(dead.pid))
guard.touch()                                # another waiter is taking it
check("a takeover under way is left to the waiter making it",
      (held._take_over(dead.pid), lock.read_text()), (False, str(dead.pid)))
os.utime(guard, (time.time() - 60, time.time() - 60))
held._take_over(dead.pid)
check("a guard a crash left behind is cleared", guard.exists(), False)
lock.write_text(str(os.getpid()))            # a new holder won it meanwhile
check("a lock taken since the dead pid was read is not removed",
      (held._take_over(dead.pid), lock.read_text()), (False, str(os.getpid())))
lock.write_text(str(dead.pid))
check("one still held by the dead pid is removed, and the guard with it",
      (held._take_over(dead.pid), lock.exists(), guard.exists()),
      (True, False, False))
lock.write_text(str(dead.pid))
with mdl.file_lock(lock, "busy", tries=5):
    check("and the lock is then taken", lock.read_text(), str(os.getpid()))
teardown(root)

sys.exit(t.done())
