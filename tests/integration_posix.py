"""POSIX process semantics: detach, orphan self-heal, SIGTERM -> SIGKILL.

Runs mdl as a real subprocess (the detach behaviour is invisible in-process).
Meant for Linux; see tests/README.md for the container one-liner.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))
import support                                  # noqa: E402

t = support.Tally("integration_posix")
check = t.check

HOME = Path(tempfile.mkdtemp(prefix="mdl-posix-"))
CONFIG = HOME / ".config" / "mdl"
CONFIG.mkdir(parents=True)
RUN = HOME / ".local" / "state" / "mdl" / "run"


def state_of(name):
    """The server's state file, one per server since 0.3."""
    return RUN / (name + ".json")

launcher = HOME / "fake-llama-server"
launcher.write_text('#!/bin/sh\nexec "%s" "%s" "$@"\n' % (sys.executable, support.FAKE))
launcher.chmod(0o755)
PORT = support.free_port()
(CONFIG / "models.toml").write_text(
    'llama_server = "%s"\n\n[demo]\nmodel = "%s"\nport = %d\n'
    % (launcher, support.FAKE, PORT))

ENV = dict(os.environ, HOME=str(HOME))
for _var in ("XDG_CONFIG_HOME", "XDG_STATE_HOME"):
    # CI sets these, and mdl rightly prefers them to HOME - so the
    # sandbox has to clear them or the suite reads the real config.
    ENV.pop(_var, None)


def mdl(*args, mode=""):
    env = dict(ENV, MDL_FAKE_MODE=mode)
    r = subprocess.run([sys.executable, str(ROOT / "mdl.py")] + list(args),
                       capture_output=True, text=True, env=env)
    if r.returncode and r.stderr.strip():
        # Otherwise a CI failure here is a bare 'expected 0, got 1'.
        print("      (mdl %s -> %d: %s)"
              % (" ".join(args), r.returncode,
                 r.stderr.strip().splitlines()[-1]))
    return r


def procstat(pid):
    try:
        raw = Path("/proc/%d/stat" % pid).read_text()
    except FileNotFoundError:
        return None
    rest = raw[raw.rindex(")") + 2:].split()
    return {"state": rest[0], "ppid": int(rest[1]), "session": int(rest[3])}


# --- detach: the server must outlive the mdl process that started it -------
r = mdl("run", "demo")
check("run exits 0", r.returncode, 0)
check("run reports ready", r.stdout.strip().split("\n")[-1].startswith("ready: demo"),
      True)
pid = json.loads(state_of("demo").read_text())["pid"]
st = procstat(pid)
check("server alive after mdl exited", st is not None and st["state"] != "Z", True)
check("start_new_session made it a session leader", st and st["session"], pid)
check("reparented away from the dead mdl", st and st["ppid"], 1)
check("ps sees it", mdl("ps").stdout.startswith("demo  pid %d" % pid), True)
check("logs -f is not required to read it", "offloaded 33/33" in mdl("logs").stdout,
      True)

# --- orphan self-heal ------------------------------------------------------
os.kill(pid, 9)
for _ in range(50):
    if procstat(pid) is None:
        break
    time.sleep(0.1)
check("ps self-heals after kill -9", mdl("ps").stdout, "nothing running\n")
check("stale state removed", state_of("demo").exists(), False)
check("run works again after a crash", mdl("run", "demo").returncode, 0)
mdl("stop")

# --- SIGTERM ignored -> escalate to SIGKILL --------------------------------
r = mdl("run", "demo", mode="stubborn")
check("stubborn server started", r.returncode, 0)
pid = json.loads(state_of("demo").read_text())["pid"]
t0 = time.time()
r = mdl("stop")
elapsed = time.time() - t0
check("stop succeeded", (r.returncode, r.stdout.strip()),
      (0, "stopped demo (pid %d)" % pid))
check("stop waited ~10s then escalated", 9.0 < elapsed < 14.0, True)
check("stubborn process is gone",
      procstat(pid) is None or procstat(pid)["state"] == "Z",
      True)
check("state cleaned up", state_of("demo").exists(), False)

# --- a wrapper script: stop must reap the tree, not orphan the server ------
# The launcher above uses exec, so mdl's pid *is* the server. Anyone whose
# llama_server sets LD_LIBRARY_PATH first has a real grandchild instead.
kidfile = HOME / "child.pid"
wrapper = HOME / "wrapper-llama-server"
wrapper.write_text('#!/bin/sh\n"%s" "%s" "$@" &\necho $! > %s\nwait\n'
                   % (sys.executable, support.FAKE, kidfile))
wrapper.chmod(0o755)
(CONFIG / "models.toml").write_text(
    'llama_server = "%s"\n\n[demo]\nmodel = "%s"\nport = %d\n'
    % (wrapper, support.FAKE, PORT))

check("wrapped server started", mdl("run", "demo").returncode, 0)
kid = int(kidfile.read_text())
parent = json.loads(state_of("demo").read_text())["pid"]
check("the server really is a grandchild", kid != parent, True)
mdl("stop")
for _ in range(50):
    if procstat(kid) is None or procstat(kid)["state"] == "Z":
        break
    time.sleep(0.1)
check("stop reaps the whole tree, not just the wrapper",
      procstat(kid) is None or procstat(kid)["state"] == "Z", True)
check("run works again straight after", mdl("run", "demo").returncode, 0)
mdl("stop")

# --- two at once, as real processes ----------------------------------------
PORT2 = support.free_port()
with open(CONFIG / "models.toml", "a") as fh:
    fh.write("\n[second]\nmodel = \"%s\"\nport = %d\n" % (support.FAKE, PORT2))

check("first starts", mdl("run", "demo").returncode, 0)
check("second starts alongside it", mdl("run", "second").returncode, 0)
listing = mdl("ps").stdout
check("ps shows both", ("demo" in listing, "second" in listing), (True, True))
pids = {n: json.loads(state_of(n).read_text())["pid"]
        for n in ("demo", "second")}
check("two live state files", sorted(pids), ["demo", "second"])
check("they are different processes", pids["demo"] != pids["second"], True)

r = mdl("stop")
check("bare stop will not guess", r.returncode, 1)
check("and says why", "several servers are running" in r.stderr, True)
check("both still up", len(json.loads(mdl("ps", "--json").stdout)), 2)

r = mdl("stop", "--all")
check("--all stops both", r.returncode, 0)
for name, pid in pids.items():
    for _ in range(50):
        if procstat(pid) is None or procstat(pid)["state"] == "Z":
            break
        time.sleep(0.1)
    check("%s is gone" % name,
          procstat(pid) is None or procstat(pid)["state"] == "Z", True)
check("ps is empty again", mdl("ps", "--json").stdout.strip(), "[]")

support.teardown(HOME)
sys.exit(t.done())
