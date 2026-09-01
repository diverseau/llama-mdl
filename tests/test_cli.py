"""mdl's four-and-a-bit commands, against a fake llama-server."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                   # noqa: E402
from support import mdl, run, sandbox, teardown  # noqa: E402

t = Tally = support.Tally("test_cli")
check = t.check


def wait_gone(pid, seconds=10):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not mdl.alive(pid):
            return True
        time.sleep(0.1)
    return False


# ---------------------------------------------------------------- mapping ---
root, port = sandbox(extra='args = ["--metrics"]\n')
models, binary = mdl.load_config()
argv = mdl.build_argv("demo", models["demo"], "LS")
check("full flag mapping", " ".join(argv[1:]),
      "-m %s -ngl 99 -c 4096 -np 1 --port %d -fa on "
      "--cache-type-k q8_0 --cache-type-v q8_0 --metrics"
      % (str(support.FAKE).replace("\\", "/"), port))
check("flash_attn=false omits -fa",
      mdl.build_argv("x", {"model": "m", "flash_attn": False}, "LS"), ["LS", "-m", "m"])
check("port defaults to 8080",
      mdl.build_argv("x", {"model": "m"}, "LS"), ["LS", "-m", "m"])

_, err, code = run(mdl.build_argv, "x", {"model": "m", "flash_atn": 1, "zz": 2}, "LS")
check("unknown keys are an error",
      (err.strip(), code), ("mdl: model 'x': unknown key(s): flash_atn, zz", 1))
_, err, code = run(mdl.build_argv, "x", {"ngl": 9}, "LS")
check("missing model key", (err.strip(), code),
      ("mdl: model 'x': missing required key 'model'", 1))
_, err, code = run(mdl.build_argv, "x", {"model": "m", "args": "-f"}, "LS")
check("args must be a list", (err.strip(), code),
      ("mdl: model 'x': 'args' must be a list of strings", 1))

os.environ["MDL_LLAMA_SERVER"] = "/opt/llama-server"
check("env var beats config", mdl.load_config()[1], "/opt/llama-server")
del os.environ["MDL_LLAMA_SERVER"]

# ------------------------------------------------------------ config file ---
mdl.CONFIG = root / "config" / "nope.toml"
_, err, code = run(mdl.cmd_list, [])
check("missing config", (err.strip(), code),
      ("mdl: no config at %s; run 'mdl init' to create one" % mdl.CONFIG, 1))
bad = root / "config" / "bad.toml"
bad.write_text("[oops\n", encoding="utf-8")
mdl.CONFIG = bad
_, err, code = run(mdl.cmd_list, [])
check("malformed toml: one line, no traceback",
      (err.count("\n"), code, err.startswith("mdl: cannot read")), (1, 1, True))
mdl.CONFIG = root / "config" / "models.toml"

out, _, _ = run(mdl.cmd_list, [])
check("list", out, "demo  %s\n" % str(support.FAKE).replace("\\", "/"))

# ------------------------------------------------------------- self-heal ----
out, _, code = run(mdl.cmd_ps, [])
check("ps when idle", (out, code), ("nothing running\n", 0))
out, _, code = run(mdl.cmd_stop, [])
check("stop when idle", (out, code), ("nothing running\n", 0))

for label, blob in (("stale pid", '{"name":"x","pid":999999,"port":1,"started":0}'),
                    ("corrupt json", "{not json"),
                    ("missing pid", '{"name":"x"}')):
    mdl.STATE.write_text(blob, encoding="utf-8")
    out, _, _ = run(mdl.cmd_ps, [])
    check("%s self-heals" % label, (out, mdl.STATE.exists()),
          ("nothing running\n", False))

check("uptime formatting", [mdl.uptime(x) for x in (0, 45, 201, 3720, 90061, -5)],
      ["0s", "45s", "3m21s", "1h02m", "25h01m", "0s"])

# ------------------------------------------------------------- pre-flight ---
_, err, code = run(mdl.spawn, "demo", {"demo": dict(models["demo"], model="/no/such.gguf")},
                   binary)
check("missing model file is caught before launch",
      (err.strip(), code), ("mdl: model file not found: /no/such.gguf", 1))
_, err, code = run(mdl.spawn, "demo", models, "/no/such/llama-server")
check("missing binary is caught before launch",
      (err.strip(), code), ("mdl: llama-server not found: /no/such/llama-server", 1))

import socket  # noqa: E402
blocker = socket.socket()
blocker.bind(("127.0.0.1", port))
blocker.listen(1)
_, err, code = run(mdl.spawn, "demo", models, binary)
check("busy port is caught before launch",
      (err.strip(), code), ("mdl: port %d is already in use" % port, 1))
blocker.close()

# ------------------------------------------------------------------ run -----
models, binary = mdl.load_config()
out, err, code = run(mdl.cmd_run, ["demo"])
state = mdl.read_state()
check("run exits 0", (code, err), (0, ""))
check("run tailed the log", "offloaded 33/33 layers" in out, True)
check("run announced readiness", out.strip().split("\n")[-1],
      "ready: demo on http://127.0.0.1:%d (pid %d)" % (port, state["pid"]))
check("state file", (state["name"], state["port"], mdl.alive(state["pid"])),
      ("demo", port, True))
check("log file written", (mdl.STATE_DIR / "demo.log").is_file(), True)
check("readiness came from /health", mdl.server_ready(port), True)

_, err, code = run(mdl.cmd_run, ["demo"])
check("second run refused",
      (err.startswith("mdl: 'demo' is already running"), "mdl stop" in err, code),
      (True, True, 1))
_, err, code = run(mdl.cmd_run, ["nope"])
check("unknown model", (err.strip().endswith("no model named 'nope' in %s" % mdl.CONFIG),
                        code), (True, 1))

out, _, _ = run(mdl.cmd_ps, [])
check("ps shows it", out.startswith("demo  pid %d  port %d  up " % (state["pid"], port)),
      True)
out, _, _ = run(mdl.cmd_logs, [])
check("logs shows the running server's log", "offloaded 33/33" in out, True)
out, _, _ = run(mdl.cmd_logs, ["demo"])
check("logs by name", "offloaded 33/33" in out, True)
_, err, code = run(mdl.cmd_logs, ["ghost"])
check("logs for an unknown name", (err.startswith("mdl: no log at"), code), (True, 1))

out, _, code = run(mdl.cmd_stop, [])
check("stop", (out, code), ("stopped demo (pid %d)\n" % state["pid"], 0))
check("process gone", wait_gone(state["pid"]), True)
check("state cleaned up", mdl.STATE.exists(), False)
_, err, code = run(mdl.cmd_logs, [])
check("logs with nothing running",
      (err.strip(), code), ("mdl: nothing running; pass a model name", 1))
teardown(root)

# ------------------------------------------- server that dies during load ---
root, port = sandbox()
os.environ["MDL_FAKE_MODE"] = "fail"
_, err, code = run(mdl.cmd_run, ["demo"])
check("dead server: exit 1", code, 1)
check("dead server: reports status", "exited with status 1" in err, True)
check("dead server: one line, no traceback", (err.count("\n"), "Traceback" in err),
      (1, False))
check("dead server: state cleaned up", mdl.STATE.exists(), False)
teardown(root)

# ------------------------------------- listening but never reporting ready ---
root, port = sandbox()
os.environ["MDL_FAKE_MODE"] = "silent"
mdl.READY_TIMEOUT = 3
_, err, code = run(mdl.cmd_run, ["demo"])
check("no /health means not ready, even though the log says listening",
      ("not ready after 3s" in err, code), (True, 1))
state = mdl.read_state()
if state:
    run(mdl.cmd_stop, [])
teardown(root)

# ------------------------------------------------------- init / paths ---
import tempfile, tomllib  # noqa: E402

home = Path(tempfile.mkdtemp(prefix="mdl-init-"))
os.environ["XDG_CONFIG_HOME"] = str(home / "cfg")
os.environ["XDG_STATE_HOME"] = str(home / "st")
import importlib  # noqa: E402
fresh = importlib.reload(mdl)
check("XDG_CONFIG_HOME is honoured", fresh.CONFIG,
      home / "cfg" / "mdl" / "models.toml")
check("XDG_STATE_HOME is honoured", fresh.STATE_DIR, home / "st" / "mdl")

_, err, code = run(fresh.cmd_list, [])
check("missing config points at init", ("run 'mdl init'" in err, code), (True, 1))

out, _, code = run(fresh.cmd_init, [])
check("init writes a config", (fresh.CONFIG.is_file(), code), (True, 0))
starter = tomllib.loads(fresh.CONFIG.read_text(encoding="utf-8"))
check("starter config is valid toml", "example" in starter, True)
check("starter uses only known keys",
      set(starter["example"]) <= fresh.KNOWN, True)
check("starter builds a real command",
      "-fa" in fresh.build_argv("example", starter["example"], "LS"), True)
_, err, code = run(fresh.cmd_init, [])
check("init refuses to clobber", ("already exists" in err, code), (True, 1))
_, err, code = run(fresh.cmd_init, ["x"])
check("init takes no arguments", (err.strip(), code), ("mdl: usage: mdl init", 1))

check("version is set", bool(fresh.VERSION), True)
for name in ("XDG_CONFIG_HOME", "XDG_STATE_HOME"):
    del os.environ[name]
import shutil as _sh  # noqa: E402
_sh.rmtree(home, ignore_errors=True)

sys.exit(t.done())
