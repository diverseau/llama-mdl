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
check("flash_attn=false is -fa off, not the build's default (B09)",
      mdl.build_argv("x", {"model": "m", "flash_attn": False}, "LS"),
      ["LS", "-m", "m", "-fa", "off"])
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

# B12: every key is checked at the boundary, one line each, before any
# socket code or subprocess sees it
for cfg, words in (
        ({"port": "8080"}, "'port' must be a number"),
        ({"port": 70000}, "'port' must be a number"),
        ({"port": 0}, "'port' must be a number"),
        ({"port": True}, "'port' must be a number"),
        ({"ctx": -1}, "'ctx' must be"),
        ({"ctx": "8k"}, "'ctx' must be"),
        ({"parallel": 0}, "'parallel' must be"),
        ({"ngl": "most"}, "'ngl' must be"),
        ({"flash_attn": "yes"}, "'flash_attn' must be true or false"),
        ({"kv_type": "q8 0"}, "'kv_type' must be"),
        ({"args": ["-t", 4]}, "'args' must be a list of strings"),
        ({"model": ""}, "'model' must be"),
        ({"args": ["--port", "9000"]}, "'--port' in args would override"),
        ({"args": ["--port=9000"]}, "'--port' in args would override"),
        ({"args": ["-m", "other.gguf"]}, "'-m' in args would override")):
    _, err, code = run(mdl.build_argv, "x", dict({"model": "m"}, **cfg), "LS")
    check("config %r is one line" % (cfg,),
          (words in err, err.count("\n"), code), (True, 1, 1))
check("ngl takes what llama.cpp takes",
      [mdl.build_argv("x", {"model": "m", "ngl": v}, "LS")[4]
       for v in (0, 99, -1, "all")], ["0", "99", "-1", "all"])
for bad in ("bad name", "a/b", "../up", ".hidden", "", "x" * 65):
    _, err, code = run(mdl.check_name, bad)
    check("the name %r is refused (B07)" % bad,
          ("bad model name" in err, code), (True, 1))
check("names with dots are quoted as TOML keys",
      [mdl.toml_key(n) for n in ("qwen3", "qwen3.5-9b")],
      ["qwen3", '"qwen3.5-9b"'])

os.environ["MDL_LLAMA_SERVER"] = "/opt/llama-server"
check("env var beats config", mdl.load_config()[1], "/opt/llama-server")
check("a model's own llama_server beats the environment",
      mdl.build_argv("x", {"model": "m", "llama_server": "prism"},
                     mdl.load_config()[1])[0], "prism")
for invalid in ("", "  ", 42, False, []):
    _, err, code = run(mdl.build_argv, "x",
                       {"model": "m", "llama_server": invalid}, "LS")
    check("a llama_server of %r is refused" % (invalid,), (err.strip(), code),
          ("mdl: model 'x': 'llama_server' must be a non-empty string", 1))
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
    mdl.run_dir().mkdir(parents=True, exist_ok=True)
    mdl.state_path("x").write_text(blob, encoding="utf-8")
    out, _, _ = run(mdl.cmd_ps, [])
    check("%s self-heals" % label, (out, mdl.state_path("x").exists()),
          ("nothing running\n", False))

check("uptime formatting", [mdl.uptime(x) for x in (0, 45, 201, 3720, 90061, -5)],
      ["0s", "45s", "3m21s", "1h02m", "25h01m", "0s"])

# ------------------------------------------------------------- pre-flight ---
_, err, code = run(mdl.spawn, "demo",
                   {"demo": dict(models["demo"], model="/no/such.gguf")},
                   binary)
check("missing model file is caught before launch",
      (err.strip(), code), ("mdl: model file not found: /no/such.gguf", 1))
_, err, code = run(mdl.spawn, "demo", models, "/no/such/llama-server")
check("missing binary is caught before launch",
      (err.strip(), code), ("mdl: llama-server not found: /no/such/llama-server", 1))

# a model with its own build runs on it even when the default is missing,
# and a missing one of its own is caught before launch
from unittest.mock import patch  # noqa: E402
with patch.object(mdl.subprocess, "Popen") as popen:
    popen.return_value.pid = 999999
    mdl.spawn("demo", {"demo": dict(models["demo"],
                                    llama_server=sys.executable)},
              "/no/such/global-server")
    check("spawn uses the model's llama_server, not the default",
          popen.call_args.args[0][0], sys.executable)
mdl.state_path("demo").unlink()
_, err, code = run(mdl.spawn, "demo",
                   {"demo": dict(models["demo"], llama_server="/no/such/prism")},
                   sys.executable)
check("a missing model llama_server is caught before launch",
      (err.strip(), code), ("mdl: llama-server not found: /no/such/prism", 1))

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
check("unknown model",
      (err.strip().endswith("no model named 'nope' in %s" % mdl.CONFIG),
                        code), (True, 1))

out, _, _ = run(mdl.cmd_ps, [])
check("ps shows it",
      out.startswith("demo  pid %d  port %d  up "
                     % (state["pid"], port)),
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
check("state cleaned up", mdl.state_path("demo").exists(), False)
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
check("dead server: state cleaned up", mdl.state_path("demo").exists(),
      False)
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
import tempfile  # noqa: E402
import tomllib  # noqa: E402

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

# a pipe on Windows is cp1252; the tables print → · ✓ and must not crash it
import io  # noqa: E402

piped = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
real_out, sys.stdout = sys.stdout, piped
try:
    mdl.safe_streams()
    print("Base → post-train · ✓")
    piped.flush()
    got = piped.buffer.getvalue().decode("cp1252")
finally:
    sys.stdout = real_out
check("a cp1252 pipe gets '?' for what it cannot encode, not a traceback",
      got.strip(), "Base ? post-train · ?")

# Hard rule 4, through the real entry point: a value the user typed wrong
# is one line on stderr and exit 1. Each of these was a traceback - an
# hf: spec the parser refused, and int() on an option's value.
import subprocess  # noqa: E402
import tempfile  # noqa: E402

home = Path(tempfile.mkdtemp(prefix="mdl-cli-errors-"))
env = dict(os.environ, XDG_CONFIG_HOME=str(home / "c"),
           XDG_STATE_HOME=str(home / "s"), XDG_CACHE_HOME=str(home / "k"),
           MDL_FIT_HOME=str(home / "f"), PYTHONIOENCODING="utf-8")
for args, words in (
        (["fit", "hf:bad"], "expected hf:org/repo"),
        (["fit", "inspect", "hf:bad"], "expected hf:org/repo"),
        (["fit", "inspect", "hf:a/b/c"], "expected hf:org/repo"),
        (["fit", "demo", "--apply", "abc"], "--apply takes the number"),
        (["fit", "demo", "--apply", "0"], "--apply takes the number"),
        (["eval", "demo", "--port", "abc"], "--port: port must be"),
        (["eval", "demo", "--port", "99999"], "--port: port must be"),
        (["find", "--top", "abc"], "--top takes a number"),
        (["find", "--top", "0"], "--top takes a number")):
    p = subprocess.run([sys.executable, str(support.ROOT / "mdl.py"), *args],
                       capture_output=True, env=env, encoding="utf-8",
                       errors="replace", timeout=60)
    lines = p.stderr.strip().splitlines()
    check("mdl %s: one line, exit 1" % " ".join(args),
          (p.returncode, len(lines), bool(lines) and lines[0].startswith(
              "mdl: ") and words in lines[0]), (1, 1, True))

# textual comes with every install since 0.13, and is still imported only
# by the dashboard: a textual that is broken, or slow to import, must cost
# the everyday commands nothing.
(home / "c" / "mdl").mkdir(parents=True, exist_ok=True)
(home / "c" / "mdl" / "models.toml").write_text(
    '[demo]\nmodel = "/nowhere/demo.gguf"\n', encoding="utf-8")
PROBE = ("import runpy, sys; sys.argv = ['mdl'] + sys.argv[1:]\n"
         "try:\n    runpy.run_path(sys.argv.pop(1), run_name='__main__')\n"
         "except SystemExit:\n    pass\n"
         "sys.stderr.write('textual=%s' % ('textual' in sys.modules))")
for args in (["--version"], ["ps"], ["list"], ["check"], ["--help"]):
    p = subprocess.run([sys.executable, "-c", PROBE,
                        str(support.ROOT / "mdl.py"), *args],
                       capture_output=True, env=env, encoding="utf-8",
                       errors="replace", timeout=60)
    check("mdl %s does not import textual" % " ".join(args),
          p.stderr.strip().splitlines()[-1:], ["textual=False"])
_sh.rmtree(home, ignore_errors=True)

sys.exit(t.done())
