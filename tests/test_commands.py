"""add / check / ps --json / log rotation / pid-reuse / ready_timeout."""
import json
import os
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                   # noqa: E402
from support import mdl, run, sandbox, teardown  # noqa: E402

t = support.Tally("test_commands")
check = t.check
GGUF = support.FAKE          # not a real gguf; layer detection must cope

# ------------------------------------------------------------------ add ----
root, port = sandbox()
_, err, code = run(mdl.cmd_add, [])
check("add needs an argument", ("usage: mdl add" in err, code), (True, 1))
_, err, code = run(mdl.cmd_add, ["/no/such/model.gguf"])
check("add rejects a missing file", ("no such file" in err, code), (True, 1))

out, _, code = run(mdl.cmd_add, [str(GGUF), "extra"])
check("add writes an entry", (code, "added [extra]" in out), (0, True))
saved = tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8"))
check("added model is in the config", "extra" in saved, True)
check("added entry has sane defaults",
      (saved["extra"]["ngl"], saved["extra"]["flash_attn"], saved["extra"]["port"]),
      (99, True, 8080))
check("added entry uses only known keys", set(saved["extra"]) <= mdl.KNOWN, True)
check("existing model untouched", "demo" in saved, True)
check("added entry builds a command",
      "-fa" in mdl.build_argv("extra", saved["extra"], "LS"), True)

_, err, code = run(mdl.cmd_add, [str(GGUF), "extra"])
check("add refuses a duplicate name", ("already in" in err, code), (True, 1))
_, err, code = run(mdl.cmd_add, [str(GGUF), "p", "notanumber"])
check("add validates the port", ("port must be a number" in err, code), (True, 1))
out, _, _ = run(mdl.cmd_add, [str(GGUF), "ported", "9001"])
check("add honours an explicit port",
      tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8"))["ported"]["port"], 9001)

check("gguf_layers returns None for a non-gguf", mdl.gguf_layers(GGUF), None)
check("human_size",
      [mdl.human_size(n) for n in (900, 4096, 5 << 20, 3 << 30, 2 << 40)],
      ["900B", "4.0K", "5.0M", "3.0G", "2.0T"])

# ---------------------------------------------------------------- check ----
# everything above wrote *valid* entries, so break one on purpose
with open(mdl.CONFIG, "a", encoding="utf-8") as fh:
    fh.write('\n[gone]\nmodel = \"/no/such/model.gguf\"\n')

out, err, code = run(mdl.cmd_check, [])
check("check flags a missing model file", "model file not found" in out, True)
check("check exits non-zero when it finds problems", code, 1)
_, err, code = run(mdl.cmd_check, ["x"])
check("check takes no arguments", (err.strip(), code), ("mdl: usage: mdl check", 1))
teardown(root)

# a config where everything is fine
root, port = sandbox()
out, err, code = run(mdl.cmd_check, [])
check("check passes a good config", (out.strip().endswith("ok"), code), (True, 0))

# ------------------------------------------------------------ ps --json ----
out, _, _ = run(mdl.cmd_ps, ["--json"])
check("ps --json is a list even when idle", out.strip(), "[]")
_, err, code = run(mdl.cmd_ps, ["--bogus"])
check("ps rejects unknown flags", (err.strip(), code),
      ("mdl: usage: mdl ps [--json]", 1))

# ------------------------------------------- run: rotation, born, timeout ---
run(mdl.cmd_run, ["demo"])
state = mdl.read_state()
check("state records the process creation time", state.get("born") is not None
      or os.uname().sysname == "Darwin" if hasattr(os, "uname") else True, True)

out, _, _ = run(mdl.cmd_ps, ["--json"])
live = json.loads(out)
check("ps --json is a list of servers", isinstance(live, list), True)
check("ps --json while running", (live[0]["name"], live[0]["port"]),
      ("demo", port))
check("ps --json includes uptime", "uptime" in live[0], True)

# a recycled pid: same pid, different creation time
if state.get("born") is not None:
    path = mdl.state_path("demo")
    path.write_text(json.dumps(dict(state, born=state["born"] + 999)),
                    encoding="utf-8")
    check("a recycled pid is not mistaken for our server",
          mdl.read_state("demo"), None)
    check("that stale state is removed", path.exists(), False)
    path.write_text(json.dumps(state), encoding="utf-8")

run(mdl.cmd_stop, [])
run(mdl.cmd_run, ["demo"])
check("previous log kept as .1", (mdl.STATE_DIR / "demo.log.1").is_file(), True)
run(mdl.cmd_stop, [])
run(mdl.cmd_run, ["demo"])
check("rotation shuffles along", (mdl.STATE_DIR / "demo.log.2").is_file(), True)
run(mdl.cmd_stop, [])
check("rotation stops at KEEP_LOGS",
      (mdl.STATE_DIR / ("demo.log.%d" % (mdl.KEEP_LOGS + 1))).exists(), False)

check("ready_timeout defaults to the constant", mdl.ready_timeout(), mdl.READY_TIMEOUT)
mdl.CONFIG_DATA["ready_timeout"] = 42
check("ready_timeout is read from the config", mdl.ready_timeout(), 42.0)
mdl.CONFIG_DATA["ready_timeout"] = "nonsense"
check("a bad ready_timeout falls back", mdl.ready_timeout(), mdl.READY_TIMEOUT)
mdl.CONFIG_DATA.pop("ready_timeout")
teardown(root)

# --------------------------------------------------- a fresh config ----
# mdl init then mdl check is the first thing anyone does. It must pass.
root, port = sandbox()
mdl.CONFIG.unlink()
run(mdl.cmd_init, [])
# init fills llama_server from PATH, so on a machine without llama-server
# it leaves a placeholder there too - a real problem, and not the one
# under test. Point it at something that certainly exists.
mdl.CONFIG.write_text(
    re.sub(r'^llama_server = .*$',
           'llama_server = "%s"' % sys.executable.replace(chr(92), '/'),
           mdl.CONFIG.read_text(encoding='utf-8'), count=1, flags=re.M),
    encoding='utf-8')
out, err, code = run(mdl.cmd_check, [])
check("a freshly initialised config passes check", (code, err), (0, ""))
check("the placeholder is called out as a to-do",
      "not filled in yet" in out, True)
check("the starter really does use the placeholder path",
      mdl.PLACEHOLDER in mdl.STARTER, True)
teardown(root)

# ---- vision -------------------------------------------------------------
root, port = sandbox()
vd = root / "vision"
vd.mkdir()
weights = vd / "Gemma-3-12B-Q4_K_M.gguf"
weights.write_bytes(b"GGUF" + bytes(64))
proj = vd / "mmproj-F16.gguf"
proj.write_bytes(b"GGUF" + bytes(32))

check("group is a known key",  "group" in mdl.KNOWN, True)
check("but it never reaches llama-server",
      mdl.build_argv("g", {"model": "m.gguf", "group": "qwen"}, "srv"),
      ["srv", "-m", "m.gguf"])

check("mmproj becomes --mmproj",
      mdl.build_argv("v", {"model": "m.gguf", "mmproj": "p.gguf"}, "srv"),
      ["srv", "-m", "m.gguf", "--mmproj", "p.gguf"])
check("and it is a known key, not a typo",
      "mmproj" in mdl.KNOWN, True)
check("add finds the projector beside the weights",
      mdl.find_mmproj(weights), proj)

out, _, code = run(mdl.cmd_add, [str(weights)])
check("add writes it into the block", code, 0)
added = tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8"))["gemma-3-12b"]
check("as an absolute path", Path(added["mmproj"]), proj)
check("and says so", "vision: mmproj-F16.gguf" in out, True)

second = vd / "mmproj-Q8_0.gguf"
second.write_bytes(b"GGUF")
check("two candidates is a choice, so it declines to guess",
      mdl.find_mmproj(weights), None)
second.unlink()

proj.unlink()
out, _, code = run(mdl.cmd_check, [])
check("check notices a projector that is gone",
      "mmproj file not found" in out, True)
check("and fails the run", code, 1)
teardown(root)

sys.exit(t.done())
