"""Doctor diagnoses config and runtime without changing either."""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support  # noqa: E402
from support import mdl, run, sandbox, teardown  # noqa: E402

t = support.Tally("test_doctor")
check = t.check
root, port = sandbox()
model = root / "model.gguf"
model.write_bytes(b"GGUF" + bytes(16))
mdl.patch_params("demo", {"model": str(model)})
original = mdl.CONFIG.read_bytes()


def diagnose(args=None):
    out, err, code = run(mdl.cmd_doctor, ["--json"] + (args or []))
    return json.loads(out), err, code


def has(report, level, text, name="demo"):
    notes = report["global"] if name is None else report["models"][name]
    return any(f["level"] == level and text in f["message"] for f in notes)


def snapshot():
    return {str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


try:
    report, err, code = diagnose()
    check("healthy fake binary and GGUF", (code, err), (0, ""))
    check("build reported", has(report, "ok", "version: 1234 (fake)"), True)
    check("JSON shape", set(report), {"global", "models"})
    check("finding shape", all(set(f) == {"level", "check", "message"}
                               for f in report["models"]["demo"]), True)
    out, _, code = run(mdl.cmd_doctor, [])
    check("text headers and summary", ("demo:\n" in out,
                                       out.endswith("0 fail, 0 warn\n")),
          (True, True))
    for changes, level, message in [
        ({"model": str(root / "missing.gguf")}, "fail", "model file not found"),
        ({"model": str(support.FAKE)}, "fail", "file is not GGUF"),
        ({"model": mdl.PLACEHOLDER}, "warn", "not filled in yet"),
        ({"oops": 1}, "fail", "unknown key(s): oops"),
        ({"port": "bad"}, "fail", "'port' must be a number from 1 to 65535"),
        ({"port": []}, "fail", "'port' must be a number from 1 to 65535"),
        ({"llama_server": "/no/such"}, "fail", "llama-server not found"),
        ({"args": ["--not-supported=yes"]}, "warn", "--not-supported"),
        ({"mmproj": str(support.FAKE)}, "fail", "mmproj file is not GGUF"),
        ({"mmproj": str(root / "gone")}, "fail", "mmproj file not found"),
        ({"mmproj": str(model)}, "ok", "mmproj GGUF header valid"),
    ]:
        mdl.CONFIG.write_bytes(original)
        mdl.patch_params("demo", changes)
        report, err, code = diagnose()
        check(message, has(report, level, message), True)
        check("exit code for " + message, code, int(level == "fail"))
        if code:
            check("failure stderr is one line", len(err.splitlines()), 1)
    mdl.CONFIG.write_bytes(original)
    with mdl.CONFIG.open("a", encoding="utf-8") as fh:
        fh.write('\n[other]\nmodel = %s\nport = %d\n' %
                 (mdl.toml_value(str(model)), port))
    real_run = subprocess.run
    with patch.object(mdl.subprocess, "run", wraps=real_run) as probe:
        report, _, code = diagnose()
        check("help cached per binary", sum(c.args[0][-1] == "--help"
                                            for c in probe.call_args_list), 1)
    check("shared port", has(report, "warn", "share port", None), True)
    report, _, _ = diagnose(["demo"])
    check("name limits presets", list(report["models"]), ["demo"])
    _, err, code = run(mdl.cmd_doctor, ["unknown"])
    check("unknown name one-line error", (code, len(err.splitlines()),
                                          "no model named" in err), (1, 1, True))
    mdl.CONFIG.write_bytes(original)
    with patch.object(mdl.subprocess, "run", side_effect=OSError("cannot execute")):
        report, _, code = diagnose()
    check("probe failures warn", (code, has(report, "warn", "cannot run --version"),
                                  has(report, "warn", "cannot run --help")),
          (0, True, True))
    with patch.object(mdl.subprocess, "run",
                      side_effect=subprocess.TimeoutExpired("fake", 10)):
        report, _, code = diagnose()
    check("probe timeouts warn", (code, has(report, "warn", "flags skipped")),
          (0, True))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", port))
        listener.listen()
        report, _, code = diagnose()
        check("untracked listener", has(report, "warn", "something else"), True)
    _, err, code = run(mdl.cmd_run, ["demo"])
    check("test server started", (code, err), (0, ""))
    try:
        before = snapshot()
        report, _, code = diagnose()
        check("running health", has(report, "ok", "/health answers"), True)
        check("doctor preserves running state and files", snapshot(), before)
        check("server still alive", mdl.server_ready(port), True)
        with patch.object(mdl, "server_ready", return_value=False):
            report, _, _ = diagnose()
        check("unhealthy live server warns",
              has(report, "warn", "/health does not answer"), True)
        with patch.object(mdl, "alive", return_value=False), \
                patch.object(mdl, "survivors", return_value=[123]):
            report, _, _ = diagnose()
        check("orphan tree warns", has(report, "warn", "server it started survives"),
              True)
    finally:
        run(mdl.cmd_stop, ["--all"])
    mdl.run_dir().mkdir(exist_ok=True)
    (mdl.run_dir() / "old.lock").write_text("0")
    (mdl.run_dir() / "live.lock").write_text(str(os.getpid()))
    mdl.state_path("demo").write_text(json.dumps({"pid": 0, "port": port}))
    (mdl.run_dir() / "broken.json").write_text("{")
    mdl.STATE.write_text(json.dumps({"name": "legacy", "pid": 0}))
    before = snapshot()
    report, _, _ = diagnose()
    check("leftover lock warns", has(report, "warn", "old.lock", None), True)
    check("live lock is not stale", has(report, "warn", "live.lock", None), False)
    check("stale and legacy state left untouched", snapshot(), before)
    with patch.object(mdl.tempfile, "TemporaryFile",
                      side_effect=PermissionError("not writable")):
        report, _, code = diagnose()
    check("unwritable directories fail", (code, has(report, "fail", "not writable",
                                                   None)), (1, True))
    mdl.CONFIG.write_text('["bad name"]\nmodel = "missing"\n')
    report, _, code = diagnose()
    check("bad name is a finding", has(report, "fail", "bad model name",
                                      "bad name"), True)
    mdl.CONFIG.write_text("[broken")
    report, err, code = diagnose()
    check("bad TOML stops with global failure", (code, report["models"],
                                                len(err.splitlines())), (1, {}, 1))
    mdl.CONFIG.unlink()
    report, _, code = diagnose()
    check("missing config fails", (code, has(report, "fail", "no config", None)),
          (1, True))
    check("doctor registered", mdl.COMMANDS.get("doctor"), mdl.cmd_doctor)
    # review fixes: a negative value is not a flag, and a pid recycled
    # onto another process is a stale state, not a live server
    root2, port2 = sandbox()
    try:
        mdl.CONFIG.write_text(mdl.CONFIG.read_text(encoding="utf-8").replace(
            "ngl = 99", "ngl = -1"), encoding="utf-8")
        report, _, _ = diagnose()
        check("-ngl -1 is not reported as an unknown flag",
              has(report, "warn", "-1"), False)
        mdl.run_dir().mkdir(parents=True, exist_ok=True)
        mdl.state_path("demo").write_text(json.dumps(
            {"name": "demo", "pid": os.getpid(), "port": port2, "born": 1}))
        report, _, _ = diagnose()
        check("a recycled pid reads as a stale state",
              has(report, "warn", "stale state"), True)
    finally:
        teardown(root2)
finally:
    teardown(root)

sys.exit(t.done())
