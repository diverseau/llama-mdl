"""Shared test helpers: a sandboxed mdl, and a tiny result tally.

Every test runs against a temp config and temp state dir. Nothing here may
touch ~/.config/mdl or ~/.local/state/mdl - the UI tests write to the config
on purpose, and doing that to a real one would be unforgivable.
"""
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAKE = Path(__file__).resolve().parent / "fake_llama_server.py"
sys.path.insert(0, str(ROOT))

import mdl  # noqa: E402


class Tally:
    def __init__(self, name):
        self.name, self.fails = name, []

    def check(self, label, got, want):
        ok = got == want
        print(("PASS " if ok else "FAIL ") + label)
        if not ok:
            print("      got:  %r" % (got,))
            print("      want: %r" % (want,))
            self.fails.append(label)

    def done(self):
        print("\n%s: %d failure(s)" % (self.name, len(self.fails)))
        return 1 if self.fails else 0


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _launcher(root):
    """A runnable stand-in for llama-server that mdl can exec directly."""
    if os.name == "nt":
        path = root / "fake-llama-server.cmd"
        path.write_text('@echo off\r\n"%s" "%s" %%*\r\n' % (sys.executable, FAKE),
                        encoding="ascii")
    else:
        path = root / "fake-llama-server"
        path.write_text('#!/bin/sh\nexec "%s" "%s" "$@"\n' % (sys.executable, FAKE),
                        encoding="ascii")
        path.chmod(0o755)
    return path


def sandbox(port=None, extra="", model=None, binary=None):
    """Point mdl at a throwaway config and state dir. Returns (root, port)."""
    root = Path(tempfile.mkdtemp(prefix="mdl-test-"))
    (root / "config").mkdir()
    (root / "state").mkdir()
    mdl.CONFIG = root / "config" / "models.toml"
    mdl.STATE_DIR = root / "state"
    mdl.STATE = root / "state" / "state.json"
    port = port or free_port()
    binary = binary or _launcher(root)
    model = model or FAKE                # any real file works as the "model"
    mdl.CONFIG.write_text(
        'llama_server = "%s"\n\n'
        "[demo]\n"
        'model = "%s"\n'
        "ngl = 99\n"
        "ctx = 4096\n"
        "flash_attn = true\n"
        'kv_type = "q8_0"\n'
        "parallel = 1\n"
        "port = %d\n%s" % (str(binary).replace("\\", "/"),
                           str(model).replace("\\", "/"), port, extra),
        encoding="utf-8")
    return root, port


def teardown(root):
    for name in ("MDL_FAKE_MODE", "MDL_LLAMA_SERVER"):
        os.environ.pop(name, None)
    shutil.rmtree(root, ignore_errors=True)


def run(fn, *args):
    """Call a cmd_* function, capturing output the way main() reports it."""
    import contextlib
    import io
    out, err, code = io.StringIO(), io.StringIO(), 0
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            fn(*args)
    except SystemExit as e:
        code = e.code
    except mdl.MdlError as e:
        err.write("mdl: " + str(e) + "\n")
        code = 1
    return out.getvalue(), err.getvalue(), code
