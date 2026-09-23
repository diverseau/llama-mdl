"""mdl config opens the active config in your editor: it never parses or
rewrites the TOML, so a config too broken to load can still be fixed."""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                        # noqa: E402
from support import mdl, run                          # noqa: E402

t = support.Tally("test_config")
check = t.check


@contextlib.contextmanager
def config(text="invalid TOML ["):
    """A throwaway config, no editor set, and the editor never started:
    (path, the stand-in for subprocess.run)."""
    with tempfile.TemporaryDirectory(prefix="mdl config ") as tmp:
        path = Path(tmp) / "models.toml"
        if text is not None:
            path.write_text(text, encoding="utf-8")
        with patch.object(mdl, "CONFIG", path), \
                patch.dict(os.environ, {"VISUAL": "", "EDITOR": ""}), \
                patch.object(mdl.subprocess, "run") as launch:
            launch.return_value.returncode = 0
            yield path, launch


def launched(launch):
    """The one command the editor was started with, or None."""
    return launch.call_args.args[0] if launch.call_count == 1 else None


with config() as (path, launch):
    with patch.dict(os.environ, {"VISUAL": "code --wait", "EDITOR": "nano"}):
        mdl.cmd_config([])
    check("VISUAL wins over EDITOR, and gets the config's path",
          launched(launch), ["code", "--wait", str(path.resolve())])
    check("and the config is neither parsed nor changed",
          path.read_text(), "invalid TOML [")

with config() as (path, launch):
    editor = (r"C:\Program Files\Editor\edit.exe" if os.name == "nt"
              else "/my editor")
    with patch.dict(os.environ, {"EDITOR": '"%s" --wait' % editor}):
        mdl.cmd_config([])
    check("an editor path with spaces, quoted, keeps its arguments",
          launched(launch), [editor, "--wait", str(path.resolve())])

with config() as (path, launch):
    mdl.cmd_config([])
    check("with neither set, the platform's editor",
          launched(launch),
          ["notepad" if os.name == "nt" else "vi", str(path.resolve())])

with config(text=None) as (path, launch):
    _, err, code = run(mdl.cmd_config, [])
    check("no config yet: says to run mdl init, and opens nothing",
          ("Run 'mdl init' first" in err, code, launch.called),
          (True, 1, False))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        mdl.cmd_config(["--path"])
    check("--path works before init",
          (out.getvalue().strip(), launch.called),
          (str(path.resolve()), False))

with config() as (path, launch):
    _, err, code = run(mdl.cmd_config, ["--unknown"])
    check("an unknown argument is a usage line, and opens nothing",
          ("usage: mdl config" in err, code, launch.called),
          (True, 1, False))

with config() as (path, launch):
    launch.side_effect = FileNotFoundError("missing editor")
    _, err, code = run(mdl.cmd_config, [])
    check("an editor that cannot start is one line",
          ("cannot open config editor" in err, code), (True, 1))
    launch.side_effect = None
    launch.return_value.returncode = 7
    _, err, code = run(mdl.cmd_config, [])
    check("and one that fails says its status",
          ("exited with status 7" in err, code), (True, 1))

with tempfile.TemporaryDirectory(prefix="mdl config ") as tmp:
    result = subprocess.run(
        [sys.executable, mdl.__file__, "config", "--path"],
        env=dict(os.environ, XDG_CONFIG_HOME=tmp),
        capture_output=True, text=True)
    check("the real CLI's --path honours XDG_CONFIG_HOME",
          (result.returncode, result.stdout.strip(), result.stderr),
          (0, str((Path(tmp) / "mdl" / "models.toml").resolve()), ""))

sys.exit(t.done())
