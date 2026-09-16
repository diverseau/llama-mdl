"""Config editing uses the active path without parsing or rewriting TOML."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mdl  # noqa: E402


class ConfigTests(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory(prefix="mdl config "))
        self.path = Path(tmp) / "models.toml"
        self.path.write_text("invalid TOML [", encoding="utf-8")
        self.enterContext(patch.object(mdl, "CONFIG", self.path))
        self.enterContext(patch.dict(os.environ, {"VISUAL": "", "EDITOR": ""}))
        self.launch = self.enterContext(patch.object(mdl.subprocess, "run"))
        self.launch.return_value.returncode = 0

    def test_visual_wins_and_config_is_not_parsed_or_changed(self):
        with patch.dict(os.environ, {"VISUAL": "code --wait", "EDITOR": "nano"}):
            mdl.cmd_config([])
        self.launch.assert_called_once_with(["code", "--wait", str(self.path)])
        self.assertEqual(self.path.read_text(), "invalid TOML [")

    def test_editor_with_quoted_executable_and_spaces(self):
        editor = (r'C:\Program Files\Editor\edit.exe'
                  if os.name == "nt" else "/my editor")
        with patch.dict(os.environ, {"EDITOR": '"%s" --wait' % editor}):
            mdl.cmd_config([])
        self.launch.assert_called_once_with([editor, "--wait", str(self.path)])

    def test_default_editor(self):
        mdl.cmd_config([])
        self.launch.assert_called_once_with(
            ["notepad" if os.name == "nt" else "vi", str(self.path)])

    def test_missing_config(self):
        self.path.unlink()
        with self.assertRaisesRegex(mdl.MdlError, "Run 'mdl init' first"):
            mdl.cmd_config([])
        self.launch.assert_not_called()

    def test_path_works_before_init(self):
        self.path.unlink()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mdl.cmd_config(["--path"])
        self.assertEqual(out.getvalue().strip(), str(self.path))
        self.launch.assert_not_called()

    def test_bad_arguments(self):
        with self.assertRaisesRegex(mdl.MdlError, "usage: mdl config"):
            mdl.cmd_config(["--unknown"])
        self.launch.assert_not_called()

    def test_editor_errors(self):
        self.launch.side_effect = FileNotFoundError("missing editor")
        with self.assertRaisesRegex(mdl.MdlError, "cannot open config editor"):
            mdl.cmd_config([])
        self.launch.side_effect = None
        self.launch.return_value.returncode = 7
        with self.assertRaisesRegex(mdl.MdlError, "exited with status 7"):
            mdl.cmd_config([])


class DispatchTests(unittest.TestCase):
    def test_path_honours_xdg_in_real_cli(self):
        with tempfile.TemporaryDirectory(prefix="mdl config ") as tmp:
            result = subprocess.run(
                [sys.executable, mdl.__file__, "config", "--path"],
                env=dict(os.environ, XDG_CONFIG_HOME=tmp),
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(),
                             str(Path(tmp) / "mdl" / "models.toml"))


if __name__ == "__main__":
    unittest.main()
