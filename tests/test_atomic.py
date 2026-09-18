"""A write that fails must leave the old file intact.

models.toml is hand-edited and lives in nobody's git. Every one of these
kills a write at the worst moment and checks the config survived it.
"""
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                        # noqa: E402
from support import mdl, run, sandbox, teardown       # noqa: E402

import mdl_ui                                         # noqa: E402

t = support.Tally("test_atomic")
check = t.check
GGUF = support.FAKE


class Boom(Exception):
    pass


def kill_rename(exc):
    """Make os.replace fail, i.e. crash in the window before the swap."""
    real = mdl.os.replace

    def broken(*a, **k):
        raise exc

    mdl.os.replace = broken
    return lambda: setattr(mdl.os, "replace", real)


def strays(root):
    return sorted(p.name for p in (root / "config").iterdir()
                  if ".tmp" in p.name)


# ------------------------------------------------------- the writer ----
root, port = sandbox()
before = mdl.CONFIG.read_text(encoding="utf-8")

for exc, label in ((OSError("disk full"), "a failed write"),
                   (KeyboardInterrupt(), "a ctrl-c")):
    restore = kill_rename(exc)
    try:
        mdl.write_atomic(mdl.CONFIG, "ruined")
    except (OSError, KeyboardInterrupt):
        pass
    finally:
        restore()
    check("%s leaves the config untouched" % label,
          mdl.CONFIG.read_text(encoding="utf-8"), before)
    check("%s leaves no temp file behind" % label, strays(root), [])

mdl.write_atomic(mdl.CONFIG, before + "\n# added\n")
check("a good write does land",
      mdl.CONFIG.read_text(encoding="utf-8").endswith("# added\n"), True)
check("and cleans up after itself", strays(root), [])

mdl.write_atomic(mdl.CONFIG, "[fresh]\n", keep_backup=True)
bak = mdl.CONFIG.with_name(mdl.CONFIG.name + ".bak")
check("a backup is kept when asked", bak.is_file(), True)
check("the backup holds what was there before",
      bak.read_text(encoding="utf-8").endswith("# added\n"), True)
teardown(root)

# ----------------------------------------------------------- mdl add ----
root, port = sandbox()
before = mdl.CONFIG.read_text(encoding="utf-8")
restore = kill_rename(OSError("disk full"))
try:
    out, err, code = run(mdl.cmd_add, [str(GGUF), "doomed"])
finally:
    restore()
check("add reports the failure", ("cannot write" in err, code), (True, 1))
check("a failed add leaves the config byte for byte",
      mdl.CONFIG.read_text(encoding="utf-8"), before)
check("a failed add leaves valid toml",
      "demo" in tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8")), True)
check("a failed add leaves no temp file", strays(root), [])

run(mdl.cmd_add, [str(GGUF), "kept"])
check("a good add still works",
      "kept" in tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8")), True)
check("add keeps a backup first",
      mdl.CONFIG.with_name(mdl.CONFIG.name + ".bak").is_file(), True)
teardown(root)

# ------------------------------------------------------ write_params ----
root, port = sandbox()
mdl.CONFIG.write_text(
    "# my notes\n"
    "llama_server = \"x\"\n\n"
    "[demo]\n"
    "model = \"m.gguf\"   # the good one\n"
    "ngl = 99\n", encoding="utf-8")
before = mdl.CONFIG.read_text(encoding="utf-8")

restore = kill_rename(OSError("disk full"))
try:
    mdl_ui.write_params("demo", {"ngl": 40})
except OSError:
    pass
finally:
    restore()
check("a failed edit leaves the config untouched",
      mdl.CONFIG.read_text(encoding="utf-8"), before)
check("a failed edit leaves no temp file", strays(root), [])

mdl_ui.write_params("demo", {"ngl": 40})
saved = mdl.CONFIG.read_text(encoding="utf-8")
check("a good edit lands", tomllib.loads(saved)["demo"]["ngl"], 40)
check("comments survive the edit", "# my notes" in saved, True)
check("an edit keeps a backup",
      mdl.CONFIG.with_name(mdl.CONFIG.name + ".bak").read_text(encoding="utf-8"),
      before)
teardown(root)

# ------------------------------------------------ patch_params (B01) ----
# `mdl fit --apply` knows the flags it tuned and nothing else. It used to
# write through the full-replace path and delete model, port, group and
# llama_server with it; the next run had no model to load.
root, port = sandbox()
mdl.CONFIG.write_text(
    "[demo]  # the fork build\n"
    "model = \"C:/m/x.gguf\"   # keep me\n"
    "mmproj = \"C:/m/mmproj.gguf\"\n"
    "llama_server = \"C:/prism/llama-server.exe\"\n"
    "port = 8181\n"
    "group = \"forks\"\n"
    "ctx = 4096\n"
    "n_cpu_moe = 12\n"
    "args = [\n"
    "  \"--jinja\",\n"
    "  \"--temp\", \"0.6\",\n"
    "]\n\n"
    "[other]\nmodel = \"o.gguf\"\n", encoding="utf-8")
mdl.patch_params("demo", {"ctx": 65536, "ngl": 99, "flash_attn": True,
                          "args": ["--jinja", "--temp", "0.6"]},
                 drop=("n_cpu_moe",))
saved = mdl.CONFIG.read_text(encoding="utf-8")
got = tomllib.loads(saved)
check("a patch keeps every key it was not about",
      {k: got["demo"][k] for k in ("model", "mmproj", "llama_server", "port",
                                   "group")},
      {"model": "C:/m/x.gguf", "mmproj": "C:/m/mmproj.gguf",
       "llama_server": "C:/prism/llama-server.exe", "port": 8181,
       "group": "forks"})
check("sets what it was given, drops only what it was told to",
      (got["demo"]["ctx"], got["demo"]["ngl"], "n_cpu_moe" in got["demo"]),
      (65536, 99, False))
check("an args array over several lines is replaced, not left dangling",
      got["demo"]["args"], ["--jinja", "--temp", "0.6"])
check("the next table is untouched", got["other"], {"model": "o.gguf"})
check("inline comments on rewritten and kept lines survive",
      ("# keep me" in saved, "# the fork build" in saved), (True, True))
models, binary = mdl.load_config()
check("and the preset still builds a command on its own server",
      mdl.build_argv("demo", models["demo"], binary)[:3],
      ["C:/prism/llama-server.exe", "-m", "C:/m/x.gguf"])
teardown(root)

# --------------------------------------- hand-written layouts (B06) ----
# valid TOML the old writer could not find, or wrote twice
root, port = sandbox()
for label, text in (
        ("a comment after the header", "[demo] # mine\nngl = 1\n"),
        ("an indented table", "  [demo]\n  ngl = 1\n  ctx = 2\n"),
        ("a quoted name", "[\"demo\"]\nngl = 1\n")):
    mdl.CONFIG.write_text(text, encoding="utf-8")
    mdl_ui.write_params("demo", {"ngl": 40, "ctx": 8})
    saved = mdl.CONFIG.read_text(encoding="utf-8")
    check("%s is edited, not refused" % label,
          tomllib.loads(saved)["demo"], {"ngl": 40, "ctx": 8})
    check("%s: no key written twice" % label, saved.count("ngl"), 1)
mdl.CONFIG.write_text("[demo]\nngl = 1\n", encoding="utf-8")
_, err, code = support.run(mdl.write_params, "nope", {"ngl": 2})
check("a missing table is one line, not a ValueError",
      (code, "no [nope] table" in err), (1, True))
mdl_ui.write_params("demo", {"ngl": 5})       # the dashboard clears by omission
mdl.CONFIG.write_text("[demo]\nngl = 1\nctx = 2\n", encoding="utf-8")
mdl_ui.write_params("demo", {"ngl": 5})
check("the dashboard still clears a field it leaves out",
      tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8"))["demo"],
      {"ngl": 5})
teardown(root)

sys.exit(t.done())
