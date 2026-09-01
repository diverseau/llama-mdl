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

sys.exit(t.done())
