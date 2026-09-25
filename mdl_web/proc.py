"""mdl as a process of its own, on the config and state this one uses.

The page starts a pull that must outlive it, and runs `mdl find`, which
is slow, apart from the server; both must see the same models.toml and
state directory as the page does, including a test's sandbox.
"""

import os
import subprocess
import sys
from pathlib import Path

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def argv(args):
    import mdl
    root = str(Path(__file__).resolve().parent.parent)
    code = ("import sys; sys.path.insert(0, %r); import mdl; "
            "from pathlib import Path; mdl.CONFIG = Path(%r); "
            "mdl.STATE_DIR = Path(%r); sys.argv = ['mdl'] + %r; mdl.main()"
            % (root, str(mdl.CONFIG), str(mdl.STATE_DIR), list(args)))
    return [sys.executable, "-c", code]


def detach(args):
    """Start `mdl args` on its own: no window, no terminal, and it lives
    on when this process ends."""
    flags = {}
    if os.name == "nt":
        flags["creationflags"] = (NO_WINDOW | subprocess.DETACHED_PROCESS
                                  | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        flags["start_new_session"] = True
    return subprocess.Popen(argv(args), stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, close_fds=True, **flags)


def output(args, timeout):
    """What `mdl args` prints, or None when it fails or runs too long."""
    try:
        r = subprocess.run(argv(args), capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout, creationflags=NO_WINDOW,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None
