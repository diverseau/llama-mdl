#!/usr/bin/env python3
"""A new user's first session, measured: from nothing to a model answering.

Runs this tree's mdl in a throwaway home - its own config, state, cache,
fit and models directories - through the commands a newcomer would type,
and prints, per step, how long it took, whether it failed, and the longest
stretch in which it printed nothing. That last number is the one that
makes a tool feel stuck, and the one a progress bar exists to fix.

    python tools/journey.py                       the default walk
    python tools/journey.py "find" "pull org/repo:Q4_K_M --run"
    python tools/journey.py --keep                leave the home behind

The default walk uses the real Hugging Face Hub and the llama-server on
PATH, downloads a 0.4 GB model and starts it, so it needs a network and a
llama.cpp build; it is not part of tests/run.py (tests/test_journey.py is
the offline one). Nothing outside a temp directory is written, except that
a model already in the Hugging Face cache is reused from there.

Output is piped, not a terminal, so what is measured is what a script or
a CI log would see; a terminal gets redrawn bars on top.
"""
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ["list", "find", "pull unsloth/Qwen3-0.6B-GGUF", "run qwen3-0-6b",
           "stop"]


def step(argv, env, echo):
    """(seconds, exit code, longest silence in seconds, output)."""
    t0 = time.monotonic()
    proc = subprocess.Popen([sys.executable, str(ROOT / "mdl.py"), *argv],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, env=env)
    chunks, last, gap = [], [t0], [0.0]

    def read():
        while True:
            b = proc.stdout.read1(4096)
            if not b:
                return
            now = time.monotonic()
            gap[0] = max(gap[0], now - last[0])
            last[0] = now
            chunks.append(b)
            if echo:
                sys.stdout.write(b.decode("utf-8", "replace"))
                sys.stdout.flush()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    code = proc.wait()
    reader.join()
    end = time.monotonic()
    gap[0] = max(gap[0], end - last[0])
    return end - t0, code, gap[0], b"".join(chunks).decode("utf-8", "replace")


def main(args):
    keep = "--keep" in args
    quiet = "--quiet" in args
    walk = [a for a in args if a not in ("--keep", "--quiet")] or DEFAULT
    home = Path(tempfile.mkdtemp(prefix="mdl-journey-"))
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               XDG_CONFIG_HOME=str(home / "config"),
               XDG_STATE_HOME=str(home / "state"),
               XDG_CACHE_HOME=str(home / "cache"),
               MDL_FIT_HOME=str(home / "fit"),
               MDL_MODELS=str(home / "models"),
               MDL_NO_UPDATE_CHECK="1")
    env.pop("MDL_LLAMA_SERVER", None)
    rows = []
    try:
        for line in walk:
            argv = shlex.split(line)
            if not quiet:
                print("\n$ mdl %s" % line, flush=True)
            secs, code, gap, out = step(argv, env, not quiet)
            # anywhere in a line: an error printed over a progress line
            # without clearing it is still an error
            errors = sum(1 for x in out.replace(chr(13), chr(10)).splitlines()
                         if "mdl: " in x)
            rows.append((line, secs, code, gap, errors))
    finally:
        # a server the walk started is not left running
        subprocess.run([sys.executable, str(ROOT / "mdl.py"), "stop", "--all"],
                       env=env, capture_output=True)
        if keep:
            print("\nhome kept at %s" % home)
        else:
            shutil.rmtree(home, ignore_errors=True)
    width = max(len(r[0]) for r in rows)
    print("\n%-*s  %8s  %4s  %8s  %s" % (width, "step", "time", "exit",
                                         "silent", "errors"))
    for line, secs, code, gap, errors in rows:
        print("%-*s  %7.1fs  %4d  %7.1fs  %d" % (width, line, secs, code, gap,
                                                errors))
    total = sum(r[1] for r in rows)
    print("%-*s  %7.1fs  %4s  %7.1fs  %d" % (
        width, "total (%d commands)" % len(rows), total, "",
        max(r[3] for r in rows), sum(r[4] for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
