#!/usr/bin/env python3
"""Run mdl's test suites.

    python tests/run.py           fast suites, no real models needed
    python tests/run.py --live    also drives a real model through the UI

integration_posix.py is not run here - it needs Linux. On any machine with
Docker:

    docker run --rm -v "$PWD:/repo:ro" python:3.12-slim \
        sh -c 'cp -r /repo /w && cd /w && python3 tests/integration_posix.py'
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAST = ["test_cli.py", "test_commands.py", "test_doctor.py",
        "test_config.py",
        "test_atomic.py", "test_multi.py",
        "test_ui.py",
        "test_chat.py",
        "test_fx.py",
        "test_fit.py",
        "test_fit_verify.py",
        "test_eval.py",
        "test_eval_main.py",
        "test_catalog.py",
        "test_catalog_resume.py",
        "test_find.py",
        "test_pull.py",
        "test_lab.py",
        "test_manifest.py",
        "test_web.py", "test_update.py",
        "test_journey.py", "test_progress.py", "test_setup.py"]
LIVE = ["test_live.py"]


def main():
    suites = FAST + (LIVE if "--live" in sys.argv else [])
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    failed = []
    for suite in suites:
        print("\n" + "=" * 64)
        print("  " + suite)
        print("=" * 64)
        if subprocess.run([sys.executable, str(HERE / suite)], env=env).returncode:
            failed.append(suite)
    print("\n" + "=" * 64)
    if failed:
        print("  FAILED: " + ", ".join(failed))
        return 1
    print("  all %d suites passed" % len(suites))
    return 0


if __name__ == "__main__":
    sys.exit(main())
