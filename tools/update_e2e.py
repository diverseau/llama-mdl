#!/usr/bin/env python3
"""End to end: `mdl update` through a real installer, offline.

Builds a wheel of this tree and one of the same tree stamped 99.0.0,
installs the first with each installer on this machine (pip always, uv
and pipx when they are on PATH) into throwaway places, points mdl at a
local index that lists both, and runs the real `mdl update`. Passes when
the installed `mdl --version` says 99.0.0 afterwards.

    python tools/update_e2e.py

Not part of tests/run.py: it builds wheels and creates environments, which
takes a minute, and uv and pipx may not be installed. Run it on Windows and
on Linux before a release that touches mdl_update.py. Nothing outside a
temp directory is written. mdl and the installers see only a local index;
building the two wheels lets pip fetch setuptools, unless it is cached,
and textual and its dependencies are downloaded once for the index.

On Linux, with Docker:

    docker run --rm -v "$PWD:/repo:ro" python:3.12-slim \\
        sh -c 'cp -r /repo /w && cd /w && python3 tools/update_e2e.py'
"""
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mdl  # noqa: E402

OLD, NEW = mdl.VERSION, "99.0.0"
FILES = ["mdl.py", "mdl_ui.py", "mdl_update.py", "pyproject.toml",
         "README.md", "LICENSE"]
BIN = "Scripts" if os.name == "nt" else "bin"
EXE = ".exe" if os.name == "nt" else ""


def sh(argv, env=None, check=True):
    r = subprocess.run([str(a) for a in argv], env=env, capture_output=True,
                       text=True, errors="replace")
    if check and r.returncode:
        sys.exit("FAILED: %s\n%s%s" % (" ".join(map(str, argv)), r.stdout,
                                       r.stderr))
    return r


def wheel(tmp, version, out):
    src = tmp / ("src-" + version)
    src.mkdir()
    for f in FILES:
        shutil.copy2(ROOT / f, src / f)
    for package in ("mdl_fit", "mdl_web"):
        shutil.copytree(ROOT / package, src / package,
                        ignore=shutil.ignore_patterns("__pycache__"))
    text = (src / "mdl.py").read_text(encoding="utf-8")
    (src / "mdl.py").write_text(text.replace('VERSION = "%s"' % OLD,
                                             'VERSION = "%s"' % version),
                                encoding="utf-8")
    sh([sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "-w", out,
        src])
    return next(out.glob("llama_mdl-%s-*.whl" % version))


def index(port_holder):
    body = json.dumps({"releases": {
        v: [{"filename": "llama_mdl-%s-py3-none-any.whl" % v,
             "requires_python": ">=3.11", "yanked": False}]
        for v in (OLD, NEW)}}).encode()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def run_update(label, mdl_exe, env):
    before = sh([mdl_exe, "--version"], env).stdout.strip()
    r = sh([mdl_exe, "update"], env, check=False)
    after = sh([mdl_exe, "--version"], env).stdout.strip()
    again = sh([mdl_exe, "update"], env, check=False)
    left = sorted(p.name for p in Path(mdl_exe).parent.glob("mdl*.old-*"))
    ok = (before == "mdl " + OLD and r.returncode == 0
          and after == "mdl " + NEW and again.returncode == 0
          and "newest release" in again.stdout and not left)
    print("%s %s: %s -> %s" % ("PASS" if ok else "FAIL", label, before, after))
    if not ok:
        print(r.stdout + r.stderr)
        print("second run:", again.stdout + again.stderr)
        print("left behind:", left)
    return ok


def main():
    tmp = Path(tempfile.mkdtemp(prefix="mdl-update-e2e-"))
    try:
        built = tmp / "built"
        built.mkdir()
        old_whl, new_whl = wheel(tmp, OLD, built), wheel(tmp, NEW, built)
        # textual and what it needs, fetched once: the installers below see
        # only a local directory, and mdl depends on textual since 0.13
        deps = tmp / "deps"
        sh([sys.executable, "-m", "pip", "download", "-q", "-d", deps,
            "textual>=3,<9"])
        server = index(None)
        base = dict(os.environ,
                    MDL_PYPI_URL="http://127.0.0.1:%d/" % server.server_address[1],
                    XDG_CACHE_HOME=str(tmp / "cache"),
                    XDG_CONFIG_HOME=str(tmp / "config"),
                    XDG_STATE_HOME=str(tmp / "state"),
                    MDL_FIT_HOME=str(tmp / "fit"),
                    UV_NO_CACHE="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
        base.pop("MDL_NO_UPDATE_CHECK", None)
        results = []

        def links(name):
            """A find-links directory holding only the old wheel until the
            install is done - an install that saw 99.0.0 would take it."""
            d = tmp / ("links-" + name)
            shutil.copytree(deps, d)
            shutil.copy2(old_whl, d)
            return d

        # pip, in a venv
        d = links("pip")
        venv = tmp / "venv"
        sh([sys.executable, "-m", "venv", venv])
        env = dict(base, PIP_NO_INDEX="1", PIP_FIND_LINKS=str(d))
        sh([venv / BIN / ("python" + EXE), "-m", "pip", "install", "-q",
            "llama-mdl"], env)
        shutil.copy2(new_whl, d)
        results.append(run_update("pip", venv / BIN / ("mdl" + EXE), env))

        if shutil.which("uv"):
            d = links("uv")
            env = dict(base, UV_TOOL_DIR=str(tmp / "uv-tools"),
                       UV_TOOL_BIN_DIR=str(tmp / "uv-bin"))
            sh(["uv", "tool", "install", "-q", "--no-index", "--find-links", d,
                "llama-mdl"], env)
            shutil.copy2(new_whl, d)
            results.append(run_update("uv tool", tmp / "uv-bin" /
                                      ("mdl" + EXE), env))
        else:
            print("SKIP uv tool: uv is not on PATH")

        if shutil.which("pipx"):
            d = links("pipx")
            env = dict(base, PIPX_HOME=str(tmp / "pipx-home"),
                       PIPX_BIN_DIR=str(tmp / "pipx-bin"))
            sh(["pipx", "install", "llama-mdl",
                "--pip-args=--no-index --find-links %s" % d], env)
            shutil.copy2(new_whl, d)
            results.append(run_update("pipx", tmp / "pipx-bin" /
                                      ("mdl" + EXE), env))
        else:
            print("SKIP pipx: pipx is not on PATH")
        server.shutdown()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("%d of %d routes passed" % (sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
