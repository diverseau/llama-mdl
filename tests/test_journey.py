"""A new user's first session, offline: from an empty home to a model
answering, through the real `mdl` command, against a fake Hub and the
fake llama-server on PATH.

tools/journey.py is the same walk against the real Hub and a real
llama.cpp, timed; this is the part that can be a gate. It holds the
promises the onboarding work made: no step of the first session is an
error, a model is one command from answering, and nothing has to be
written into models.toml by hand first.
"""
import hashlib
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                                 # noqa: E402

t = support.Tally("test_journey")
check = t.check

HOME = Path(tempfile.mkdtemp(prefix="mdl-journey-"))
REPO = "someorg/Tiny-Model-GGUF"
SHA = "89abcdef0123456789abcdef0123456789abcdef"
model = support.llama(HOME / "src.gguf", n_layer=2, embd=64, ff=128,
                      vocab=64)
FILES = {"Tiny-Model-Q8_0.gguf": model.read_bytes(),
         "Tiny-Model-Q4_K_M.gguf": model.read_bytes()[:-7] + b"quant4k",
         "README.md": b"# tiny"}


class Hub(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", headers=()):
        self.send_response(code)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/models/%s/revision/main" % REPO:
            return self._send(200, json.dumps({"sha": SHA}).encode())
        if self.path.startswith("/api/models/%s/tree/%s" % (REPO, SHA)):
            return self._send(200, json.dumps([
                {"type": "file", "path": k, "size": len(v), "oid": "git%d" % i,
                 "lfs": {"size": len(v),
                         "oid": hashlib.sha256(v).hexdigest()}}
                for i, (k, v) in enumerate(FILES.items())]).encode())
        prefix = "/%s/resolve/%s/" % (REPO, SHA)
        data = FILES.get(self.path[len(prefix):]) if self.path.startswith(
            prefix) else None
        if data is None:
            return self._send(404)
        rng = self.headers.get("Range")
        if not rng:
            return self._send(200, data)
        lo, _, hi = rng.split("=")[1].partition("-")
        lo, hi = int(lo), min(int(hi) if hi else len(data) - 1, len(data) - 1)
        self._send(206, data[lo:hi + 1], [(
            "Content-Range", "bytes %d-%d/%d" % (lo, hi, len(data)))])


hub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hub)
threading.Thread(target=hub.serve_forever, daemon=True).start()

# llama-server on PATH, as a package manager would leave it
bin_dir = HOME / "bin"
bin_dir.mkdir()
launcher = support._launcher(bin_dir)
launcher.rename(bin_dir / ("llama-server" + launcher.suffix))

ENV = dict(os.environ, PYTHONIOENCODING="utf-8",
           PATH=str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
           XDG_CONFIG_HOME=str(HOME / "config"),
           XDG_STATE_HOME=str(HOME / "state"),
           XDG_CACHE_HOME=str(HOME / "cache"),
           MDL_FIT_HOME=str(HOME / "fit"), MDL_MODELS=str(HOME / "models"),
           MDL_HF_ENDPOINT="http://127.0.0.1:%d" % hub.server_port,
           HF_HUB_CACHE=str(HOME / "hf-cache"), HF_HOME=str(HOME / "hf"),
           MDL_NO_UPDATE_CHECK="1", MDL_RECORD="off", MDL_FAKE_LOAD_S="0.3")
for var in ("MDL_LLAMA_SERVER", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    ENV.pop(var, None)
CONFIG = HOME / "config" / "mdl" / "models.toml"


def mdl(*args):
    p = subprocess.run([sys.executable, str(support.ROOT / "mdl.py"), *args],
                       capture_output=True, env=ENV, encoding="utf-8",
                       errors="replace", stdin=subprocess.DEVNULL,
                       timeout=180)
    errors = [x for x in (p.stdout + p.stderr).replace("\r", "\n").splitlines()
              if "mdl: " in x]
    return p.returncode, p.stdout, p.stderr, errors


try:
    code, out, err, errors = mdl("list")
    check("list, before anything: not an error, and says how to get a model",
          (code, errors, "mdl pull" in err), (0, [], True))
    code, out, err, errors = mdl("doctor")
    check("doctor, before anything: no failure; the missing config is a "
          "warning", (code, errors, "no config at" in out), (0, [], True))

    code, out, err, errors = mdl("pull", REPO, "--run")
    check("one command, from nothing to a model answering",
          (code, errors, out.strip().splitlines()[-1:][0].startswith(
              "ready: tiny-model on http://127.0.0.1:")), (0, [], True))
    check("it said which quant it picked", "picked Q" in out, True)
    check("and wrote the config itself, with no placeholder table to trip on",
          (CONFIG.is_file(), "\n[example]" in CONFIG.read_text(
              encoding="utf-8")), (True, False))
    code, out, err, errors = mdl("ps")
    check("ps shows it", (code, out.startswith("tiny-model")), (0, True))
    code, out, err, errors = mdl("check")
    check("check has nothing to complain about", (code, errors), (0, []))
    code, out, err, errors = mdl("stop")
    check("and stop stops it", (code, errors), (0, []))
finally:
    mdl("stop", "--all")
    hub.shutdown()
    shutil.rmtree(HOME, ignore_errors=True)

sys.exit(t.done())
