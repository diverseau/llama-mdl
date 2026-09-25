"""mdl pull and the page's downloads: against a fake Hub, the files come
pinned, checked, resumed, adopted from the cache, and become a preset;
the page shows a pull as a card and offers find's picks."""
import hashlib
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                                 # noqa: E402
from support import mdl, run, sandbox, teardown               # noqa: E402

from mdl_fit import pull                                       # noqa: E402

t = support.Tally("test_pull")
check = t.check

TMP = Path(tempfile.mkdtemp(prefix="mdl-pull-"))
REPO = "someorg/Tiny-Model-GGUF"
SHA = "0123456789abcdef0123456789abcdef01234567"
model = support.llama(TMP / "src.gguf", n_layer=2, embd=64, ff=128, vocab=64)
FILES = {"Tiny-Model-Q8_0.gguf": model.read_bytes(),
         "Tiny-Model-Q4_K_M.gguf": model.read_bytes()[:-7] + b"quant4k",
         "mmproj-Q8_0.gguf": b"GGUF-projector-q8",
         "mmproj-F16.gguf": b"GGUF-projector-f16",
         "README.md": b"# tiny"}
MODE = {}                  # file -> "half" (drop mid-way) or "bad" (wrong bytes)
HITS = []


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
            listing = [{"type": "file", "path": k, "size": len(v),
                        "oid": "git%d" % i,
                        "lfs": {"size": len(v),
                                "oid": hashlib.sha256(v).hexdigest()}}
                       for i, (k, v) in enumerate(FILES.items())]
            listing += MODE.get("extra", [])
            return self._send(200, json.dumps(listing).encode())
        prefix = "/%s/resolve/%s/" % (REPO, SHA)
        if not self.path.startswith(prefix):
            return self._send(404)
        name = self.path[len(prefix):]
        data = FILES.get(name)
        if data is None:
            return self._send(404)
        HITS.append((name, self.headers.get("Range")))
        mode = MODE.get(name)
        if mode == "bad":
            data = b"X" * len(data)
        lo = 0
        rng = self.headers.get("Range")
        if rng:
            lo = int(rng.split("=")[1].split("-")[0])
        body = data[lo:]
        if mode == "half":
            MODE.pop(name)             # once: the next try gets the rest
            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[:len(body) // 2])
            self.wfile.flush()
            self.connection.close()
            return
        headers = [("Content-Range", "bytes %d-%d/%d" % (lo, len(data) - 1,
                                                         len(data)))] if rng else []
        self._send(206 if rng else 200, body, headers)


hub_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hub)
threading.Thread(target=hub_server.serve_forever, daemon=True).start()
os.environ["MDL_HF_ENDPOINT"] = "http://127.0.0.1:%d" % hub_server.server_port
os.environ["MDL_MODELS"] = str(TMP / "models")
os.environ["HF_HUB_CACHE"] = str(TMP / "hf-cache")
for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    os.environ.pop(var, None)
os.environ["HF_HOME"] = str(TMP / "hf-home")          # no token file either

# ------------------------------------------------------------ the plan --
check("a preset's name from the repo",
      [pull.default_name(REPO), pull.default_name("a/Qwen3.6-35B-A3B-GGUF")],
      ["tiny-model", "qwen3-6-35b-a3b"])
try:
    pull.plan(REPO)
    check("two quants and none named: says which there are", "planned", "error")
except pull.PullError as e:
    check("two quants and none named: says which there are",
          ("Q4_K_M" in str(e), "Q8_0" in str(e), "mdl pull %s:" % REPO in str(e)),
          (True, True, True))
sha, key, need = pull.plan(REPO, "Q8_0")
check("a quant named: pinned to the commit, the model then its projector, "
      "at full precision",
      (sha, key, [f["path"] for f in need]),
      (SHA, "Tiny-Model-Q8_0.gguf", ["Tiny-Model-Q8_0.gguf", "mmproj-F16.gguf"]))
MODE["extra"] = [{"type": "file", "path": "../evil-Q2_K.gguf", "size": 1,
                  "lfs": {"size": 1, "oid": "x"}}]
try:
    pull.plan(REPO, "Q2_K")
    check("a path out of the folder is refused", "planned", "error")
except pull.PullError as e:
    check("a path out of the folder is refused", "unsafe path" in str(e), True)
MODE.pop("extra")

# ------------------------------------------------------------ the pull --
root, port = sandbox()
folder = TMP / "models" / "someorg--Tiny-Model-GGUF"
MODE["Tiny-Model-Q8_0.gguf"] = "half"
out, err, code = run(pull.main, ["%s:Q8_0" % REPO])
check("a dropped connection fails, saying to run again",
      (code, "run again to resume" in err), (1, True))
part = folder / "Tiny-Model-Q8_0.gguf.part"
check("and keeps what it had", 0 < part.stat().st_size < len(
    FILES["Tiny-Model-Q8_0.gguf"]), True)
check("a failed pull leaves its reason for the page",
      pull.read_all()["tiny-model"]["state"], "error")
HITS.clear()
out, err, code = run(pull.main, ["%s:Q8_0" % REPO])
check("run again: it resumes where it stopped",
      (code, [h for h in HITS if h[0] == "Tiny-Model-Q8_0.gguf"][0][1]),
      (0, "bytes=%d-" % (len(FILES["Tiny-Model-Q8_0.gguf"]) // 2)))
check("the files, whole, with where they came from",
      ((folder / "Tiny-Model-Q8_0.gguf").read_bytes() == FILES["Tiny-Model-Q8_0.gguf"],
       (folder / "mmproj-F16.gguf").is_file(), part.exists(),
       json.loads((folder / ".mdl-pull.json").read_text())["revision"]),
      (True, True, False, SHA))
models, _ = mdl.load_config()
cfg = models.get("tiny-model", {})
check("a preset for it: the model, its projector, --metrics",
      (Path(cfg.get("model", "")).name, Path(cfg.get("mmproj", "")).name,
       "--metrics" in cfg.get("args", [])),
      ("Tiny-Model-Q8_0.gguf", "mmproj-F16.gguf", True))
check("on a port no other preset has", cfg.get("port") != port, True)
check("and it says how to run it", "mdl run tiny-model" in out, True)
check("the page's status is gone once done", pull.read_all(), {})

HITS.clear()
out, err, code = run(pull.main, ["%s:Q8_0" % REPO])
check("pull it again: nothing downloaded, nothing added twice",
      (code, [h for h in HITS], sorted(mdl.load_config()[0])),
      (0, [], ["demo", "tiny-model"]))

MODE["Tiny-Model-Q4_K_M.gguf"] = "bad"
out, err, code = run(pull.main, ["%s:Q4_K_M" % REPO, "--name", "tiny-q4"])
check("a file that does not match the Hub's is not kept",
      (code, "does not match" in err,
       (folder / "Tiny-Model-Q4_K_M.gguf").exists(),
       (folder / "Tiny-Model-Q4_K_M.gguf.part").exists()),
      (1, True, False, False))
MODE.pop("Tiny-Model-Q4_K_M.gguf")
_, err, code = run(pull.main, ["%s:Q4_K_M" % REPO, "--name", "tiny-model"])
check("a name another preset has is refused",
      (code, "already in" in err), (1, True))
pull.stop("tiny-model")
teardown(root)

# -- a verified copy in the Hugging Face cache is used where it is ---------------
root, port = sandbox()
for f in ("Tiny-Model-Q8_0.gguf", "mmproj-F16.gguf"):
    p = pull.cached(REPO, SHA, f)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(FILES[f])
HITS.clear()
out, err, code = run(pull.main, ["%s:Q8_0" % REPO, "--name", "cached"])
cfg = mdl.load_config()[0].get("cached", {})
check("the cache's copy: nothing downloaded, the preset points at it",
      (code, HITS, Path(cfg.get("model", "")) == pull.cached(REPO, SHA,
                                                              "Tiny-Model-Q8_0.gguf")),
      (0, [], True))
from mdl_web import snapshot                                   # noqa: E402
check("and the page knows its weights from the cache's path",
      snapshot.weights(cfg.get("model")), [{"repository": REPO, "revision": SHA}])

# -- what the page shows of a pull -------------------------------------------------
st = pull.Status("going", REPO, ["0"], quiet=True, spec="hf:%s:Q8_0" % REPO)
st("download", "3 of 14 GB", 21)
s = snapshot.build()
d = next((d for d in s["deployments"] if d["id"] == "going"), {})
check("a pull is a card: downloading, how far",
      (d.get("state"), d.get("detail"), d.get("percent")),
      ("download", "3 of 14 GB", 21))
st.base["pid"] = 2 ** 22 + 12345              # a process that is not there
st("download", "3 of 14 GB", 21)
d = next((d for d in snapshot.build()["deployments"] if d["id"] == "going"), {})
check("one whose process went is crashed, to run again",
      (d.get("state"), "run again" in d.get("error", "")), ("error", True))
check("dismissed, it is gone", (pull.stop("going"), pull.read_all()), (True, {}))
check("nothing to stop", pull.stop("going"), False)

# -- find's picks as recipes for a card's Config -----------------------------------
snapshot.picks_path().parent.mkdir(parents=True, exist_ok=True)
snapshot.picks_path().write_text(json.dumps({"rows": [
    {"spec": "hf:other/Qwen3-8B-GGUF:Q4_K_M", "model": "Qwen/Qwen3-8B",
     "repo": "other/Qwen3-8B-GGUF", "file": "Qwen3-8B-Q4_K_M.gguf",
     "quant": "Q4_K_M", "size": 5 * 2 ** 30, "ctx": 131072},
    {"spec": "hf:other/Qwen3-8B-GGUF:Q8_0", "model": "Qwen/Qwen3-8B",
     "repo": "other/Qwen3-8B-GGUF", "file": "Qwen3-8B-Q8_0.gguf",
     "quant": "Q8_0", "size": 9 * 2 ** 30, "ctx": 131072},
    {"spec": "hf:%s:Q8_0" % REPO, "model": "someorg/Tiny",
     "repo": REPO, "file": "Tiny-Model-Q8_0.gguf", "quant": "Q8_0",
     "size": 1, "ctx": 4096},
    {"spec": "C:/models/local.gguf", "model": "local", "repo": None,
     "file": None, "quant": "Q4_K_M", "size": 1, "ctx": 4096}]}))
ms = snapshot.build()["kinds"][0]["models"]
check("Config offers find's picks after your own: one a repo, none pulled",
      [(m["id"], m.get("pull", False)) for m in ms],
      [("demo", False), ("cached", False),
       ("hf:other/Qwen3-8B-GGUF:Qwen3-8B-Q4_K_M.gguf", True)])
pick = ms[-1]
check("a pick as the panel's recipe",
      (pick["name"], pick["family"], pick["format"], pick["ctx"],
       pick["sizeGb"], pick["weights"]),
      ("Qwen3-8B", "qwen", "GGUF · Q4_K_M", 131072, 5.0,
       [{"repository": "other/Qwen3-8B-GGUF", "revision": "main"}]))

# -- Run on a pick, from the page: download, add, start --------------------------
from mdl_web import server                                     # noqa: E402

hub = server.Hub()
code, body = server.act(hub, {"verb": "run", "name": "hf:%s:Q4_K_M" % REPO,
                              "keys": "0"})
check("run a pick: it starts downloading", (code, body), (200, {"ok": True}))
up = None
end = time.monotonic() + 90
while time.monotonic() < end and not up:
    up = mdl.read_states().get("tiny-model")
    time.sleep(0.5)
check("and it runs, as its preset", bool(up), True)
check("run it again while it runs is refused",
      server.act(hub, {"verb": "run", "name": "hf:%s:Q4_K_M" % REPO})[0], 409)
for name, st in mdl.read_states().items():
    mdl.stop_one(name, st)
teardown(root)

# ------------------------------------------------------------------ cli --
_, err, code = run(pull.main, ["--bogus"])
check("a usage line for what it does not know",
      (code, "usage: mdl pull" in err), (1, True))
_, err, code = run(pull.main, ["not-a-repo"])
check("a repo is org/name", (code, "expected hf:org/repo" in err), (1, True))

hub_server.shutdown()
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(t.done())
