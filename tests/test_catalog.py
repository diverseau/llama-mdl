"""mdl catalog: the crawl, the noise gate, lineage, incremental builds,
the snapshot pull and the queries, against a stand-in Hugging Face."""
import http.server
import io
import json
import os
import sys
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                   # noqa: E402

from mdl_fit import catalog, hw, remote          # noqa: E402

t = support.Tally("test_catalog")
check = t.check
TMP = Path(tempfile.mkdtemp(prefix="mdl-catalog-test-"))
os.environ["MDL_FIT_HOME"] = str(TMP / "home")
os.environ.pop("HF_TOKEN", None)
catalog.SLEEP = lambda s: sleeps.append(s)
sleeps = []


def model(mid, downloads=1000, likes=0, bases=(), gguf=False, card=None,
          evals=None, arch="qwen3", total=8_000_000_000):
    tags = ["text-generation"] + (["gguf"] if gguf else [])
    tags += ["base_model:%s:%s" % (rel, b) for b, rel in bases]
    return {"id": mid, "downloads": downloads, "likes": likes, "tags": tags,
            "pipeline_tag": "text-generation", "cardData": card or {},
            "lastModified": "2026-09-01T00:00:00.000Z",
            "createdAt": "2026-08-01T00:00:00.000Z",
            "gguf": {"architecture": arch, "total": total,
                     "context_length": 40960} if gguf else None,
            "safetensors": None if gguf else {"total": total},
            "evalResults": evals or []}


HUB = {
    # the tracked org: a base, its official post-train, and the org's own
    # GGUF repo (a quant, which must not become a node)
    "Fam/Base-8B": model("Fam/Base-8B", 5000),
    "Fam/Base-8B-Instruct": model(
        "Fam/Base-8B-Instruct", 90000, bases=[("Fam/Base-8B", "finetune")],
        evals=[{"verified": True, "data": {
            "dataset": {"id": "org/coding-bench", "task_id": "pass1"},
            "value": 61.5, "date": "2026-07-01",
            "source": {"name": "leaderboard"}}}]),
    "Fam/Base-8B-Instruct-GGUF": model(
        "Fam/Base-8B-Instruct-GGUF", 40000, gguf=True,
        bases=[("Fam/Base-8B-Instruct", "quantized")]),
    # fine-tunes: one popular, one junk, one popular with only a
    # GGUF of its own, and a merge of a fine-tune
    # a card may name several licenses, as a list
    "coder/Coder-8B": model("coder/Coder-8B", 3000,
                            bases=[("Fam/Base-8B-Instruct", "finetune")],
                            card={"datasets": ["openai/gsm8k"],
                                  "license": ["apache-2.0", "other"]}),
    "junk/test-upload": model("junk/test-upload", 3,
                              bases=[("Fam/Base-8B-Instruct", "finetune")]),
    "solo/Solo-8B-GGUF": model("solo/Solo-8B-GGUF", 900, gguf=True,
                               bases=[("Fam/Base-8B-Instruct", "finetune")]),
    "mix/Merge-8B": model("mix/Merge-8B", 50, likes=25,
                          bases=[("coder/Coder-8B", "merge")]),
    "quanter/Coder-8B-GGUF": model("quanter/Coder-8B-GGUF", 2000, gguf=True,
                                   bases=[("coder/Coder-8B", "quantized")]),
    "quanter/Merge-8B-GGUF": model("quanter/Merge-8B-GGUF", 100, gguf=True,
                                   bases=[("mix/Merge-8B", "quantized")]),
    "nogguf/Popular-8B": model("nogguf/Popular-8B", 99999,
                               bases=[("Fam/Base-8B-Instruct", "finetune")]),
}
FILES = {
    "Fam/Base-8B-Instruct-GGUF": ["Base-8B-Instruct-Q4_K_M.gguf",
                                  "Base-8B-Instruct-Q8_0.gguf",
                                  "Base-8B-Instruct-UD-Q3_K_XL-00001-of-00002"
                                  ".gguf",
                                  "Base-8B-Instruct-UD-Q3_K_XL-00002-of-00002"
                                  ".gguf", "mmproj-F16.gguf", "README.md"],
    "quanter/Coder-8B-GGUF": ["Coder-8B.IQ3_XXS.gguf", "Coder-8B.Q6_K.gguf"],
    "quanter/Merge-8B-GGUF": ["Merge-8B-Q4_K_M.gguf"],
    "solo/Solo-8B-GGUF": ["Solo-8B-q4_k_m.gguf"],
}
hits = []
SNAP = {"etag": '"v1"', "body": b""}


class Hub(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, headers=()):
        body = json.dumps(obj).encode()
        self.send_response(200)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        hits.append(self.path)
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        if url.path.startswith("/datasets/"):
            if self.headers.get("If-None-Match") == SNAP["etag"]:
                self.send_response(304)
                self.end_headers()
                return
            if not SNAP["body"]:
                self.send_response(401)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("ETag", SNAP["etag"])
            self.send_header("Content-Length", str(len(SNAP["body"])))
            self.end_headers()
            if SNAP.get("cut"):             # a connection that drops
                self.wfile.write(SNAP["body"][:len(SNAP["body"]) // 2])
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(SNAP["body"])
            return
        if url.path.endswith("/tree/main"):
            repo = url.path[len("/api/models/"):-len("/tree/main")]
            self._json([{"type": "file", "path": p, "size": 1000 + i,
                         "oid": "o%d" % i, "lfs": {"size": 4_000_000_000 + i,
                                                   "oid": "sha%d" % i}}
                        for i, p in enumerate(FILES.get(repo, []))])
            return
        if url.path.startswith("/api/models/"):
            m = HUB.get(url.path[len("/api/models/"):])
            if m is None:
                self.send_response(404)
                self.end_headers()
                return
            self._json(m)
            return
        if "expand" not in q:
            self.send_response(400)
            self.end_headers()
            return
        if "author" in q:
            got = [m for m in HUB.values()
                   if m["id"].split("/")[0] == q["author"][0]]
        else:
            filt = q.get("filter", [])        # the hub ignores other= here
            want = [f for f in filt if f.startswith("base_model:")]
            got = [m for m in HUB.values() if want and want[0] in m["tags"]
                   and ("gguf" not in filt or "gguf" in m["tags"])]
        got.sort(key=lambda m: -m["downloads"])
        limit = min(2, int(q.get("limit", ["100"])[0]))   # small pages
        start = int(q.get("cursor", ["0"])[0])
        page = got[start:start + limit]
        headers = []
        if start + limit < len(got):             # paginate like the hub
            nq = dict(q, cursor=[str(start + limit)])
            headers.append(("Link", '<http://%s%s?%s>; rel="next"' % (
                self.headers["Host"], url.path,
                urllib.parse.urlencode(nq, doseq=True))))
        headers.append(("RateLimit", '"api";r=400;t=100'))
        self._json(page, headers)


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hub)
threading.Thread(target=srv.serve_forever, daemon=True).start()
os.environ["MDL_HF_ENDPOINT"] = "http://127.0.0.1:%d" % srv.server_port

# =============================================================== util ===

check("quant labels come out of file names",
      [catalog.quant_of(n) for n in (
          "Qwen3-8B-UD-Q4_K_XL.gguf", "LFM2.5-8B-A1B-Q8_0.gguf",
          "Qwen3.8-27B-GSQ-RCO-IQ2_XS-mtp.gguf", "model.q4_k_m.gguf",
          "gpt-oss-20b-MXFP4.gguf", "Model-BF16.gguf", "Qwen3-8B.fp16.gguf",
          "types/NVFP4.gguf", "weird.gguf",
          "Bonsai-2-27B-PQ2_0-CRACK.gguf",
          "Laguna-XS-2.1-APEX-I-Balanced.gguf", "M-APEX-Quality-MTP.gguf")],
      ["UD-Q4_K_XL", "Q8_0", "IQ2_XS", "Q4_K_M", "MXFP4", "BF16", "F16",
       "NVFP4", "?", "PQ2_0", "APEX-I-Balanced", "APEX-Quality"])
# Every name here was in a published snapshot, listed as a quant of a 27B.
AUX = ["MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "mtp-Qwen3.8-27B-BF16.gguf",
       "mtp-RVN.gguf", "Qwen3.8-27B-Uncensored-draft-Q8_0.gguf",
       "Qwen3.8-27B-DFlash2-Q4_K_M.gguf",
       "dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf",
       "Qwen3.8-27B-Fable-5-Coding-Distilled.mmproj-Q8_0.gguf",
       "gemma-4-E2B-it-mmproj.gguf", "vision-projector.gguf",
       "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf",
       "doctors/TAARDIS-27B-Doctors-V3.lora.gguf", "mmproj-F16.gguf",
       "imatrix.gguf", "imatrix_unsloth.gguf",
       "dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf"]
MODELS = ["RVN-Q4_K_M-mtp.gguf", "RVN-IQ1_S-multilingual-mtp.gguf",
          "Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf",
          "Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S-mtp.gguf",
          "RVN-Q3_K_M-vision.gguf", "Bonsai-27B-Q1_0.gguf",
          "Llama-3-8B-LoRA-merged-Q4_K_M.gguf", "UD-TQ2_0/Kimi-K3-UD-TQ2_0.gguf",
          "Qwen3.8-27B-Q4_0.gguf", "Model-imatrix-Q4_K_M.gguf"]
check("drafts, MTP heads, projectors and adapters are not quants",
      [n for n in AUX + MODELS if remote.auxiliary(n)], AUX)
check("files sharing a quant are told apart, the rest keep it",
      catalog.variant_labels([
          {"quant": q, "file": f} for q, f in (
              ("Q4_K_M", "RVN-Q4_K_M.gguf"),
              ("Q4_K_M", "RVN-Q4_K_M-mtp.gguf"),
              ("Q8_0", "RVN-Q8_0.gguf"),
              ("IQ3_S", "RVN-IQ3_S-multilingual.gguf"))]),
      {"RVN-Q4_K_M.gguf": "Q4_K_M", "RVN-Q4_K_M-mtp.gguf": "Q4_K_M-mtp",
       "RVN-Q8_0.gguf": "Q8_0", "RVN-IQ3_S-multilingual.gguf": "IQ3_S"})
check("a quant repo is not a model", [
    catalog.is_quant_repo(HUB["Fam/Base-8B-Instruct-GGUF"]),
    catalog.is_quant_repo(HUB["solo/Solo-8B-GGUF"])], [True, False])

# ============================================================== build ===

path = TMP / "cat.sqlite"
lines = []
meta = catalog.build(path, orgs=["Fam"], log=lines.append, per_org=10)
cat = catalog.Catalog(path)
ids = sorted(r["id"] for r in cat.nodes())
check("the gate lets in the org, popular and allowlisted fine-tunes and "
      "liked merges; quant repos, junk and GGUF-less leaves stay out", ids,
      ["Fam/Base-8B", "Fam/Base-8B-Instruct", "coder/Coder-8B",
       "mix/Merge-8B", "solo/Solo-8B-GGUF"])
inst = cat.node("Fam/Base-8B-Instruct")
check("an official post-train is linked to its base from its card",
      (inst["parent"], inst["relation"], inst["method"], inst["official"]),
      ("Fam/Base-8B", "finetune", "card", 1))
merge = cat.node("mix/Merge-8B")
check("a merge two generations down, with HF's own relation",
      (merge["parent"], merge["relation"], merge["method"], merge["depth"],
       merge["root"]), ("coder/Coder-8B", "merge", "hf-tree", 2,
                        "Fam/Base-8B"))
q = {r["file"]: r for r in cat.ggufs("Fam/Base-8B-Instruct")}
check("quants per node: split shards as one, projectors left out",
      sorted((r["quant"], json.loads(r["shards"]).__len__()) for r in
             q.values()), [("Q4_K_M", 1), ("Q8_0", 1), ("UD-Q3_K_XL", 2)])
check("sizes are the LFS sizes, shards added up",
      q["Base-8B-Instruct-UD-Q3_K_XL.gguf"]["size"], 2 * 4_000_000_000 + 5)
check("a fine-tune that only ships GGUF lists its own quants",
      [r["quant"] for r in cat.ggufs("solo/Solo-8B-GGUF")], ["Q4_K_M"])
ev = cat.evals("Fam/Base-8B-Instruct")
check("eval results come with their verified flag",
      [(e["benchmark"], e["value"], e["verified"]) for e in ev],
      [("org/coding-bench", 61.5, 1)])
check("a card's training sets are kept, for the contamination check",
      json.loads(cat.node("coder/Coder-8B")["datasets"]), ["openai/gsm8k"])
check("several licenses become text the --license filter can match",
      cat.node("coder/Coder-8B")["license"], "apache-2.0, other")
check("pagination is followed", any("cursor=" in h for h in hits), True)
check("the build says what it did", (meta["nodes"], meta["ggufs"] >= 6,
                                     meta["requests"] > 5), (5, True, True))
check("quiet rate-limit headers cost no waiting", sleeps, [])

tree = io.StringIO()
catalog.show_tree(cat, "Fam/Base-8B-Instruct", tree.write)
check("one query: every GGUF quant of every fine-tune of a base",
      [s in tree.getvalue() for s in ("coder/Coder-8B", "IQ3_XXS", "Q6_K",
                                      "mix/Merge-8B", "solo/Solo-8B-GGUF",
                                      "4 models")],
      [True] * 6)
check("lineage walks to the root", [r["id"] for r in cat.lineage(
    "mix/Merge-8B")], ["mix/Merge-8B", "coder/Coder-8B",
                       "Fam/Base-8B-Instruct", "Fam/Base-8B"])

# --------------------------------------------------------- incremental --

hits.clear()
catalog.build(TMP / "cat2.sqlite", orgs=["Fam"], prev=path, per_org=10)
check("an unchanged repo's file list is reused, not fetched again",
      [h for h in hits if "/tree/" in h], [])
cat2 = catalog.Catalog(TMP / "cat2.sqlite")
check("and the result is the same", len(cat2.ggufs("coder/Coder-8B")),
      len(cat.ggufs("coder/Coder-8B")))

# --------------------------------------------------------- base-only ---

catalog.build(TMP / "cat3.sqlite", orgs=[], bases=["coder/Coder-8B"])
cat3 = catalog.Catalog(TMP / "cat3.sqlite")
check("--base crawls just that model's tree",
      sorted(r["id"] for r in cat3.nodes()),
      ["coder/Coder-8B", "mix/Merge-8B"])

# ----------------------------------------------------------- 429 once --

state = {"n": 0}
orig = Hub.do_GET


def flaky(self):
    if "author=" in self.path and state["n"] == 0:
        state["n"] += 1
        self.send_response(429)
        self.send_header("Retry-After", "7")
        self.end_headers()
        return
    orig(self)


Hub.do_GET = flaky
catalog.build(TMP / "cat4.sqlite", orgs=["Fam"])
Hub.do_GET = orig
check("a 429 waits as long as the hub asks, then carries on",
      (sleeps, catalog.Catalog(TMP / "cat4.sqlite").node("Fam/Base-8B")
       is not None), ([7], True))

# ================================================================ pull ===

try:
    catalog.pull(TMP / "pulled.sqlite")
    err = None
except catalog.CatalogError as e:
    err = str(e)
check("nothing published yet: a clear message", "no published catalog"
      in (err or ""), True)
SNAP["body"] = path.read_bytes()
check("pull fetches the snapshot", catalog.pull(TMP / "pulled.sqlite"),
      "fresh")
check("and asks with the ETag next time",
      catalog.pull(TMP / "pulled.sqlite"), "unchanged")
check("what was pulled is a catalog",
      catalog.Catalog(TMP / "pulled.sqlite").meta()["nodes"], 5)
SNAP.update(etag='"v2"', body=b"not a database at all" * 100)
try:
    catalog.pull(TMP / "pulled.sqlite")
    err = None
except catalog.CatalogError as e:
    err = str(e)
check("a broken download never replaces a good snapshot",
      (err is not None, catalog.Catalog(TMP / "pulled.sqlite").meta()
       ["nodes"]), (True, 5))
SNAP.update(etag='"v3"', body=path.read_bytes(), cut=True)
try:
    catalog.pull(TMP / "pulled.sqlite")
    err = None
except catalog.CatalogError as e:
    err = str(e)
SNAP["cut"] = False
check("a download cut off halfway is an error, not a traceback, and leaves "
      "no .part behind", (err is not None, (TMP / "pulled.sqlite.part")
                          .exists(), catalog.Catalog(TMP / "pulled.sqlite")
                          .meta()["nodes"]), (True, False, 5))

# find's own fetch: once when there is none, then only once a week
os.environ["MDL_FIT_HOME"] = str(TMP / "home-ensure")
SNAP.update(etag='"v4"', body=path.read_bytes())
said = []
check("no snapshot yet: fetched, and it says so",
      (catalog.ensure(said.append), said, catalog.default_path().is_file()),
      ("fresh", ["fetching the model catalog, once"], True))
check("fetched today: nothing asked", catalog.ensure(said.append), None)
week_ago = time.time() - catalog.REFRESH_S - 60
os.utime(catalog.default_path(), (week_ago, week_ago))
check("a week old: asked again, and a 304 counts as checked",
      (catalog.ensure(said.append), catalog.ensure(said.append),
       "over a week old" in said[-1]), ("unchanged", None, True))

# ================================================================= cli ===

out = io.StringIO()
os.environ["MDL_FIT_HOME"] = str(TMP / "home2")
hw_cache = hw.cache_dir()
hw_cache.mkdir(parents=True, exist_ok=True)
(hw_cache / "catalog.sqlite").write_bytes(path.read_bytes())
catalog.main(["search", "Coder"], out)
catalog.main(["stats"], out)
check("search and stats read the local snapshot",
      ["coder/Coder-8B" in out.getvalue(), "models   5" in out.getvalue()],
      [True, True])
check("the llama.cpp arch check reads the library, and says when it "
      "cannot", hw.arch_supported(str(TMP / "no-such-binary"), "qwen3"),
      None)
lib = TMP / "bin"
lib.mkdir()
(lib / "llama.dll").write_bytes(b"\x00llama\x00qwen35moe\x00lfm2\x00")
(lib / "llama-server").write_bytes(b"")
check("and knows which archs a build loads",
      [hw.arch_supported(str(lib / "llama-server"), a)
       for a in ("qwen35moe", "lfm2", "gemma9")], [True, True, False])

sys.exit(t.done())
