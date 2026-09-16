"""Budget, pagination and refresh tests against a local stand-in Hub.

The clock advances per request, so a forty-minute cutoff costs no real
waiting. Every resumed run opens a new database connection, like CI does.
"""
import http.server
import io
import json
import os
import sqlite3
import sys
import tempfile
import threading
import urllib.parse
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                  # noqa: E402
from mdl_fit import catalog, catalog_crawl, hw   # noqa: E402

t = support.Tally("test_catalog_resume")
check = t.check
root = Path(tempfile.mkdtemp(prefix="mdl-resume-test-"))
os.environ["MDL_FIT_HOME"] = str(root / "config")
os.environ["HF_HOME"] = str(root / "hf")
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)


def model(name, source=None, modified="v1"):
    return {"id": name, "tags": ["gguf", "text-generation"] + (
        ["base_model:quantized:" + source] if source else []),
        "downloads": 100, "lastModified": modified,
        "pipeline_tag": "text-generation", "gguf": {"architecture": "llama",
        "total": 1_000_000_000, "context_length": 32768}}


old = model("quant/old", "lab/old")
both = model("quant/both", "lab/shared")
new = model("quant/new", "lab/shared")
extra = model("quant/extra", "lab/extra")
lanes = {"downloads": [old, both, extra], "createdAt": [new, both]}
hits = []
fail = set()
low_quota = set()
denied = set()
revision = [1]


class Hub(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        q = urllib.parse.parse_qs(url.query)
        hits.append(self.path)
        if url.path in denied:
            self.send_response(403)
            self.end_headers()
            return
        if url.path in fail:
            self.send_response(429)
            self.send_header("Retry-After", "300")
            self.end_headers()
            return
        headers = {}
        if url.path == "/api/models":
            sort = q["sort"][0]
            start = int(q.get("cursor", [0])[0])
            data = lanes[sort][start:start + 1]   # force many seed pages
            if start + 1 < len(lanes[sort]):
                q["cursor"] = [str(start + 1)]
                headers["link"] = '<http://%s%s?%s>; rel="next"' % (
                    self.headers["Host"], url.path,
                    urllib.parse.urlencode(q, doseq=True))
        elif url.path.endswith("/tree/main"):
            page = int(q.get("page", [1])[0])
            data = [{"type": "file", "path":
                     "model-Q4_K_M-%05d-of-00002.gguf" % page,
                     "lfs": {"size": 100_000_000 * revision[0],
                             "oid": "file-%d" % page}}]
            if page == 1:
                headers["link"] = '<http://%s%s?page=2>; rel="next"' % (
                    self.headers["Host"], url.path)
        else:
            ident = url.path.removeprefix("/api/models/")
            data = {"id": ident, "tags": ["text-generation"],
                    "evalResults": [{"verified": True, "data": {
                        "dataset": {"id": "bench"}, "value": 70}}]}
        if url.path in low_quota:
            headers["ratelimit"] = '"api";r=1;t=300'
        body = json.dumps(data).encode()
        self.send_response(200)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hub)
threading.Thread(target=srv.serve_forever, daemon=True).start()
os.environ["MDL_HF_ENDPOINT"] = "http://127.0.0.1:%d" % srv.server_port


class ClockBudget(catalog_crawl.Budget):
    def __init__(self, requests):
        self.now = 0
        super().__init__(requests, clock=lambda: self.now, sleep=self.advance)

    def advance(self, seconds):
        self.now += seconds

    def before_request(self):
        result = super().before_request()
        self.now += 1
        return result


def build(path, n=100, **kwargs):
    return catalog_crawl.build(path, popular=2, recent=2,
                               budget=ClockBudget(n), **kwargs)


def rows(path, table):
    db = sqlite3.connect(path)
    try:
        return db.execute("SELECT * FROM " + table + " ORDER BY 1,2").fetchall()
    finally:
        db.close()


# A complete run establishes the result that all interrupted runs must reach.
full = root / "full.sqlite"
meta = build(full)
check("mixed seed has one old, one new, and one overlap, not four repos",
      sorted(r[0] for r in rows(full, "ggufs")),
      ["quant/both", "quant/new", "quant/old"])
check("quants share a model rather than becoming duplicate recommendations",
      sorted(r[0] for r in rows(full, "nodes")), ["lab/old", "lab/shared"])
check("source benchmark evidence is retained", len(rows(full, "evals")), 2)
check("source metadata fetched once even across both seed lanes",
      sum(h.startswith("/api/models/lab/shared?") for h in hits), 1)
check("never asks for descendants or quantized children",
      any("base_model" in h for h in hits), False)
check("complete snapshot has no pending tasks", (meta["complete"],
                                                meta["pending"]), (True, 0))
check("pagination consumes only each lane's quota, not its full listing",
      any("lab/extra" in h for h in hits), False)

# Reopen a published snapshot at every possible network boundary. Even a
# stop between the two shard pages must resume without duplicating a page.
partial = root / "partial.sqlite"
hits.clear()
meta = build(partial, 1)
check("budget saves a readable partial immediately", (meta["complete"],
                                                       meta["stop_reason"]),
      (False, "budget"))
staged = False
for _ in range(30):
    if meta["complete"]:
        break
    staged = staged or bool(rows(partial, "crawl_files"))
    meta = build(partial, 1)
check("file pagination really was interrupted", staged, True)
check("one-request resumptions finish", meta["complete"], True)
check("resume does not refetch successful HTTP pages", len(hits),
      len(set(hits)))
for table in ("nodes", "ggufs", "evals"):
    check("resumed %s equals uninterrupted result" % table,
          rows(partial, table), rows(full, table))

# Resume from the published file on a different runner/output path.
download = root / "download.sqlite"
meta = build(download, 2)
hits.clear()
resumed = root / "other-runner.sqlite"
meta = build(resumed, prev=download)
check("downloaded checkpoint resumes on another runner", meta["complete"], True)
check("resumption starts at file inventory, not the first seed",
      hits[0].startswith("/api/models/quant/old/tree/main"), True)

# The successful low-quota response must commit before a too-long wait.
low_quota.add("/api/models")
rate = root / "rate.sqlite"
meta = build(rate, 10)
check("a successful page survives a quota wait longer than the budget",
      (meta["requests"], len(rows(rate, "nodes")), meta["stop_reason"]),
      (1, 1, "budget"))
low_quota.clear()
meta = build(rate)
check("next run continues after deferred quota wait", meta["complete"], True)

# 429 before any response also saves a checkpoint and never sleeps 300s.
fail.add("/api/models")
blocked = root / "blocked.sqlite"
meta = build(blocked, 10)
check("429 backoff respects wall-clock budget", (meta["complete"],
      meta["requests"], meta["stop_reason"]), (False, 1, "budget"))
fail.clear()
check("429 task is retried on the next run", build(blocked)["complete"], True)

denied.add("/api/models/quant/old/tree/main")
gated = root / "gated.sqlite"
meta = build(gated)
check("one gated repo does not strand the rest of the catalog",
      (meta["complete"], meta["unavailable"], len(rows(gated, "ggufs"))),
      (True, 1, 2))
denied.clear()
check("unavailable repo is retried next cycle", build(gated)["unavailable"], 0)
denied.add("/api/models")
meta = build(root / "denied-seed.sqlite")
check("denied seed is a partial with an explicit error, not an empty success",
      (meta["complete"], meta["stop_reason"].startswith("request failed:")),
      (False, True))
denied.clear()

# Refresh a completed catalog without shrinking it on a partial run.
before = rows(full, "ggufs")
hits.clear()
meta = build(full, 1)
check("refresh partial keeps the previously published inventory",
      rows(full, "ggufs"), before)
meta = build(full)
check("unchanged repos reuse their file inventories",
      any("/tree/main" in h for h in hits), False)
check("refresh completes after resuming", meta["complete"], True)

# A snapshot published before drafts and MTP heads were recognised loses
# them on the next run, even one the budget stops before any repo refreshes.
db = sqlite3.connect(full)
with db:
    db.execute("INSERT INTO ggufs SELECT repo, node, 'Q4_0', "
               "'MTP/mtp-Model-Q4_0.gguf', 1, shards, downloads, modified "
               "FROM ggufs LIMIT 1")
db.close()
build(full, 1)
check("an older snapshot's drafts are dropped, its quants kept",
      rows(full, "ggufs"), before)
build(full)                             # finish that cycle before going on

# Changed file inventories replace their old rows only on the last page.
old["lastModified"] = "v2"
revision[0] = 2
meta = build(full, 3)  # seed, source, first file page
check("a partial changed inventory does not replace complete shards",
      rows(full, "ggufs"), before)
meta = build(full)
check("completed inventory replaces old sizes",
      next(r[4] for r in rows(full, "ggufs") if r[0] == "quant/old"),
      400_000_000)

# Changed settings start a new cycle; only completion drops stale rows.
lanes["downloads"] = [both]
lanes["createdAt"] = [new]
meta = catalog_crawl.build(full, popular=1, recent=1, budget=ClockBudget(1))
check("scope change does not erase the catalog before it finishes",
      len(rows(full, "ggufs")), 3)
meta = catalog_crawl.build(full, popular=1, recent=1)
check("completed scope change prunes old repositories",
      sorted(r[0] for r in rows(full, "ggufs")), ["quant/both", "quant/new"])

# A crash rolls back the current page and leaves committed work recoverable.
crash = root / "crash.sqlite"
real_files = catalog_crawl.Crawl.files


def explode(self, task):
    real_files(self, task)
    raise RuntimeError("simulated process failure")


try:
    with patch.object(catalog_crawl.Crawl, "files", explode):
        build(crash)
except RuntimeError:
    pass
check("failed build has a local checkpoint", Path(str(crash) + ".building")
      .is_file(), True)
check("crash does not publish uncommitted rows", crash.exists(), False)
check("committed checkpoint recovers after a failed page",
      build(crash)["complete"], True)

# Readers report partials, and old readers can still query their usual tables.
out = io.StringIO()
cached = hw.cache_dir() / "catalog.sqlite"
build(cached, 1)
catalog.main(["stats"], out)
check("stats labels partial snapshots", "crawl    partial" in out.getvalue(),
      True)

# Only an absent snapshot may start from scratch in CI. Authentication,
# network and corrupt-download failures must not silently reset the queue.
for status, missing in ((404, True), (401, False), (403, False), (503, False)):
    error = urllib.error.HTTPError("http://test", status, "test", {}, None)
    with patch("urllib.request.urlopen", side_effect=error):
        try:
            catalog.pull(root / "unpublished.sqlite")
        except catalog.CatalogError as e:
            check("restore HTTP %d distinguishes bootstrap from failure" % status,
                  isinstance(e, catalog.CatalogNotPublished), missing)
for args in (["build", "--budget-minutes", "nan"],
             ["build", "--popular", "-1"],
             ["build", "--org", "X", "--budget-minutes", "1"]):
    _, err, code = support.run(catalog.main, args)
    check("invalid options fail politely: " + " ".join(args),
          (code, bool(err)), (1, True))

srv.shutdown()
sys.exit(t.done())
