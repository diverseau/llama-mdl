"""mdl catalog - every model worth considering, as a lineage graph.

The graph is Hugging Face's own model tree: base -> official post-train
-> fine-tunes and merges, with quantization edges saying which GGUF
repos exist for each node. It lives in one SQLite file (stdlib sqlite3).

A crawl builds it (`mdl catalog build`, or nightly in CI, which
publishes the file as an HF dataset) and `mdl find` reads it. Users
pull the snapshot with an ETag check (`mdl catalog pull`), so only the
crawler ever walks the API.

The noise gate: a node gets in if it is from a tracked org, or has
MIN_DOWNLOADS downloads in the last 30 days, or MIN_LIKES likes, or is
from an allowlisted org - and it, or a quant of it, has GGUF files.
Every lineage edge records how it was known: hf-tree (HF's own
relation), card (the card's base_model), gguf-meta (a GGUF's
general.base_model keys), fingerprint (same arch and parameter count as
a tracked model), or name (a last resort).
"""

import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import hw, remote

SCHEMA_VERSION = 1
FILE = "catalog.sqlite"
# Orgs whose own releases are tracked: their models are the roots.
FAMILIES = ["Qwen", "meta-llama", "google", "mistralai", "microsoft",
            "deepseek-ai", "LiquidAI", "ibm-granite", "zai-org", "openai",
            "moonshotai", "nvidia", "allenai", "HuggingFaceTB", "tiiuae",
            "CohereLabs", "internlm", "tencent", "baidu", "stepfun-ai",
            "MiniMaxAI", "inclusionAI", "ServiceNow-AI", "swiss-ai",
            "arcee-ai", "ByteDance-Seed", "XiaomiMiMo"]
# Fine-tuners let through the gate whatever their numbers.
ALLOW = ["NousResearch", "cognitivecomputations", "open-thoughts",
         "agentica-org", "PrimeIntellect", "Menlo", "all-hands",
         "SWE-bench", "nvidia", "allenai", "arcee-ai"]
MIN_DOWNLOADS, MIN_LIKES = 500, 20
EXPAND = ["downloads", "likes", "gguf", "evalResults", "safetensors",
          "cardData", "createdAt", "lastModified", "tags", "gated",
          "pipeline_tag"]
PIPELINES = {"text-generation", "image-text-to-text"}
RELATIONS = ("finetune", "merge")          # adapters rarely come as GGUF
SLEEP = time.sleep                          # tests replace it

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS nodes(
  id TEXT PRIMARY KEY, author TEXT, official INTEGER, root TEXT,
  parent TEXT, relation TEXT, method TEXT, depth INTEGER,
  arch TEXT, params INTEGER, ctx INTEGER, license TEXT, pipeline TEXT,
  created TEXT, modified TEXT, downloads INTEGER, likes INTEGER,
  gated INTEGER, tags TEXT, datasets TEXT);
CREATE TABLE IF NOT EXISTS ggufs(
  repo TEXT, node TEXT, quant TEXT, file TEXT, size INTEGER,
  shards TEXT, downloads INTEGER, modified TEXT,
  PRIMARY KEY(repo, file));
CREATE TABLE IF NOT EXISTS evals(
  node TEXT, benchmark TEXT, task TEXT, value REAL, verified INTEGER,
  source TEXT, date TEXT);
CREATE INDEX IF NOT EXISTS nodes_parent ON nodes(parent);
CREATE INDEX IF NOT EXISTS ggufs_node ON ggufs(node);
CREATE INDEX IF NOT EXISTS evals_node ON evals(node);
"""

QUANT = re.compile(
    r"(?i)(?:^|[-_.])((?:UD-)?(?:IQ\d_[A-Z0-9]+(?:_[A-Z0-9]+)?"
    r"|Q\d(?:_[A-Z0-9]+)*|TQ\d_\d|BF16|F16|F32|FP16|FP32|MXFP4(?:_MOE)?))"
    r"(?=[-_.]|$)")
SPELLINGS = {"FP16": "F16", "FP32": "F32"}


class CatalogError(Exception):
    pass


class CatalogAccessError(CatalogError):
    """A private/gated repo must not block every other queued model."""


class CatalogNotPublished(CatalogError):
    """Only a 404 may bootstrap CI; other pull failures must keep its state."""


def default_path():
    return hw.cache_dir() / FILE


def repo_name():
    return os.environ.get("MDL_CATALOG_REPO", "diversemate/mdl-catalog")


def quant_of(name):
    """'Qwen3-8B-UD-Q4_K_XL.gguf' -> 'UD-Q4_K_XL'."""
    stem = Path(name).name
    stem = stem[:-5] if stem.lower().endswith(".gguf") else stem
    found = QUANT.findall(stem)
    if not found:
        return "?"
    q = found[-1].upper()
    return SPELLINGS.get(q, q)


def _text(v):
    """A card field is a string, or a list of them, or a dict. sqlite
    binds none of the last two, and a crawl is too long to lose to one."""
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v) or None
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return v


# ---------------------------------------------------------------- http --

def _headers(extra=None):
    h = {"User-Agent": remote.UA}
    tok = remote.token()
    if tok:
        h["Authorization"] = "Bearer " + tok
    h.update(extra or {})
    return h


def _header(headers, name):
    # urllib's header object is case-insensitive; dict(r.headers) is not.
    return next((v for k, v in headers.items() if k.lower() == name.lower()), "")


def _pace(headers, budget=None):
    """HF says how many calls are left in the window; wait it out when
    they run low rather than eat a 429."""
    m = re.search(r"r=(\d+);\s*t=(\d+)", _header(headers, "RateLimit"))
    if m and int(m.group(1)) < 3:
        if budget is None:
            SLEEP(int(m.group(2)) + 1)
        else:
            # Commit the response before waiting for the next request.
            budget.defer(int(m.group(2)) + 1)


def get_json(url, tries=6, budget=None):
    """(parsed JSON or None when not there, headers). Retries on 429 and
    5xx with the server's Retry-After, and on network errors."""
    for attempt in range(tries):
        timeout = budget.before_request() if budget else 60
        req = urllib.request.Request(url, headers=_headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body, headers = r.read(), dict(r.headers)
            _pace(headers, budget)
            return json.loads(body), headers
        except urllib.error.HTTPError as e:
            if e.code == 404 or (budget is None and e.code in (401, 403)):
                return None, {}
            if e.code in (401, 403):
                raise CatalogAccessError("Hub access denied (HTTP %d)" % e.code
                                         ) from None
            if e.code == 429 or e.code >= 500:
                wait = e.headers.get("Retry-After")
                m = re.search(r"t=(\d+)", e.headers.get("RateLimit", "") or "")
                seconds = min(300, int(wait) if wait and wait.isdigit()
                              else int(m.group(1)) + 1 if m
                              else 5 * 2 ** attempt)
                (budget.sleep if budget else SLEEP)(seconds)
                continue
            raise CatalogError("%s: HTTP %d" % (url, e.code)) from None
        except (urllib.error.URLError, OSError, ValueError):
            (budget.sleep if budget else SLEEP)(2 ** attempt)
    raise CatalogError("gave up on %s after %d tries" % (url, tries))


def _next(headers):
    m = re.search(r'<([^>]+)>;\s*rel="next"', _header(headers, "Link"))
    return m.group(1) if m else None


# ------------------------------------------------------------------ db --

def connect(path, fresh=False):
    path = Path(path)
    if fresh and path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path))
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def _tags_value(tags, prefix):
    for t in tags or []:
        if t.startswith(prefix):
            return t[len(prefix):]
    return None


def base_of(m):
    """[(parent id, relation)] a repo declares, from HF's computed
    base_model tags first, then the card."""
    out = []
    for t in m.get("tags") or []:
        parts = t.split(":", 2)
        if len(parts) == 3 and parts[0] == "base_model":
            out.append((parts[2], parts[1]))
    if not out:
        card = m.get("cardData") or {}
        base = card.get("base_model")
        rel = card.get("base_model_relation") or "finetune"
        for b in [base] if isinstance(base, str) else base or []:
            out.append((b, rel))
    return out


def is_quant_repo(m):
    """A repo of GGUF quants of another model, not a model of its own."""
    return "gguf" in (m.get("tags") or []) and any(
        rel == "quantized" for _, rel in base_of(m))


def is_text_model(m):
    tags = set(m.get("tags") or [])
    return (m.get("pipeline_tag") in PIPELINES or "text-generation" in tags
            or "conversational" in tags)


# --------------------------------------------------------------- crawl --

class Crawler:
    def __init__(self, db, orgs=None, allow=None, per_org=60,
                 max_children=100, max_depth=3, max_nodes=5000,
                 ggufs_per_node=3, min_downloads=MIN_DOWNLOADS,
                 min_likes=MIN_LIKES, prev=None, log=None):
        self.db = db
        self.orgs = list(FAMILIES if orgs is None else orgs)
        self.allow = set(ALLOW if allow is None else allow) | set(self.orgs)
        self.per_org, self.max_children = per_org, max_children
        self.max_depth, self.max_nodes = max_depth, max_nodes
        self.ggufs_per_node = ggufs_per_node
        self.min_downloads, self.min_likes = min_downloads, min_likes
        self.prev = prev
        self.log = log or (lambda s: None)
        self.ids, self.raw, self.requests = set(), {}, 0

    # -- api --
    def list_models(self, params, limit):
        url = "%s/api/models?%s" % (remote.endpoint(), urllib.parse.urlencode(
            params + [("limit", str(min(100, limit)))]
            + [("expand", e) for e in EXPAND]))
        out = []
        while url and len(out) < limit:
            data, headers = get_json(url)
            self.requests += 1
            if not isinstance(data, list):
                break
            out += data
            url = _next(headers)
        return out[:limit]

    def model(self, repo):
        url = "%s/api/models/%s?%s" % (remote.endpoint(), repo,
                                       urllib.parse.urlencode(
                                           [("expand", e) for e in EXPAND]))
        data, _ = get_json(url)
        self.requests += 1
        return data if isinstance(data, dict) else None

    def files(self, repo):
        url = "%s/api/models/%s/tree/main?recursive=true" % (
            remote.endpoint(), repo)
        out = []
        while url:
            data, headers = get_json(url)
            self.requests += 1
            for item in data or []:
                if item.get("type") == "file":
                    lfs = item.get("lfs") or {}
                    out.append({"path": item["path"],
                                "size": lfs.get("size", item.get("size", 0)),
                                "oid": lfs.get("oid") or item.get("oid", "")})
            url = _next(headers)
        return out

    # -- gate --
    def worth(self, m):
        return (m.get("downloads", 0) >= self.min_downloads
                or m.get("likes", 0) >= self.min_likes
                or m["id"].split("/")[0] in self.allow)

    # -- rows --
    def add(self, m, official, parent=None, relation=None, method=None,
            depth=0):
        mid = m["id"]
        if mid in self.ids:
            return False
        self.ids.add(mid)
        self.raw[mid] = m
        card = m.get("cardData") or {}
        gg, st = m.get("gguf") or {}, m.get("safetensors") or {}
        lic = card.get("license") or _tags_value(m.get("tags"), "license:")
        self.db.execute(
            "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, mid.split("/")[0], int(official), None, parent, relation,
             method, depth, _text(gg.get("architecture")),
             st.get("total") or gg.get("total"), gg.get("context_length"),
             _text(lic), _text(m.get("pipeline_tag")), m.get("createdAt"),
             m.get("lastModified"), m.get("downloads", 0), m.get("likes", 0),
             int(bool(m.get("gated"))),
             json.dumps([t for t in m.get("tags") or []
                         if not t.startswith(("region:", "endpoints_",
                                              "deploy:"))]),
             json.dumps(card.get("datasets") or [])))
        self.add_evals(mid, m)
        return True

    def add_evals(self, mid, m):
        rows = []
        for e in m.get("evalResults") or []:
            d = e.get("data") or {}
            ds = d.get("dataset") or {}
            src = d.get("source") or {}
            if isinstance(d.get("value"), (int, float)) and ds.get("id"):
                rows.append((mid, ds["id"], ds.get("task_id"), d["value"],
                             int(bool(e.get("verified"))),
                             src.get("name") or src.get("url") or "eval_results",
                             d.get("date")))
        index = (m.get("cardData") or {}).get("model-index") or []
        for entry in index if isinstance(index, list) else []:
            for res in entry.get("results") or []:
                ds = res.get("dataset") or {}
                for metric in res.get("metrics") or []:
                    v = metric.get("value")
                    if isinstance(v, (int, float)) and (ds.get("name")
                                                        or ds.get("type")):
                        rows.append((mid, ds.get("type") or ds.get("name"),
                                     metric.get("type"), v,
                                     int(bool(metric.get("verified"))),
                                     "model card", None))
        self.db.executemany("INSERT INTO evals VALUES (?,?,?,?,?,?,?)", rows)

    def add_gguf_repo(self, node, m):
        repo = m["id"]
        gg = m.get("gguf") or {}
        if gg.get("architecture"):              # a safetensors node learns
            self.db.execute(                    # its arch from its quants
                "UPDATE nodes SET arch = COALESCE(arch, ?), ctx = COALESCE("
                "ctx, ?), params = COALESCE(params, ?) WHERE id = ?",
                (gg["architecture"], gg.get("context_length"),
                 gg.get("total"), node))
        if self.prev is not None:                 # unchanged since last time
            old = self.prev.execute(
                "SELECT * FROM ggufs WHERE repo = ? AND modified = ?",
                (repo, m.get("lastModified"))).fetchall()
            if old:
                self.db.executemany(
                    "INSERT OR REPLACE INTO ggufs VALUES (?,?,?,?,?,?,?,?)",
                    [(r["repo"], node, r["quant"], r["file"], r["size"],
                      r["shards"], m.get("downloads", 0), r["modified"])
                     for r in old])
                return
        try:
            groups = {k: v for k, v in remote.gguf_groups(
                self.files(repo)).items()        # not stray templates or
                if sum(s["size"] for s in v) >= 50_000_000}  # vocab files
        except CatalogError as e:
            self.log("skip %s: %s" % (repo, e))
            return
        self.db.executemany(
            "INSERT OR REPLACE INTO ggufs VALUES (?,?,?,?,?,?,?,?)",
            [(repo, node, quant_of(key), key, sum(s["size"] for s in shards),
              json.dumps(shards), m.get("downloads", 0),
              m.get("lastModified")) for key, shards in groups.items()])

    def is_official(self, mid):
        """A family's own release, whichever orgs this build crawls: a
        --base Qwen/Qwen3-8B is still Qwen's."""
        return mid.split("/")[0] in set(self.orgs) | set(FAMILIES)

    # -- walk --
    def run(self, bases=()):
        queue = []
        for org in self.orgs:
            got = self.list_models([("author", org), ("sort", "downloads"),
                                    ("direction", "-1")], self.per_org * 2)
            n = 0
            for m in got:
                if n >= self.per_org:
                    break
                if is_quant_repo(m) or not is_text_model(m):
                    continue
                if self.add(m, official=True):
                    queue.append((m["id"], 0))
                    n += 1
            self.log("%-16s %d models" % (org, n))
        for b in bases:
            m = self.model(b)
            if m is None:
                raise CatalogError("%s: not found on the hub" % b)
            if self.add(m, official=self.is_official(b)):
                queue.append((b, 0))
        done = set()
        while queue:
            nid, depth = queue.pop(0)
            if nid in done:
                continue
            done.add(nid)
            self.expand(nid, depth, queue)
            if len(done) % 25 == 0:
                self.log("%d nodes walked, %d known, %d requests" % (
                    len(done), len(self.ids), self.requests))
        self.link()
        self.prune()
        self.db.commit()

    def expand(self, nid, depth, queue):
        if depth < self.max_depth and len(self.ids) < self.max_nodes:
            for rel in RELATIONS:
                for m in self.list_models(
                        [("filter", "base_model:%s:%s" % (rel, nid)),
                         ("sort", "downloads"), ("direction", "-1")],
                        self.max_children):
                    if m["id"] in self.ids or is_quant_repo(m) \
                            or not self.worth(m):
                        continue
                    if len(self.ids) >= self.max_nodes:
                        break
                    self.add(m, official=self.is_official(m["id"]),
                             parent=nid, relation=rel, method="hf-tree",
                             depth=depth + 1)
                    queue.append((m["id"], depth + 1))
        # both conditions go in `filter`: the hub ignores `other=` here and
        # hands back the most popular GGUF repos of all
        repos = self.list_models([("filter", "base_model:quantized:%s" % nid),
                                  ("filter", "gguf"), ("sort", "downloads"),
                                  ("direction", "-1")], self.ggufs_per_node)
        own = self.raw.get(nid)
        if own and "gguf" in (own.get("tags") or []):
            repos.insert(0, own)                  # it ships as GGUF itself
        for m in repos[:self.ggufs_per_node + 1]:
            self.add_gguf_repo(nid, m)

    def link(self):
        """Parents for the roots from their cards (an official post-train
        names its base), then each node's root: its topmost ancestor."""
        for nid, m in self.raw.items():
            row = self.db.execute("SELECT parent FROM nodes WHERE id = ?",
                                  (nid,)).fetchone()
            if row["parent"]:
                continue
            for base, rel in base_of(m):
                if base in self.ids and base != nid and rel != "quantized":
                    self.db.execute("UPDATE nodes SET parent = ?, relation = "
                                    "?, method = 'card' WHERE id = ?",
                                    (base, rel, nid))
                    break
        parents = dict(self.db.execute("SELECT id, parent FROM nodes"))
        for nid in parents:
            root, seen = nid, {nid}
            while parents.get(root) and parents[root] not in seen:
                root = parents[root]
                seen.add(root)
            self.db.execute("UPDATE nodes SET root = ? WHERE id = ?",
                            (root, nid))

    def prune(self):
        """Keep the tracked orgs' models (they anchor the lineage) and
        anything with GGUF in its own subtree; drop the rest."""
        parents = dict(self.db.execute("SELECT id, parent FROM nodes"))
        keep = {r[0] for r in self.db.execute(
            "SELECT id FROM nodes WHERE official = 1")}
        for (node,) in self.db.execute("SELECT DISTINCT node FROM ggufs"):
            while node and node not in keep:
                keep.add(node)
                node = parents.get(node)
        drop = [n for n in parents if n not in keep]
        self.db.executemany("DELETE FROM nodes WHERE id = ?",
                            [(n,) for n in drop])
        self.db.executemany("DELETE FROM evals WHERE node = ?",
                            [(n,) for n in drop])


def build(path, bases=(), prev=None, log=None, **kw):
    """Crawl into a fresh file at `path`. `prev`: an older snapshot whose
    file listings are reused for repos not modified since."""
    tmp = Path(str(path) + ".building")
    db = connect(tmp, fresh=True)
    old = None
    if prev and Path(prev).is_file():
        old = sqlite3.connect(str(prev))
        old.row_factory = sqlite3.Row
    crawler = Crawler(db, prev=old, log=log, **kw)
    t0 = time.time()
    crawler.run(bases)
    counts = {k: db.execute("SELECT COUNT(*) FROM %s" % k).fetchone()[0]
              for k in ("nodes", "ggufs", "evals")}
    meta = {"schema": SCHEMA_VERSION,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "orgs": crawler.orgs, "bases": list(bases),
            "min_downloads": crawler.min_downloads,
            "min_likes": crawler.min_likes, "requests": crawler.requests,
            "seconds": round(time.time() - t0), **counts}
    db.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                   [(k, json.dumps(v)) for k, v in meta.items()])
    db.commit()
    db.close()
    if old is not None:
        old.close()
    os.replace(tmp, path)
    return meta


# ---------------------------------------------------------------- pull --

def pull(path=None, repo=None):
    """Fetch the published snapshot if it changed. 'fresh', 'unchanged'."""
    path = Path(path or default_path())
    repo = repo or repo_name()
    etag_file = path.with_suffix(".etag")
    url = "%s/datasets/%s/resolve/main/%s" % (remote.endpoint(), repo, FILE)
    extra = {}
    if path.is_file() and etag_file.is_file():
        extra["If-None-Match"] = etag_file.read_text().strip()
    req = urllib.request.Request(url, headers=_headers(extra))
    tmp = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
        con = sqlite3.connect(str(tmp))
        try:
            con.execute("SELECT 1 FROM meta").fetchall()
        finally:
            con.close()
        os.replace(tmp, path)
        if etag:
            etag_file.write_text(etag)
        return "fresh"
    except urllib.error.HTTPError as e:
        if e.code == 304:
            return "unchanged"
        if e.code in (401, 403, 404):
            error = CatalogNotPublished if e.code == 404 else CatalogError
            raise error(
                "no published catalog at %s yet (or it is private); build "
                "one here with: mdl catalog build" % repo) from None
        raise CatalogError("%s: HTTP %d" % (url, e.code)) from None
    except (urllib.error.URLError, OSError) as e:
        raise CatalogError("%s: %s" % (url, getattr(e, "reason", e))) from None
    except sqlite3.DatabaseError as e:
        tmp.unlink(missing_ok=True)
        raise CatalogError("the downloaded catalog is not a catalog: %s"
                           % e) from None


# ---------------------------------------------------------------- read --

class Catalog:
    def __init__(self, path=None):
        path = Path(path or default_path())
        if not path.is_file():
            raise CatalogError("no catalog at %s; run: mdl catalog pull "
                               "(or mdl catalog build)" % path)
        self.path = path
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row

    def meta(self):
        return {r["key"]: json.loads(r["value"])
                for r in self.db.execute("SELECT * FROM meta")}

    def node(self, nid):
        return self.db.execute("SELECT * FROM nodes WHERE id = ? COLLATE "
                               "NOCASE", (nid,)).fetchone()

    def nodes(self):
        return self.db.execute("SELECT * FROM nodes").fetchall()

    def children(self, nid):
        return self.db.execute("SELECT * FROM nodes WHERE parent = ? ORDER "
                               "BY downloads DESC", (nid,)).fetchall()

    def lineage(self, nid):
        """[node, parent, ..., root]."""
        out, seen = [], set()
        row = self.node(nid)
        while row is not None and row["id"] not in seen:
            out.append(row)
            seen.add(row["id"])
            row = self.node(row["parent"]) if row["parent"] else None
        return out

    def ggufs(self, nid):
        return self.db.execute("SELECT * FROM ggufs WHERE node = ? ORDER BY "
                               "downloads DESC, size DESC", (nid,)).fetchall()

    def evals(self, nid):
        return self.db.execute("SELECT * FROM evals WHERE node = ?",
                               (nid,)).fetchall()

    def all_evals(self):
        return self.db.execute("SELECT * FROM evals").fetchall()

    def subtree(self, nid):
        """[(node, depth below nid)], nid first, breadth-first."""
        root = self.node(nid)
        if root is None:
            return []
        out, frontier, seen = [(root, 0)], [root["id"]], {root["id"]}
        depth = 0
        while frontier:
            depth += 1
            nxt = []
            for p in frontier:
                for c in self.children(p):
                    if c["id"] not in seen:
                        seen.add(c["id"])
                        out.append((c, depth))
                        nxt.append(c["id"])
            frontier = nxt
        return out

    def search(self, text, limit=20):
        return self.db.execute("SELECT * FROM nodes WHERE id LIKE ? ORDER BY "
                               "downloads DESC LIMIT ?",
                               ("%" + text + "%", limit)).fetchall()


# ----------------------------------------------------------------- cli --

USAGE = """\
usage: mdl catalog pull                 fetch the published snapshot
       mdl catalog build [options]      crawl the hub into a local one
       mdl catalog tree <org/repo>      cataloged relationships and GGUF quants
       mdl catalog search <text>        find a model by name
       mdl catalog stats

build options:
  --popular N        GGUF repos by downloads (default 800)
  --recent N         GGUF repos by creation date (default 200)
  --budget-minutes N  stop and save a resumable partial (default 40)
                     These apply to the default mixed GGUF seed.
  --org NAME         use the original lineage crawl for this org (repeatable)
  --base ORG/REPO    crawl the tree under this model only (repeatable)
  --per-org N        models per org (default 60)
  --max-nodes N      stop adding nodes at N (default 5000)
  --depth N          fine-tune generations to follow (default 3)
  --min-downloads N  the noise gate (default 500 in 30 days)
  --min-likes N      ... or this many likes (default 20)
  --from PATH        resume a partial snapshot, or refresh a completed one
                     (lineage mode only reuses unchanged file lists)
  --out PATH         where to write (default: the local cache)

The snapshot lives in %s. $MDL_CATALOG_REPO names the
dataset it is pulled from (default diversemate/mdl-catalog).
"""


def die(msg):
    import mdl
    mdl.die(msg)


def _opts(args, values, flags=()):
    o, pos, i = {}, [], 0
    while i < len(args):
        a = args[i]
        if a in values:
            if i + 1 >= len(args):
                die("%s needs a value" % a)
            o.setdefault(a[2:], []).append(args[i + 1])
            i += 2
        elif a in flags:
            o[a[2:]] = True
            i += 1
        elif a.startswith("--"):
            die("unknown option %s" % a)
        else:
            pos.append(a)
            i += 1
    return o, pos


def human(n):
    for unit in ("", "k", "M", "G"):
        if abs(n) < 1000:
            return "%g%s" % (round(n, 1), unit)
        n /= 1000
    return "%gT" % round(n, 1)


def progress(meta):
    if "complete" not in meta:
        return ""
    text = "%s; %d tasks pending; %s" % (
        "complete" if meta["complete"] else "partial",
        meta.get("pending", 0), meta.get("stop_reason", "unknown"))
    if meta.get("unavailable"):
        text += "; %d unavailable tasks" % meta["unavailable"]
    return text


def main(args, out=None):
    out = out or sys.stdout
    w = out.write
    if not args or args[0] in ("-h", "--help"):
        w(USAGE % default_path())
        return None
    cmd, rest = args[0], args[1:]
    try:
        if cmd == "pull":
            state = pull()
            w("catalog  %s (%s)\n" % (state, default_path()))
            cat = Catalog()
            try:
                detail = progress(cat.meta())
            finally:
                cat.db.close()
            if detail:
                w("crawl    %s\n" % detail)
            return state
        if cmd == "build":
            o, _ = _opts(rest, {"--org", "--base", "--per-org", "--max-nodes",
                                "--depth", "--min-downloads", "--min-likes",
                                "--from", "--out", "--popular", "--recent",
                                "--budget-minutes"})
            kw = {}
            for key, name in (("per-org", "per_org"),
                              ("max-nodes", "max_nodes"),
                              ("depth", "max_depth"),
                              ("min-downloads", "min_downloads"),
                              ("min-likes", "min_likes")):
                if key in o:
                    kw[name] = int(o[key][-1])
            bases = o.get("base", [])
            if "org" in o or bases:
                kw["orgs"] = o.get("org", [])
            path = Path(o["out"][-1]) if "out" in o else default_path()
            args = {"prev": (o.get("from") or [None])[-1],
                    "log": lambda s: (w("  %s\n" % s), out.flush())}
            if "org" in o or bases:
                if any(k in o for k in ("popular", "recent", "budget-minutes")):
                    die("mixed-seed options cannot be combined with --org/--base")
                meta = build(path, bases=bases, **args, **kw)
            else:
                from . import catalog_crawl
                if kw:
                    die("lineage options require --org or --base")
                meta = catalog_crawl.build(
                    path, popular=int(o.get("popular", [800])[-1]),
                    recent=int(o.get("recent", [200])[-1]),
                    minutes=float(o.get("budget-minutes", [40])[-1]), **args)
            w("built    %s: %d models, %d GGUF quants, %d eval results, "
              "%d requests, %ds\n" % (path, meta["nodes"], meta["ggufs"],
                                      meta["evals"], meta["requests"],
                                      meta["seconds"]))
            if progress(meta):
                w("crawl    %s\n" % progress(meta))
            return meta
        cat = Catalog()
        if cmd == "stats":
            m = cat.meta()
            w("catalog  %s\nbuilt    %s\nmodels   %d · GGUF quants %d · eval "
              "results %d\n" % (cat.path, m.get("built_at"), m.get("nodes"),
                                m.get("ggufs"), m.get("evals")))
            if progress(m):
                w("crawl    %s\n" % progress(m))
            return m
        if cmd == "search" and rest:
            for r in cat.search(rest[0]):
                w("%-60s %8s dl  %s\n" % (r["id"], human(r["downloads"] or 0),
                                          r["relation"] or "root"))
            return None
        if cmd == "tree" and rest:
            return show_tree(cat, rest[0], w)
    except (CatalogError, ValueError) as e:
        die(str(e))
    die(USAGE % default_path())
    return None


def show_tree(cat, nid, w):
    tree = cat.subtree(nid)
    if not tree:
        die("%s is not in the catalog" % nid)
    total = 0
    for row, depth in tree:
        quants = cat.ggufs(row["id"])
        total += len(quants)
        w("%s%s%s  %s dl%s\n" % ("  " * depth, "└ " if depth else "",
                                 row["id"], human(row["downloads"] or 0),
                                 "  (%s)" % row["relation"] if depth else ""))
        by_repo = {}
        for q in quants:
            by_repo.setdefault(q["repo"], []).append(q)
        for repo, qs in by_repo.items():
            w("%s    %s: %s\n" % ("  " * depth, repo, " ".join(
                "%s %.1fG" % (q["quant"], q["size"] / 1e9)
                for q in sorted(qs, key=lambda q: q["size"]))))
    w("\n%d models, %d GGUF quants\n" % (len(tree), total))
    return tree
