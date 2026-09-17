"""A bounded GGUF catalog, with its work queue inside the published SQLite.

Two seeds (downloads and creation date) share a deduplicated queue. Every
HTTP page and its next cursor commit together. A partial is a normal
catalog: readers ignore the extra tables, and the next builder resumes it.
Only direct quantization sources are fetched; this never walks descendants.
"""

import json
import math
import os
import sqlite3
import time
import urllib.parse
from pathlib import Path

from . import catalog, remote

STATE_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS crawl_state(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS crawl_tasks(
  key TEXT PRIMARY KEY, kind TEXT, payload TEXT, done INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS crawl_seen(kind TEXT, id TEXT,
  PRIMARY KEY(kind, id));
CREATE TABLE IF NOT EXISTS crawl_files(repo TEXT, path TEXT, payload TEXT,
  PRIMARY KEY(repo, path));
"""


class BudgetExpired(Exception):
    pass


class Budget:
    def __init__(self, seconds, clock=time.monotonic, sleep=None):
        self.clock = clock
        self.sleeper = sleep or catalog.SLEEP
        self.end = clock() + seconds
        self.wait_until = 0
        self.requests = 0

    def check(self):
        if self.clock() >= self.end:
            raise BudgetExpired

    def sleep(self, seconds):
        # A wait that cannot finish in this run belongs to the next run.
        if self.clock() + seconds >= self.end:
            raise BudgetExpired
        self.sleeper(seconds)
        self.check()

    def defer(self, seconds):
        self.wait_until = max(self.wait_until, self.clock() + seconds)

    def before_request(self):
        self.check()
        if self.wait_until > self.clock():
            self.sleep(self.wait_until - self.clock())
        self.requests += 1
        return max(0.001, min(60, self.end - self.clock()))


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _values(db, table):
    return {r["key"]: json.loads(r["value"])
            for r in db.execute("SELECT * FROM " + table)}


def _put(db, table, values):
    db.executemany("INSERT OR REPLACE INTO " + table + " VALUES (?, ?)",
                   [(k, json.dumps(v)) for k, v in values.items()])


class Crawl:
    def __init__(self, db, budget, log):
        self.db, self.budget, self.log = db, budget, log
        self.state = _values(db, "crawl_state")

    def enqueue(self, key, kind, payload):
        self.db.execute("INSERT OR IGNORE INTO crawl_tasks "
                        "(key, kind, payload) VALUES (?, ?, ?)",
                        (key, kind, json.dumps(payload)))

    def seen(self, kind, ident):
        self.db.execute("INSERT OR IGNORE INTO crawl_seen VALUES (?, ?)",
                        (kind, ident))

    def request(self, url):
        # A downloaded checkpoint can contain URLs. Never send an HF token
        # to another origin because a snapshot or pagination link says to.
        parsed = urllib.parse.urlsplit(url)
        origin = urllib.parse.urlsplit(remote.endpoint())
        if ((parsed.scheme, parsed.netloc) != (origin.scheme, origin.netloc)
                or not parsed.path.startswith("/api/models")):
            raise catalog.CatalogError("catalog cursor is outside the Hub API")
        return catalog.get_json(url, budget=self.budget)

    def store_model(self, m):
        mid = m["id"]
        old = self.db.execute("SELECT arch, params, ctx FROM nodes WHERE id=?",
                              (mid,)).fetchone()
        self.db.execute("DELETE FROM nodes WHERE id=?", (mid,))
        self.db.execute("DELETE FROM evals WHERE node=?", (mid,))
        parents = [(p, r) for p, r in catalog.base_of(m) if r != "quantized"]
        parent, relation = parents[0] if parents else (None, None)
        writer = catalog.Crawler(self.db, orgs=[])
        writer.add(m, official=writer.is_official(mid), parent=parent,
                   relation=relation, method="card" if parent else None)
        if old:
            self.db.execute("UPDATE nodes SET arch=COALESCE(arch, ?), "
                            "params=COALESCE(params, ?), ctx=COALESCE(ctx, ?) "
                            "WHERE id=?", (*old, mid))
        self.seen("node", mid)

    def seed(self, task):
        p = json.loads(task["payload"])
        data, headers = self.request(p["url"])
        if not isinstance(data, list):
            raise catalog.CatalogError("Hub model listing is not a list")
        page = data[:p["remaining"]]
        for m in page:
            if not catalog.is_text_model(m):
                continue
            repo = m["id"]
            if self.db.execute("SELECT 1 FROM crawl_seen WHERE kind='repo' "
                               "AND id=?", (repo,)).fetchone():
                continue
            self.seen("repo", repo)
            sources = [b for b, r in catalog.base_of(m) if r == "quantized"]
            node = sources[0] if sources else repo
            self.seen("node", node)
            if sources:
                if not self.db.execute("SELECT 1 FROM nodes WHERE id=?",
                                       (node,)).fetchone():
                    # Do not attribute a quantizer's scores/license to the
                    # source. This placeholder only carries tensor metadata.
                    self.store_model({"id": node, "gguf": m.get("gguf")})
                self.enqueue("source:" + node, "source", {"node": node})
            else:
                self.store_model(m)
            gg = m.get("gguf") or {}
            self.db.execute("UPDATE nodes SET arch=COALESCE(arch, ?), "
                            "params=COALESCE(params, ?), ctx=COALESCE(ctx, ?) "
                            "WHERE id=?", (catalog._text(gg.get("architecture")),
                                           gg.get("total"),
                                           gg.get("context_length"), node))
            changed = m.get("lastModified")
            old = self.db.execute("SELECT 1 FROM ggufs WHERE repo=? "
                                  "AND modified=?", (repo, changed)).fetchone()
            if changed and old:
                self.db.execute("UPDATE ggufs SET node=?, downloads=? "
                                "WHERE repo=?", (node, m.get("downloads", 0),
                                                 repo))
            else:
                self.enqueue("files:" + repo, "files", {
                    "repo": repo, "node": node, "modified": changed,
                    "downloads": m.get("downloads", 0),
                    "url": "%s/api/models/%s/tree/main?recursive=true" % (
                        remote.endpoint(), urllib.parse.quote(repo, safe="/"))})
        p["remaining"] -= len(page)
        nxt = catalog._next(headers)
        self.state[p["sort"] + "_listed"] = (
            self.state.get(p["sort"] + "_listed", 0) + len(page))
        if p["remaining"] > 0 and nxt and data:
            p["url"] = nxt
            self.update(task, p)
        else:
            self.done(task)

    def source(self, task):
        node = json.loads(task["payload"])["node"]
        url = "%s/api/models/%s?%s" % (
            remote.endpoint(), urllib.parse.quote(node, safe="/"),
            urllib.parse.urlencode([("expand", e) for e in catalog.EXPAND]))
        m, _ = self.request(url)
        if isinstance(m, dict):
            self.store_model(m)
        self.done(task)

    def files(self, task):
        p = json.loads(task["payload"])
        data, headers = self.request(p["url"])
        if data is not None and not isinstance(data, list):
            raise catalog.CatalogError("Hub file listing is not a list")
        for item in data or []:
            if item.get("type") != "file":
                continue
            lfs = item.get("lfs") or {}
            f = {"path": item["path"], "size": lfs.get("size",
                                                        item.get("size", 0)),
                 "oid": lfs.get("oid") or item.get("oid", "")}
            self.db.execute("INSERT OR REPLACE INTO crawl_files VALUES (?,?,?)",
                            (p["repo"], f["path"], json.dumps(f)))
        nxt = catalog._next(headers)
        if nxt:
            p["url"] = nxt
            self.update(task, p)
            return
        files = [json.loads(r[0]) for r in self.db.execute(
            "SELECT payload FROM crawl_files WHERE repo=?", (p["repo"],))]
        groups = remote.gguf_groups(files)
        # Until the last page, keep the previous complete inventory. Never
        # expose half of a sharded model as if it were a usable GGUF.
        self.db.execute("DELETE FROM ggufs WHERE repo=?", (p["repo"],))
        self.db.executemany("INSERT INTO ggufs VALUES (?,?,?,?,?,?,?,?)", [
            (p["repo"], p["node"], catalog.quant_of(key), key,
             sum(s["size"] for s in shards), json.dumps(shards),
             p["downloads"], p["modified"])
            for key, shards in groups.items()
            if sum(s["size"] for s in shards) >= 50_000_000])
        self.db.execute("DELETE FROM crawl_files WHERE repo=?", (p["repo"],))
        self.done(task)

    def update(self, task, payload):
        self.db.execute("UPDATE crawl_tasks SET payload=? WHERE key=?",
                        (json.dumps(payload), task["key"]))

    def done(self, task):
        self.db.execute("UPDATE crawl_tasks SET done=1 WHERE key=?",
                        (task["key"],))

    def run(self):
        while True:
            # Finish useful inventories between seed pages. Alternate the
            # two seed lanes so a partial includes recent models too.
            task = self.db.execute("SELECT * FROM crawl_tasks WHERE done=0 "
                                   "ORDER BY CASE kind WHEN 'source' THEN 0 "
                                   "WHEN 'files' THEN 1 ELSE 2 END, rowid "
                                   "LIMIT 1").fetchone()
            if task is None:
                return
            self.budget.check()
            try:
                with self.db:
                    try:
                        getattr(self, task["kind"])(task)
                    except catalog.CatalogAccessError:
                        if task["kind"] == "seed":
                            raise
                        self.state["unavailable"] = (
                            self.state.get("unavailable", 0) + 1)
                        self.log("access denied: %s; retry next cycle" %
                                 task["key"])
                        self.done(task)
                    if task["kind"] == "seed":
                        # Move this lane behind the other without changing
                        # its cursor. Both lanes retain their own quotas.
                        self.db.execute("UPDATE crawl_tasks SET rowid="
                                        "(SELECT MAX(rowid)+1 FROM crawl_tasks) "
                                        "WHERE key=?", (task["key"],))
                    _put(self.db, "crawl_state", self.state)
            except Exception:
                self.state = _values(self.db, "crawl_state")
                raise
            if self.budget.requests % 25 == 0:
                self.log("%d requests; %d tasks pending" % (
                    self.budget.requests, self.pending()))

    def pending(self):
        return self.db.execute("SELECT COUNT(*) FROM crawl_tasks WHERE done=0"
                               ).fetchone()[0]


def build(path, prev=None, popular=3000, recent=500, minutes=40, log=None,
          budget=None):
    if popular < 0 or recent < 0 or popular + recent < 1:
        raise catalog.CatalogError("seed counts must be non-negative, "
                                   "with at least one repository")
    if not math.isfinite(minutes) or minutes <= 0:
        raise catalog.CatalogError("budget minutes must be a positive number")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".building")
    settings = {"version": STATE_VERSION, "popular": popular, "recent": recent,
                "endpoint": remote.endpoint()}
    # A .building file contains committed pages if the process was killed.
    # In CI, the downloaded snapshot carries exactly the same checkpoint.
    if not tmp.exists():
        source = Path(prev) if prev else path
        if source.is_file():
            old = sqlite3.connect(str(source))
            try:
                copy = sqlite3.connect(str(tmp))
                try:
                    old.backup(copy)
                finally:
                    copy.close()
            finally:
                old.close()
    db = catalog.connect(tmp)
    try:
        db.executescript(SCHEMA)
        state, previous = _values(db, "crawl_state"), _values(db, "meta")
        if state.get("settings") != settings or state.get("complete"):
            with db:
                for table in ("crawl_state", "crawl_tasks", "crawl_seen",
                              "crawl_files"):
                    db.execute("DELETE FROM " + table)
                _put(db, "crawl_state", {"settings": settings,
                                         "started_at": _now(),
                                         "complete": False})
        t0 = time.monotonic()
        budget = budget or Budget(minutes * 60)
        crawl = Crawl(db, budget, log or (lambda s: None))
        with db:
            for sort, count in (("downloads", popular), ("createdAt", recent)):
                if count:
                    url = "%s/api/models?%s" % (remote.endpoint(),
                        urllib.parse.urlencode(
                            [("filter", "gguf"), ("sort", sort),
                             ("direction", "-1"), ("limit", min(100, count))]
                            + [("expand", e) for e in catalog.EXPAND]))
                    crawl.enqueue("seed:" + sort, "seed", {
                        "url": url, "sort": sort, "remaining": count})
        reason = "complete"
        try:
            crawl.run()
        except BudgetExpired:
            reason = "budget"
        except catalog.CatalogError as e:
            # Preserve completed pages and retry this task next run. The
            # reason is published too; a partial never masquerades as done.
            reason = "request failed: " + str(e)
        complete = crawl.pending() == 0
        with db:
            if complete:
                # Only a completed refresh may remove the previous cycle's
                # rows. Partial refreshes remain useful throughout the run.
                db.execute("DELETE FROM ggufs WHERE repo NOT IN "
                           "(SELECT id FROM crawl_seen WHERE kind='repo')")
                db.execute("DELETE FROM nodes WHERE id NOT IN "
                           "(SELECT id FROM crawl_seen WHERE kind='node')")
                db.execute("DELETE FROM evals WHERE node NOT IN "
                           "(SELECT id FROM nodes)")
            # Recompute roots from the direct relationships we actually
            # know. No recursive network calls are needed for this.
            catalog.Crawler(db, orgs=[]).link()
            # A snapshot from before remote.auxiliary() carries drafts and
            # MTP heads as quants; removing them is safe in a partial too.
            db.executemany("DELETE FROM ggufs WHERE repo=? AND file=?", [
                r for r in db.execute("SELECT repo, file FROM ggufs").fetchall()
                if remote.auxiliary(r[1])])
            counts = {k: db.execute("SELECT COUNT(*) FROM " + k).fetchone()[0]
                      for k in ("nodes", "ggufs", "evals")}
            now = _now()
            meta = {"schema": catalog.SCHEMA_VERSION, "built_at": now,
                    "scope": "gguf-mixed", "popular": popular, "recent": recent,
                    "complete": complete, "stop_reason": reason,
                    "pending": crawl.pending(), "requests": budget.requests,
                    "unavailable": crawl.state.get("unavailable", 0),
                    "seconds": round(time.monotonic() - t0),
                    "cycle_started_at": crawl.state["started_at"],
                    "last_complete_at": now if complete else
                        previous.get("last_complete_at"), **counts}
            crawl.state["complete"] = complete
            _put(db, "crawl_state", crawl.state)
            _put(db, "meta", meta)
    finally:
        db.close()
    os.replace(tmp, path)
    return meta
