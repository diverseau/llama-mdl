"""mdl find - the best model you can run, at the quant and config you
would run it at, for what you do.

  1. load the catalog snapshot (mdl catalog pull), and every model in
     models.toml
  2. hard filter: llama.cpp at your build loads the arch, GGUF exists,
     the license and tag filters pass
  3. a generous bound: drop only what cannot fit even at its smallest
     quant with everything it can spare in RAM
  4. the fit engine on each shortlisted quant, via header fetch
     (cached): the best config per (model, quant) for the profile
  5. a quality estimate per (model, quant, KV type), from quality.py
  6. rank: quality is the objective, speed a floor. Unrated models never
     outrank rated ones; they go to 'worth testing' when their upper
     band beats the current #1.

A shortlisted model's quants share a header layout, so one header is
fetched per model and the others are sized from it; the rows shown get
their own header before they are.
"""

import concurrent.futures
import json
import re
import sys
import time
from pathlib import Path

from . import (catalog, evalrun, evalsuite, gguf, hw, progress, quality,
               remote, search)

# Floors on top of the profile's own: an agent turn slower than this
# is not an agent you will use; a chat below this is not a chat.
FLOORS = {"agent": ("s_turn", 90.0), "chat": ("decode_d", 20.0)}
SHORTLIST = 24
QUANTS_PER_MODEL = 3
WORKERS = 12          # header fetches at once: 6 read 8 MB/s, 12 read 10, 16 no more
ROWS = 8
TIE = 0.5              # points of quality a faster quant of one model may give up
K = 1024
GiB = 1 << 30

USAGE = """\
usage: mdl find [--profile agent|chat|max-ctx|speed] [options]

The best model this machine can run for the profile, across the
catalog (mdl catalog pull) and everything in models.toml: at the quant
and config it would run at, ranked by expected quality, with speed as
a floor.

options:
  --profile P     agent (default: ctx >= 128k, s/turn <= 90 s), chat
                  (decode >= 20 t/s), max-ctx, speed
  --top N         models to fit from the catalog (default 24)
  --license TEXT  only licenses containing TEXT (e.g. apache)
  --tag TEXT      only models tagged TEXT (e.g. code)
  --kv-floor T    most quantised KV allowed (default q8_0)
  --new           only what has appeared since the last --new
  --no-fetch      use cached GGUF headers only; nothing downloaded
  --catalog PATH  a catalog file other than the pulled one
  --why MODEL     explain one catalog model or models.toml name
  --pull N        fetch row N of the table and add a preset for it
  --run N         the same, then start it (one already here just starts)
  --json          machine-readable

Scores are on the public scale (50 + 15 z against the catalog) until
three models have both public results and local ones from mdl eval;
then on your local scale. `~/.config/mdl/find.toml` sets bench_allow,
bench_block, bench_domain and stale.
"""


def die(msg):
    import mdl
    mdl.die(msg)


class Cand:
    """One (model, quant): where it comes from, and what it came to."""

    def __init__(self, node, label, quant, size, repo=None, key=None,
                 shards=None, local=None, path=None):
        self.node, self.label, self.quant, self.size = node, label, quant, size
        self.repo, self.key, self.shards = repo, key, shards or []
        self.local, self.path = local, path   # a models.toml name and file
        self.binary = None       # a preset's own llama_server, if it has one
        self.inv = self.fit = self.shape = self.q = None
        self.exact = self.refined = False
        self.reject = None
        self.header_error = None
        self.hash = None

    @property
    def why(self):
        return self.reject[1] if self.reject else None

    @why.setter
    def why(self, message):
        # Keep callers that clear a stale reason working as before.
        self.reject = (self.reject[0] if self.reject else "header", message) \
            if message is not None else None

    @property
    def rankable(self):
        """A fit that clears the floors, a quality estimate, no reason to
        turn it away - all from the same evaluation."""
        return (self.fit is not None and self.q is not None
                and self.why is None)

    @property
    def spec(self):
        return str(self.path) if self.local else "hf:%s:%s" % (self.repo,
                                                               self.quant)


# ------------------------------------------------------------- headers --

def fetch(c, cache_only=False):
    """The candidate's own header (cached by content hash)."""
    if cache_only and not remote._cache_path(c.repo, c.key,
                                             c.shards).is_file():
        return None
    return remote.inventory(c.repo, c.key, c.shards)


SAME_MODEL = 3.0       # header params may differ from the card's by this factor


def same_model(inv, qm, nid):
    """Whether a header can be the model the catalog filed it under.

    Repos hold files that are not the model - speculative drafters under
    new names every month, calibration data - and names cannot keep up.
    An 11 GB DSpark drafter in a DeepSeek-V4-Flash repo was ranked as that
    304B model, on the GPU, at 8 s a turn. The header's own parameter count
    cannot be fooled that way. The bound is loose because a card counts a
    vision tower the GGUF leaves in its projector.
    """
    want, have = card_params(qm, nid), inv.n_params
    if not want or not have:
        return True
    return 1 / SAME_MODEL <= have / want <= SAME_MODEL


def card_params(qm, nid):
    node = qm.nodes.get(nid)
    try:
        return node["params"] if node else None
    except (KeyError, IndexError):          # a local node carries none
        return None


def impostor(inv, qm, nid):
    return "its header holds %.1fB parameters, not %s's %.1fB" % (
        inv.n_params / 1e9, nid, card_params(qm, nid) / 1e9)


def rescale(inv, size, source):
    """Another quant of the same model, from one header: the same tensors,
    each scaled to the file. Close enough to rank on."""
    ratio = size / max(1, inv.file_size)
    tensors = []
    for t in inv.tensors:
        c = gguf.Tensor(t.name, t.dims, t.type, t.offset, t.shard)
        c.nbytes = int(t.nbytes * ratio)
        tensors.append(c)
    out = gguf.Inventory(source, inv.meta, tensors,
                         [int(s * ratio) for s in inv.file_sizes],
                         inv.data_starts)
    out.warnings = list(inv.warnings)
    return out


# ------------------------------------------------------------- lineage --

def link_local(inv, nodes):
    """(identity, parent, method) for a local GGUF against the catalog:
    the repo it names as its source; else its general.base_model keys,
    a tracked model of the same arch and size, or its name."""
    meta = inv.meta
    for key in ("general.source.huggingface.repository", "general.source.url",
                "general.url", "general.repo_url"):
        m = re.search(r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)$",
                      str(meta.get(key) or "").strip().rstrip("/"))
        if m and m.group(1) in nodes:
            return m.group(1), None, "gguf-meta"
    for i in range(int(meta.get("general.base_model.count", 0) or 0) or 4):
        m = re.search(r"huggingface\.co/([^/\s]+/[^/\s]+)",
                      str(meta.get("general.base_model.%d.repo_url" % i) or ""))
        if m and m.group(1) in nodes:
            return None, m.group(1), "gguf-meta"
    n = inv.n_params
    same = [r for r in nodes.values() if r["official"] and r["params"]
            and r["arch"] == inv.arch and abs(r["params"] - n) <= 0.02 * n]
    if same:
        return None, max(same, key=lambda r: r["downloads"] or 0)["id"], \
            "fingerprint"
    stem = re.sub(r"[^a-z0-9]", "", inv.name.lower())
    best = None
    for r in nodes.values():
        short = re.sub(r"[^a-z0-9]", "", r["id"].split("/")[-1].lower())
        if len(short) >= 5 and short in stem and (
                best is None or len(short) > len(best[1])):
            best = (r["id"], short)
    return (None, best[0], "name") if best else (None, None, None)


def lineage_words(qm, node):
    chain, seen = [], set()
    while node and node not in seen and node in qm.nodes:
        seen.add(node)
        chain.append(qm.nodes[node])
        node = qm.nodes[node]["parent"]
    if not chain:
        return "?"
    if len(chain) == 1:
        return "base" if chain[0]["official"] else "no known parent"
    hops = []
    for r in reversed(chain[:-1]):
        hops.append("merge" if r["relation"] == "merge" else
                    "post-train" if r["official"] else "FT")
    if len(hops) > 2:
        hops = ["…"] + hops[-2:]
    return " → ".join([chain[-1]["id"].split("/")[-1]] + hops)


# ------------------------------------------------------------- fitting --

def floor_ok(fit, profile):
    floor = FLOORS.get(profile)
    if not floor:
        return True
    key, limit = floor
    v = getattr(fit.speed, key)
    return v <= limit if key == "s_turn" else v >= limit


def best_fit(inv, mach, opts, profile):
    ctx = search.Context(inv, mach)
    res = search.solve(ctx, opts)
    ok = [f for f in res.everything
          if search.meets(f, opts) and floor_ok(f, profile)]
    if res.relaxed or not ok:
        reasons = []
        if res.relaxed:
            reasons.append(res.relaxed)
        eligible = [f for f in res.everything if search.meets(f, opts)]
        if eligible and not any(floor_ok(f, profile) for f in eligible):
            key, limit = FLOORS[profile]
            reasons.append("s/turn > %g s" % limit if key == "s_turn"
                           else "decode < %g t/s" % limit)
        ctx.floor_reason = "; ".join(reasons) or "misses the floors"
        return None, ctx
    return max(ok, key=lambda f: search.score(f, opts.goal)), ctx


def evaluate(c, qm, mach, opts, profile, binary):
    # everything derived goes first: a refinement that turns a sized
    # guess into a miss used to keep the guess's score, rank on it, and
    # crash reading the fit it no longer had
    c.fit = c.shape = c.q = c.why = None
    if c.repo and not same_model(c.inv, qm, c.node):
        c.reject = ("impostor", impostor(c.inv, qm, c.node))
        return
    # a preset that names its own build is judged by that build: a fork
    # made for an arch upstream lacks must not be turned away for it
    if hw.arch_supported(c.binary or binary, c.inv.arch) is False:
        c.reject = ("arch", "llama.cpp here does not load %s" % c.inv.arch)
        return
    c.fit, ctx = best_fit(c.inv, mach, opts, profile)
    c.shape = ctx.shape
    if c.fit is None:
        c.reject = ("floors", getattr(ctx, "floor_reason", "misses the floors"))
        return
    est = qm.profile(c.node, profile)
    pen = qm.penalty(c.inv.bpw, c.inv.n_params, c.fit.flags.ctk,
                     c.fit.flags.ctv)
    c.q = est.minus(pen)


# -------------------------------------------------------------- choose --

def pick_quants(rows, mach, params, dropped=None):
    """The biggest quant that fits on the card, the biggest that fits at
    all, and a ~4.5 bpw middle - the three a person would weigh."""
    rows = sorted(rows, key=lambda r: r["size"])
    original = rows
    if params:
        # F16/BF16 is for converting, not running: Q8_0 is as good, faster
        rows = [r for r in rows if r["size"] * 8 / params <= 8.6] or rows
    picks = []
    gpu = [r for r in rows if r["size"] <= mach.vram_usable * 0.92]
    if gpu:
        picks.append(gpu[-1])
    picks.append(rows[-1])
    if params:
        mids = [r for r in rows if 4.3 <= r["size"] * 8 / params <= 5.2]
        if mids:
            picks.append(mids[-1])
    out, seen = [], set()
    for r in sorted(picks, key=lambda r: -r["size"]):
        if r["file"] not in seen:
            seen.add(r["file"])
            out.append(r)
    out = out[:QUANTS_PER_MODEL]
    if dropped is not None:
        for r in original:
            if r not in out:
                message = ("above the 8.6 bpw cap" if r not in rows else
                           "not among the %d representative quants kept"
                           % QUANTS_PER_MODEL)
                dropped.append((r["node"], r, "quant-cap", message))
    return out


def catalog_cands(cat, qm, mach, profile, binary, o, notes, since=None,
                  decisions=None):
    decisions = decisions if decisions is not None else []

    def record(nid, rows, code, message):
        decisions.extend((nid, r, code, message) for r in rows)

    budget = mach.vram_usable + mach.ram_usable
    by_node = {}
    for r in cat.db.execute("SELECT * FROM ggufs"):
        if not remote.auxiliary(r["file"]):     # older snapshots carry them
            by_node.setdefault(r["node"], []).append(r)
        else:
            record(r["node"], [r], None, "auxiliary file, not a model quant")
    lic = (o.get("license") or "").lower()
    tag = (o.get("tag") or "").lower()
    dropped = {"arch": 0, "size": 0, "filter": 0}
    pool = []
    for nid, rows in by_node.items():
        node = qm.nodes.get(nid)
        if node is None:
            continue
        if since and (node["created"] or "") <= since:
            record(nid, rows, None, "outside --new")
            continue
        if (lic and lic not in (node["license"] or "").lower()) or (
                tag and tag not in (node["tags"] or "").lower()):
            dropped["filter"] += 1
            record(nid, rows, None, "excluded by --license or --tag")
            continue
        if node["arch"] and hw.arch_supported(binary, node["arch"]) is False:
            dropped["arch"] += 1
            record(nid, rows, "arch", "llama.cpp here does not load %s"
                   % node["arch"])
            continue
        record(nid, [r for r in rows if not r["size"]], None,
               "no size in the catalog")
        record(nid, [r for r in rows if r["size"] and r["size"] > budget],
               "too-big", "too big even with everything spare in RAM")
        rows = [r for r in rows if r["size"] and r["size"] <= budget]
        if not rows:
            dropped["size"] += 1
            continue
        pool.append((qm.profile(nid, profile), nid, rows))
    top = int(o.get("top", SHORTLIST))
    rated = sorted([p for p in pool if p[0].rated],
                   key=lambda p: -p[0].mean)[:top]
    unrated = sorted([p for p in pool if not p[0].rated],
                     key=lambda p: -p[0].hi)[:max(4, top // 2)]
    kept = {nid for _, nid, _ in rated + unrated}
    for _, nid, rows in pool:
        if nid not in kept:
            record(nid, rows, None, "outside the --top shortlist")
    if dropped["arch"]:
        notes.append("%d models need an architecture this llama.cpp build "
                     "does not load" % dropped["arch"])
    if dropped["size"]:
        notes.append("%d models are too big even at their smallest quant "
                     "with everything spare in RAM" % dropped["size"])
    cands = []
    for _, nid, rows in rated + unrated:
        dl = {}
        for r in rows:
            dl[r["repo"]] = max(dl.get(r["repo"], 0), r["downloads"] or 0)
        repo = max(dl, key=dl.get)
        record(nid, [r for r in rows if r["repo"] != repo], None,
               "another quant repository has more downloads")
        for r in pick_quants([r for r in rows if r["repo"] == repo], mach,
                             qm.nodes[nid]["params"], decisions):
            cands.append(Cand(nid, nid, r["quant"], r["size"], r["repo"],
                              r["file"], json.loads(r["shards"])))
    return cands


def local_cands(models, qm, decisions=None):
    """Every GGUF in models.toml, linked into the catalog where it can
    be. One with no catalog identity becomes its own node, under the
    parent it was linked to, so its lineage still lends it a prior."""
    out, links = [], {}
    for name, cfg in models.items():
        p = Path(str(cfg.get("model", "")))
        if not (p.is_file() and p.suffix.lower() == ".gguf"):
            continue
        try:
            inv = gguf.load(p)
        except (gguf.NotGGUF, gguf.Truncated, OSError, ValueError) as e:
            if decisions is not None:
                decisions.append(("local-name:" + name,
                                  {"quant": "?", "size": None,
                                   "repo": None, "file": str(p)},
                                  "header", " ".join(str(e).splitlines())))
            continue
        c = Cand(None, name, inv.quant_label, inv.file_size, local=name,
                 path=p)
        if isinstance(cfg.get("llama_server"), str):
            c.binary = cfg["llama_server"]
        c.inv, c.exact = inv, True
        c.hash = evalrun.file_hash(p)
        ident, parent, method = link_local(inv, qm.nodes)
        if ident:
            c.node = ident
        else:
            c.node = "local:" + c.hash
            qm.nodes[c.node] = {"id": c.node, "parent": parent,
                                "relation": "finetune", "official": 0,
                                "downloads": 0, "datasets": "[]",
                                "method": method}
        links[c.hash] = c.node
        out.append(c)
    return out, links


def fit_all(cands, qm, mach, opts, profile, binary, cache_only, notes,
            spin=None):
    """Headers for one quant per model (in parallel), the rest sized
    from it; then the fit engine and the quality estimate on each.
    `spin`, a progress.Spinner, is told how far the headers are."""
    first = {}
    for c in cands:
        if not c.inv:
            first.setdefault(c.node, c)
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        jobs = {pool.submit(fetch, c, cache_only): c for c in first.values()}
        for n, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            if spin:
                spin.label = ("reading model headers from the Hub: %d of %d"
                              % (n, len(jobs)))
            c = jobs[job]
            try:
                c.inv = job.result()
                c.exact = c.inv is not None
            except (remote.RemoteError, gguf.NotGGUF, gguf.Truncated,
                    ValueError, OSError) as e:
                c.reject = ("header", " ".join(str(e).splitlines()))
                failed += 1
    for c in cands:
        base = first.get(c.node)
        # a sibling sized from an impostor's header would inherit its lie
        if (c.inv is None and base and base.inv
                and same_model(base.inv, qm, c.node)):
            base = first[c.node]
            c.inv = rescale(base.inv, c.size, "hf:%s/%s" % (c.repo, c.key))
        if c.inv is None and c.reject is None:
            if base and base.inv and not same_model(base.inv, qm, c.node):
                c.reject = ("impostor", impostor(base.inv, qm, c.node))
            else:
                c.reject = (base.reject if base and base.reject else
                            ("header", "header not cached (--no-fetch)"
                             if cache_only else "no header returned"))
    if failed:
        notes.append("%d headers could not be fetched" % failed)
    if spin:
        spin.label = "fitting %d quants to this machine" % len(cands)
    for c in cands:
        if c.inv is not None and c.why is None:
            evaluate(c, qm, mach, opts, profile, binary)


def choose(cands, qm):
    """(main rows, explore rows): per model its best quant; rated ones
    ranked by expected quality, unrated ones only in explore, and only
    when their upper band beats the #1."""
    by_node = {}
    for c in cands:
        if c.rankable:
            by_node.setdefault(c.node, []).append(c)
    best = {}
    for nid, cs in by_node.items():
        # F16 over Q8_0 buys a hundredth of a point; take the faster quant
        # whenever the quality is a wash
        top = max(c.q.mean for c in cs)
        best[nid] = min([c for c in cs if c.q.mean >= top - TIE],
                        key=lambda c: c.fit.speed.s_turn)
    rows = sorted(best.values(), key=lambda c: -c.q.mean)
    main = [c for c in rows if c.q.rated]
    bar = main[0].q.mean if main else float("-inf")
    explore = sorted([c for c in rows if "local" not in c.q.kinds
                      and c.q.hi > bar and (not c.q.rated or c not in
                                            main[:ROWS])],
                     key=lambda c: -c.q.hi)
    return main, explore


def choose_quant(repo, groups, binary, profile="agent", revision="main"):
    """The quant of one repo to pull for this machine, when none was
    named: (label, why, [(label, size, verdict)]).

    The same rule find ranks by, for one model and without the catalog:
    one header is read and the other quants are sized from it; quality
    is the quant penalty alone; F16 and above are left out, as they are
    for converting, not running. The best quant that clears the
    profile's floors wins, then the fastest within TIE of it. When none
    clears them, the best that runs here at all, and the why says so.
    Raises remote.RemoteError when the header cannot be had, and
    ValueError when no quant runs here.
    """
    mach = hw.probe(binary)
    rows = sorted(((k, sum(s["size"] for s in v)) for k, v in groups.items()),
                  key=lambda r: r[1])
    mid = rows[len(rows) // 2][0]
    inv = remote.inventory(repo, mid, groups[mid], revision)
    params = inv.n_params or 0
    capped = [r for r in rows if params and r[1] * 8 / params <= 8.6] or rows
    opts = search.Options(profile)
    if mach.bench.get("threads"):
        opts.threads = int(mach.bench["threads"])
    good, runs, verdicts = [], [], []
    for key, size in capped:
        q_inv = inv if key == mid else rescale(inv, size, "hf:%s/%s" % (
            repo, key))
        res = search.solve(search.Context(q_inv, mach), opts)
        fit = res.best
        if fit is None:
            verdicts.append((key, size, "does not fit"))
            continue
        quality = -(quality_penalty(q_inv, fit))
        row = (quality, fit, key, size)
        if not res.relaxed and floor_ok(fit, profile):
            good.append(row)
            verdicts.append((key, size, "fits"))
        else:
            verdicts.append((key, size, "runs, below the %s floors"
                             % profile))
        runs.append(row)
    for key, size in rows:
        if (key, size) not in capped:
            verdicts.append((key, size, "for converting, not running"))
    pool = good or runs
    if not pool:
        raise ValueError(
            "none of its %d quants fits this machine; the smallest is %s at "
            "%.1f G" % (len(rows), _label(rows[0][0]), rows[0][1] / GiB))
    top = max(r[0] for r in pool)
    speed = ((lambda f: -f.speed.decode_d) if profile == "chat"
             else (lambda f: f.speed.s_turn))
    quality, fit, key, size = min([r for r in pool if r[0] >= top - TIE],
                                  key=lambda r: speed(r[1]))
    if good:
        why = "the best quant that clears the %s floors here" % profile
    else:
        why = ("none clears the %s floors here, so the best that runs"
               % profile)
    verdicts.sort(key=lambda v: v[1])
    return key, why, verdicts


def quality_penalty(inv, fit):
    """Points a quant and its KV types cost, as find charges them before
    local evals have taught it better."""
    return (quality.quant_penalty(inv.bpw, inv.n_params)
            + quality.kv_penalty(fit.flags.ctk, fit.flags.ctv))


def _label(key):
    q = catalog.quant_of(key)
    return q if q != "?" else Path(key).name


def refine(rows, qm, mach, opts, profile, binary, cache_only):
    """The rows about to be shown get their own header, if they were
    sized from a sibling's. Each is tried once, so a header that cannot
    be had does not stop the rows from settling."""
    todo = [c for c in rows if not c.exact and c.repo and not c.refined]
    for c in todo:
        c.refined = True
        try:
            inv = fetch(c, cache_only)
        except (remote.RemoteError, gguf.NotGGUF, gguf.Truncated,
                ValueError, OSError) as e:
            c.header_error = " ".join(str(e).splitlines())
            continue
        if inv is None:
            c.header_error = "header not cached (--no-fetch)"
        if inv is not None:
            c.inv, c.exact = inv, True
            evaluate(c, qm, mach, opts, profile, binary)
    return bool(todo)


# ---------------------------------------------------------------- show --

def config_words(c):
    f = c.fit.flags
    return "%s · %dk" % (search.offload_label(f, c.shape), f.ctx // K)


def eval_minutes(c, mach, items):
    return evalrun.minutes(evalrun.estimate(items, c.shape, c.fit.flags,
                                            mach, None, False))


def get_words(c):
    """The command that gets a row running: it is here, or pull it."""
    if c.local:
        return "mdl run %s" % c.local
    return "mdl pull %s:%s --run" % (c.repo, c.key)


def get(c, run):
    """--pull N / --run N: the row's model fetched with a fitted preset,
    and started if asked. Returns the preset's name."""
    import mdl

    from . import pull
    if c.local:
        if run:
            mdl.cmd_run([c.local])
        else:
            print("%s is already here; run it with: mdl run %s"
                  % (c.local, c.local))
        return c.local
    try:
        return pull.main(["%s:%s" % (c.repo, c.key)] + (["--run"] if run
                                                         else []))
    except pull.PullError as e:
        die(str(e))


def next_step(c):
    if c.local:
        return "mdl eval %s" % c.local
    name = re.sub(r"[^a-z0-9-]+", "-", c.node.split("/")[-1].lower()).strip("-")
    # pull fetches it and writes a preset fitted to this machine
    return "mdl pull %s:%s --name %s && mdl eval %s" % (c.repo, c.key, name,
                                                         name)


def clip(s, n):
    return s if len(s) <= n else s[:n - 1] + "…"


def label_words(c, n):
    """The repo name, dropping the org before cutting into the name."""
    return clip(c.label if len(c.label) <= n else c.label.split("/")[-1], n)


def show(main, explore, qm, mach, opts, profile, meta, notes, w, items):
    floor = FLOORS.get(profile)
    floors = ["ctx ≥ %dk" % (opts.min_ctx // K)] if opts.min_ctx else []
    if opts.min_tps:
        floors.append("decode ≥ %g t/s" % opts.min_tps)
    if floor:
        floors.append("s/turn ≤ %.0f s" % floor[1] if floor[0] == "s_turn"
                      else "decode ≥ %.0f t/s" % floor[1])
    cat = ("catalog %s (%s models)" % (meta.get("built_at", "?")[:10],
                                       "{:,}".format(meta.get("nodes", 0)))
           if meta else "no catalog: models.toml only")
    w("machine  %s · %s\n" % (mach.summary(), cat))
    w("floors   %s\n" % " · ".join(floors))
    weights = quality.WEIGHTS.get(profile, quality.WEIGHTS["agent"])
    w("score    %s, on the %s scale%s\n" % (
        " + ".join("%s %d%%" % (d, round(x * 100)) for d, x in
                   weights.items()), qm.scale_name,
        "" if qm.map else " (local once 3 models have evals)"))
    for n in notes:
        w("note     %s\n" % n)
    w("\n")
    if not main:
        w("nothing rated clears the floors here%s\n" % (
            "; see what is worth testing below" if explore else ""))
    else:
        speed = "decode" if profile == "chat" else "s/turn"
        w(" #  %-30s %-30s %-11s %-17s %-7s %-8s %s\n" % (
            "model", "lineage", "quant", "config", speed, "score",
            "evidence"))
        for i, c in enumerate(main[:ROWS], 1):
            sp = ("%.0f t/s" % c.fit.speed.decode_d if profile == "chat"
                  else "%.0f s" % c.fit.speed.s_turn)
            w(" %d  %-30s %-30s %-11s %-17s %-7s %-8s %s%s\n" % (
                i, label_words(c, 30), clip(lineage_words(qm, c.node), 30),
                c.quant[:11], config_words(c), sp,
                "%.0f ±%.0f" % (c.q.mean, 1.64 * c.q.sd), c.q.evidence(),
                "" if c.exact else " (sized)"))
        w("\nget #1   %s\n" % get_words(main[0]))
        if len(main[:ROWS]) > 1:
            w("         or another row: mdl find --run N\n")
        flagged = [(c, f) for c in main[:ROWS] for f in c.q.flags]
        if flagged:
            w("\n")
        for c, f in flagged:
            w(" ⚑ %s %s → down-weighted\n" % (c.label, f))
    if explore:
        w("\nworth testing\n")
        for c in explore[:5]:
            w("    %-30s %-28s est. %.0f–%.0f  %-14s ~%s → %s\n" % (
                label_words(c, 30), clip(lineage_words(qm, c.node), 28),
                c.q.mean - 1.64 * c.q.sd, c.q.hi,
                "no evals" if not c.q.rated else c.q.evidence(),
                eval_minutes(c, mach, items), next_step(c)))


def as_json(main, explore, qm, profile):
    def one(c):
        f = c.fit.flags
        return {"model": c.label, "node": c.node, "spec": c.spec,
                "quant": c.quant, "lineage": lineage_words(qm, c.node),
                "flags": f.as_dict(), "gpu": c.fit.gpu,
                "s_turn": c.fit.speed.s_turn,
                "decode": c.fit.speed.decode_d, "score": c.q.mean,
                "sd": c.q.sd, "evidence": sorted(c.q.kinds),
                "warnings": c.q.flags, "exact": c.exact,
                # what mdl pull and the web UI need: the repo, the file
                # and its size, and the context it would run at
                "repo": c.repo, "file": c.key, "size": c.size,
                "ctx": c.fit.flags.ctx}
    return {"profile": profile, "scale": qm.scale_name,
            "main": [one(c) for c in main[:ROWS]],
            "explore": [one(c) for c in explore[:5]]}


# ----------------------------------------------------------------- why --

def match_model(query, qm, cands, models):
    """Exact names win; aliases of one node are not ambiguous matches."""
    names = {nid: nid for nid in qm.nodes}
    names.update({name: "local-name:" + name for name in models})
    names.update({c.local: c.node for c in cands if c.local})
    if query in names:
        return names[query]
    matches = sorted(name for name in names if query.casefold() in
                     name.casefold())
    nodes = {names[name] for name in matches}
    if not nodes:
        die("no model matches %r" % query)
    if len(nodes) > 1:
        die("ambiguous model %r: %s" % (query, ", ".join(matches[:5])))
    return nodes.pop()


def explain(node, cands, decisions, main, explore, qm, profile, cat=None):
    """One explanation shared by text and JSON; never rerank or refit."""
    cs = [c for c in cands if c.node == node]
    shown = main[:ROWS] + explore[:5]
    selected = next((c for c in shown if c.node == node), None)
    status, rank = "not shown", None
    if selected is not None:
        table = main[:ROWS] if selected in main[:ROWS] else explore[:5]
        status = "main table" if selected in main[:ROWS] else "worth testing"
        rank = table.index(selected) + 1
    est = qm.profile(node, profile)
    details = qm.evidence_details(node, profile, cat)
    details.update({"heuristic": True, "mean": est.mean,
                    "lo": est.mean - 1.64 * est.sd, "hi": est.hi,
                    "rated": est.rated, "kinds": sorted(est.kinds),
                    "flags": est.flags})
    quants = []
    for c in cs:
        row = {"quant": c.quant, "size": c.size, "repo": c.repo,
               "file": c.key or (str(c.path) if c.path else None),
               "local": c.local, "header": "exact" if c.exact else
               "sized from a sibling" if c.inv else "unavailable",
               "header_error": c.header_error, "reject":
               {"code": c.reject[0], "message": c.reject[1]}
               if c.reject else None, "fit": None, "penalty": None,
               "quality": None, "selection": [], "beaten_by": []}
        if c.rankable:
            f = c.fit.flags
            row["fit"] = {"ctx": f.ctx, "decode": c.fit.speed.decode_d,
                          "s_turn": c.fit.speed.s_turn,
                          "kv": "%s/%s" % (f.ctk, f.ctv)}
            row["penalty"] = qm.penalty(c.inv.bpw, c.inv.n_params,
                                         f.ctk, f.ctv)
            row["quality"] = {"mean": c.q.mean,
                              "lo": c.q.mean - 1.64 * c.q.sd,
                              "hi": c.q.hi}
            siblings = [x for x in cs if x.rankable]
            top = max(x.q.mean for x in siblings)
            winner = min([x for x in siblings if x.q.mean >= top - TIE],
                         key=lambda x: x.fit.speed.s_turn)
            if winner is not c:
                row["selection"].append(
                    "lost to %s of itself: faster quant within %.1f points "
                    "(TIE)" % (winner.quant, TIE) if c.q.mean >= top - TIE
                    else "lost to %s of itself on quality" % winner.quant)
            if c not in shown:
                competitors = main[:ROWS] if c.q.rated else explore[:5]
                for other in competitors:
                    if other.node == node:
                        continue
                    metric = "mean" if c.q.rated else "hi"
                    gap = getattr(other.q, metric) - getattr(c.q, metric)
                    if gap >= 0:
                        row["beaten_by"].append({"node": other.node,
                                                 "quant": other.quant,
                                                 "by": gap,
                                                 "metric": metric})
                if not c.q.rated and main and c.q.hi <= main[0].q.mean:
                    row["selection"].append(
                        "upper band does not beat %s's mean (%.2f points)"
                        % (main[0].node, main[0].q.mean - c.q.hi))
                if "local" in c.q.kinds and c not in main[:ROWS]:
                    row["selection"].append(
                        "already locally evaluated; outside worth testing")
        quants.append(row)
    for nid, r, code, message in decisions:
        if nid == node:
            quants.append({"quant": r["quant"], "size": r["size"],
                           "repo": r["repo"], "file": r["file"],
                           "local": None, "header": "not fetched",
                           "header_error": None, "fit": None,
                           "penalty": None, "quality": None,
                           "reject": {"code": code, "message": message}
                           if code else None,
                           "selection": [] if code else [message],
                           "beaten_by": []})
    return {"node": node, "lineage": lineage_words(qm, node),
            "profile": profile, "scale": qm.scale_name, "status": status,
            "rank": rank, "quality": details, "quants": quants}


def show_why(data, w):
    # a model with no catalog identity is a hash; the name it goes by in
    # models.toml is what anyone asking about it typed
    names = sorted({r["local"] for r in data["quants"] if r.get("local")})
    label = data["node"]
    if names and label.startswith("local:"):
        label = "%s (%s)" % (", ".join(names), label)
    w("%s · %s\n" % (label, data["lineage"]))
    w("%s%s\n" % (data["status"], " · rank #%d" % data["rank"]
                   if data["rank"] is not None else ""))
    if not data["quants"]:
        w("no eligible GGUFs considered under these options\n")
    for r in data["quants"]:
        size = "%.2f GiB" % (r["size"] / GiB) if r["size"] else "size unknown"
        words = []
        if r["reject"]:
            words.append("%s: %s" % (r["reject"]["code"],
                                     r["reject"]["message"]))
        if r["fit"]:
            f = r["fit"]
            speed = ("decode %.0f t/s" % f["decode"]
                     if data["profile"] == "chat" else
                     "s/turn %.0f s" % f["s_turn"])
            words.append("ctx %dk · %s · KV %s" % (f["ctx"] // K, speed,
                                                   f["kv"]))
        words.extend(r["selection"])
        source = r["local"] or r["repo"] or r["file"] or "?"
        w("  %s [%s] · %s · %s · %s\n" % (r["quant"], source,
                                      size, r["header"],
                                      "; ".join(words)))
        if r["header_error"]:
            w("    own header unavailable: %s\n" % r["header_error"])
        if r["quality"]:
            q = r["quality"]
            w("    heuristic %.2f (%.2f..%.2f); quant/KV penalty %.2f\n"
              % (q["mean"], q["lo"], q["hi"], r["penalty"]))
        for other in r["beaten_by"]:
            w("    beaten by %s %s by %.2f points (%s)\n" % (
                other["node"], other["quant"], other["by"], other["metric"]))
    q = data["quality"]
    w("quality  heuristic %.2f (%.2f..%.2f), before quant/KV penalty; "
      "%s · %s scale · %s\n" % (q["mean"], q["lo"], q["hi"],
                                "rated" if q["rated"] else "unrated",
                                data["scale"], ", ".join(q["kinds"])))
    # a parent's estimate is always blended in as a prior; it is only
    # "inherited" when the model has no evidence of its own to outweigh it
    own = set(q["kinds"]) & {"verified", "reported", "local"}
    chain = " → ".join(q["parents"])
    w("parent   %s\n" % ("none" if not q["parents"] else
                           chain + " blended in as a prior" if own else
                           "estimate inherited from " + chain))
    for b in q["benchmarks"]:
        w("  %s · %s = %g · %s\n" % (b["node"], b["benchmark"], b["value"],
                                      "verified" if b["verified"] else
                                      "self-reported"))
    for flag in q["flags"]:
        w(" ⚑ %s %s → down-weighted\n" % (data["node"], flag))


# ---------------------------------------------------------------- main --

def parse(args):
    o, i = {}, 0
    values = {"--profile", "--top", "--license", "--tag", "--kv-floor",
              "--catalog", "--why", "--pull", "--run"}
    flags = {"--new", "--no-fetch", "--json"}
    while i < len(args):
        a = args[i]
        if a in values:
            if i + 1 >= len(args):
                die("%s needs a value" % a)
            o[a[2:]] = args[i + 1]
            i += 2
        elif a in flags:
            o[a[2:]] = True
            i += 1
        else:
            die("unknown argument %s\n%s" % (a, USAGE))
    if "top" in o:
        if not re.fullmatch(r"[1-9]\d*", o["top"]):
            die("--top takes a number of models, 1 or more")
        o["top"] = int(o["top"])
    for key in ("pull", "run"):
        if key in o:
            if not re.fullmatch(r"[1-9]\d*", o[key]):
                die("--%s takes a row number from the table, 1 or more" % key)
            o[key] = int(o[key])
    if "pull" in o and "run" in o:
        die("--pull and --run both name a row; pick one")
    return o


def seen_path():
    return hw.config_dir() / "find.json"


def main(args, out=None):
    out = out or sys.stdout
    w = out.write
    if args and args[0] in ("-h", "--help"):
        w(USAGE)
        return None
    o = parse(args)
    import mdl
    try:
        models, binary = mdl.load_config(missing_ok=True)
    except mdl.MdlError:
        models, binary = {}, "llama-server"
    profile = o.get("profile", "agent")
    try:
        opts = search.Options(profile, kv_floor=o.get("kv-floor", "q8_0"))
    except ValueError as e:
        die(str(e))
    mach = hw.probe(binary)
    if mach.bench.get("threads"):
        opts.threads = int(mach.bench["threads"])
    notes, cat, meta = [], None, {}
    if not o.get("catalog") and not o.get("no-fetch"):
        # without the catalog there is nothing to find but models.toml,
        # which on a first run is nothing at all
        try:
            catalog.ensure(say=lambda line: sys.stderr.write(line + "\n"),
                           bar=True)
        except catalog.CatalogError as e:
            notes.append("could not fetch the catalog: %s" % e)
    try:
        cat = catalog.Catalog(o.get("catalog"))
        meta = cat.meta()
        if meta.get("complete") is False:
            notes.append("catalog crawl " + catalog.progress(meta))
    except catalog.CatalogError as e:
        notes.append(str(e))
    qm = quality.Model(cat)
    decisions = []
    locals_, links = local_cands(models, qm, decisions)
    qm.add_local(evalrun.load(), links)
    since = None
    if o.get("new"):
        try:
            since = json.loads(seen_path().read_text()).get("seen")
        except (OSError, ValueError):
            since = ""
    cands = [] if o.get("new") and cat is None else list(
        [] if o.get("new") else locals_)
    if cat is not None:
        cands += catalog_cands(cat, qm, mach, profile, binary, o, notes,
                               since, decisions)
    # the first run reads a header per model from the Hub, which is the
    # slow part; the spinner says how far it is, on stderr
    spin = progress.Spinner("sizing %d models against this machine" % len(
        {c.node for c in cands}), line=progress.Line()
    ) if sys.stderr.isatty() else None
    if spin:
        spin.start()
    try:
        fit_all(cands, qm, mach, opts, profile, binary, o.get("no-fetch"),
                notes, spin)
        main_rows, explore = choose(cands, qm)
        # a row that fails on its own header drops out and lets another
        # in, which may itself only be sized from a sibling - so until it
        # settles
        if spin:
            spin.label = "reading the headers of the rows to show"
        while refine(main_rows[:ROWS] + explore[:5], qm, mach, opts, profile,
                     binary, o.get("no-fetch")):
            main_rows, explore = choose(cands, qm)
    finally:
        if spin:
            spin.stop()
    if o.get("new"):
        seen_path().parent.mkdir(parents=True, exist_ok=True)
        mdl.write_atomic(seen_path(), json.dumps({"seen": time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.gmtime())}))
    if "why" in o:
        node = match_model(o["why"], qm, locals_, models)
        data = explain(node, cands, decisions, main_rows, explore, qm,
                       profile, cat)
        if o.get("json"):
            w(json.dumps(data, indent=1) + "\n")
        else:
            show_why(data, w)
        return main_rows, explore
    if o.get("json"):
        w(json.dumps(as_json(main_rows, explore, qm, profile), indent=1)
          + "\n")
        return main_rows, explore
    row = o.get("pull") or o.get("run")
    if row:
        if row > len(main_rows[:ROWS]):
            die("the table has %d row%s; --%s %d is not one of them" % (
                len(main_rows[:ROWS]), "" if len(main_rows[:ROWS]) == 1
                else "s", "run" if o.get("run") else "pull", row))
        c = main_rows[row - 1]
        w("#%d  %s  %s  %s\n" % (row, c.label, c.quant, config_words(c)))
        get(c, bool(o.get("run")))
        return main_rows, explore
    items = evalsuite.build(evalsuite.SUITES, evalsuite.secret())
    show(main_rows, explore, qm, mach, opts, profile, meta, notes, w, items)
    return main_rows, explore
