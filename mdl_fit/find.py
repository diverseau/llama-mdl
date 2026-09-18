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

from . import (catalog, evalrun, evalsuite, gguf, hw, quality, remote,
               search)

# Floors on top of the profile's own: an agent turn slower than this
# is not an agent you will use; a chat below this is not a chat.
FLOORS = {"agent": ("s_turn", 90.0), "chat": ("decode_d", 20.0)}
SHORTLIST = 24
QUANTS_PER_MODEL = 3
WORKERS = 6
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
        self.inv = self.fit = self.shape = self.q = None
        self.exact = self.refined = False
        self.why = None
        self.hash = None

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
        return None, ctx
    return max(ok, key=lambda f: search.score(f, opts.goal)), ctx


def evaluate(c, qm, mach, opts, profile, binary):
    # everything derived goes first: a refinement that turns a sized
    # guess into a miss used to keep the guess's score, rank on it, and
    # crash reading the fit it no longer had
    c.fit = c.shape = c.q = c.why = None
    if c.repo and not same_model(c.inv, qm, c.node):
        c.why = impostor(c.inv, qm, c.node)
        return
    if hw.arch_supported(binary, c.inv.arch) is False:
        c.why = "llama.cpp here does not load %s" % c.inv.arch
        return
    c.fit, ctx = best_fit(c.inv, mach, opts, profile)
    c.shape = ctx.shape
    if c.fit is None:
        c.why = "misses the floors"
        return
    est = qm.profile(c.node, profile)
    pen = qm.penalty(c.inv.bpw, c.inv.n_params, c.fit.flags.ctk,
                     c.fit.flags.ctv)
    c.q = est.minus(pen)


# -------------------------------------------------------------- choose --

def pick_quants(rows, mach, params):
    """The biggest quant that fits on the card, the biggest that fits at
    all, and a ~4.5 bpw middle - the three a person would weigh."""
    rows = sorted(rows, key=lambda r: r["size"])
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
    return out[:QUANTS_PER_MODEL]


def catalog_cands(cat, qm, mach, profile, binary, o, notes, since=None):
    budget = mach.vram_usable + mach.ram_usable
    by_node = {}
    for r in cat.db.execute("SELECT * FROM ggufs"):
        if not remote.auxiliary(r["file"]):     # older snapshots carry them
            by_node.setdefault(r["node"], []).append(r)
    lic = (o.get("license") or "").lower()
    tag = (o.get("tag") or "").lower()
    dropped = {"arch": 0, "size": 0, "filter": 0}
    pool = []
    for nid, rows in by_node.items():
        node = qm.nodes.get(nid)
        if node is None:
            continue
        if since and (node["created"] or "") <= since:
            continue
        if (lic and lic not in (node["license"] or "").lower()) or (
                tag and tag not in (node["tags"] or "").lower()):
            dropped["filter"] += 1
            continue
        if node["arch"] and hw.arch_supported(binary, node["arch"]) is False:
            dropped["arch"] += 1
            continue
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
        for r in pick_quants([r for r in rows if r["repo"] == repo], mach,
                             qm.nodes[nid]["params"]):
            cands.append(Cand(nid, nid, r["quant"], r["size"], r["repo"],
                              r["file"], json.loads(r["shards"])))
    return cands


def local_cands(models, qm):
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
        except (gguf.NotGGUF, OSError, ValueError):
            continue
        c = Cand(None, name, inv.quant_label, inv.file_size, local=name,
                 path=p)
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


def fit_all(cands, qm, mach, opts, profile, binary, cache_only, notes):
    """Headers for one quant per model (in parallel), the rest sized
    from it; then the fit engine and the quality estimate on each."""
    first = {}
    for c in cands:
        if not c.inv:
            first.setdefault(c.node, c)
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        jobs = {pool.submit(fetch, c, cache_only): c for c in first.values()}
        for job in concurrent.futures.as_completed(jobs):
            c = jobs[job]
            try:
                c.inv = job.result()
                c.exact = c.inv is not None
            except (remote.RemoteError, gguf.NotGGUF, gguf.Truncated,
                    ValueError, OSError) as e:
                c.why = str(e)
                failed += 1
    for c in cands:
        base = first.get(c.node)
        # a sibling sized from an impostor's header would inherit its lie
        if (c.inv is None and base and base.inv
                and same_model(base.inv, qm, c.node)):
            base = first[c.node]
            c.inv = rescale(base.inv, c.size, "hf:%s/%s" % (c.repo, c.key))
    if failed:
        notes.append("%d headers could not be fetched" % failed)
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
                ValueError, OSError):
            continue
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


def next_step(c):
    if c.local:
        return "mdl eval %s" % c.local
    name = re.sub(r"[^a-z0-9-]+", "-", c.node.split("/")[-1].lower()).strip("-")
    return "mdl fit hf:%s:%s --write %s && mdl eval %s" % (
        c.repo, c.quant, name, name)


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
                "warnings": c.q.flags, "exact": c.exact}
    return {"profile": profile, "scale": qm.scale_name,
            "main": [one(c) for c in main[:ROWS]],
            "explore": [one(c) for c in explore[:5]]}


# ---------------------------------------------------------------- main --

def parse(args):
    o, i = {}, 0
    values = {"--profile", "--top", "--license", "--tag", "--kv-floor",
              "--catalog"}
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
        models, binary = mdl.load_config()
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
    try:
        cat = catalog.Catalog(o.get("catalog"))
        meta = cat.meta()
        if meta.get("complete") is False:
            notes.append("catalog crawl " + catalog.progress(meta))
    except catalog.CatalogError as e:
        notes.append(str(e))
    qm = quality.Model(cat)
    locals_, links = local_cands(models, qm)
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
                               since)
    fit_all(cands, qm, mach, opts, profile, binary, o.get("no-fetch"), notes)
    main_rows, explore = choose(cands, qm)
    # a row that fails on its own header drops out and lets another in,
    # which may itself only be sized from a sibling - so until it settles
    while refine(main_rows[:ROWS] + explore[:5], qm, mach, opts, profile,
                 binary, o.get("no-fetch")):
        main_rows, explore = choose(cands, qm)
    if o.get("new"):
        seen_path().parent.mkdir(parents=True, exist_ok=True)
        seen_path().write_text(json.dumps({"seen": time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.gmtime())}))
    if o.get("json"):
        w(json.dumps(as_json(main_rows, explore, qm, profile), indent=1)
          + "\n")
        return main_rows, explore
    items = evalsuite.build(evalsuite.SUITES, evalsuite.secret())
    show(main_rows, explore, qm, mach, opts, profile, meta, notes, w, items)
    return main_rows, explore
