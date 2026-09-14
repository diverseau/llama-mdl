"""Quality estimates for mdl find: what a (model, quant, KV type) is
likely to score, per domain, as a value and a band.

Evidence is combined by inverse-variance weighting, strongest first:

  local     your own `mdl eval`, at the exact file, quant and KV type
  verified  public results Hugging Face marks as verified
  reported  self-reported results (model cards, .eval_results files) -
            discounted, and discounted hard when flagged
  lineage   a fine-tune starts from its parent's estimate, wider
  traction  downloads against its siblings; it only moves models with
            nothing else to go on

Public numbers are put on a 'public scale', 50 + 15 z, z taken against
every model in the catalog that reports that benchmark. Once three
models have both local results and public ones, a per-domain linear map
puts public estimates on your local scale (0-100, from mdl eval); until
then local results are read as public-scale numbers with a wider band,
and `mdl find` says which scale it is on.

Two flags down-weight a public number (its band triples). Benchmaxxing:
a score far above what the model's other scores predict, from a
leave-one-out regression fitted across the catalog. Contamination: the
card lists a training set tied to a benchmark it reports.
"""

import json
import math
import re
import statistics
import tomllib

from . import hw

DOMAINS = ("coding", "agentic", "long-context", "general", "reasoning")
WEIGHTS = {"agent": {"agentic": 0.4, "coding": 0.35, "long-context": 0.25},
           "chat": {"general": 0.5, "reasoning": 0.25, "coding": 0.25},
           "max-ctx": {"long-context": 0.5, "general": 0.5},
           "speed": {"general": 0.4, "coding": 0.3, "reasoning": 0.3}}
# benchmark id (lower case) -> domains. A benchmark nothing matches is
# not counted: a number nobody can place is not evidence.
BENCH = [
    (r"swe-?bench|terminal-?bench|tau2?-?bench|bfcl|toolbench|nexus|agent"
     r"|toolcall|function.?call", ("agentic",)),
    (r"swe-?bench|humaneval|mbpp|livecodebench|\blcb\b|bigcodebench|aider"
     r"|codeforces|evalplus|\bcode", ("coding",)),
    (r"gpqa|aime|hmmt|\bmath|gsm8k|\bbbh|musr|arc[-_ ]?c|logic|reason",
     ("reasoning",)),
    (r"ruler|longbench|helmet|niah|needle|loong|mrcr|long.?context",
     ("long-context",)),
    (r"mmlu|ifeval|ifstruct|ifbench|arena|alpaca|mt-?bench|simpleqa"
     r"|truthful|\bbbq|global", ("general",)),
]
# Saturated: every model scores near the top, so a gap means nothing.
STALE = r"hellaswag|winogrande|arc[-_ ]?easy|piqa|boolq|\bgsm8k"
BLOCK = ["deepswe"]
# benchmark -> training-set names that would teach to it
TIED = {"gsm8k": r"gsm8k", "humaneval": r"humaneval", "mbpp": r"mbpp",
        "math": r"competition_math|hendrycks.?math|/math\b",
        "mmlu": r"mmlu", "swe-bench": r"swe-?bench", "aime": r"aime",
        "gpqa": r"gpqa", "ifeval": r"ifeval", "livecodebench": r"livecodebench"}
VAR = {"verified": 8 ** 2, "reported": 15 ** 2, "public_floor": 6 ** 2,
       "local_floor": 3 ** 2, "unmapped_local": 12 ** 2,
       "hop_official": 8 ** 2, "hop_finetune": 10 ** 2, "hop_merge": 12 ** 2,
       "unknown": 25 ** 2}
FLAG_SD = 3.0
MIN_BENCH_N = 5            # below this a benchmark is read as 50 +- 20
QUANT_SCALE = 25.0         # quant penalty, points, before local evals move it
KV_PENALTY = {"k": {"q8_0": 0.3, "q5_1": 0.8, "q5_0": 0.8, "iq4_nl": 1.8,
                    "q4_1": 2.0, "q4_0": 2.0},
              "v": {"q8_0": 0.1, "q5_1": 0.3, "q5_0": 0.3, "iq4_nl": 0.7,
                    "q4_1": 0.8, "q4_0": 0.8}}


def settings():
    """~/.config/mdl/find.toml: bench_allow, bench_block, bench_domain
    (a table of benchmark -> [domains]), and stale."""
    cfg = {"allow": [], "block": list(BLOCK), "domain": {}, "stale": STALE}
    try:
        data = tomllib.loads((hw.config_dir() / "find.toml").read_text(
            encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return cfg
    cfg["allow"] = [str(x).lower() for x in data.get("bench_allow", [])]
    cfg["block"] = [str(x).lower() for x in data.get("bench_block", BLOCK)]
    cfg["domain"] = {str(k).lower(): list(v) for k, v in
                     (data.get("bench_domain") or {}).items()}
    if data.get("stale"):
        cfg["stale"] = "|".join(re.escape(str(s).lower())
                                for s in data["stale"])
    return cfg


def domains_of(bench, cfg=None):
    cfg = cfg or {"allow": [], "block": BLOCK, "domain": {}, "stale": STALE}
    b = bench.lower()
    if any(x in b for x in cfg["block"]):
        return ()
    for key, doms in cfg["domain"].items():
        if key in b:
            return tuple(doms)
    allowed = any(a in b for a in cfg["allow"])
    if re.search(cfg["stale"], b) and not allowed:
        return ()
    out = set()
    for pat, doms in BENCH:
        if re.search(pat, b):
            out.update(doms)
    return tuple(sorted(out))


def quant_penalty(bpw, params, scale=QUANT_SCALE):
    """Points lost to quantization: steep below ~4 bpw, gentler for big
    models, which tolerate low bpw better."""
    if not bpw:
        return 0.0
    billions = params / 1e9 if params else 8.0
    return scale * math.exp(-(bpw - 2.0) / 0.9) * (
        8.0 / max(billions, 0.5)) ** 0.3


def kv_penalty(ctk, ctv):
    return KV_PENALTY["k"].get(ctk, 0.0) + KV_PENALTY["v"].get(ctv, 0.0)


class Est:
    __slots__ = ("mean", "var", "kinds", "flags")

    def __init__(self, mean, var, kinds=(), flags=()):
        self.mean, self.var = mean, var
        self.kinds, self.flags = set(kinds), list(flags)

    @property
    def sd(self):
        return math.sqrt(self.var)

    @property
    def hi(self):
        return self.mean + 1.64 * self.sd

    @property
    def rated(self):
        return bool(self.kinds & {"local", "verified", "reported"})

    def minus(self, points):
        return Est(self.mean - points, self.var, self.kinds, self.flags)

    def evidence(self):
        for kind, words in (("local", "local eval ✓"),
                            ("verified", "verified benchmarks"),
                            ("reported", "self-reported"),
                            ("lineage", "lineage only"),
                            ("none", "nothing yet")):
            if kind in self.kinds:
                return words
        return "?"


def combine(items):
    """Inverse-variance mean of (mean, var) pairs."""
    w = sum(1.0 / v for _, v in items)
    return sum(m / v for m, v in items) / w, 1.0 / w


def _ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-9:
        return 0.0, my
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    return slope, my - slope * mx


class Model:
    """Every estimate for one catalog and one set of local results."""

    def __init__(self, cat=None, cfg=None):
        self.cfg = cfg or settings()
        self.nodes = {r["id"]: r for r in cat.nodes()} if cat else {}
        self.public = {}                  # node -> domain -> [(pub, var, ...)]
        self.flags = {}                   # node -> [words]
        self.local = {}                   # node -> domain -> [(score, var)]
        self.map = {}                     # domain -> (a, b, residual var)
        self.scale = QUANT_SCALE
        self.memo = {}
        self._public(cat.all_evals() if cat else [])
        self.siblings = {}
        for r in self.nodes.values():
            self.siblings.setdefault(r["parent"], []).append(
                r["downloads"] or 0)

    # -- public --
    def _public(self, rows):
        by_bench, claimed = {}, {}
        for r in rows:
            v = r["value"]
            claimed.setdefault(r["node"], set()).add(r["benchmark"].lower())
            if v is None or not domains_of(r["benchmark"], self.cfg):
                continue
            v = v * 100 if 0 <= v <= 1.0 else v
            by_bench.setdefault(r["benchmark"].lower(), {}).setdefault(
                r["node"], []).append((v, r["verified"]))
        z = {}
        for b, per in by_bench.items():
            vals = {n: (statistics.mean(v for v, _ in lst),
                        max(ver for _, ver in lst)) for n, lst in per.items()}
            xs = [v for v, _ in vals.values()]
            if len(xs) >= MIN_BENCH_N:
                mu, sd = statistics.mean(xs), max(statistics.pstdev(xs), 1.0)
            else:
                mu, sd = 50.0, 20.0
            for n, (v, ver) in vals.items():
                z.setdefault(n, {})[b] = ((v - mu) / sd, ver)
        flagged = self._benchmaxx(z)
        # a claim is suspect even on a benchmark that does not count here
        for n, bz in claimed.items():
            try:
                sets = [d.lower() for d in json.loads(
                    (self.nodes.get(n) or {"datasets": "[]"})["datasets"]
                    or "[]")]
            except (ValueError, TypeError):
                sets = []
            for b in bz:
                for key, pat in TIED.items():
                    if key in b and any(re.search(pat, d) for d in sets):
                        flagged[(n, b)] = ("trained on data tied to %s, which "
                                           "it reports" % b)
        for (n, _b), words in sorted(flagged.items()):
            self.flags.setdefault(n, []).append(words)
        for n, bz in z.items():
            for b, (zz, ver) in bz.items():
                var = VAR["verified" if ver else "reported"]
                if (n, b) in flagged:
                    var *= FLAG_SD ** 2
                for d in domains_of(b, self.cfg):
                    self.public.setdefault(n, {}).setdefault(d, []).append(
                        (50 + 15 * zz, var, ver))

    @staticmethod
    def _benchmaxx(z):
        """{(node, bench): words} for scores far above what the model's
        other scores predict. Per benchmark, a regression of its z on the
        mean of the model's other z's, fitted across the catalog."""
        pairs = {}
        for n, bz in z.items():
            if len(bz) < 3:
                continue
            for b, (zb, _) in bz.items():
                others = [v for k, (v, _) in bz.items() if k != b]
                pairs.setdefault(b, []).append((n, sum(others) / len(others),
                                                zb))
        flagged = {}
        for b, ps in pairs.items():
            for n, x, y in ps:
                # fitted without the model under test, or a spike drags
                # the line up to meet itself
                rest = [(x2, y2) for n2, x2, y2 in ps if n2 != n]
                if len(rest) >= 4:
                    slope, icpt = _ols([r[0] for r in rest],
                                       [r[1] for r in rest])
                    spread = max(statistics.pstdev(
                        [y2 - (icpt + slope * x2) for x2, y2 in rest]), 0.5)
                else:                     # a model's scores move together
                    slope, icpt, spread = 1.0, 0.0, 0.75
                pred = icpt + slope * x
                if y - pred > max(2 * spread, 1.0):
                    flagged[(n, b)] = ("claims %+.0f on %s; its other scores "
                                       "predict %+.0f" % (15 * y, b,
                                                          15 * pred))
        return flagged

    # -- local --
    def add_local(self, records, links):
        """records: evalrun results; links: {file hash: node id}. A local
        result is booked at full quality (its quant and KV penalties added
        back), so it can speak for the model at any quant."""
        # Scores only mean the same thing when the questions were the
        # same. Keep the newest generation of items and drop the rest:
        # an older run is history, not evidence about this model.
        usable = [r for r in records
                  if not r.get("partial") and r.get("domains")]
        newest = max((r.get("suite_version", 0) for r in usable), default=0)
        current = [r for r in usable if r.get("suite_version", 0) == newest]
        latest = max(current, key=lambda r: r.get("at", ""), default=None)
        want = latest.get("items_hash") if latest else None
        by_node = {}
        for rec in current:
            if rec.get("items_hash") != want:
                continue
            node = links.get(rec.get("hash")) or "local:" + str(rec.get("hash"))
            back = self.penalty(rec.get("bpw"), rec.get("params"),
                                *(rec.get("kv", "f16/f16").split("/") + [""])[:2])
            for d, v in rec["domains"].items():
                half = (v["hi"] - v["lo"]) / 2 * 100
                var = max((half / 1.96) ** 2, VAR["local_floor"])
                self.local.setdefault(node, {}).setdefault(d, []).append(
                    (100 * v["score"] + back, var))
                by_node.setdefault((node, d), []).append(
                    (rec.get("bpw"), rec.get("params"), 100 * v["score"]))
        self._learn_scale(by_node)
        self._fit_map()
        self.memo.clear()

    def _learn_scale(self, by_node):
        """The quant curve's height, from the same model evaluated at two
        quants; shrunk towards the default so one pair cannot swing it."""
        num, den = QUANT_SCALE * 4.0, 4.0
        for runs in by_node.values():
            runs = [r for r in runs if r[0]]
            for i in range(len(runs)):
                for j in range(i + 1, len(runs)):
                    (b1, p1, s1), (b2, p2, s2) = runs[i], runs[j]
                    unit = quant_penalty(b2, p2, 1.0) - quant_penalty(
                        b1, p1, 1.0)
                    if abs(unit) > 0.05:
                        num += (s1 - s2) * unit
                        den += unit * unit
        self.scale = max(2.0, min(80.0, num / den))

    def _fit_map(self):
        self.map = {}
        for d in DOMAINS:
            xs, ys = [], []
            for n, doms in self.local.items():
                if d in doms and self.public.get(n, {}).get(d):
                    pub = combine([(m, v) for m, v, _ in self.public[n][d]])[0]
                    xs.append(pub)
                    ys.append(combine(doms[d])[0])
            if len(xs) >= 3:
                slope, icpt = _ols(xs, ys)
                slope = max(slope, 0.05)
                resid = [y - (icpt + slope * x)
                         for x, y in zip(xs, ys, strict=True)]
                self.map[d] = (icpt, slope, max(statistics.pvariance(resid),
                                                VAR["local_floor"]))

    @property
    def scale_name(self):
        return "local" if self.map else "public"

    # -- estimates --
    def penalty(self, bpw, params, ctk="f16", ctv="f16"):
        return quant_penalty(bpw, params, self.scale) + kv_penalty(ctk, ctv)

    def estimate(self, node, domain, _depth=0):
        key = (node, domain)
        if key in self.memo:
            return self.memo[key]
        items, kinds = [], set()
        mapped = self.map.get(domain)
        pub = self.public.get(node, {}).get(domain, [])
        if pub:
            m, v = combine([(x, var) for x, var, _ in pub])
            if mapped:
                a, b, rv = mapped
                m, v = a + b * m, b * b * v + rv
            items.append((m, max(v, VAR["public_floor"])))
            kinds.add("verified" if any(ver for _, _, ver in pub)
                      else "reported")
        loc = self.local.get(node, {}).get(domain, [])
        if loc:
            m, v = combine(loc)
            items.append((m, max(v, VAR["local_floor"])
                          + (0 if mapped or not self.map and not pub
                             else VAR["unmapped_local"])))
            kinds.add("local")
        row = self.nodes.get(node)
        if row is not None and row["parent"] and _depth < 16:
            p = self.estimate(row["parent"], domain, _depth + 1)
            hop = VAR["hop_merge" if row["relation"] == "merge" else
                      "hop_official" if row["official"] else "hop_finetune"]
            items.append((p.mean, p.var + hop))
            if not kinds:
                kinds.add("lineage" if "none" not in p.kinds else "none")
        if not items:
            items.append((50.0, VAR["unknown"]))
            kinds.add("none")
        mean, var = combine(items)
        if kinds <= {"lineage", "none"} and row is not None:
            sib = self.siblings.get(row["parent"]) or [0]
            ratio = ((row["downloads"] or 0) + 1) / (statistics.median(sib) + 1)
            mean += max(-4.0, min(4.0, 1.5 * math.log2(ratio)))
        est = Est(mean, var, kinds, self.flags.get(node, []))
        self.memo[key] = est
        return est

    def profile(self, node, profile):
        """The profile's domains, weighted: its score and band."""
        parts = [(w, self.estimate(node, d))
                 for d, w in WEIGHTS.get(profile, WEIGHTS["agent"]).items()]
        kinds, flags = set(), []
        for _, e in parts:
            kinds |= e.kinds
            flags = e.flags
        return Est(sum(w * e.mean for w, e in parts),
                   sum(w * w * e.var for w, e in parts), kinds, flags)
