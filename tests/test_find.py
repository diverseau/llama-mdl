"""mdl find and the quality model: evidence weighting, the benchmaxx and
contamination flags, lineage priors, the public -> local map, the quant
penalty, and the ranking, over a catalog and headers made up here."""
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                   # noqa: E402

from mdl_fit import catalog, find, gguf, hw, quality  # noqa: E402

t = support.Tally("test_find")
check = t.check
TMP = Path(tempfile.mkdtemp(prefix="mdl-find-test-"))
os.environ["MDL_FIT_HOME"] = str(TMP / "home")
GiB = 1 << 30
Q8_0, Q4_K, F32 = 8, 12, 0

# ============================================================ quality ===

check("benchmarks map to domains; blocked and stale ones do not count",
      [quality.domains_of(b) for b in ("SWE-bench_Verified", "livecodebench",
                                       "gpqa_diamond", "RULER-128k",
                                       "LiquidAI/ifstruct-v1.0",
                                       "DeepSWE", "hellaswag", "vibes")],
      [("agentic", "coding"), ("coding",), ("reasoning",),
       ("long-context",), ("general",), (), (), ()])
check("quant penalty: steeper at low bpw, gentler for big models",
      [quality.quant_penalty(8.5, 8e9) < quality.quant_penalty(4.8, 8e9)
       < quality.quant_penalty(3.5, 8e9) < quality.quant_penalty(2.6, 8e9),
       quality.quant_penalty(3.5, 70e9) < quality.quant_penalty(3.5, 8e9)],
      [True, True])
check("4-bit K costs more than 4-bit V, and q8_0 next to nothing",
      [quality.kv_penalty("q4_0", "q8_0") > quality.kv_penalty("q8_0",
                                                                "q4_0"),
       quality.kv_penalty("q8_0", "q8_0") < 0.5,
       quality.kv_penalty("f16", "f16")], [True, True, 0.0])
check("inverse-variance: the sure number wins",
      [round(x, 2) for x in quality.combine([(60, 4), (40, 100)])],
      [59.23, 3.85])

# a catalog, made by hand
path = TMP / "cat.sqlite"
db = catalog.connect(path, fresh=True)


def node(nid, parent=None, relation=None, official=0, params=30_000_000,
         downloads=1000, arch="llama", datasets=(), created="2026-08-01"):
    db.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
               "?,?)", (nid, nid.split("/")[0], official, None, parent,
                        relation, "hf-tree", 0, arch, params, 131072,
                        "apache-2.0", "text-generation", created, created,
                        downloads, 0, 0, "[]", json.dumps(list(datasets))))


def quants(nid, repo, sizes):
    for q, size in sizes.items():
        db.execute("INSERT INTO ggufs VALUES (?,?,?,?,?,?,?,?)",
                   (repo, nid, q, "%s-%s.gguf" % (nid.split("/")[1], q), size,
                    json.dumps([{"path": "%s-%s.gguf" % (nid.split("/")[1], q),
                                 "size": size, "oid": "x" + q}]), 100, "m"))


def ev(nid, bench, value, verified=0):
    db.execute("INSERT INTO evals VALUES (?,?,?,?,?,?,?)",
               (nid, bench, None, value, verified, "test", None))


node("Fam/Base", official=1)
node("Fam/Base-Instruct", "Fam/Base", "finetune", official=1)
node("ft/Coder", "Fam/Base-Instruct", "finetune",
     datasets=["openai/gsm8k"])
node("new/Fresh", "Fam/Base-Instruct", "finetune", downloads=90000,
     created="2026-09-10")
node("big/Huge", "Fam/Base-Instruct", "finetune", downloads=5000)
node("odd/Alien", None, None, arch="alienarch", downloads=5000)
for i, other in enumerate(("o/A", "o/B", "o/C", "o/D")):   # a field
    node(other, official=1)
    for j, (b, v) in enumerate((("humaneval", 60), ("mmlu", 60),
                                ("gpqa", 40), ("ifeval", 70), ("bfcl", 55))):
        ev(other, b, v + 2 * i + (i * j) % 3)
for b, v in (("humaneval", 70), ("mmlu", 68), ("gpqa", 50), ("ifeval", 78),
             ("bfcl", 64)):
    ev("Fam/Base-Instruct", b, v, verified=1)
for b, v in (("humaneval", 97), ("mmlu", 55), ("gpqa", 35), ("ifeval", 62),
             ("gsm8k-cot", 90)):
    ev("ft/Coder", b, v)
quants("Fam/Base-Instruct", "Fam/Base-Instruct-GGUF",
       {"Q8_0": 32_000_000, "Q4_K_M": 18_000_000})
quants("ft/Coder", "q/Coder-GGUF", {"Q8_0": 32_000_000})
quants("new/Fresh", "q/Fresh-GGUF", {"Q8_0": 32_000_000})
quants("big/Huge", "q/Huge-GGUF", {"Q4_K_M": 900 * GiB})
quants("odd/Alien", "odd/Alien", {"Q8_0": 32_000_000})
db.executemany("INSERT INTO meta VALUES (?, ?)",
               [("built_at", json.dumps("2026-09-11T00:00:00Z")),
                ("nodes", json.dumps(10))])
db.commit()
db.close()
cat = catalog.Catalog(path)
qm = quality.Model(cat)

inst = qm.estimate("Fam/Base-Instruct", "coding")
coder = qm.estimate("ft/Coder", "coding")
check("verified results make a rated estimate with a band",
      (inst.rated, "verified" in inst.kinds, 0 < inst.sd < 12), (True, True,
                                                                 True))
check("a spike far above a model's other scores is flagged",
      any("humaneval" in f and "claims" in f for f in coder.flags), True)
check("and so is reporting a benchmark it trained for",
      any("trained on data tied to gsm8k" in f for f in
          qm.flags.get("ft/Coder", [])), True)
check("so the flagged coder does not inherit its claimed lead",
      coder.mean < 50 + 15 * 2.0, True)
fresh = qm.estimate("new/Fresh", "coding")
check("an unrated fine-tune starts at its parent, wider",
      (fresh.rated, "lineage" in fresh.kinds, fresh.sd > inst.sd,
       abs(fresh.mean - inst.mean) <= 4.01), (False, True, True, True))
check("traction moves only models with nothing else to go on",
      fresh.mean > inst.mean, True)
agent = qm.profile("Fam/Base-Instruct", "agent")
check("a profile mixes its domains", (agent.rated, round(sum(
    quality.WEIGHTS["agent"].values()), 6)), (True, 1.0))


def local_rec(h, score, bpw=8.5, params=30_000_000, kv="f16/f16"):
    return {"hash": h, "bpw": bpw, "params": params, "kv": kv,
            "domains": {d: {"score": score, "lo": score - 0.05,
                            "hi": score + 0.05} for d in quality.DOMAINS}}


qm2 = quality.Model(cat)
qm2.add_local([local_rec("h1", 0.9)], {"h1": "new/Fresh"})
f2 = qm2.estimate("new/Fresh", "coding")
check("a local eval makes a model rated, and outweighs its lineage",
      (f2.rated, "local" in f2.kinds, f2.sd < fresh.sd), (True, True, True))
check("no map yet: the public scale", qm2.scale_name, "public")
qm3 = quality.Model(cat)
qm3.add_local([local_rec("a", 0.40), local_rec("b", 0.55),
               local_rec("c", 0.70)], {"a": "o/A", "b": "o/B",
                                       "c": "Fam/Base-Instruct"})
check("three models with both kinds of evidence map public onto local, "
      "in the domains they have public results for",
      (qm3.scale_name, sorted(qm3.map)),
      ("local", ["agentic", "coding", "general", "reasoning"]))
qm4 = quality.Model(cat)
qm4.add_local([local_rec("q8", 0.80, bpw=8.5),
               local_rec("q3", 0.50, bpw=3.5)], {"q8": "new/Fresh",
                                                 "q3": "new/Fresh"})
check("the same model at two quants teaches the penalty curve",
      qm4.scale > quality.QUANT_SCALE, True)

# ============================================================== find ===


def inv_of(source, qtype=Q8_0, layers=8, embd=512, ff=1536, vocab=4000,
           arch="llama", base_url=None):
    meta = {"general.architecture": arch, "%s.block_count" % arch: layers,
            "%s.embedding_length" % arch: embd,
            "%s.context_length" % arch: 131072,
            "%s.attention.head_count" % arch: 8,
            "%s.attention.head_count_kv" % arch: 2,
            "tokenizer.ggml.tokens": {"_len": vocab}}
    if base_url:
        meta["general.base_model.count"] = 1
        meta["general.base_model.0.repo_url"] = base_url
    ts = []

    def add(name, dims, ty=qtype):
        x = gguf.Tensor(name, dims, ty, 0)
        x.nbytes = gguf.type_bytes(ty, x.n_elements)
        ts.append(x)
    add("token_embd.weight", [embd, vocab])
    for il in range(layers):
        b = "blk.%d." % il
        add(b + "attn_norm.weight", [embd], F32)
        add(b + "attn_q.weight", [embd, embd])
        add(b + "attn_k.weight", [embd, 128])
        add(b + "attn_v.weight", [embd, 128])
        add(b + "attn_output.weight", [embd, embd])
        add(b + "ffn_norm.weight", [embd], F32)
        add(b + "ffn_gate.weight", [embd, ff])
        add(b + "ffn_up.weight", [embd, ff])
        add(b + "ffn_down.weight", [ff, embd])
    add("output_norm.weight", [embd], F32)
    add("output.weight", [embd, vocab])
    return gguf.Inventory(source, meta, ts, [sum(x.nbytes for x in ts)], [0])


fetched = []


def fake_fetch(c, cache_only=False):
    fetched.append((c.repo, c.quant))
    if cache_only:
        return None
    return inv_of("hf:%s/%s" % (c.repo, c.key),
                  Q4_K if c.quant.startswith("Q4") else Q8_0,
                  arch="alienarch" if c.repo == "odd/Alien" else "llama")


find.fetch = fake_fetch
lib = TMP / "bin"
lib.mkdir()
(lib / "llama.dll").write_bytes(b"\x00llama\x00qwen3\x00")
binary = str(lib / "llama-server")
Path(binary).write_bytes(b"")
mach = hw.Machine(gpu_name="NVIDIA GeForce RTX 3060", backend="CUDA",
                  vram_total=3 * GiB, vram_free=2 * GiB, margin=0,
                  ram_total=8 * GiB, ram_avail=6 * GiB, os_headroom=0,
                  ram_reserve=2 * GiB, cores=(16, 10, 6),
                  pcie=[4, 4, 16, 16], build={"build": 1})
opts = find.search.Options("agent")
notes = []
qm = quality.Model(cat)
cands = find.catalog_cands(cat, qm, mach, "agent", binary, {}, notes)
check("the generous bound drops only what cannot fit anywhere",
      ("big/Huge" not in {c.node for c in cands},
       any("too big" in n for n in notes)), (True, True))
check("an arch this build does not load is dropped before any fetch",
      ("odd/Alien" not in {c.node for c in cands},
       any("architecture" in n for n in notes)), (True, True))
check("up to three quants per model, biggest first",
      [c.quant for c in cands if c.node == "Fam/Base-Instruct"],
      ["Q8_0", "Q4_K_M"])
find.fit_all(cands, qm, mach, opts, "agent", binary, False, notes)
check("one header per model; its other quants are sized from it",
      (len([f for f in fetched if f[0] == "Fam/Base-Instruct-GGUF"]),
       sum(1 for c in cands if not c.exact)), (1, 1))
main, explore = find.choose(cands, qm)
check("rated models rank by expected quality, each at its best quant",
      [(c.node, c.quant) for c in main[:2]],
      [("Fam/Base-Instruct", "Q8_0"), ("ft/Coder", "Q8_0")])
check("an unrated model never enters the main list",
      "new/Fresh" in {c.node for c in main}, False)
check("but is worth testing when its upper band beats the #1",
      [c.node for c in explore][:1], ["new/Fresh"])
check("every row shown has a config that clears the floors",
      all(c.fit.flags.ctx >= opts.min_ctx and c.fit.speed.s_turn <= 90
          for c in main), True)
find.refine(main + explore, qm, mach, opts, "agent", binary, False)
check("rows about to be shown get their own header",
      all(c.exact for c in main + explore), True)
out = io.StringIO()
items = find.evalsuite.build(find.evalsuite.SUITES, "seed")
find.show(main, explore, qm, mach, opts, "agent", cat.meta(), notes,
          out.write, items)
text = out.getvalue()
check("the table: floors, lineage, config, evidence, flags, next steps",
      [s in text for s in ("s/turn ≤ 90 s", "Base → post-train → FT",
                           "verified benchmarks", "⚑ ft/Coder",
                           "worth testing", "mdl eval fresh",
                           "public scale")], [True] * 7)
check("lineage in words", [find.lineage_words(qm, n) for n in (
    "Fam/Base", "Fam/Base-Instruct", "ft/Coder")],
    ["base", "Base → post-train", "Base → post-train → FT"])

# local files link into the catalog
nodes = {r["id"]: r for r in cat.nodes()}
check("a local GGUF finds its parent from general.base_model",
      find.link_local(inv_of("x.gguf", base_url="https://huggingface.co/"
                             "Fam/Base-Instruct"), nodes),
      (None, "Fam/Base-Instruct", "gguf-meta"))
check("else by being the same arch and size as a tracked model",
      find.link_local(inv_of("x.gguf"), {k: dict(v, params=inv_of(
          "y").n_params) if k == "Fam/Base" else v for k, v in nodes.items()}),
      (None, "Fam/Base", "fingerprint"))
check("else by name", find.link_local(inv_of("my-Coder-q8.gguf", layers=2),
                                      nodes)[1:], ("ft/Coder", "name"))
# A drafter or calibration file filed under a giant must not be ranked as
# it: an 11 GB DSpark drafter was shown as DeepSeek-V4-Flash, all on GPU.
fq = type("Q", (), {"nodes": {
    "big/Giant": {"params": 304_000_000_000},
    "Fam/Base-Instruct": nodes["Fam/Base-Instruct"],
    "local:abc": {"id": "local:abc", "parent": None}}})()
small = inv_of("draft.gguf")
check("a header far smaller than its model's card is not that model",
      [find.same_model(small, fq, n) for n in (
          "big/Giant", "Fam/Base-Instruct", "local:abc", None)],
      [False, True, True, True])
real_fetch, fetched_now = find.fetch, []
find.fetch = lambda c, cache_only=False: fetched_now.append(c.key) or small
try:
    drafts = [find.Cand("big/Giant", "big/Giant", q, size, "q/Giant-GGUF",
                        "Giant-%s.gguf" % q, [])
              for q, size in (("BF16", 11 * GiB), ("Q8_0", 10 * GiB))]
    find.fit_all(drafts, fq, mach, opts, "agent", binary, False, [])
finally:
    find.fetch = real_fetch
check("it is turned away with the reason in its header",
      "not big/Giant's 304.0B" in (drafts[0].why or ""), True)
check("and its sibling is not sized from the impostor's header",
      (fetched_now, drafts[1].inv), (["Giant-BF16.gguf"], None))

rescaled = find.rescale(inv_of("a"), 16_000_000, "b")
check("a sized sibling keeps the shapes and takes the file size",
      (len(rescaled.tensors), rescaled.file_size <= 16_000_000,
       rescaled.n_params == inv_of("a").n_params),
      (len(inv_of("a").tensors), True, True))


# B05: a sized guess that passes, then fails on its own header, must stop
# ranking - it used to keep its score and crash choose() on c.fit.speed
guess = next(c for c in cands if c.q is not None and c.repo)
real_best = find.best_fit
find.best_fit = lambda inv, *a: (None, type("C", (), {"shape": None})())
try:
    guess.exact = guess.refined = False
    find.refine([guess], qm, mach, opts, "agent", binary, False)
finally:
    find.best_fit = real_best
check("a row that misses on its own header loses its score and reason",
      (guess.fit, guess.q, guess.why), (None, None, "misses the floors"))
again, _ = find.choose(cands, qm)
check("and choose() passes over it rather than crashing",
      guess in again, False)
guess.why = None
guess.q = qm.profile(guess.node, "agent")       # a stale score, no fit
check("nothing without a fit is ranked, whatever else it carries",
      guess in find.choose(cands, qm)[0], False)
check("a row is tried once, so refinement settles",
      find.refine([guess], qm, mach, opts, "agent", binary, False), False)

remote_row = next(c for c in cands if c.repo)
check("find's next step for a remote quant names the file to fetch, and "
      "never an hf: --write that writes nothing (B08)",
      (remote_row.key in find.next_step(remote_row),
       "hf:" in find.next_step(remote_row)), (True, False))

sys.exit(t.done())
