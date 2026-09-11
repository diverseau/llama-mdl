"""mdl fit: parser, placement, memory, search, emit, explain, remote, CLI.

Everything runs on GGUFs this file writes itself - headers with the real
layout and zero-filled weights - against a fixed machine, so none of it
needs a model, a GPU, llama.cpp or the network. The numbers the rules
encode were checked against llama-fit-params on real models; the tests
pin the rules.
"""
import http.server
import json
import os
import struct
import sys
import tempfile
import threading
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                   # noqa: E402
from support import mdl, run, sandbox, teardown  # noqa: E402

from mdl_fit import (calib, emit, explain, gguf, hw, model,  # noqa: E402
                     perf, remote, search)

t = support.Tally("test_fit")
check = t.check
MiB, GiB = model.MiB, model.GiB
TMP = Path(tempfile.mkdtemp(prefix="mdl-fit-test-"))
os.environ["MDL_FIT_HOME"] = str(TMP / "home")   # never the real ~/.config


# ------------------------------------------------------- a GGUF writer --

def _val(v):
    """(type id, packed bytes) for a metadata value."""
    if isinstance(v, bool):
        return 7, struct.pack("<?", v)
    if isinstance(v, int):
        return 4, struct.pack("<I", v)
    if isinstance(v, float):
        return 6, struct.pack("<f", v)
    if isinstance(v, str):
        b = v.encode()
        return 8, struct.pack("<Q", len(b)) + b
    if isinstance(v, list):
        etype = _val(v[0])[0] if v else 4
        body = b"".join(_val(x)[1] for x in v)
        return 9, struct.pack("<IQ", etype, len(v)) + body
    raise TypeError(v)


def write_gguf(path, meta, tensors, align=32):
    """tensors: [(name, dims, ggml_type)]; data is zeros, laid out the way
    llama.cpp's writer does it (each tensor padded to the alignment)."""
    head = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(meta))]
    for k, v in meta.items():
        kb = k.encode()
        vt, vb = _val(v)
        head.append(struct.pack("<Q", len(kb)) + kb + struct.pack("<I", vt)
                    + vb)
    offset, sizes = 0, []
    for name, dims, ty in tensors:
        n = 1
        for d in dims:
            n *= d
        size = gguf.type_bytes(ty, n)
        nb = name.encode()
        head.append(struct.pack("<Q", len(nb)) + nb
                    + struct.pack("<I", len(dims))
                    + b"".join(struct.pack("<Q", d) for d in dims)
                    + struct.pack("<IQ", ty, offset))
        sizes.append(size)
        offset += -(-size // align) * align
    raw = b"".join(head)
    raw += bytes(-(-len(raw) // align) * align - len(raw))
    with open(path, "wb") as fh:
        fh.write(raw)
        fh.write(bytes(offset))
    return path


F32, F16, Q8_0, Q4_K = 0, 1, 8, 12


def llama(path, n_layer=4, embd=256, heads=4, kv_heads=2, head=64, ff=512,
          vocab=1000, tied=False, arch="llama", extra_meta=None,
          layer_extra=None, ctx=8192, experts=0, used=0):
    """A small model of any of the shapes the engine knows about."""
    meta = {"general.architecture": arch, "general.name": "t",
            "%s.block_count" % arch: n_layer,
            "%s.context_length" % arch: ctx,
            "%s.embedding_length" % arch: embd,
            "%s.attention.head_count" % arch: heads,
            "%s.attention.head_count_kv" % arch: kv_heads,
            "tokenizer.ggml.tokens": ["t%d" % i for i in range(vocab)]}
    if experts:
        meta["%s.expert_count" % arch] = experts
        meta["%s.expert_used_count" % arch] = used
    meta.update(extra_meta or {})
    ts = [("token_embd.weight", [embd, vocab], Q8_0),
          ("output_norm.weight", [embd], F32)]
    if not tied:
        ts.append(("output.weight", [embd, vocab], Q8_0))
    for il in range(n_layer):
        b = "blk.%d." % il
        ts += [(b + "attn_norm.weight", [embd], F32),
               (b + "attn_q.weight", [embd, heads * head], Q8_0),
               (b + "attn_k.weight", [embd, kv_heads * head], Q8_0),
               (b + "attn_v.weight", [embd, kv_heads * head], Q8_0),
               (b + "attn_output.weight", [heads * head, embd], Q8_0),
               (b + "ffn_norm.weight", [embd], F32)]
        if experts:
            ts += [(b + "ffn_gate_inp.weight", [embd, experts], F32),
                   (b + "ffn_gate_exps.weight", [embd, ff, experts], Q4_K),
                   (b + "ffn_up_exps.weight", [embd, ff, experts], Q4_K),
                   (b + "ffn_down_exps.weight", [ff, embd, experts], Q4_K),
                   (b + "ffn_up_shexp.weight", [embd, ff], Q8_0)]
        else:
            ts += [(b + "ffn_gate.weight", [embd, ff], Q8_0),
                   (b + "ffn_up.weight", [embd, ff], Q8_0),
                   (b + "ffn_down.weight", [ff, embd], Q8_0)]
        ts += (layer_extra or (lambda il: []))(il)
    return write_gguf(path, meta, ts)


class FakeMachine(hw.Machine):
    pass


def machine(vram=8 * GiB, ram=16 * GiB, backend="CUDA"):
    return FakeMachine(gpu_name="NVIDIA GeForce RTX 3060", backend=backend,
                       vram_total=vram + GiB, vram_free=vram, margin=0,
                       ram_total=ram + 4 * GiB, ram_avail=ram, os_headroom=0,
                       cores=(16, 10, 6), pcie=[4, 4, 16, 16],
                       build={"build": 1, "load_mode": True,
                              "fit_flag": True})


# ================================================================ M0 ====

dense = llama(TMP / "Tiny-Dense-Q8_0.gguf")
inv = gguf.load(dense)
total = sum(x.nbytes for x in inv.tensors) + sum(inv.data_starts)
check("tensor sizes + header add up to the file, exactly",
      total, dense.stat().st_size)
check("sizes come from offsets and agree with the type table",
      inv.find("blk.0.attn_q.weight").nbytes,
      gguf.type_bytes(Q8_0, 256 * 256))
check("hyperparameters", (inv.arch, inv.n_layer, inv.n_embd, inv.n_vocab,
                           inv.n_ctx_train), ("llama", 4, 256, 1000, 8192))
check("roles", sorted({x.role for x in inv.tensors}),
      ["attn", "embd", "ffn", "norm", "output", "output_norm"])
check("a quant label comes off the file name", inv.quant_label, "Q8_0")

big_vocab = llama(TMP / "Big-Vocab.gguf", vocab=gguf.KEEP_ARRAY + 10,
                  n_layer=1)
check("a long token list is kept as a length, and still counts",
      gguf.load(big_vocab).n_vocab, gguf.KEEP_ARRAY + 10)

raw = dense.read_bytes()
try:
    gguf.parse_header(raw[:200])
    check("a short buffer says it is short", "no error", "Truncated")
except gguf.Truncated as e:
    check("a short buffer says how much it needs", int(e.args[0]) > 200, True)
try:
    gguf.parse_header(b"NOPE" + raw[4:])
    check("not a GGUF is refused", "no error", "NotGGUF")
except gguf.NotGGUF:
    check("not a GGUF is refused", True, True)

check("classify: routed experts", gguf.classify("blk.3.ffn_up_exps.weight"),
      (3, "exps"))
check("classify: expert bias travels as an expert",
      gguf.classify("blk.3.ffn_down_exps.bias"), (3, "exps"))
check("classify: shared expert", gguf.classify("blk.0.ffn_gate_shexp.weight"),
      (0, "shexp"))
check("classify: router", gguf.classify("blk.0.ffn_gate_inp.weight"),
      (0, "router"))
check("classify: recurrent", gguf.classify("blk.1.ssm_conv1d.weight"),
      (1, "recurrent"))
check("classify: MTP head", gguf.classify("blk.9.nextn.eh_proj.weight"),
      (9, "mtp"))
check("classify: unknown is 'other', not a crash",
      gguf.classify("blk.2.mystery.weight"), (2, "other"))

# split shards: every shard has its own tensor table
a = write_gguf(TMP / "Split-00001-of-00002.gguf",
               {"general.architecture": "llama", "llama.block_count": 1,
                "split.count": 2},
               [("token_embd.weight", [64, 100], F16)])
write_gguf(TMP / "Split-00002-of-00002.gguf", {"split.count": 2},
           [("blk.0.attn_k.weight", [64, 64], F16)])
sp = gguf.load(a)
check("split shards are unioned", sorted(x.name for x in sp.tensors),
      ["blk.0.attn_k.weight", "token_embd.weight"])
check("and sized against their own file", sp.find(
    "blk.0.attn_k.weight").nbytes, 64 * 64 * 2)
check("round-trips through the cache format",
      gguf.Inventory.from_json(sp.to_json()).file_size, sp.file_size)

# ================================================================ M1 ====

shape = model.Shape(inv)
f = model.Flags(ctx=4096, ctk="f16", ctv="f16")
gkv, hkv, grs, hrs = model.cache(shape, f)
check("KV: 4 layers x 4096 cells x (128+128) x 2 bytes",
      (gkv, hkv), (4 * 4096 * 256 * 2, 0))
f8 = f.replace(ctk="q8_0", ctv="q4_0")
check("KV type sizes: q8_0 34/32, q4_0 18/32",
      model.cache(shape, f8)[0], int(4 * 4096 * 128 * (34 / 32 + 18 / 32)))
check("context is padded to 256 cells",
      model.cache(shape, f.replace(ctx=4000))[0], gkv)

gw, hw_, _, _ = model.weights(shape, model.Flags())
embd = inv.find("token_embd.weight").nbytes
check("token embeddings stay on the host, the rest goes up",
      (hw_, gw), (embd, inv.file_size - sum(inv.data_starts) - embd))
# -ngl counts the output layer: -ngl 2 is the last block plus the head
f2 = model.Flags(ngl=2)
check("-ngl L puts the last L-1 blocks and the output on the GPU",
      shape.gpu_start(f2), 3)
gw2, hw2, _, hl = model.weights(shape, f2)
per_layer = shape.w_dense[0]
check("so three blocks stay on the host", (hl, hw2),
      (3, embd + 3 * per_layer))
check("and their KV goes with them",
      model.cache(shape, f.replace(ngl=2))[1], 3 * 4096 * 256 * 2)

tied = gguf.load(llama(TMP / "Tied.gguf", tied=True))
ts = model.Shape(tied)
tw = model.weights(ts, model.Flags())
check("tied embeddings are counted on both sides",
      (tw[1], ts.w_output >= tied.find("token_embd.weight").nbytes),
      (tied.find("token_embd.weight").nbytes, True))

moe = gguf.load(llama(TMP / "Tiny-MoE-Q4_K.gguf", experts=8, used=2, ff=1024))
ms = model.Shape(moe)
exps0 = ms.w_exps[0]
g0, h0, he0, _ = model.weights(ms, model.Flags())
g3, h3, he3, hl3 = model.weights(ms, model.Flags(ncmoe=3))
check("--n-cpu-moe moves only routed experts of the first N layers",
      (he3, h3 - h0, g0 - g3, hl3), (3 * exps0, 3 * exps0, 3 * exps0, 3))
check("router and shared expert stay on the GPU",
      ms.w_dense[0] > 0 and exps0 > ms.w_dense[0], True)
check("active params count experts at used/total",
      round(ms.exp_frac, 3), 0.25)

swa = gguf.load(llama(TMP / "Swa.gguf", arch="gemma3", extra_meta={
    "gemma3.attention.sliding_window": 512,
    "gemma3.attention.sliding_window_pattern": [True, True, True, False]}))
ss = model.Shape(swa)
fs = model.Flags(ctx=32768, ub=512, ctk="f16", ctv="f16")
check("SWA cells are window + ubatch, padded", ss.cells(fs), (32768, 1024))
per = 256 * 2
check("SWA KV: 3 windowed layers at 1024 cells, 1 full at 32768",
      model.cache(ss, fs)[0], 3 * 1024 * per + 32768 * per)
check("--swa-full gives every layer the full context",
      model.cache(ss, fs.replace(swa_full=True))[0], 4 * 32768 * per)
check("two slots, two windows", ss.cells(fs.replace(np=2))[1], 2 * 1024)

def ssm_layers(il):
    if il % 2:
        return []
    b = "blk.%d." % il
    return [(b + "ssm_conv1d.weight", [4, 512], F32),
            (b + "ssm_out.weight", [512, 256], Q8_0)]

hyb = gguf.load(llama(TMP / "Hybrid.gguf", arch="qwen35", extra_meta={
    "qwen35.ssm.conv_kernel": 4, "qwen35.ssm.inner_size": 512,
    "qwen35.ssm.state_size": 64, "qwen35.ssm.group_count": 2},
    layer_extra=ssm_layers))
hs = model.Shape(hyb)
check("layers with recurrent tensors carry no KV",
      [bool(k) for k in hs.kv_k], [False, True, False, True])
rs = (3 * (512 + 2 * 2 * 64) + 64 * 512) * 4
check("recurrent state per layer per sequence, f32",
      model.cache(hs, model.Flags(ctx=4096, np=3))[2], 2 * rs * 3)

shared = gguf.load(llama(TMP / "Shared.gguf", arch="gemma4", extra_meta={
    "gemma4.attention.shared_kv_layers": 1}))
check("shared-KV layers allocate none of their own",
      [bool(k) for k in model.Shape(shared).kv_k], [True, True, True, False])

mtp = gguf.load(llama(TMP / "Mtp.gguf", n_layer=3, extra_meta={
    "llama.nextn_predict_layers": 1}, layer_extra=lambda il: [
        ("blk.%d.nextn.eh_proj.weight" % il, [512, 256], Q8_0)] if il == 2
    else []))
mt = model.Shape(mtp)
check("an unused MTP head is not loaded", model.weights(mt, model.Flags())[0]
      < model.weights(mt, model.Flags(mtp=True))[0], True)
check("and gets no KV", model.cache(mt, model.Flags(ctx=1024))[0],
      2 * 1024 * 256 * 2)

f_big = model.Flags(ub=4096)
lg = 4 * 4096 * inv.n_vocab
check("logits past the backend's largest buffer are computed on the CPU",
      model.compute(shape, f_big, max_alloc=lg)[1] >= lg, True)
check("and stay on the card when they fit",
      model.compute(shape, f_big, max_alloc=lg + 1)[0] >= lg, True)
check("the host compute buffer holds the KQ mask",
      model.compute(shape, model.Flags(ctx=65536, ub=512))[1]
      >= 512 * 65536 * 2, True)

argv = ["llama-server", "-m", "m.gguf", "-c", "8192", "--cache-type-k",
        "q8_0", "--cache-type-v", "q8_0", "-ngl", "99", "--n-cpu-moe", "4",
        "-fa", "on", "--cache-type-k", "q4_0", "--load-mode", "none",
        "-ub", "1024", "-np", "2", "--jinja", "--spec-type", "draft-mtp"]
pf, notes, mp, _ = model.parse_argv(argv)
check("parse_argv: later flags win, like llama.cpp",
      (pf.ctk, pf.ctv, pf.ncmoe, pf.ub, pf.np, pf.mmap, pf.mtp, mp),
      ("q4_0", "q8_0", 4, 1024, 2, False, True, "m.gguf"))
check("-cmoe is every layer", model.parse_argv(["-cmoe"])[0].ncmoe, 999)
check("-ot is flagged, not silently ignored",
      any("-ot" in n for n in model.parse_argv(["-ot", "x=CPU"])[1]), True)

# ============================================================= search ===

m8 = machine(vram=int(0.02 * GiB), ram=GiB)
ctx_obj = search.Context(inv, m8, residuals=calib.Residuals(entries=[]))
opts = search.Options("chat", min_ctx=4096)
res = search.solve(ctx_obj, opts)
best = res.best
check("the search finds a config", best is not None, True)
check("every pick fits the card",
      all(p.gpu <= m8.vram_usable for p in res.picks), True)
check("and clears the context floor",
      all(p.flags.ctx >= 4096 for p in res.picks), True)
check("contexts are whole kilotokens", best.flags.ctx % 1024, 0)
check("a pick one step larger would not fit",
      ctx_obj.fits(ctx_obj.memory(best.flags.replace(
          ctx=best.flags.ctx + 1024))) and best.flags.ctx < inv.n_ctx_train,
      False)
check("the context never passes what the model was trained for",
      max(p.flags.ctx for p in res.everything) <= inv.n_ctx_train, True)
check("the default KV floor never goes below q8_0",
      {p.flags.ctv for p in res.everything} <= {"f16", "q8_0"}, True)

floor = search.solve(ctx_obj, search.Options("agent"))
check("an unreachable floor is reported, not refused",
      (bool(floor.picks), "most context" in (floor.relaxed or "")),
      (True, True))

tiny_ram = machine(vram=GiB, ram=int(0.001 * GiB))
nope = search.solve(search.Context(inv, tiny_ram, calib.Residuals([])),
                    search.Options("chat"))
check("nothing fitting names the budget that ran out",
      (nope.picks, "RAM is the limit" in (nope.relaxed or "")), ([], True))

# a card between 'everything on it' and 'no experts on it', so the only
# way in is to move some experts off
lean = model.Flags(ctx=2048, ctk="q8_0", ctv="q8_0")
room = (model.memory(ms, lean).gpu
        + model.memory(ms, lean.replace(ncmoe=4)).gpu) // 2
mctx = search.Context(moe, machine(vram=room, ram=GiB), calib.Residuals([]))
mres = search.solve(mctx, search.Options("speed", min_ctx=2048))
check("a MoE that does not fit whole sheds experts before layers",
      (mres.best.flags.ncmoe > 0, mres.best.flags.ngl), (True, 999))

pl = perf.Placement(ms, model.Flags(ncmoe=2))
check("decode reads host experts at n_used/n_expert",
      round(pl.raw_cpu), round(2 * exps0 * 0.25))
check("and a K-quant reads slower than its bytes on the CPU",
      pl.a_cpu > pl.raw_cpu, True)
iq = gguf.Tensor("blk.0.ffn_up_exps.weight", [256, 256, 8], 18, 0)
q8 = gguf.Tensor("blk.0.ffn_up_exps.weight", [256, 256, 8], 8, 0)
iq.nbytes = q8.nbytes = 1 << 20
check("an IQ3 byte costs more time than a Q8_0 byte, on both sides",
      [model.eff_bytes(iq, tb) > model.eff_bytes(q8, tb)
       for tb in (model.TYPE_EFF_GPU, model.TYPE_EFF_CPU)], [True, True])
p = perf.seeds(machine())
check("a slower placement decodes slower",
      perf.decode_time(perf.Placement(ms, model.Flags(ncmoe=4)), p, 0)
      > perf.decode_time(perf.Placement(ms, model.Flags()), p, 0), True)
check("decode slows with KV depth",
      perf.decode_time(pl, p, 32768) > perf.decode_time(pl, p, 0), True)
check("a bigger ubatch prefills host experts faster",
      perf.prefill_time(pl, p, 8192, 2048) < perf.prefill_time(pl, p, 8192,
                                                               512), True)

# ======================================================= calibration ===

entries = [{"kind": "oracle", "sig": "x|1", "arch": "llama",
            "flags": {"ub": 512, "ncmoe": 0, "ctx": c},
            "pred": [0, 0, 100], "actual": [0, 0, 100 + miss]}
           for c, miss in ((8192, 10 * MiB), (65536, 50 * MiB))]
r = calib.Residuals(entries)
f512 = model.Flags(ub=512)
check("residuals interpolate linearly over context",
      round(r.lookup("x|1", "llama", f512.replace(ctx=36864)) / MiB), 30)
check("and hold flat past the ends",
      round(r.lookup("x|1", "llama", f512.replace(ctx=200000)) / MiB), 50)
check("another file of the arch borrows the median",
      r.lookup("y|2", "llama", f512.replace(ctx=8192)) > 0, True)
check("an unseen arch borrows nothing", r.lookup("z", "qwen", f512), 0)
check("bands widen as calibration thins",
      (r.band("x|1", "llama", 0) < r.band("y|2", "llama", 0)
       < r.band("z", "qwen", 0)), True)

check("oracle output parses to bytes",
      calib.parse_oracle("noise\nVulkan0 8784 1502 996 \nHost 3803 0 528\n"),
      {"Vulkan0": [8784 * MiB, 1502 * MiB, 996 * MiB],
       "Host": [3803 * MiB, 0, 528 * MiB]})
bench_out = ('build: x\n[{"n_prompt": 2048, "n_gen": 0, "n_depth": 0, '
             '"avg_ts": 812.5}, {"n_prompt": 0, "n_gen": 64, "n_depth": 0, '
             '"avg_ts": 41.0}, {"n_prompt": 0, "n_gen": 64, "n_depth": 49152,'
             ' "avg_ts": 33.0}]')
check("llama-bench json parses, by depth",
      calib.parse_bench(bench_out), {"pp": 812.5, "tg": {0: 41.0, 49152: 33.0}})
log = ("load_tensors:      Vulkan0 model buffer size =  8784.00 MiB\n"
       "llama_kv_cache:    Vulkan0 KV buffer size =  1502.00 MiB\n"
       "ggml_vulkan: failed to allocate Vulkan0 buffer of size 1073741824\n")
found = calib.scrape_log(log)
check("a load log gives away its buffers",
      found["buffers"]["Vulkan0"], {"model": 8784 * MiB, "KV": 1502 * MiB})
check("and which allocation failed", found["failed"],
      ("Vulkan0", 1073741824))
check("oracle argv pins every flag and turns --fit off",
      calib.oracle_argv("fp", "m.gguf", model.Flags(ctx=8192))[-4:],
      ["-fitp", "on", "-fit", "off"])

# ================================================================ emit ===

feats = {"load_mode": True, "fit_flag": True}
fl = model.Flags(ctx=131072, ncmoe=15, ctk="q8_0", ctv="q4_0", ub=2048,
                 mmap=False)
av = emit.server_argv("llama-server", "m.gguf", fl, 40, feats)
check("the command says -np and -fa out loud",
      ("-np" in av and "-fa" in av and av[av.index("-fa") + 1]), "on")
check("and the mixed KV pair, mmap choice and --fit off",
      [x in av for x in ("--cache-type-v", "none", "off")], [True] * 3)
check("-ngl 99 means all of it", av[av.index("-ngl") + 1], "99")
keys, drop = emit.table(fl, "C:\\m\\x.gguf", 40, feats,
                        keep_args=["--jinja", "-ub", "512", "--temp", "0.6",
                                   "--no-mmap"])
check("a mixed KV pair goes to args and kv_type is dropped",
      ("kv_type" in keys, "kv_type" in drop), (False, True))
check("owned flags are replaced, the rest kept",
      keys["args"][:4], ["--metrics", "--jinja", "--temp", "0.6"])
check("paths get forward slashes", keys["model"], "C:/m/x.gguf")
same, _ = emit.table(fl.replace(ctv="q8_0"), "m.gguf", 40, feats)
check("a matched pair is kv_type", same.get("kv_type"), "q8_0")
check("the block parses as TOML",
      tomllib.loads(emit.block("x", keys, "# stamp"))["x"]["n_cpu_moe"], 15)

# ================================================================ CLI ===

hw.probe = lambda *a, **k: machine(vram=int(0.02 * GiB), ram=GiB)
root, port = sandbox(model=dense)
out, err, code = run(mdl.cmd_fit, [str(dense), "--no-oracle", "--json",
                                   "--profile", "chat", "--min-ctx", "2k"])
check("mdl fit <gguf> --json", code, 0)
data = json.loads(out)
check("reports picks with memory and speed",
      (len(data["picks"]) >= 1, "gpu_total" in data["picks"][0],
       data["picks"][0]["speed"]["decode0"] > 0), (True, True, True))
out, err, code = run(mdl.cmd_fit, [str(dense), "--no-oracle", "--min-ctx",
                                   "2k", "--profile", "chat"])
check("human output has a table, placement and the command",
      [s in out for s in ("decode", "placement", "llama-server")],
      [True] * 3)
out, err, code = run(mdl.cmd_fit, ["demo", "--no-oracle", "--min-ctx", "2k",
                                   "--profile", "chat", "--write", "tuned"])
cfg = tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8"))
check("--write appends a new entry mdl can run",
      ("tuned" in cfg, set(cfg["tuned"]) <= mdl.KNOWN), (True, True))
check("which builds a command", "-fa" in mdl.build_argv(
    "tuned", cfg["tuned"], "srv"), True)
_, err, code = run(mdl.cmd_fit, ["demo", "--no-oracle", "--write", "tuned"])
check("--write refuses a name that exists", ("already in" in err, code),
      (True, 1))

# a config that is far over: explain it, and apply the first fix
with open(mdl.CONFIG, "a", encoding="utf-8") as fh:
    fh.write('\n[huge]\nmodel = "%s"\nctx = 8192\nflash_attn = true\n'
             'kv_type = "f16"\nargs = ["--temp", "0.7"]\n'
             % str(dense).replace("\\", "/"))
out, err, code = run(mdl.cmd_fit, ["huge", "--explain", "--no-oracle",
                                   "--apply", "1"])
check("--explain says it is over", "over by" in out, True)
check("and lists fixes", "fix " in out, True)
after = tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8"))["huge"]
check("--apply rewrote the entry and kept its other args",
      ("--temp" in after["args"], after != {"model": 1}), (True, True))
check("the old config was kept", (mdl.CONFIG.with_name(
    "models.toml.bak")).exists(), True)
_, err, code = run(mdl.cmd_fit, ["nosuch"])
check("an unknown target is one line", ("no such file" in err, code),
      (True, 1))
out, _, code = run(mdl.cmd_fit, ["inspect", str(moe.source)])
check("inspect prints layers and says the sizes are exact",
      ("exact" in out, "exps" in out), (True, True))
teardown(root)

# ============================================================== remote ===

REPO = "someorg/Some-Model-GGUF"
files = {"Some-Model-Q8_0.gguf": dense.read_bytes(),
         "Some-Model-Q4_K-00001-of-00002.gguf": a.read_bytes(),
         "Some-Model-Q4_K-00002-of-00002.gguf":
             (TMP / "Split-00002-of-00002.gguf").read_bytes(),
         "mmproj-F16.gguf": b"GGUF"}
hits = []


class Hub(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        hits.append(self.path)
        if self.path.startswith("/api/models/%s/tree/main" % REPO):
            body = json.dumps([{"type": "file", "path": k, "size": len(v),
                                "oid": "o%d" % i, "lfs": {"size": len(v),
                                                          "oid": "sha%d" % i}}
                               for i, (k, v) in enumerate(files.items())])
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body.encode())
            return
        name = self.path.rsplit("/", 1)[-1].replace("%20", " ")
        data = files.get(name)
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        lo, hi = 0, len(data) - 1
        rng = self.headers.get("Range")
        if rng:
            lo, hi = (int(x) for x in rng.split("=")[1].split("-"))
            hi = min(hi, len(data) - 1)
        self.send_response(206 if rng else 200)
        self.end_headers()
        self.wfile.write(data[lo:hi + 1])


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hub)
threading.Thread(target=server.serve_forever, daemon=True).start()
os.environ["MDL_HF_ENDPOINT"] = "http://127.0.0.1:%d" % server.server_port
os.environ.pop("HF_TOKEN", None)

listing = remote.list_files(REPO)
groups = remote.gguf_groups(listing)
check("quants group, shards together, projectors left out",
      sorted(groups), ["Some-Model-Q4_K.gguf", "Some-Model-Q8_0.gguf"])
check("shards in order", [s["path"] for s in groups["Some-Model-Q4_K.gguf"]],
      ["Some-Model-Q4_K-00001-of-00002.gguf",
       "Some-Model-Q4_K-00002-of-00002.gguf"])
remote.FIRST = 64                       # force the grow-and-refetch path
before = len(hits)
rinv = remote.inventory(REPO, "Some-Model-Q8_0.gguf",
                        groups["Some-Model-Q8_0.gguf"])
check("a remote inventory matches the local one, from headers only",
      (rinv.file_size, rinv.n_layer, rinv.n_vocab),
      (inv.file_size, inv.n_layer, inv.n_vocab))
check("a short first fetch grows until the header parses",
      len(hits) - before > 1, True)
before = len(hits)
remote.inventory(REPO, "Some-Model-Q8_0.gguf", groups["Some-Model-Q8_0.gguf"])
check("the second look costs no requests", len(hits) - before, 0)
check("a selector narrows to one quant",
      list(remote.select(groups, "q8_0")), ["Some-Model-Q8_0.gguf"])
check("hf spec parsing", remote.parse_spec("hf:a/b:Q4_K"), ("a/b", "Q4_K"))
root, port = sandbox()
out, err, code = run(mdl.cmd_fit, ["hf:" + REPO, "--profile", "chat",
                                   "--min-ctx", "1k"])
check("mdl fit hf:repo tables every quant without downloading",
      (code, "Some-Model-Q8_0" in out, "nothing downloaded" in out),
      (0, True, True))
teardown(root)
server.shutdown()

# ================================================================ TUI ===

try:
    import mdl_ui
except ImportError:              # textual is an extra; the engine is not
    mdl_ui = None
if mdl_ui is not None:
    placed = mdl_ui.fit_placement(mdl.build_argv(
        "d", {"model": str(dense), "ctx": 4096, "flash_attn": True}, "srv"))
    check("the dashboard reads placement from the fit engine",
          placed is not None and placed[0].gpu > 0, True)
    text = mdl_ui.placement_text(placed, 12288).plain
    check("and shows what sits where, and how fast",
          [s in text for s in ("weights", "kv", "compute", "t/s")],
          [True] * 4)
    check("a file that is not a GGUF falls back to the old estimate",
          mdl_ui.fit_placement(["srv", "-m", str(support.FAKE)]), None)

# ============================================================ explain ===

# a card that holds the q8_0 cache at this context but not the f16 one
over = model.Flags(ctx=8192, ctk="f16", ctv="f16", ub=512)
room = (model.memory(shape, over).gpu
        + model.memory(shape, over.replace(ctk="q8_0", ctv="q8_0")).gpu) // 2
fx_ctx = search.Context(inv, machine(vram=room, ram=GiB), calib.Residuals([]))
base, fixes, _ = explain.fixes(fx_ctx, over, search.Options("chat",
                                                              min_ctx=1024))
check("fixes all fit", all(fx_ctx.fits(x.fit.mem) for x in fixes), True)
check("a context cut is offered, and ranked last",
      (fixes[-1].kind if fixes else None), "ctx")
check("the KV step is the first thing it tries",
      [x.kind for x in fixes][:1], ["kv"])

sys.exit(t.done())
