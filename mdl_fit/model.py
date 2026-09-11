"""Placement and memory: where llama.cpp puts every tensor, and what it costs.

Every rule here was checked against `llama-fit-params -fitp on` on build
10424; the ones that disagree with folklore say so. Weights, KV and
recurrent state come out exact. The compute buffer is the one term that
is a product of llama.cpp's graph allocator rather than of the model, so
it is a seed formula plus whatever calibration has learned (calib.py).
"""

import re

MiB = 1 << 20
GiB = 1 << 30
N_PAD = 256                      # llama.cpp pads KV cell counts to this

# bytes per element of a KV cache type
KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32,
            "q5_1": 24 / 32, "q5_0": 22 / 32, "q4_1": 20 / 32,
            "q4_0": 18 / 32, "iq4_nl": 18 / 32}
# best first; quantising K costs more quality than quantising V
KV_RANK = {"f16": 0, "bf16": 0, "f32": 0, "q8_0": 1, "q5_1": 2, "q5_0": 2,
           "q4_1": 3, "q4_0": 3, "iq4_nl": 3}


# How close to memory bandwidth a matmul on each weight type gets, per
# side, relative to Q8_0. The IQ types are compute-bound - dequantising
# them costs more than reading them - and on the CPU far more so. Seeds
# read off llama-bench on an RTX 3060 (Vulkan) and a 12600KF:
# IQ2_XS decodes at ~0.4 of Q8_0's bytes-per-second on the card, IQ3_S
# experts at ~0.33 on the CPU. `--verify` learns the rest per arch.
TYPE_EFF_GPU = {"F32": 1.0, "F16": 1.0, "BF16": 1.0, "Q8_0": 1.0,
                "Q4_0": 1.0, "Q4_1": 1.0, "Q5_0": 1.0, "Q5_1": 1.0,
                "Q4_K": 0.9, "Q5_K": 0.9, "Q6_K": 0.9, "Q2_K": 0.8,
                "Q3_K": 0.8, "IQ4_XS": 0.8, "IQ4_NL": 0.8, "MXFP4": 0.9,
                "IQ3_XXS": 0.55, "IQ3_S": 0.55, "IQ2_XXS": 0.42,
                "IQ2_XS": 0.42, "IQ2_S": 0.42, "IQ1_S": 0.4, "IQ1_M": 0.4,
                "TQ1_0": 0.5, "TQ2_0": 0.5}
TYPE_EFF_CPU = {"F32": 0.9, "F16": 0.9, "BF16": 0.8, "Q8_0": 1.0,
                "Q4_0": 1.0, "Q4_1": 1.0, "Q5_0": 0.9, "Q5_1": 0.9,
                "Q4_K": 0.8, "Q5_K": 0.8, "Q6_K": 0.8, "Q2_K": 0.7,
                "Q3_K": 0.7, "IQ4_XS": 0.6, "IQ4_NL": 0.6, "MXFP4": 0.8,
                "IQ3_XXS": 0.5, "IQ3_S": 0.35, "IQ2_XXS": 0.45,
                "IQ2_XS": 0.45, "IQ2_S": 0.45, "IQ1_S": 0.4, "IQ1_M": 0.4,
                "TQ1_0": 0.4, "TQ2_0": 0.4}


def eff_bytes(t, table):
    """A tensor's bytes as seconds-equivalent at Q8_0 speed."""
    from .gguf import type_name
    return t.nbytes / table.get(type_name(t.type), 0.7)


def pad(n, to=N_PAD):
    return -(-int(n) // to) * to


class Flags:
    """The llama-server flags that change placement, memory or speed."""

    DEFAULTS = {"ctx": 4096, "ngl": 999, "ncmoe": 0, "fa": True,
                "ctk": "f16", "ctv": "f16", "b": 2048, "ub": 512, "np": 1,
                "mmap": True, "swa_full": False, "kvu": False, "mmproj": 0,
                "mmproj_offload": True, "threads": 0, "mtp": False}
    __slots__ = tuple(DEFAULTS)

    def __init__(self, **kw):
        for key, value in self.DEFAULTS.items():
            setattr(self, key, kw.pop(key, value))
        if kw:
            raise TypeError("unknown flag(s): " + ", ".join(sorted(kw)))

    def as_dict(self):
        return {k: getattr(self, k) for k in self.DEFAULTS}

    def replace(self, **kw):
        d = self.as_dict()
        d.update(kw)
        return Flags(**d)

    def __eq__(self, other):
        return isinstance(other, Flags) and self.as_dict() == other.as_dict()

    def __repr__(self):
        return "Flags(%s)" % ", ".join("%s=%r" % kv for kv in
                                       self.as_dict().items())

    @property
    def kv_label(self):
        return "%s/%s" % (self.ctk, self.ctv)


_LOAD_MODE_MMAP = {"auto": True, "mmap": True, "mlock": True,
                   "mmap+mlock": True, "none": False, "dio": False}


def parse_argv(argv):
    """Flags from a llama-server command line, plus notes on what it skipped.

    Later flags win, the way llama.cpp's own parser does it - which is
    what makes kv_type plus a --cache-type-k in args behave.
    """
    f, notes, model, mmproj_path = Flags(), [], None, None
    i = 1 if argv and not argv[0].startswith("-") else 0
    while i < len(argv):
        flag = argv[i]
        val = argv[i + 1] if i + 1 < len(argv) else ""
        took = 2
        try:
            if flag in ("-m", "--model"):
                model = val
            elif flag in ("-c", "--ctx-size"):
                f.ctx = int(val)
            elif flag in ("-ngl", "--n-gpu-layers", "--gpu-layers"):
                f.ngl = 999 if val in ("all", "auto") else int(val)
            elif flag in ("-ncmoe", "--n-cpu-moe"):
                f.ncmoe = int(val)
            elif flag in ("-cmoe", "--cpu-moe"):
                f.ncmoe, took = 999, 1
            elif flag in ("-fa", "--flash-attn"):
                if val in ("on", "off", "auto", "1", "0"):
                    f.fa = val != "off" and val != "0"
                else:
                    f.fa, took = True, 1
            elif flag in ("-ctk", "--cache-type-k"):
                f.ctk = val
            elif flag in ("-ctv", "--cache-type-v"):
                f.ctv = val
            elif flag in ("-b", "--batch-size"):
                f.b = int(val)
            elif flag in ("-ub", "--ubatch-size"):
                f.ub = int(val)
            elif flag in ("-np", "--parallel"):
                f.np = int(val)
            elif flag in ("-t", "--threads"):
                f.threads = int(val)
            elif flag == "--no-mmap":
                f.mmap, took = False, 1
            elif flag == "--mmap":
                f.mmap, took = True, 1
            elif flag in ("-lm", "--load-mode"):
                f.mmap = _LOAD_MODE_MMAP.get(val, True)
            elif flag == "--swa-full":
                f.swa_full, took = True, 1
            elif flag in ("-kvu", "--kv-unified"):
                f.kvu, took = True, 1
            elif flag in ("-mm", "--mmproj"):
                mmproj_path = val
            elif flag == "--no-mmproj-offload":
                f.mmproj_offload, took = False, 1
            elif flag == "--spec-type":
                f.mtp = "mtp" in val
                if not f.mtp:
                    notes.append("draft model (%s) is not sized" % val)
            elif flag in ("-ot", "--override-tensor"):
                notes.append("-ot %s is not modelled; placement ignores it"
                             % val)
            elif flag in ("-md", "--model-draft"):
                notes.append("draft model is not sized")
            else:
                took = 1
        except ValueError:
            notes.append("could not read %s %s" % (flag, val))
        i += took
    return f, notes, model, mmproj_path


# ------------------------------------------------------------- the shape --

def _kv_dims(inv, il, tensors):
    """(K elements, V elements) per KV cell for one layer, or (0, 0)."""
    by = {t.name.split(".", 2)[2]: t for t in tensors}
    if any(t.role == "recurrent" for t in tensors):
        return 0, 0
    k = by.get("attn_k.weight")
    if k is not None:
        v = by.get("attn_v.weight")
        kd = k.dims[-1]
        # No attn_v: V is K (Gemma 4's full layers), and still cached.
        return kd, (v.dims[-1] if v is not None else kd)
    if "attn_kv_a_mqa.weight" in by:                 # MLA: a compressed latent
        rank = int(inv.hp("attention.kv_lora_rank", 512) or 512)
        rope = int(inv.hp("rope.dimension_count", 64) or 64)
        return rank + rope, rank          # errs high: newer builds skip V
    if "attn_qkv.weight" in by:
        n_kv = inv.per_layer("attention.head_count_kv", il, 0) or \
            inv.per_layer("attention.head_count", il, 0)
        head = inv.n_embd // max(1, int(inv.per_layer(
            "attention.head_count", il, 1) or 1))
        dk = int(inv.hp("attention.key_length", head) or head)
        dv = int(inv.hp("attention.value_length", dk) or dk)
        return int(n_kv) * dk, int(n_kv) * dv
    return 0, 0


def _recurrent_floats(inv):
    """f32 elements of state per recurrent layer per sequence (r + s)."""
    embd = inv.n_embd
    if inv.hp("ssm.conv_kernel") is not None:
        conv = int(inv.hp("ssm.conv_kernel") or 0)
        inner = int(inv.hp("ssm.inner_size") or 0)
        state = int(inv.hp("ssm.state_size") or 0)
        groups = int(inv.hp("ssm.group_count") or 0)
        return max(conv - 1, 0) * (inner + 2 * groups * state) + state * inner
    if inv.hp("shortconv.l_cache") is not None:
        return (int(inv.hp("shortconv.l_cache")) - 1) * embd
    if inv.hp("wkv.head_size") is not None:
        shift = int(inv.hp("token_shift_count", 2) or 2)
        return shift * embd + embd * int(inv.hp("wkv.head_size"))
    return 0


def _swa_layers(inv):
    """Which layers use the sliding window, from the metadata."""
    n = inv.n_layer
    window = int(inv.hp("attention.sliding_window", 0) or 0)
    if not window:
        return [False] * n, 0
    pattern = inv.hp("attention.sliding_window_pattern")
    if isinstance(pattern, list):
        return [bool(pattern[i]) if i < len(pattern) else False
                for i in range(n)], window
    if isinstance(pattern, int) and pattern > 1:
        return [(i % pattern) < pattern - 1 for i in range(n)], window
    # Older GGUFs leave the pattern to llama.cpp's per-arch table.
    table = {"gemma2": 2, "gemma3": 6, "gemma3n": 5, "cohere2": 4,
             "gpt-oss": 2, "llama4": 4, "exaone4": 4}
    p = table.get(inv.arch)
    if p:
        return [(i % p) < p - 1 for i in range(n)], window
    return [True] * n, window            # every layer windowed (Mistral-style)


class Shape:
    """Per-layer weights and cache sizes, precomputed once per model so
    the search can evaluate a config in constant-ish time."""

    def __init__(self, inv):
        self.inv = inv
        self.memo = {}               # per-placement sums the search reuses
        n = self.n_layer = inv.n_layer
        self.n_embd, self.n_vocab = inv.n_embd, inv.n_vocab
        self.arch = inv.arch
        n_mtp = int(inv.hp("nextn_predict_layers", 0) or 0)
        self.mtp_from = n - n_mtp if n_mtp else n
        layers = inv.layer_tensors()
        self.w_dense = [0] * n       # everything in the layer but routed experts
        self.w_exps = [0] * n        # routed experts: what --n-cpu-moe moves
        self.kv_k = [0] * n
        self.kv_v = [0] * n
        self.recurrent = [False] * n
        self.swa, self.n_swa = _swa_layers(inv)
        shared = int(inv.hp("attention.shared_kv_layers", 0) or 0)
        # the same sums, weighted by how fast each side reads each type
        self.eff = {"gpu": ([0.0] * n, [0.0] * n), "cpu": ([0.0] * n,
                                                          [0.0] * n)}
        for il in range(n):
            ts = layers.get(il, [])
            for t in ts:
                exps = t.role == "exps"
                if exps:
                    self.w_exps[il] += t.nbytes
                else:
                    self.w_dense[il] += t.nbytes
                for side, table in (("gpu", TYPE_EFF_GPU),
                                    ("cpu", TYPE_EFF_CPU)):
                    self.eff[side][1 if exps else 0][il] += eff_bytes(t, table)
            self.recurrent[il] = any(t.role == "recurrent" for t in ts)
            if il < n - shared:  # Gemma 3n/4 reuse earlier layers' KV
                self.kv_k[il], self.kv_v[il] = _kv_dims(inv, il, ts)
        self.rs_floats = _recurrent_floats(inv)
        # n_head x head size per attention layer, for prefill's attention
        self.attn_width = [0] * n
        for il in range(n):
            if self.kv_k[il]:
                heads = int(inv.per_layer("attention.head_count", il, 0) or 0)
                kv_heads = int(inv.per_layer("attention.head_count_kv", il, 0)
                               or 0) or heads or 1
                self.attn_width[il] = max(1, heads) * self.kv_k[il] // kv_heads
        self.w_host_fixed = 0                  # token embeddings and friends
        self.w_output = 0                      # output head, output norm
        self.w_first = 0                       # rides with layer 0's device
        for t in inv.tensors:
            if t.layer is not None:
                continue
            if t.role == "embd":
                self.w_host_fixed += t.nbytes
            elif t.name.startswith(("per_layer_model_proj",
                                    "per_layer_proj_norm")):
                self.w_first += t.nbytes       # checked: -ngl 10 on Gemma 4
            else:
                self.w_output += t.nbytes
        if inv.tied_embeddings:
            embd = inv.find("token_embd.weight")
            if embd is not None:  # a copy is made for the output device
                self.w_output += embd.nbytes
        out = [t for t in inv.tensors if t.layer is None and t.role in (
            "output", "output_norm")]
        if inv.tied_embeddings and inv.find("token_embd.weight") is not None:
            out.append(inv.find("token_embd.weight"))
        self.w_output_eff = {"gpu": sum(eff_bytes(t, TYPE_EFF_GPU)
                                        for t in out),
                             "cpu": sum(eff_bytes(t, TYPE_EFF_CPU)
                                        for t in out)}
        self.n_head = int(inv.per_layer("attention.head_count", 0, 0) or 0)
        # active bytes per token for the perf model: routed experts count
        # at n_used / n_expert, everything else in full
        ne, nu = inv.n_expert, inv.n_expert_used
        self.exp_frac = (nu / ne) if ne and nu else 1.0
        self.n_params = inv.n_params
        self.active_params = sum(
            t.n_elements * (self.exp_frac if t.role == "exps" else 1.0)
            for t in inv.tensors
            if t.role != "embd" and (t.layer is None or t.layer < self.mtp_from))
        self.is_moe = inv.is_moe

    def live(self, il, flags):
        """False for layers llama.cpp does not load (an unused MTP head)."""
        return il < self.mtp_from or flags.mtp

    def gpu_start(self, flags):
        """First layer on the GPU. -ngl counts the output layer too, so
        -ngl L puts the last L-1 blocks and the output head on the GPU."""
        return max(self.n_layer + 1 - flags.ngl, 0)

    def output_on_gpu(self, flags):
        return flags.ngl >= 1

    def cells(self, flags):
        """(full-attention cells, SWA cells), both summed over streams."""
        ctx = pad(flags.ctx)
        if not self.n_swa or flags.swa_full:
            return ctx, ctx
        if flags.kvu or flags.np <= 1:
            return ctx, min(ctx, pad(self.n_swa * flags.np + flags.ub))
        per = pad(flags.ctx // flags.np)
        return ctx, flags.np * min(per, pad(self.n_swa + flags.ub))

    def max_host_exps_layer(self, flags):
        top = 0
        for il in range(min(flags.ncmoe, self.n_layer)):
            if self.live(il, flags):
                top = max(top, self.w_exps[il])
        return top


class Memory:
    """Bytes per component per side. Every number decomposes."""

    FIELDS = ("gpu_weights", "gpu_kv", "gpu_rs", "gpu_compute", "gpu_mmproj",
              "host_weights", "host_kv", "host_rs", "host_compute",
              "host_mmproj")
    __slots__ = FIELDS + ("host_exps", "host_exps_layers", "host_layers",
                          "host_embd")

    def __init__(self):
        for k in self.__slots__:
            setattr(self, k, 0)

    @property
    def gpu(self):
        return (self.gpu_weights + self.gpu_kv + self.gpu_rs
                + self.gpu_compute + self.gpu_mmproj)

    @property
    def host(self):
        return (self.host_weights + self.host_kv + self.host_rs
                + self.host_compute + self.host_mmproj)

    @property
    def gpu_context(self):
        """KV + recurrent state: llama.cpp's 'context' column."""
        return self.gpu_kv + self.gpu_rs

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


def weights(shape, flags):
    """(gpu bytes, host bytes, host expert bytes, layers with anything on host)."""
    key = ("w", flags.ngl, flags.ncmoe, flags.mtp)
    if key not in shape.memo:
        shape.memo[key] = _weights(shape, flags)
    return shape.memo[key]


def _weights(shape, flags):
    start = shape.gpu_start(flags)
    gpu = host = host_exps = 0
    host_layers = 0
    for il in range(shape.n_layer):
        if not shape.live(il, flags):
            continue
        dense, exps = shape.w_dense[il], shape.w_exps[il]
        if il < start:
            host += dense + exps
            host_exps += exps
            host_layers += 1
        elif il < flags.ncmoe:
            gpu += dense
            host += exps
            host_exps += exps
            host_layers += 1 if exps else 0
        else:
            gpu += dense + exps
    if shape.output_on_gpu(flags):
        gpu += shape.w_output
    else:
        host += shape.w_output
    if start == 0:
        gpu += shape.w_first
    else:
        host += shape.w_first
    host += shape.w_host_fixed
    return gpu, host, host_exps, host_layers


def cache_coeffs(shape, flags):
    """Per side: [bytes per full-attention cell, per SWA cell, recurrent
    state bytes for one sequence]. Memoised: the search asks for the same
    placement at a dozen context sizes."""
    key = ("kv", flags.ngl, flags.ctk, flags.ctv, flags.mtp)
    if key in shape.memo:
        return shape.memo[key]
    sk, sv = KV_BYTES.get(flags.ctk, 2.0), KV_BYTES.get(flags.ctv, 2.0)
    start = shape.gpu_start(flags)
    out = [[0.0, 0.0, 0], [0.0, 0.0, 0]]
    for il in range(shape.n_layer):
        if not shape.live(il, flags):
            continue
        side = out[0 if il >= start else 1]
        if shape.recurrent[il]:
            side[2] += shape.rs_floats * 4
        elif shape.kv_k[il]:
            per = shape.kv_k[il] * sk + shape.kv_v[il] * sv
            side[1 if shape.swa[il] else 0] += per
    shape.memo[key] = out
    return out


def cache(shape, flags):
    """(gpu kv, host kv, gpu recurrent state, host recurrent state) bytes."""
    full, swa = shape.cells(flags)
    (gf, gs, gr), (hf, hs, hr) = cache_coeffs(shape, flags)
    seqs = max(1, flags.np)
    return (int(full * gf + swa * gs), int(full * hf + swa * hs),
            gr * seqs, hr * seqs)


# Seed compute-buffer coefficients, read off 217 oracle runs across five
# archs on build 10424. Where a term had to be guessed it was guessed
# high: an uncalibrated prediction may be pessimistic, never optimistic.
SEED = {"out_extra": 16 * MiB, "kappa": 1.0}
BIG_ALLOC = 1 << 62


def beta(shape):
    """Bytes per token of the block-side peak at a large ubatch.

    Tracks the vocab on the models measured, and the widest activation
    on the ones where that is bigger (qwen35 27B's 6144-wide delta net).
    """
    if "beta" in shape.memo:
        return shape.memo["beta"]
    inv = shape.inv
    widths = [shape.n_embd, int(inv.hp("ssm.inner_size", 0) or 0)]
    for il in range(shape.n_layer):
        if shape.kv_k[il]:
            heads = int(inv.per_layer("attention.head_count", il, 0) or 0)
            kv_heads = int(inv.per_layer("attention.head_count_kv", il, 0)
                           or 0) or 1
            widths.append(heads * shape.kv_k[il] // kv_heads)
    shape.memo["beta"] = max(shape.n_vocab, 48 * max(widths))
    return shape.memo["beta"]


def host_peak(shape, flags):
    """The biggest slab of host-resident weights one layer sends up to the
    card when a large ubatch is offloaded for compute."""
    key = ("peak", flags.ngl, flags.ncmoe, flags.mtp)
    if key in shape.memo:
        return shape.memo[key]
    top = shape.max_host_exps_layer(flags)
    for il in range(min(shape.gpu_start(flags), shape.n_layer)):
        if shape.live(il, flags):
            top = max(top, shape.w_dense[il] + shape.w_exps[il])
    shape.memo[key] = top
    return top


def big_ub(shape):
    """The ubatch from which the big-ubatch phase shows up. Measured at
    2048 on every arch but the sliding-window ones, where it is 1024."""
    return 1024 if any(shape.swa) else 2048


def compute(shape, flags, coef=None, max_alloc=BIG_ALLOC):
    """(gpu, host) compute-buffer bytes.

    The GPU peak is the largest of three phases, and which one wins moves
    with the ubatch - that is why one formula with fitted slopes kept
    failing against the oracle:

      output   logits for the whole ubatch, 4 x ub x n_vocab, plus change
      offload  one layer's host weights copied up for compute, plus the mask
      big-ub   from ubatch 2048 up, logits + beta x ub + the mask together

    The KQ mask (f16, ub x cells per stream) lives on the host as an
    input and is copied to the card. A logits tensor over the backend's
    largest allocation (Vulkan's 4 GiB) is computed on the CPU instead.
    """
    c = dict(SEED)
    if coef:
        c.update(coef)
    ub = min(flags.ub, flags.b)
    full, swa = shape.cells(flags)
    streams = 1 if flags.kvu else max(1, flags.np)
    has_swa = any(shape.swa) and shape.n_swa and not flags.swa_full
    mask = ub * 2 * (full + (swa if has_swa else 0)) // streams
    ple = int(shape.inv.hp("embedding_length_per_layer_input", 0) or 0)
    logits = 4 * ub * shape.n_vocab
    # exactly 4 GiB is already too big: gemma's 262144 x 4096 x 4 moves
    out_gpu = shape.output_on_gpu(flags) and logits < max_alloc
    host = mask + ub * (8 * shape.n_embd + 4 * ple * shape.n_layer)
    if shape.gpu_start(flags) > 0:
        host += 4 * ub * shape.n_embd
    if not out_gpu:
        host += logits
    on_card = logits if out_gpu else 0
    phases = [on_card + c["out_extra"]]
    peak = host_peak(shape, flags)
    if peak and ub >= 32:
        phases.append(c["kappa"] * peak + mask)
    if ub >= c.get("big_ub", big_ub(shape)):
        phases.append(on_card + c.get("beta", beta(shape)) * ub + mask)
    gpu = max(phases)
    if not flags.fa:
        # without flash attention the KQ matrix is materialised per head
        gpu += ub * (full // streams) * 4 * max(1, shape.n_head)
    return int(gpu), int(host)


def mmproj_cost(flags):
    """A projector is weights plus its own compute buffer. Flat add, sized
    from one measurement (a 0.9 G BF16 projector cost 1.6 G on the card)."""
    if not flags.mmproj:
        return 0
    return int(flags.mmproj * 1.85)


def memory(shape, flags, coef=None, max_alloc=BIG_ALLOC, residual=0):
    """The whole bill. residual is what calibration has learned this exact
    file gets wrong, added to the GPU compute term where it lives."""
    m = Memory()
    gw, hw, hexps, hlayers = weights(shape, flags)
    gkv, hkv, grs, hrs = cache(shape, flags)
    gc, hc = compute(shape, flags, coef, max_alloc)
    gc = max(0, gc + int(residual))
    m.gpu_weights, m.host_weights = gw, hw
    m.gpu_kv, m.host_kv, m.gpu_rs, m.host_rs = gkv, hkv, grs, hrs
    m.gpu_compute, m.host_compute = gc, hc
    if flags.mmproj_offload:
        m.gpu_mmproj = mmproj_cost(flags)
    else:
        m.host_mmproj = mmproj_cost(flags)
    m.host_exps, m.host_layers = hexps, hlayers
    m.host_exps_layers = min(flags.ncmoe, shape.n_layer)
    m.host_embd = shape.w_host_fixed
    return m


def human(nbytes, digits=1):
    return ("%%.%df" % digits) % (nbytes / GiB)


_KV_NAME = re.compile(r"^(f16|bf16|f32|q8_0|q5_1|q5_0|q4_1|q4_0|iq4_nl)$")


def valid_kv(name):
    return bool(_KV_NAME.match(str(name)))
