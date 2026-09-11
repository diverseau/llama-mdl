"""Speed: decode at a KV depth, prefill, and seconds per agent turn.

Decode is bytes over bandwidth, per side, plus a hop cost for every layer
that makes the token cross between GPU and CPU. Prefill for a large
ubatch is dominated by shipping host weights over PCIe once per ubatch,
which is the coupling a VRAM guesser cannot see: a bigger -ub means
fewer trips and faster prefill, and a bigger compute buffer, and less
room, and more experts on the CPU, and slower decode.

The bandwidths are effective ones - what llama.cpp achieves on this box -
which is what `mdl fit hw` measures. Until it has, spec-sheet numbers
with a derating stand in, and the confidence band says so.
"""

import math
import re

from . import model

GB = 1e9

# Memory bandwidth, GB/s, spec sheet. Only a seed; `mdl fit hw` replaces it.
GPU_BW = [
    (r"5090", 1792), (r"5080", 960), (r"5070 ti", 896), (r"5070", 672),
    (r"5060 ti", 448), (r"5060", 448), (r"4090", 1008), (r"4080", 717),
    (r"4070 ti super", 672), (r"4070 ti", 504), (r"4070 super", 504),
    (r"4070", 504), (r"4060 ti", 288), (r"4060", 272), (r"3090", 936),
    (r"3080 ti", 912), (r"3080", 760), (r"3070 ti", 608), (r"3070", 448),
    (r"3060 ti", 448), (r"3060", 360), (r"3050", 224), (r"2080 ti", 616),
    (r"2080", 448), (r"2070", 448), (r"2060", 336), (r"a6000", 768),
    (r"a5000", 768), (r"a4000", 448), (r"a100", 1935), (r"h100", 3350),
    (r"l4\b", 300), (r"t4\b", 320), (r"7900 xtx", 960), (r"7900 xt", 800),
    (r"7800 xt", 624), (r"6800 xt", 512), (r"6700 xt", 384),
]
# FP16 dense tensor throughput, TFLOPS, spec sheet, for the prefill seed.
GPU_TFLOPS = [
    (r"5090", 210), (r"5080", 113), (r"5070", 62), (r"4090", 165),
    (r"4080", 97), (r"4070", 58), (r"4060", 30), (r"3090", 71),
    (r"3080", 59), (r"3070", 40), (r"3060", 25), (r"2080", 40),
    (r"2070", 30), (r"2060", 26),
]
PCIE_GBPS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.97, 5: 3.94}   # per lane


def _lookup(table, name, default):
    name = (name or "").lower()
    for pattern, value in table:
        if re.search(pattern, name):
            return value
    return default


def seeds(machine):
    """Effective numbers for an unmeasured machine. Deratings are what
    llama.cpp tends to reach of the spec, erring slow."""
    bw = _lookup(GPU_BW, machine.gpu_name, 300) * GB * 0.70
    tf = _lookup(GPU_TFLOPS, machine.gpu_name, 20) * 1e12 * 0.35
    gen, width = 3, 16
    if machine.pcie and machine.pcie[0]:
        gen = machine.pcie[0] or 3
        width = machine.pcie[2] or 16
    pcie = PCIE_GBPS.get(gen, 0.985) * width * GB * 0.75
    return {"bw_gpu": bw, "bw_cpu": 32 * GB, "t_hop": 0.00030,
            "bw_pcie": pcie, "f_gpu": tf, "f_cpu": 0.4e12, "t0": 0.0015}


def params(machine):
    p = seeds(machine)
    for key, value in (machine.bench or {}).items():
        if key in p and isinstance(value, (int, float)) and value > 0:
            p[key] = float(value)
    return p


class Placement:
    """Bytes read per token on each side, for one Shape and Flags.

    a_gpu and a_cpu are weighted by each tensor type's efficiency on that
    side (model.TYPE_EFF_*), so they divide by a Q8_0-equivalent
    bandwidth; raw_gpu and raw_cpu are the plain byte counts.
    """

    def __init__(self, shape, flags):
        start = shape.gpu_start(flags)
        frac = shape.exp_frac
        (gd, ge), (cd, ce) = shape.eff["gpu"], shape.eff["cpu"]
        a_gpu = a_cpu = raw_gpu = raw_cpu = 0.0
        host_layers = 0
        for il in range(shape.n_layer):
            if not shape.live(il, flags):
                continue
            dense, exps = shape.w_dense[il], shape.w_exps[il] * frac
            if il < start:
                a_cpu += cd[il] + ce[il] * frac
                raw_cpu += dense + exps
                host_layers += 1
            elif il < flags.ncmoe and shape.w_exps[il]:
                a_gpu += gd[il]
                a_cpu += ce[il] * frac
                raw_gpu += dense
                raw_cpu += exps
                host_layers += 1
            else:
                a_gpu += gd[il] + ge[il] * frac
                raw_gpu += dense + exps
        if shape.output_on_gpu(flags):
            a_gpu += shape.w_output_eff["gpu"]
            raw_gpu += shape.w_output
        else:
            a_cpu += shape.w_output_eff["cpu"]
            raw_cpu += shape.w_output
        self.a_gpu, self.a_cpu, self.host_layers = a_gpu, a_cpu, host_layers
        self.raw_gpu, self.raw_cpu = raw_gpu, raw_cpu
        # host weights a large ubatch touches: every expert gets hit
        _, host_w, _, _ = model.weights(shape, flags)
        self.w_host_touched = max(0, host_w - shape.w_host_fixed)
        # per-token KV bytes, per side, for the KV_read term
        sk = model.KV_BYTES.get(flags.ctk, 2.0)
        sv = model.KV_BYTES.get(flags.ctv, 2.0)
        self.kv_full = [0.0, 0.0]
        self.kv_swa = [0.0, 0.0]
        for il in range(shape.n_layer):
            if not shape.live(il, flags) or not shape.kv_k[il]:
                continue
            side = 0 if il >= start else 1
            per = shape.kv_k[il] * sk + shape.kv_v[il] * sv
            if shape.swa[il] and not flags.swa_full:
                self.kv_swa[side] += per
            else:
                self.kv_full[side] += per
        self.n_swa = shape.n_swa
        self.width_full = self.width_swa = 0
        for il in range(shape.n_layer):
            if not shape.live(il, flags) or not shape.kv_k[il]:
                continue
            if shape.swa[il] and not flags.swa_full:
                self.width_swa += shape.attn_width[il]
            else:
                self.width_full += shape.attn_width[il]
        self.active = shape.active_params


def decode_time(pl, p, depth, eff=1.0):
    """Seconds for one token at KV depth `depth`, single slot."""
    swa_d = min(depth, pl.n_swa) if pl.n_swa else depth
    kv_gpu = depth * pl.kv_full[0] + swa_d * pl.kv_swa[0]
    kv_cpu = depth * pl.kv_full[1] + swa_d * pl.kv_swa[1]
    t = (pl.a_gpu / p["bw_gpu"] + pl.a_cpu / p["bw_cpu"]
         + kv_gpu / p["bw_gpu"] + kv_cpu / p["bw_cpu"]
         + pl.host_layers * p["t_hop"] + p["t0"])
    return t / eff


def prefill_time(pl, p, n_tokens, ub, depth=0, eff=1.0):
    """Seconds to prefill n_tokens on top of `depth` already cached."""
    n_ub = math.ceil(n_tokens / max(1, ub))
    flops = 2.0 * pl.active * ub
    if ub >= 32:
        per_ub = pl.w_host_touched / p["bw_pcie"] + flops / p["f_gpu"]
    else:                                   # small batches stay on the CPU
        per_ub = flops / p["f_cpu"]
    # QK^T and PV: 4 x width flops per (query, key) pair; a windowed
    # layer only ever sees its window of keys
    span = depth + n_tokens / 2.0
    swa_span = min(span, pl.n_swa) if pl.n_swa else span
    attn = 4.0 * n_tokens * (pl.width_full * span
                             + pl.width_swa * swa_span) / p["f_gpu"]
    return (n_ub * per_ub + attn) / eff


class Speed:
    __slots__ = ("decode0", "decode_d", "prefill", "s_turn", "depth")

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


# Agent turn: a big new prompt every turn, a short reply. OpenCode/Hermes.
AGENT = {"p_new": 16384, "n_gen": 800, "depth": 49152}


def speed(shape, flags, machine, depth=8192, eff=None, agent=None):
    agent = agent or AGENT
    eff = eff or {}
    p = params(machine)
    pl = Placement(shape, flags)
    s = Speed()
    e_tg, e_pp = eff.get("tg", 1.0), eff.get("pp", 1.0)
    s.depth = depth
    s.decode0 = 1.0 / decode_time(pl, p, 0, e_tg)
    s.decode_d = 1.0 / decode_time(pl, p, depth, e_tg)
    s.prefill = 2048 / prefill_time(pl, p, 2048, flags.ub, 0, e_pp)
    d = min(agent["depth"], flags.ctx)
    new = min(agent["p_new"], max(0, flags.ctx - agent["n_gen"]))
    s.s_turn = (prefill_time(pl, p, new, flags.ub, max(0, d - new), e_pp)
                + agent["n_gen"] * decode_time(pl, p, d, e_tg))
    return s


def band(machine, verified_runs=0):
    """Speed uncertainty, as a fraction."""
    if verified_runs >= 3:
        return 0.10
    if machine.calibrated:
        return 0.15
    return 0.35
