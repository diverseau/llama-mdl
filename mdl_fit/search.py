"""The search: every config worth trying, the feasible ones, the winners.

Brute force. The space is KV pair x ubatch x n-cpu-moe (or -ngl for a
dense model). For each discrete combo the largest context that fits is a
binary search over a function that is linear in context, and speed does
not depend on context at all - only on how deep the KV is when you read
it - so each combo is worth exactly one candidate, at its max context.
A few hundred combos, microseconds each.
"""

from . import calib, model, perf

K = 1024
# (K type, V type). Quantising K costs more quality than quantising V,
# so the mixed pair keeps K at q8_0.
KV_PAIRS = {"f16": [("f16", "f16")],
            "q8_0": [("f16", "f16"), ("q8_0", "q8_0")],
            "q4_0": [("f16", "f16"), ("q8_0", "q8_0"), ("q8_0", "q4_0"),
                     ("q4_0", "q4_0")]}
UBATCHES = (512, 1024, 2048, 4096)
CTX_STEP = 1024

PROFILES = {
    "agent": {"min_ctx": 128 * K, "min_tps": 0, "depth": 48 * K,
              "goal": "s_turn", "blurb": "ctx ≥ 128k"},
    "chat": {"min_ctx": 32 * K, "min_tps": 0, "depth": 8 * K,
             "goal": "decode_d", "blurb": "ctx ≥ 32k"},
    "max-ctx": {"min_ctx": 8 * K, "min_tps": 15, "depth": 8 * K,
                "goal": "ctx", "blurb": "decode ≥ 15 t/s"},
    "speed": {"min_ctx": 8 * K, "min_tps": 0, "depth": 0,
              "goal": "decode0", "blurb": "ctx ≥ 8k"},
}


class Options:
    def __init__(self, profile="agent", min_ctx=None, min_tps=None,
                 kv_floor="q8_0", np=1, max_ctx=None, mmproj=0,
                 mmproj_offload=True, threads=0, ubatches=UBATCHES):
        if profile not in PROFILES:
            raise ValueError("unknown profile %r (have: %s)" % (
                profile, ", ".join(PROFILES)))
        p = PROFILES[profile]
        self.profile = profile
        self.min_ctx = p["min_ctx"] if min_ctx is None else min_ctx
        self.min_tps = p["min_tps"] if min_tps is None else min_tps
        self.depth, self.goal = p["depth"], p["goal"]
        self.kv_floor = kv_floor if kv_floor in KV_PAIRS else "q8_0"
        self.np, self.max_ctx = max(1, np), max_ctx
        self.mmproj, self.mmproj_offload = mmproj, mmproj_offload
        self.threads = threads
        self.ubatches = tuple(ubatches)

    @property
    def blurb(self):
        parts = []
        if self.min_ctx:
            parts.append("ctx ≥ %dk" % (self.min_ctx // K))
        if self.min_tps:
            parts.append("decode ≥ %g t/s" % self.min_tps)
        return " · ".join(parts) or "no floors"


class Fit:
    """One config, with everything the output shows about it."""

    def __init__(self, flags, mem, speed, residual=0):
        self.flags, self.mem, self.speed = flags, mem, speed
        self.residual = residual
        self.oracle = None          # the oracle's gpu [model, ctx, compute]
        self.note = ""

    @property
    def gpu(self):
        if self.oracle:
            return sum(self.oracle) + self.mem.gpu_mmproj
        return self.mem.gpu

    def key(self):
        f = self.flags
        return (f.ctk, f.ctv, f.ub, f.ncmoe, f.ngl, f.ctx)


class Context:
    """Everything a search needs that is not the flags."""

    def __init__(self, inv, machine, residuals=None, build=None):
        self.inv = inv
        self.shape = model.Shape(inv)
        self.machine = machine
        self.build = build or (machine.build or {}).get("build")
        self.res = residuals if residuals is not None else calib.Residuals(
            build=self.build)
        self.sig = calib.signature(inv)
        self.eff = calib.efficiency(inv.arch)

    def residual(self, flags):
        return self.res.lookup(self.sig, self.inv.arch, flags)

    def memory(self, flags):
        return model.memory(self.shape, flags, None, self.machine.max_alloc,
                            self.residual(flags))

    def fits(self, mem):
        return (mem.gpu <= self.machine.vram_usable
                and mem.host <= self.machine.ram_usable)

    def tight(self, fit):
        """Inside the free VRAM, but eating the safety margin: it loads
        today, and a browser tab opening later can push it over."""
        return (self.machine.vram_usable < fit.gpu <= self.machine.vram_free
                and fit.mem.host <= self.machine.ram_usable)

    def evaluate(self, flags, depth):
        mem = self.memory(flags)
        sp = perf.speed(self.shape, flags, self.machine, depth, self.eff)
        return Fit(flags, mem, sp, self.residual(flags))

    def mem_band(self, fit):
        if fit.oracle:
            return 64 * model.MiB
        return self.res.band(self.sig, self.inv.arch, fit.mem.gpu_compute)

    def speed_band(self):
        return perf.band(self.machine, self.eff.get("runs", 0))


def placements(shape):
    """[(ngl, ncmoe)] from all-on-GPU to all-on-CPU, without repeats.

    A MoE sheds routed experts layer by layer first; only when every
    expert is already on the CPU does it start giving up whole layers.
    A dense model only has whole layers to give.
    """
    out, seen = [], set()
    full = shape.n_layer + 1
    if shape.is_moe:
        for n in range(0, shape.n_layer + 1):
            host = sum(shape.w_exps[:n])
            if host in seen:
                continue
            seen.add(host)
            out.append((999, n))
        for ngl in range(shape.n_layer, -1, -1):
            out.append((ngl, shape.n_layer))
    else:
        for ngl in range(full, -1, -1):
            out.append((999 if ngl == full else ngl, 0))
    return out


def max_ctx(ctx_obj, base, lo, hi):
    """Largest context in [lo, hi] (CTX_STEP multiples) that fits, or 0."""
    step = CTX_STEP
    lo, hi = max(step, lo // step * step), hi // step * step
    if hi < lo or not ctx_obj.fits(ctx_obj.memory(base.replace(ctx=lo))):
        return 0
    if ctx_obj.fits(ctx_obj.memory(base.replace(ctx=hi))):
        return hi
    a, b = lo // step, hi // step          # a fits, b does not
    while b - a > 1:
        mid = (a + b) // 2
        if ctx_obj.fits(ctx_obj.memory(base.replace(ctx=mid * step))):
            a = mid
        else:
            b = mid
    return a * step


def ctx_cap(ctx_obj, opts):
    cap = ctx_obj.inv.n_ctx_train or 128 * K
    if opts.max_ctx:
        cap = min(cap, opts.max_ctx)
    return cap


def candidates(ctx_obj, opts, floor_ctx=None):
    """One Fit per feasible (kv, ub, placement) combo, at its max context."""
    shape, out = ctx_obj.shape, []
    cap = ctx_cap(ctx_obj, opts)
    lo = min(floor_ctx if floor_ctx is not None else opts.min_ctx, cap)
    lo = max(lo, CTX_STEP)
    for ctk, ctv in KV_PAIRS[opts.kv_floor]:
        for ub in opts.ubatches:
            if 4 * ub * shape.n_vocab >= ctx_obj.machine.max_alloc:
                continue                  # logits would land on the CPU
            prev_ctx = None
            for ngl, ncmoe in placements(shape):
                base = model.Flags(ngl=ngl, ncmoe=ncmoe, ctk=ctk, ctv=ctv,
                                   ub=ub, b=max(2048, ub), np=opts.np,
                                   mmproj=opts.mmproj,
                                   mmproj_offload=opts.mmproj_offload,
                                   threads=opts.threads, fa=True)
                ctx = max_ctx(ctx_obj, base, lo, cap)
                if not ctx:
                    continue
                # Moving more onto the CPU only helps if it bought context;
                # once the context is capped, every further step is a loss.
                if prev_ctx is not None and ctx <= prev_ctx and ctx >= cap:
                    break
                prev_ctx = ctx
                flags = base.replace(ctx=ctx)
                flags.mmap = choose_mmap(ctx_obj, flags)
                out.append(ctx_obj.evaluate(flags, opts.depth))
    return out


def choose_mmap(ctx_obj, flags):
    """--no-mmap when the host weights fit with room to spare (llama.cpp
    warns that CPU-resident experts plus mmap is slower); mmap when RAM is
    tight or nothing lives on the host anyway."""
    _, host_w, host_exps, _ = model.weights(ctx_obj.shape, flags)
    if not host_exps and flags.ngl >= ctx_obj.shape.n_layer + 1:
        return True
    return not (host_w * 1.1 < ctx_obj.machine.ram_usable)


def score(fit, goal):
    """Bigger is better, for every goal."""
    s, f = fit.speed, fit.flags
    kvq = -(model.KV_RANK.get(f.ctk, 0) * 2 + model.KV_RANK.get(f.ctv, 0))
    main = {"s_turn": -s.s_turn, "decode_d": s.decode_d, "decode0": s.decode0,
            "ctx": f.ctx}[goal]
    return (round(main, 3), kvq, f.ctx, s.prefill)


def meets(fit, opts):
    return fit.flags.ctx >= opts.min_ctx and fit.speed.decode_d >= opts.min_tps


def pareto(fits):
    """Fits nobody beats on context, decode at depth, prefill and KV quality."""
    def dims(f):
        return (f.flags.ctx, f.speed.decode_d, f.speed.prefill,
                -(model.KV_RANK.get(f.flags.ctk, 0) * 2
                  + model.KV_RANK.get(f.flags.ctv, 0)))
    pts = [(dims(f), f) for f in fits]
    out = []
    for d, f in pts:
        if not any(all(x >= y for x, y in zip(e, d, strict=True)) and e != d
                   for e, _ in pts):
            out.append(f)
    return out


def differences(a, b):
    """[(category, words)] for how runner-up b differs from winner a."""
    out = []
    if b.flags.ctx > a.flags.ctx * 1.1:
        out.append(("ctx+", "+%dk ctx" % ((b.flags.ctx - a.flags.ctx) // K)))
    elif b.flags.ctx < a.flags.ctx * 0.9:
        out.append(("ctx-", "less ctx"))
    dd = b.speed.decode_d - a.speed.decode_d
    if abs(dd) >= 1:
        out.append(("tps+" if dd > 0 else "tps-",
                    "%s%.0f t/s" % ("+" if dd > 0 else "−", abs(dd))))
    rank = model.KV_RANK
    for side, x, y in (("K", a.flags.ctk, b.flags.ctk),
                       ("V", a.flags.ctv, b.flags.ctv)):
        if rank.get(y, 0) > rank.get(x, 0):
            out.append(("kv-", "%s quantized" % side))
        elif rank.get(y, 0) < rank.get(x, 0):
            out.append(("kv+", "%s %s" % (side, y)))
    if b.speed.prefill > a.speed.prefill * 1.15:
        out.append(("pp+", "faster prefill"))
    return out


def pick(fits, opts, n=3):
    """Winner plus runner-ups that each differ from it in a different way.

    Two runner-ups that are both 'less context, slower' tell you nothing
    the first one did not, so the second is dropped whatever its numbers.
    """
    if not fits:
        return []
    ranked = sorted(fits, key=lambda f: score(f, opts.goal), reverse=True)
    best = ranked[0]
    front = [f for f in pareto(fits) if f is not best]
    front.sort(key=lambda f: score(f, opts.goal), reverse=True)
    chosen, kinds = [best], set()
    for f in front + ranked[1:]:
        if len(chosen) >= n:
            break
        diff = differences(best, f)
        kind = frozenset(c for c, _ in diff)
        # 'strictly worse on everything shown' is never worth a row, and
        # neither is giving up a fifth of the speed without buying context
        slow = f.speed.decode_d < 0.8 * best.speed.decode_d
        if not diff or kind in kinds or kind <= {"ctx-", "tps-", "kv-"} \
                or (slow and "ctx+" not in kind) \
                or any(f.key() == c.key() for c in chosen):
            continue
        kinds.add(kind)
        f.note = ", ".join(words for _, words in diff[:3])
        chosen.append(f)
    best.note = "★"
    return chosen


class Result:
    def __init__(self, ctx_obj, opts, picks, relaxed=None, everything=None,
                 closest=None):
        self.ctx, self.opts, self.picks = ctx_obj, opts, picks
        self.relaxed = relaxed          # why the floors had to give, if they did
        self.everything = everything or []
        self.closest = closest          # (flags, memory) when nothing fits

    @property
    def best(self):
        return self.picks[0] if self.picks else None


def offload_label(flags, shape):
    """'ncmoe 15', 'ngl 30' or 'all', the way the tables show it."""
    if flags.ngl < shape.n_layer + 1:
        return "ngl %d" % flags.ngl
    if shape.is_moe and flags.ncmoe:
        return "ncmoe %d" % min(flags.ncmoe, shape.n_layer)
    return "all"


def closest(ctx_obj, opts):
    """When nothing fits: which budget runs out, and where.

    Every placement at 4k context and the leanest KV allowed is costed.
    The ones the card can hold and the ones RAM can hold never overlap -
    that is what 'nothing fits' means - so the useful answer is where
    each boundary sits and what the other budget would need there.
    """
    mach, shape = ctx_obj.machine, ctx_obj.shape
    ctk, ctv = KV_PAIRS[opts.kv_floor][-1]
    rows = []
    for ngl, ncmoe in placements(shape):
        f = model.Flags(ngl=ngl, ncmoe=ncmoe, ctk=ctk, ctv=ctv, ctx=4 * K,
                        ub=512, np=opts.np, mmproj=opts.mmproj,
                        mmproj_offload=opts.mmproj_offload)
        rows.append((f, ctx_obj.memory(f)))
    card = [r for r in rows if r[1].gpu <= mach.vram_usable]
    ram = [r for r in rows if r[1].host <= mach.ram_usable]
    gib = model.GiB
    if card and not ram:
        f, m = min(card, key=lambda r: r[1].host)
        why = ("RAM is the limit: the card holds it from %s up, which needs "
               "%.1f G of host memory, and %.1f G is usable (%.0f G total, "
               "%.1f G kept for the system); past that it pages from disk "
               "on every token" % (
                   offload_label(f, shape), m.host / gib,
                   mach.ram_usable / gib, mach.ram_total / gib,
                   mach.ram_reserve / gib))
    elif ram and not card:
        f, m = min(ram, key=lambda r: r[1].gpu)
        why = ("VRAM is the limit: even at %s it needs %.1f G on the card, "
               "%.1f G usable" % (offload_label(f, shape), m.gpu / gib,
                                  mach.vram_usable / gib))
    elif card and ram:
        f, m = min(card, key=lambda r: r[1].host)
        g2, m2 = min(ram, key=lambda r: r[1].gpu)
        why = ("the card holds it from %s up, which needs %.1f G of RAM "
               "(%.1f usable); RAM holds it down to %s, which needs %.1f G "
               "on the card (%.1f usable)" % (
                   offload_label(f, shape), m.host / gib,
                   mach.ram_usable / gib, offload_label(g2, shape),
                   m2.gpu / gib, mach.vram_usable / gib))
    else:
        f, m = min(rows, key=lambda r: r[1].gpu + r[1].host)
        why = ("both are short at every placement: the leanest needs %.1f G "
               "on the card and %.1f G of RAM" % (m.gpu / gib, m.host / gib))
    return why, f, m


def solve(ctx_obj, opts):
    """Winner and runner-ups for a profile. Tuning first: if the floors
    cannot be met at this quant, say what the best it can do is rather
    than refusing."""
    fits = candidates(ctx_obj, opts)
    ok = [f for f in fits if meets(f, opts)]
    if ok:
        return Result(ctx_obj, opts, pick(ok, opts), everything=fits)
    loose = candidates(ctx_obj, opts, floor_ctx=4 * K)
    if not loose:
        why, f, m = closest(ctx_obj, opts)
        return Result(ctx_obj, opts, [], relaxed=why, everything=[],
                      closest=(f, m))
    top_ctx = max(f.flags.ctx for f in loose)
    top_tps = max(f.speed.decode_d for f in loose)
    why = []
    if top_ctx < opts.min_ctx:
        why.append("the most context this quant can hold here is %dk "
                   "(floor %dk)" % (top_ctx // K, opts.min_ctx // K))
    if opts.min_tps and top_tps < opts.min_tps:
        why.append("the fastest config decodes %.0f t/s (floor %g)" % (
            top_tps, opts.min_tps))
    relaxed_opts = Options(opts.profile, min_ctx=0, min_tps=0,
                           kv_floor=opts.kv_floor, np=opts.np,
                           max_ctx=opts.max_ctx, mmproj=opts.mmproj,
                           mmproj_offload=opts.mmproj_offload,
                           threads=opts.threads, ubatches=opts.ubatches)
    relaxed_opts.goal = opts.goal if opts.goal != "s_turn" else "ctx"
    return Result(ctx_obj, opts, pick(loose, relaxed_opts),
                  relaxed="; ".join(why) or "floors relaxed", everything=loose)


def verify_picks(ctx_obj, opts, result, fit_bin, rounds=2):
    """Put the picks in front of the oracle. Anything it says does not
    fit is booked as a residual and the search runs again with it."""
    if not fit_bin or not str(ctx_obj.inv.source) or \
            str(ctx_obj.inv.source).startswith("hf:"):
        return result
    for _ in range(rounds):
        moved = False
        for fit in result.picks:
            entry = calib.check(fit_bin, ctx_obj.inv, ctx_obj.shape,
                                fit.flags, ctx_obj.build,
                                max_alloc=ctx_obj.machine.max_alloc)
            if not entry:
                return result               # no oracle on this build
            fit.oracle = entry["actual"]
            ctx_obj.res = calib.Residuals(build=ctx_obj.build)
            if fit.gpu > ctx_obj.machine.vram_usable:
                moved = True
        if not moved:
            return result
        result = solve(ctx_obj, opts)
    for fit in result.picks:                # the last round's picks, checked
        if fit.oracle is None:
            entry = calib.check(fit_bin, ctx_obj.inv, ctx_obj.shape,
                                fit.flags, ctx_obj.build,
                                max_alloc=ctx_obj.machine.max_alloc)
            if entry:
                fit.oracle = entry["actual"]
    return result
