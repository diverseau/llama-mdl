"""Why a config does not fit - predicted, or seen dying in a load log -
and the cheapest ways out, ranked by what each one costs you."""

from pathlib import Path

from . import calib, model, search

K = 1024
KV_STEPS = [("f16", "f16"), ("q8_0", "q8_0"), ("q8_0", "q4_0"),
            ("q4_0", "q4_0")]


def _kv_rank(ctk, ctv):
    return model.KV_RANK.get(ctk, 0) * 2 + model.KV_RANK.get(ctv, 0)


class Fix:
    def __init__(self, label, fit, base, kind):
        self.label, self.fit, self.base, self.kind = label, fit, base, kind

    @property
    def vram(self):
        return self.fit.gpu - self.base.gpu

    @property
    def decode(self):
        return self.fit.speed.decode0 - self.base.speed.decode0

    @property
    def turn(self):
        b = self.base.speed.s_turn
        return (self.fit.speed.s_turn - b) / b if b else 0.0

    @property
    def lossy(self):
        """A 4-bit K or V costs quality you can measure; q8_0 does not."""
        f = self.fit.flags
        return self.kind == "kv" and max(model.KV_RANK.get(f.ctk, 0),
                                         model.KV_RANK.get(f.ctv, 0)) >= 3

    def cost(self):
        """Context cuts last (the spec's own rule: never lead with them),
        then 4-bit KV, then whatever slows the agent turn least. f16 ->
        q8_0 is close enough to free that it competes on speed alone."""
        return (self.kind == "ctx", self.lossy, round(self.turn, 3))


def overshoot(ctx_obj, fit):
    return (fit.gpu - ctx_obj.machine.vram_usable,
            fit.mem.host - ctx_obj.machine.ram_usable)


def headline(fit):
    """The component most responsible, in words."""
    m, f = fit.mem, fit.flags
    parts = [(m.gpu_kv, "KV %s @ %dk" % (f.kv_label, f.ctx // K)),
             (m.gpu_weights, "weights on the GPU"),
             (m.gpu_compute, "compute buffer at ubatch %d" % f.ub),
             (m.gpu_mmproj, "vision projector")]
    size, what = max(parts)
    return "%s is %.1f G of the total" % (what, size / model.GiB)


def _first_fitting(ctx_obj, tries, depth):
    for label, flags, kind in tries:
        fit = ctx_obj.evaluate(flags, depth)
        if ctx_obj.fits(fit.mem):
            return label, fit, kind
    return None


def _merge(flags, a, b):
    """b, with whatever a changed relative to flags applied on top."""
    here = flags.as_dict()
    return b.replace(**{k: v for k, v in a.as_dict().items() if v != here[k]})


def fixes(ctx_obj, flags, opts):
    """(evaluated config, fixes cheapest first, notes).

    Single changes first, each the smallest step in its direction that
    makes the config fit. When no single change will do, pairs of them.
    A context cut is always offered, and always ranked last.
    """
    base = ctx_obj.evaluate(flags, opts.depth)
    shape, out, notes = ctx_obj.shape, [], []
    here = _kv_rank(flags.ctk, flags.ctv)
    kv = [("kv %s → %s/%s" % (flags.kv_label, k, v),
           flags.replace(ctk=k, ctv=v), "kv")
          for k, v in KV_STEPS if _kv_rank(k, v) > here]
    moe = []
    if shape.is_moe and flags.ncmoe < shape.n_layer:
        moe = [("n-cpu-moe %d → %d" % (flags.ncmoe, n),
                flags.replace(ncmoe=n), "offload")
               for n in range(flags.ncmoe + 1, shape.n_layer + 1)]
    top = min(flags.ngl, shape.n_layer + 1)
    ngl = [("ngl %d → %d" % (top, n), flags.replace(ngl=n), "offload")
           for n in range(top - 1, -1, -1)] if not shape.is_moe else []
    ub = [("ubatch %d → %d" % (flags.ub, u), flags.replace(ub=u), "ub")
          for u in (1024, 512, 256) if u < flags.ub]
    extra = []
    if flags.np > 1:
        extra.append(("parallel %d → 1" % flags.np, flags.replace(np=1),
                      "np"))
    if flags.mmproj and flags.mmproj_offload:
        extra.append(("projector off the GPU (--no-mmproj-offload)",
                      flags.replace(mmproj_offload=False), "mmproj"))
    groups = {"kv": kv, "offload": moe + ngl, "ub": ub, "extra": extra}
    for group in groups.values():
        hit = _first_fitting(ctx_obj, group, opts.depth)
        if hit:
            out.append(Fix(hit[0], hit[1], base, hit[2]))
    if not out:                        # nothing alone will do: pair them
        names = list(groups)
        for i, ga in enumerate(names):
            for gb in names[i + 1:]:
                for la, fa, _ in groups[ga]:
                    tries = [(la + " + " + lb, _merge(flags, fa, fb), kb)
                             for lb, fb, kb in groups[gb]]
                    hit = _first_fitting(ctx_obj, tries, opts.depth)
                    if hit:
                        out.append(Fix(hit[0], hit[1], base, hit[2]))
                        break
    ctx = search.max_ctx(ctx_obj, flags, 4 * K, flags.ctx)
    if ctx and ctx < flags.ctx:
        fit = ctx_obj.evaluate(flags.replace(ctx=ctx), opts.depth)
        out.append(Fix("ctx %dk → %dk" % (flags.ctx // K, ctx // K), fit,
                       base, "ctx"))
    for label, f2, _ in ub:            # a smaller ubatch buys back context
        c2 = search.max_ctx(ctx_obj, f2, 4 * K, flags.ctx)
        if c2 and ctx and c2 > ctx and c2 < flags.ctx:
            fit = ctx_obj.evaluate(f2.replace(ctx=c2), opts.depth)
            out.append(Fix("%s + ctx %dk → %dk" % (label, flags.ctx // K,
                                                   c2 // K), fit, base, "ctx"))
            break
    if moe and not any(fx.kind == "offload" for fx in out):
        lean = ctx_obj.memory(flags.replace(ncmoe=shape.n_layer))
        if lean.host > ctx_obj.machine.ram_usable:
            notes.append("more experts on the CPU would need %.1f G of RAM; "
                         "%.1f G is usable" % (
                             lean.host / model.GiB,
                             ctx_obj.machine.ram_usable / model.GiB))
    out.sort(key=Fix.cost)
    return base, out, notes


def log_evidence(name, state_dir):
    """What the last load log of <name> says went wrong, if anything."""
    for suffix in ("", ".1", ".2"):
        path = Path(state_dir) / ("%s.log%s" % (name, suffix))
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        found = calib.scrape_log(text)
        lines = [ln.strip() for ln in text.splitlines()
                 if "failed to allocate" in ln or "out of memory" in ln.lower()
                 or "ErrorOutOfDeviceMemory" in ln]
        if found["failed"] or lines:
            return path, found, lines[:4]
    return None, None, []
