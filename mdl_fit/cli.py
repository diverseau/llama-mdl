"""mdl fit - the command line. Everything here is presentation; the
numbers come from model, perf, search and calib."""

import difflib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from . import calib, emit, explain, gguf, hw, model, perf, remote, search

K = 1024
GiB = model.GiB
MiB = model.MiB

USAGE = """\
usage: mdl fit <model.gguf | hf:org/repo[:quant] | name> [options]
       mdl fit inspect <model.gguf | hf:org/repo:quant>
       mdl fit hw [--probe]
       mdl fit hw --idle | --idle-vram 0.5G --idle-ram 35% | --idle-reset
       mdl fit calibrate <model.gguf | name>

options:
  --profile P        agent (default), chat, max-ctx or speed
  --min-ctx N        context floor, e.g. 64k
  --min-tps N        decode floor at the profile's depth
  --max-ctx N        never propose more context than this
  --kv-floor T       most quantised KV allowed: f16, q8_0 (default), q4_0
  --np N             parallel slots (default 1)
  --mmproj PATH      count a vision projector on the card
  --explain          why <name> does not fit, and the cheapest fixes
  --apply N          with --explain, apply fix N to <name>; without it,
                     write pick N into <name>
  --write NAME       append the winner to models.toml as [NAME]
  --dry-run          preview --apply or --write without editing the config
  --verify           run the config: oracle + llama-bench, and learn
  --profiles         every measured profile for this model: what each
                     configuration did here, from --verify and mdl eval
  --no-oracle        analytic only; skip llama-fit-params
  --now              plan for the machine as it is this minute, open apps
                     and all, instead of at idle
  --json             machine-readable

VRAM and RAM are planned for the machine at idle: apps open during the
scan are taken back off. `mdl fit hw --idle` books the machine as it is
now as its idle state; --idle-vram/--idle-ram say it outright.
"""

VALUE_OPTS = {"--profile", "--min-ctx", "--min-tps", "--max-ctx",
              "--kv-floor", "--np", "--mmproj", "--apply", "--write",
              "--depth"}
BOOL_OPTS = {"--explain", "--verify", "--no-oracle", "--json", "--yes",
             "--now", "--profiles", "--dry-run"}


def die(msg):
    import mdl
    mdl.die(msg)


def number(text):
    """'128k' -> 131072, '1.5k' -> 1536, '4096' -> 4096."""
    m = re.fullmatch(r"\s*([\d.]+)\s*([kKmM]?)\s*", str(text))
    if not m:
        die("not a number: %r" % text)
    mult = {"": 1, "k": K, "m": K * K}[m.group(2).lower()]
    return int(float(m.group(1)) * mult)


def parse(args):
    opts, pos, i = {}, [], 0
    while i < len(args):
        a = args[i]
        if a in VALUE_OPTS:
            if i + 1 >= len(args):
                die("%s needs a value" % a)
            opts[a[2:]] = args[i + 1]
            i += 2
        elif a in BOOL_OPTS:
            opts[a[2:]] = True
            i += 1
        elif a.startswith("--"):
            die("unknown option %s\n%s" % (a, USAGE))
        else:
            pos.append(a)
            i += 1
    if opts.get("dry-run") and not (opts.get("apply") or opts.get("write")):
        die("--dry-run needs --apply N or --write NAME")
    if opts.get("dry-run") and (opts.get("verify") or (
            opts.get("explain") and not opts.get("apply"))):
        die("--dry-run needs an edit; cannot combine with these options")
    if len(pos) != 1:
        die(USAGE.rstrip())
    opts["target"] = pos[0]
    return opts


# ------------------------------------------------------------- targets --

class Target:
    """What was asked about: a file, a repo, or a models.toml entry."""

    def __init__(self):
        self.kind = self.name = self.cfg = self.flags = None
        self.inv = self.model_path = self.mmproj = None
        self.notes = []
        self.binary = "llama-server"


def _config():
    """(models, binary) without dying when there is no config yet."""
    import mdl
    try:
        return mdl.load_config()
    except mdl.MdlError:
        return {}, "llama-server"


def resolve(spec):
    t = Target()
    models, t.binary = _config()
    if spec.startswith("hf:"):
        t.kind = "hf"
        return t
    path = Path(spec).expanduser()
    if path.is_file():
        t.kind, t.model_path = "file", path
        t.inv = _load(path)
        return t
    if spec in models:
        import mdl
        t.kind, t.name, t.cfg = "name", spec, models[spec]
        argv = mdl.build_argv(spec, t.cfg, t.binary)
        t.binary = argv[0]              # the build this model runs on
        t.flags, t.notes, model_path, mmproj = model.parse_argv(argv)
        t.model_path = Path(model_path)
        if not t.model_path.is_file():
            die("model file not found: %s" % model_path)
        if mmproj:
            t.mmproj = Path(mmproj)
            if t.mmproj.is_file():
                t.flags.mmproj = t.mmproj.stat().st_size
        t.inv = _load(t.model_path)
        return t
    die("no such file, and no model named %r in the config" % spec)


def _load(path):
    try:
        return gguf.load(path)
    except (gguf.NotGGUF, OSError) as e:
        die("%s: %s" % (path, e))


def kv_floor_of(target):
    """A config that already runs 4-bit KV has made its quality call; the
    search may go as far. Everything else stops at q8_0."""
    f = target.flags if target else None
    if f and max(model.KV_RANK.get(f.ctk, 0), model.KV_RANK.get(f.ctv, 0)) >= 3:
        return "q4_0"
    return "q8_0"


def options(o, target=None):
    try:
        opts = search.Options(
            o.get("profile", "agent"),
            min_ctx=number(o["min-ctx"]) if "min-ctx" in o else None,
            min_tps=float(o["min-tps"]) if "min-tps" in o else None,
            kv_floor=o.get("kv-floor", kv_floor_of(target)),
            np=int(o.get("np", target.flags.np if target and target.flags
                         else 1)),
            max_ctx=number(o["max-ctx"]) if "max-ctx" in o else None)
    except ValueError as e:
        die(str(e))
    if "depth" in o:
        opts.depth = number(o["depth"])
    if o.get("mmproj"):
        p = Path(o["mmproj"])
        if not p.is_file():
            die("no such file: %s" % p)
        opts.mmproj = p.stat().st_size
    elif target and target.flags and target.flags.mmproj:
        opts.mmproj = target.flags.mmproj
        opts.mmproj_offload = target.flags.mmproj_offload
    return opts


# ------------------------------------------------------------ describe --

def describe(inv):
    """'qwen35moe · MoE 256×8 · hybrid (10/40 full attention) · 12.3 G'"""
    shape = model.Shape(inv)
    bits = [inv.arch]
    if inv.is_moe:
        bits.append("MoE %d×%d" % (inv.n_expert, inv.n_expert_used))
    else:
        bits.append("dense")
    attn = sum(1 for k in shape.kv_k if k)
    rec = sum(shape.recurrent)
    if rec:
        bits.append("hybrid (%d/%d attention)" % (attn, inv.n_layer))
    if any(shape.swa):
        bits.append("SWA %d (%d/%d windowed)" % (
            shape.n_swa, sum(1 for i, s in enumerate(shape.swa)
                             if s and shape.kv_k[i]), attn))
    bits.append("%.1f G" % (inv.file_size / GiB))
    return " · ".join(bits)


def g(nbytes):
    return "%.1f" % (nbytes / GiB)


def kctx(n):
    return "%dk" % round(n / K)


offload = search.offload_label


def table(ctx_obj, fits, opts):
    shape, mach = ctx_obj.shape, ctx_obj.machine
    d = kctx(opts.depth) if opts.depth else "0"
    head = (" #  ctx   kv          offload   ub    VRAM          RAM     "
            "decode 0/%-5s prefill   s/turn" % d)
    rows = [head]
    for i, f in enumerate(fits, 1):
        s = f.speed
        vram = "%s / %s%s" % (g(f.gpu), g(mach.vram_usable),
                              " ✓" if f.oracle else "")
        rows.append(" %d  %-5s %-11s %-9s %-5d %-13s %-7s %-14s %-9s %s  %s" % (
            i, kctx(f.flags.ctx), f.flags.kv_label, offload(f.flags, shape),
            f.flags.ub, vram, g(f.mem.host) + " G",
            "%.0f / %.0f t/s" % (s.decode0, s.decode_d),
            "%.0f t/s" % s.prefill, "%.0f s" % s.s_turn, f.note))
    return "\n".join(rows)


def placement(ctx_obj, fit, label="#1"):
    m, f, shape = fit.mem, fit.flags, ctx_obj.shape
    free = ctx_obj.machine.vram_free - fit.gpu
    gpu = ["weights %s" % g(m.gpu_weights), "kv %s" % g(m.gpu_kv + m.gpu_rs),
           "compute %s" % g(m.gpu_compute)]
    if m.gpu_mmproj:
        gpu.append("mmproj %s" % g(m.gpu_mmproj))
    gpu.append("free %s" % g(max(0, free)))
    cpu = []
    if f.ncmoe and m.host_exps:
        last = min(f.ncmoe, shape.n_layer) - 1
        cpu.append("experts L0–%d %s" % (last, g(m.host_exps)))
    rest = m.host_weights - m.host_exps - m.host_embd
    if rest > 64 * MiB:
        cpu.append("layers %s" % g(rest))
    cpu.append("embd %s" % g(m.host_embd))
    if m.host_kv + m.host_rs:
        cpu.append("kv %s" % g(m.host_kv + m.host_rs))
    cpu.append("compute %s" % g(m.host_compute))
    return "placement (%s)\n  GPU   %s\n  CPU   %s" % (
        label, "   ".join(gpu), "   ".join(cpu))


def confidence(ctx_obj, fit):
    lvl = ctx_obj.res.level(ctx_obj.sig, ctx_obj.inv.arch)
    mem = ctx_obj.mem_band(fit) // MiB
    what = ("oracle-checked" if fit.oracle else
            {"oracle": "file calibrated", "arch": "arch calibrated",
             "none": "uncalibrated"}[lvl])
    band = ctx_obj.speed_band()
    why = ("%d verified runs" % ctx_obj.eff["runs"] if ctx_obj.eff["runs"]
           else "measured box" if ctx_obj.machine.calibrated
           else "uncalibrated - mdl fit hw")
    return "confidence   memory ±%d MB (%s) · speed ±%d%% (%s)" % (
        mem, what, band * 100, why)


# ---------------------------------------------------------------- fit --

def machine_for(target, o):
    return hw.probe(target.binary, now=bool(o.get("now")))


def busy_words(mach):
    """What keeps the machine from idle, in words, or ''."""
    bits = []
    if mach.cpu_load is not None and mach.cpu_load > 0.25:
        bits.append("CPU %d%% busy" % round(mach.cpu_load * 100))
    held = mach.held("vram")
    if sum(b for _, b in held) > GiB:
        bits.append("open apps hold %s G of VRAM (%s)" % (
            g(sum(b for _, b in held)), ", ".join(n for n, _ in held[:3])))
    return "; ".join(bits)


def machine_notes(mach):
    notes = list(mach.notes)
    if mach.cpu_load is not None and mach.cpu_load > 0.3:
        notes.append("CPU %d%% busy right now; the speeds assume it idle"
                     % round(mach.cpu_load * 100))
    return notes


def with_threads(opts, mach):
    """The thread count `mdl fit hw` measured as fastest, if it has."""
    if not opts.threads and (mach.bench or {}).get("threads"):
        opts.threads = int(mach.bench["threads"])
    return opts


def oracle_check(ctx_obj, fit, fit_bin):
    """The config as it stands, through the oracle. Its residual then
    calibrates everything costed after it, the search included."""
    if not fit_bin or str(ctx_obj.inv.source).startswith("hf:"):
        return
    entry = calib.check(fit_bin, ctx_obj.inv, ctx_obj.shape, fit.flags,
                        ctx_obj.build, max_alloc=ctx_obj.machine.max_alloc)
    if entry:
        fit.oracle = entry["actual"]
        ctx_obj.res = calib.Residuals(build=ctx_obj.build)


def verdict(ctx_obj, fit):
    mach = ctx_obj.machine
    over = fit.gpu - mach.vram_usable
    ram_over = fit.mem.host - mach.ram_usable
    short = []
    if over > 0 and not ctx_obj.tight(fit):
        short.append("VRAM over by %s G" % g(over))
    if ram_over > 0:
        short.append("RAM over by %s G" % g(ram_over))
    if short:
        return "✗ " + ", ".join(short)
    if over > 0:
        return "fits, tight: %d MiB left, inside the %d MiB safety margin" % (
            (mach.vram_free - fit.gpu) // MiB, mach.margin // MiB)
    if -over < GiB:
        return "fits, %d MiB to spare" % (-over // MiB)
    return "fits, %s G to spare" % g(-over)


def ram_note(ctx_obj, fit):
    """Over what is free at idle but under the limit: it loads, the OS
    just has to page idle programs out first."""
    mach = ctx_obj.machine
    if mach.ram_free_idle < fit.mem.host <= mach.ram_usable:
        return ("needs %s G of RAM and %s G is free %s; loading it pages "
                "idle programs out to make room" % (
                    g(fit.mem.host), g(mach.ram_free_idle),
                    "at idle" if mach.plan == "idle" else "right now"))
    return None


def now_note(ctx_obj, fit):
    """The plan is for the machine at idle; say what has to close first
    when this minute's machine is short."""
    mach = ctx_obj.machine
    if mach.plan != "idle" or fit.gpu <= mach.vram_free_now:
        return None
    held = [(n, b) for n, b in mach.held("vram") if b >= 64 * MiB]
    who = (", ".join("%s %s G" % (n, g(b)) for n, b in held[:4]) if held
           else "what is open")
    short = fit.gpu - mach.vram_free_now
    return ("right now %s G of VRAM is free, %s short: close %s first, "
            "or plan for the machine as it is with --now" % (
                g(mach.vram_free_now),
                "%s G" % g(short) if short >= GiB // 10
                else "%d MiB" % max(1, short // MiB), who))


def fit_notes(ctx_obj, fit):
    return [n for n in (now_note(ctx_obj, fit), ram_note(ctx_obj, fit)) if n]


def kv_hint(ctx_obj, o, target, result, mach):
    """What 4-bit KV would buy over the q8_0 picks - as a hint, not a
    pick: a 4-bit K cache costs quality you can measure on long agent
    contexts, and that call is the user's. Shown only when it buys
    context or speed."""
    if "kv-floor" in o or result.opts.kv_floor == "q4_0":
        return None
    opts = with_threads(options(dict(o, **{"kv-floor": "q4_0"}), target), mach)
    lean = search.solve(ctx_obj, opts)
    if not lean.picks or (lean.relaxed and not result.relaxed
                          and result.picks):
        return None
    b = lean.best
    if max(model.KV_RANK.get(b.flags.ctk, 0),
           model.KV_RANK.get(b.flags.ctv, 0)) < 3:
        return None                        # the q4_0 search kept q8_0 anyway
    gains = []
    if result.picks:
        a = result.best
        if b.flags.ctx > a.flags.ctx * 1.1:
            gains.append("+%dk ctx" % ((b.flags.ctx - a.flags.ctx) // K))
        if b.speed.decode_d - a.speed.decode_d >= 1:
            gains.append("+%.0f t/s" % (b.speed.decode_d - a.speed.decode_d))
        if b.speed.s_turn < a.speed.s_turn * 0.97:
            gains.append("%.0f%% faster turns" % (
                (1 - b.speed.s_turn / a.speed.s_turn) * 100))
        if not gains:
            return None
    return ("4-bit KV: ctx %s · kv %s · %s · %.0f t/s%s. Costs some "
            "quality; --kv-floor q4_0 to allow it" % (
                kctx(b.flags.ctx), b.flags.kv_label,
                offload(b.flags, ctx_obj.shape), b.speed.decode_d,
                " (%s over #1)" % ", ".join(gains) if gains else ""))


def fit_one(target, o, out):
    mach = machine_for(target, o)
    opts = with_threads(options(o, target), mach)
    ctx_obj = search.Context(target.inv, mach)
    fit_bin = None if o.get("no-oracle") else calib.env_fit_bin(target.binary)
    current = None
    if target.flags:
        current = ctx_obj.evaluate(target.flags, opts.depth)
        oracle_check(ctx_obj, current, fit_bin)
    result = search.solve(ctx_obj, opts)
    result = search.verify_picks(ctx_obj, opts, result, fit_bin)
    if o.get("json"):
        out.write(json.dumps(as_json(ctx_obj, opts, result, target, current),
                             indent=1) + "\n")
        return result, ctx_obj, opts
    w = out.write
    w("%-44s %s\n" % (target.inv.name, describe(target.inv)))
    w("machine   %s\n" % mach.summary())
    for note in machine_notes(mach) + target.notes + target.inv.warnings:
        w("note      %s\n" % note)
    w("profile   %s  (%s)\n\n" % (opts.profile, opts.blurb))
    if current is not None:
        w("now       %s · ctx %s · kv %s · %s · ub %d · VRAM %s G%s · %s\n" % (
            target.name, kctx(target.flags.ctx), target.flags.kv_label,
            offload(target.flags, ctx_obj.shape), target.flags.ub,
            g(current.gpu), " ✓" if current.oracle else "",
            verdict(ctx_obj, current)))
        for note in fit_notes(ctx_obj, current):
            w("          %s\n" % note)
        for line in measured_lines(ctx_obj, target, mach):
            w(line + "\n")
        w("\n")
    hint = kv_hint(ctx_obj, o, target, result, mach)
    if not result.picks:
        w("nothing fits. %s.\n" % result.relaxed)
        if result.closest:
            f, m = result.closest
            w("closest   %s · ctx 4k · kv %s · VRAM %s / %s G · RAM %s / %s G\n"
              % (offload(f, ctx_obj.shape), f.kv_label, g(m.gpu),
                 g(mach.vram_usable), g(m.host), g(mach.ram_usable)))
        if hint:
            w("hint      %s\n" % hint)
        return result, ctx_obj, opts
    if result.relaxed:
        w("floors not met: %s. Best this quant can do:\n\n" % result.relaxed)
    w(table(ctx_obj, result.picks, opts) + "\n\n")
    if hint:
        w("hint      %s\n\n" % hint)
    best = result.best
    w(placement(ctx_obj, best) + "\n")
    for note in fit_notes(ctx_obj, best):
        w("note      %s\n" % note)
    w("\n")
    w(confidence(ctx_obj, best) + "\n\n")
    feats = mach.build or {}
    argv = emit.server_argv(Path(target.binary).name, target.model_path,
                            best.flags, ctx_obj.shape.n_layer, feats,
                            target.mmproj)
    w(" ".join(_q(a) for a in argv) + "\n\n")
    name = target.name or o.get("write") or _default_name(target.inv)
    keys, _ = emit.table(best.flags, target.model_path, ctx_obj.shape.n_layer,
                         feats, target.mmproj,
                         keep_args=(target.cfg or {}).get("args", []),
                         port=(target.cfg or {}).get("port", 8080))
    w(emit.block(name, keys, emit.stamp(opts.profile, best.gpu,
                                        mach.vram_usable, best.speed.decode0,
                                        feats.get("build"))))
    return result, ctx_obj, opts


def _q(a):
    return '"%s"' % a if (" " in a or '"' in a) else a


def _default_name(inv):
    """Foo-Bar-Q4_K_M.gguf -> foo-bar, the way mdl add names things."""
    stem = re.sub(r"[-_.]?(ud-)?(i?q\d+[_0-9a-z]*|f16|f32|bf16)$", "",
                  Path(inv.name).stem, flags=re.I)
    return re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-").lower() or "model"


def as_json(ctx_obj, opts, result, target, current):
    mach = ctx_obj.machine

    def one(f):
        return {"flags": f.flags.as_dict(), "memory": f.mem.as_dict(),
                "gpu_total": f.gpu, "oracle": f.oracle,
                "speed": f.speed.as_dict(), "note": f.note,
                "memory_band": ctx_obj.mem_band(f),
                "speed_band": ctx_obj.speed_band(),
                "argv": emit.server_argv("llama-server", target.model_path,
                                         f.flags, ctx_obj.shape.n_layer,
                                         mach.build or {}, target.mmproj)}
    return {"model": str(target.model_path), "arch": ctx_obj.inv.arch,
            "machine": {"gpu": mach.gpu_name, "backend": mach.backend,
                        "plan": mach.plan,
                        "vram_free": mach.vram_free,
                        "vram_free_now": mach.vram_free_now,
                        "vram_usable": mach.vram_usable,
                        "ram_avail": mach.ram_avail,
                        "ram_avail_now": mach.ram_avail_now,
                        "ram_usable": mach.ram_usable,
                        "idle": mach.idle and {
                            "vram": mach.idle.vram,
                            "vram_how": mach.idle.vram_how,
                            "ram": mach.idle.ram,
                            "ram_how": mach.idle.ram_how},
                        "cpu_load": mach.cpu_load,
                        "calibrated": mach.calibrated},
            "profile": opts.profile, "relaxed": result.relaxed,
            "current": one(current) if current is not None else None,
            "picks": [one(f) for f in result.picks]}


# --------------------------------------------------------------- write --

def preview(name, old, new, out=None):
    """Show the exact planned text and commands without touching the config."""
    import mdl
    commands = []
    for text in (old, new):
        data = tomllib.loads(text)
        if name not in data:
            commands.append("(new preset)")
            continue
        binary = (os.environ.get("MDL_LLAMA_SERVER")
                  or data.get("llama_server") or mdl.DEFAULT_BIN)
        argv = mdl.build_argv(name, data[name], str(binary))
        # Match native shell quoting: list2cmdline on Windows, shlex on POSIX.
        commands.append(subprocess.list2cmdline(argv) if os.name == "nt"
                        else shlex.join(argv))
    out = out or sys.stdout
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(),
                                     "models.toml", "models.toml (proposed)",
                                     n=3, lineterm=""):
        out.write(line + "\n")
    out.write("before: %s\nafter: %s\ndry run: nothing written\n"
              % tuple(commands))


def write_new(name, keys, comment, dry_run=False, out=None):
    import mdl
    mdl.check_name(name)
    models, _ = mdl.load_config()
    if name in models:
        die("%s is already in %s; pick another name, or use --apply "
            "on it" % (name, mdl.CONFIG))
    text = mdl.CONFIG.read_text(encoding="utf-8")
    _, proposed = mdl.plan_append_table(text.rstrip("\n") + "\n\n",
                                         emit.block(name, keys, comment), name)
    if dry_run:
        return preview(name, text, proposed, out)
    mdl.write_atomic(mdl.CONFIG, proposed, keep_backup=True)
    print("added [%s] to %s" % (name, mdl.CONFIG))


def apply_to(target, flags, ctx_obj, label, dry_run=False, out=None):
    import mdl
    keys, drop = emit.table(flags, target.model_path, ctx_obj.shape.n_layer,
                            ctx_obj.machine.build or {}, target.mmproj,
                            keep_args=target.cfg.get("args", []))
    keys.pop("model", None)
    keys.pop("mmproj", None)
    old, proposed = mdl.plan_patch_params(target.name, keys, drop=drop)
    if dry_run:
        return preview(target.name, old, proposed, out)
    mdl.write_atomic(mdl.CONFIG, proposed, keep_backup=True)
    print("applied %s to [%s] in %s (old config kept as .bak)" % (
        label, target.name, mdl.CONFIG))


# ------------------------------------------------------------ profiles --

def _binary_path(binary):
    return str(Path(shutil.which(binary) or binary).resolve())


def _model_hash(target):
    from . import evalrun
    try:
        return evalrun.file_hash(target.model_path)
    except OSError:
        return None


def _preset_argv(target):
    """The preset's whole command, for the profile key; None for a file,
    which has flags but no command of its own."""
    import mdl
    if target.kind != "name" or not target.cfg:
        return None
    try:
        return mdl.build_argv(target.name, target.cfg, target.binary)
    except mdl.MdlError:
        return None


def _predicted(ctx_obj, mach, flags, depths):
    p = perf.params(mach)
    pl = perf.Placement(ctx_obj.shape, flags)
    return ({d: 1 / perf.decode_time(pl, p, d) for d in depths},
            2048 / perf.prefill_time(pl, p, 2048, flags.ub))


def _speeds(e, pred_tg):
    parts = []
    for d, tps in sorted(((int(k), v) for k, v in e["tg"].items())):
        pred = pred_tg.get(d)
        parts.append("%s %.1f t/s%s" % (
            kctx(d) if d else "0", tps, " (predicted %.1f, %+.0f%%)" % (
                pred, (pred / tps - 1) * 100) if pred else ""))
    return parts


def measured_lines(ctx_obj, target, mach):
    """What this exact configuration measured here, beside what the
    model predicts for it - or nothing, when it has not been measured."""
    if target.flags is None or target.model_path is None:
        return []
    mh = _model_hash(target)
    if not mh:
        return []
    have = calib.profiles(mh)
    key = calib.profile_key(mh, ctx_obj.build, _binary_path(target.binary),
                            target.flags, _preset_argv(target))
    mine = next((e for e in have if e.get("key") == key), None)
    out = []
    if mine:
        depths = sorted(int(k) for k in mine["tg"])
        pred_tg, pred_pp = _predicted(ctx_obj, mach, target.flags, depths)
        out.append("measured  decode " + " · ".join(_speeds(mine, pred_tg)))
        if mine.get("pp"):
            out.append("          prefill %.0f t/s (predicted %.0f, %+.0f%%)"
                       % (mine["pp"], pred_pp,
                          (pred_pp / mine["pp"] - 1) * 100))
        out.append("          from %s, %s" % (
            "mdl eval" if mine.get("source") == "eval"
            else "llama-bench (--verify)", mine.get("at", "?")[:10]))
    others = len(have) - (1 if mine else 0)
    if others:
        out.append("          %d other measured configuration%s of this "
                   "model: mdl fit %s --profiles" % (
                       others, "" if others == 1 else "s",
                       target.name or target.model_path))
    return out


def cmd_profiles(target, o, out):
    """Every measured profile of this model's bytes, newest first."""
    w = out.write
    mh = _model_hash(target)
    have = calib.profiles(mh) if mh else []
    if o.get("json"):
        w(json.dumps(have, indent=1) + "\n")
        return have
    if not have:
        w("no measured profiles for %s yet: mdl fit %s --verify, or "
          "mdl eval, measures one\n" % (target.model_path.name,
                                        target.name or target.model_path))
        return have
    mach = machine_for(target, o)
    ctx_obj = search.Context(target.inv, mach)
    current = (calib.profile_key(mh, ctx_obj.build,
                                 _binary_path(target.binary), target.flags,
                                 _preset_argv(target))
               if target.flags else None)
    w("%d measured profile%s of %s on this machine - speeds only; they say "
      "nothing about quality\n\n" % (len(have), "" if len(have) == 1 else "s",
                                     target.model_path.name))
    for e in have:
        f = model.Flags(**{k: v for k, v in e["flags"].items()
                           if k in model.Flags.DEFAULTS})
        pred_tg, _ = _predicted(ctx_obj, mach, f,
                                [int(k) for k in e["tg"]])
        w("%s %s · %s · ctx %s · kv %s · %s · ub %d · build %s%s\n" % (
            e.get("at", "?")[:10], e.get("source", "?"),
            offload(f, ctx_obj.shape), kctx(f.ctx), f.kv_label,
            "fa on" if f.fa else "fa off", f.ub, e.get("build") or "?",
            "  <- current config" if e.get("key") == current else ""))
        w("           decode %s\n" % " · ".join(_speeds(e, pred_tg)))
        if e.get("pp"):
            w("           prefill %.0f t/s\n" % e["pp"])
    return have


# ------------------------------------------------------------- explain --

def cmd_explain(target, o, out):
    import mdl
    if target.kind != "name":
        die("--explain takes a model name from the config")
    mach = machine_for(target, o)
    opts = options(o, target)
    ctx_obj = search.Context(target.inv, mach)
    fit_bin = None if o.get("no-oracle") else calib.env_fit_bin(target.binary)
    entry = None
    if fit_bin:                      # the oracle's view of the config as-is
        entry = calib.check(fit_bin, target.inv, ctx_obj.shape, target.flags,
                            ctx_obj.build, max_alloc=mach.max_alloc)
        if entry:
            ctx_obj.res = calib.Residuals(build=ctx_obj.build)
    base, fixes, notes = explain.fixes(ctx_obj, target.flags, opts)
    if fit_bin and entry:
        base.oracle = entry["actual"]
    gpu_over, host_over = explain.overshoot(ctx_obj, base)
    w = out.write
    path, found, lines = explain.log_evidence(target.name, mdl.STATE_DIR)
    if (gpu_over <= 0 or ctx_obj.tight(base)) and host_over <= 0 and not lines:
        w("%s  ✓  %s · VRAM %s G · RAM %s G\n" % (
            target.name, verdict(ctx_obj, base), g(base.gpu),
            g(base.mem.host)))
        for note in fit_notes(ctx_obj, base):
            w("note  %s\n" % note)
        w(placement(ctx_obj, base, target.name) + "\n")
        return
    if gpu_over > 0:
        w("%s  ✗  over by %s G   (%s)\n" % (target.name, g(gpu_over),
                                          explain.headline(base)))
    elif host_over > 0:
        w("%s  ✗  RAM over by %s G   (host weights %s G + compute %s G)\n" % (
            target.name, g(host_over), g(base.mem.host_weights),
            g(base.mem.host_compute)))
    else:
        w("%s  predicted to fit, but its last load failed:\n" % target.name)
    for ln in lines:
        w("  log  %s\n" % ln)
    if path:
        w("       (%s)\n" % path)
    w("\n" + placement(ctx_obj, base, "as configured") + "\n\n")
    for note in notes:
        w("note  %s\n" % note)
    if not fixes:
        w("no change of one or two flags makes it fit; try: mdl fit %s\n"
          % target.name)
        return
    width = max(30, max(len(fx.label) for fx in fixes) + 2)
    w("  %-*s VRAM     decode @0   s/turn\n" % (width + 3, "fix"))
    for i, fx in enumerate(fixes, 1):
        turn = ("ctx cap" if fx.kind == "ctx" else
                "%+.0f%%" % (fx.turn * 100))
        dec = "±0" if abs(fx.decode) < 0.5 else "%+.0f t/s" % fx.decode
        w("  %d  %-*s %+5.1f G  %-11s %s\n" % (
            i, width, fx.label, fx.vram / GiB, dec, turn))
    choice = o.get("apply")
    # both ends: Windows calls NUL a terminal, so stdin alone is not proof
    if not choice and sys.stdin.isatty() and sys.stdout.isatty() \
            and not o.get("json"):
        try:
            reply = input("  apply #1? [y/n] ").strip().lower()
        except EOFError:
            reply = ""
        choice = "1" if reply in ("y", "yes") else None
    if choice:
        n = int(choice)
        if not 1 <= n <= len(fixes):
            die("no fix #%s" % choice)
        apply_to(target, fixes[n - 1].fit.flags, ctx_obj,
                 "fix #%d (%s)" % (n, fixes[n - 1].label),
                 dry_run=o.get("dry-run", False), out=out)


# -------------------------------------------------------------- verify --

def cmd_verify(target, o, out):
    """Run it for real: oracle, then llama-bench at the config. Learn."""
    mach = machine_for(target, o)
    opts = options(o, target)
    ctx_obj = search.Context(target.inv, mach)
    flags = target.flags
    if flags is None:
        result = search.solve(ctx_obj, opts)
        if not result.picks:
            die("nothing fits to verify: %s" % result.relaxed)
        flags = result.best.flags
    w = out.write
    fit_bin = calib.env_fit_bin(target.binary)
    if fit_bin:
        entry = calib.check(fit_bin, target.inv, ctx_obj.shape, flags,
                            ctx_obj.build, max_alloc=mach.max_alloc)
        if entry:
            p, a = entry["pred"], entry["actual"]
            w("memory   predicted %s G  oracle %s G  (model %+d  ctx %+d  "
              "compute %+d MiB)\n" % (g(sum(p)), g(sum(a)),
                                      (p[0] - a[0]) // MiB,
                                      (p[1] - a[1]) // MiB,
                                      (p[2] - a[2]) // MiB))
    else:
        w("memory   no llama-fit-params next to %s; skipped\n" % target.binary)
    bench_bin = calib.env_bench_bin(target.binary)
    if not bench_bin:
        w("speed    no llama-bench found; skipped\n")
        return
    depth = min(opts.depth, max(0, flags.ctx - 256))
    depths = sorted({0, depth})
    w("speed    running llama-bench (pp2048 at ub %d, tg64 at depth %s)...\n"
      % (flags.ub, "/".join(kctx(d) if d else "0" for d in depths)))
    out.flush()
    got, why_pp = calib.bench(bench_bin, target.model_path, flags,
                              n_prompt=2048, n_gen=0, reps=1)
    tg, why_tg = calib.bench(bench_bin, target.model_path, flags, n_prompt=0,
                             n_gen=64, reps=1,
                             depth=",".join(map(str, depths)))
    if not got and not tg and "oom" not in (why_pp, why_tg):
        # a timeout, a flag this build does not take, a crash: none of it
        # says the config did not fit, so none of it moves the margin
        die("llama-bench did not complete (%s); nothing learned. Run it "
            "by hand to see why: %s" % (
                why_tg if why_tg == why_pp else "%s, %s" % (why_pp, why_tg),
                " ".join(calib.bench_argv(bench_bin, target.model_path,
                                          flags, n_prompt=0, n_gen=64))))
    if not got and not tg:
        mach_saved = hw.load_saved()
        margins = mach_saved.setdefault("margin_arch", {})
        margins[target.inv.arch] = margins.get(target.inv.arch,
                                               hw.DEFAULT_MARGIN) + 256 * MiB
        hw.save(mach_saved)
        die("llama-bench ran out of memory. The margin for "
            "%s is now %d MiB; run mdl fit again" % (
                target.inv.arch, margins[target.inv.arch] // MiB))
    p = perf.params(mach)
    pl = perf.Placement(ctx_obj.shape, flags)
    pred_tg = {d: 1 / perf.decode_time(pl, p, d) for d in depths}
    pred_pp = 2048 / perf.prefill_time(pl, p, 2048, flags.ub)
    meas_tg = (tg or {}).get("tg", {})
    entry = {"kind": "bench", "arch": target.inv.arch,
             "sig": calib.signature(target.inv), "build": ctx_obj.build,
             "flags": flags.as_dict(),
             "tg": {str(k): v for k, v in meas_tg.items()},
             "pred_tg": {str(k): v for k, v in pred_tg.items()},
             "pp": (got or {}).get("pp"), "pred_pp": pred_pp}
    calib.append(entry)
    calib.record_profile(
        "bench", target.model_path, _model_hash(target), ctx_obj.build,
        _binary_path(target.binary), flags,
        {"tg": meas_tg, "pp": entry["pp"], "n": {}} if meas_tg or entry["pp"]
        else None,
        # the preset's command only when it is the preset's config that
        # was benchmarked, not a pick
        argv=_preset_argv(target) if flags is target.flags else None)
    for d in depths:
        if d in meas_tg:
            w("decode @%-5s predicted %5.1f  measured %5.1f t/s  (%+.0f%%)\n"
              % (kctx(d) if d else "0", pred_tg[d], meas_tg[d],
                 (pred_tg[d] / meas_tg[d] - 1) * 100))
            if (d == 0 and meas_tg[d] < 0.4 * pred_tg[d]
                    and mach.backend == "CUDA"):
                w("  ! decode is far under prediction; if VRAM is pinned at "
                  "the cap this is the sysmem fallback - see mdl fit hw\n")
    if entry["pp"]:
        w("prefill   predicted %5.0f  measured %5.0f t/s  (%+.0f%%)\n" % (
            pred_pp, entry["pp"], (pred_pp / entry["pp"] - 1) * 100))
    w("recorded; the next fit for %s uses it\n" % target.inv.arch)


# ----------------------------------------------------------------- hf --

def cmd_hf(spec, o, out):
    repo, selector = remote.parse_spec(spec)
    try:
        files = remote.list_files(repo)
        groups = remote.select(remote.gguf_groups(files), selector)
    except remote.RemoteError as e:
        die(str(e))
    if not groups:
        die("no GGUF files in %s" % repo)
    models, binary = _config()
    mach = hw.probe(binary)
    opts = options(o)
    w = out.write
    rows = []
    for i, (key, shards) in enumerate(sorted(groups.items(), key=lambda kv:
                                             sum(s["size"] for s in kv[1])), 1):
        sys.stderr.write("\rreading headers %d/%d " % (i, len(groups)))
        sys.stderr.flush()
        try:
            inv = remote.inventory(repo, key, shards)
        except (remote.RemoteError, gguf.NotGGUF) as e:
            rows.append((key, None, None, str(e)))
            continue
        ctx_obj = search.Context(inv, mach)
        rows.append((key, inv, search.solve(ctx_obj, opts), None))
    sys.stderr.write("\r" + " " * 30 + "\r")
    if o.get("json"):
        out.write(json.dumps([{"file": k, "bpw": inv.bpw if inv else None,
                               "error": err, "picks": [
                                   {"flags": f.flags.as_dict(),
                                    "gpu_total": f.gpu,
                                    "speed": f.speed.as_dict()}
                                   for f in (res.picks if res else [])],
                               "relaxed": res.relaxed if res else None}
                              for k, inv, res, err in rows], indent=1) + "\n")
        return
    w("hf:%s   %d quant%s · headers only, nothing downloaded\n" % (
        repo, len(rows), "" if len(rows) == 1 else "s"))
    w("machine   %s\n" % mach.summary())
    w("profile   %s  (%s)\n\n" % (opts.profile, opts.blurb))
    w("  %-40s %-7s %-5s %-18s %-12s %-14s %s\n" % (
        "quant", "size", "bpw", "best config", "VRAM", "decode 0/d",
        "s/turn"))
    good = [r for r in rows if r[2] and r[2].picks and not r[2].relaxed]
    best_key = max(good, key=lambda r: r[1].bpw)[0] if good else None
    for key, inv, res, err in sorted(rows, key=lambda r: -(r[1].bpw if r[1]
                                                            else 0)):
        name = Path(key).name[:40]
        if err:
            w("  %-40s %s\n" % (name, err))
            continue
        size = "%.1f G" % (inv.file_size / GiB)
        if not res.picks:
            w("  %-40s %-7s %-5.2f %s\n" % (name, size, inv.bpw,
                                            "✗ " + res.relaxed))
            continue
        f = res.best
        cfg = "%s · %s" % (offload(f.flags, model.Shape(inv)),
                           kctx(f.flags.ctx))
        mark = " ★" if key == best_key else (" ✗ floors" if res.relaxed
                                             else "")
        w("  %-40s %-7s %-5.2f %-18s %-12s %-14s %.0f s%s\n" % (
            name, size, inv.bpw, cfg, "%s / %s" % (g(f.gpu),
                                                   g(mach.vram_usable)),
            "%.0f / %.0f t/s" % (f.speed.decode0, f.speed.decode_d),
            f.speed.s_turn, mark))
    if best_key:
        w("\n★ the highest-quality quant that clears the floors. "
          "Details: mdl fit hf:%s:%s\n" % (repo, Path(best_key).name))
    else:
        w("\nno quant clears the floors on this machine; the rows show "
          "the best each can do\n")


# ------------------------------------------------------------- inspect --

def cmd_inspect(args, out):
    if len(args) != 1:
        die("usage: mdl fit inspect <model.gguf | hf:org/repo:quant>")
    spec = args[0]
    if spec.startswith("hf:"):
        repo, sel = remote.parse_spec(spec)
        try:
            groups = remote.select(remote.gguf_groups(
                remote.list_files(repo)), sel)
        except remote.RemoteError as e:
            die(str(e))
        if len(groups) != 1:
            die("name one quant: %s" % ", ".join(sorted(
                Path(k).name for k in groups)))
        key, shards = next(iter(groups.items()))
        inv = remote.inventory(repo, key, shards)
    else:
        path = Path(spec)
        if not path.is_file():
            t = resolve(spec)
            inv = t.inv
        else:
            inv = _load(path)
    shape = model.Shape(inv)
    w = out.write
    w("%s\n%s\n\n" % (inv.name, describe(inv)))
    total = sum(t.nbytes for t in inv.tensors)
    exact = total + sum(inv.data_starts) == inv.file_size
    w("tensors  %d · %s G of weights + %d KiB header = %s G file %s\n" % (
        len(inv.tensors), g(total), sum(inv.data_starts) // K,
        g(inv.file_size), "✓ exact" if exact else "(sizes from type table)"))
    w("hparams  layers %d · embd %d · vocab %d · ctx %dk · bpw %.2f (%s)\n" % (
        inv.n_layer, inv.n_embd, inv.n_vocab, inv.n_ctx_train // K, inv.bpw,
        inv.quant_label))
    roles = {}
    for t in inv.tensors:
        roles[t.role] = roles.get(t.role, 0) + t.nbytes
    w("roles    %s\n\n" % "  ".join("%s %s" % (r, g(b)) for r, b in sorted(
        roles.items(), key=lambda kv: -kv[1])))
    w(" layer  kind        weights   experts   kv/cell (f16)\n")
    for il in range(shape.n_layer):
        kind = ("recurrent" if shape.recurrent[il] else
                "attn-swa" if shape.swa[il] and shape.kv_k[il] else
                "attn" if shape.kv_k[il] else
                "mtp" if il >= shape.mtp_from else "shared-kv")
        kv = (shape.kv_k[il] + shape.kv_v[il]) * 2
        w(" %5d  %-10s %6.0f M  %6.0f M   %s\n" % (
            il, kind, shape.w_dense[il] / MiB, shape.w_exps[il] / MiB,
            "%d B" % kv if kv else "-"))
    for warn in inv.warnings:
        w("note     %s\n" % warn)


# ------------------------------------------------------------------ hw --

def size(text, total):
    """'0.5G', '512M', '35%' (of total) -> bytes."""
    m = re.fullmatch(r"\s*([\d.]+)\s*(%|[kmgt])?i?b?\s*", str(text), re.I)
    if not m:
        die("not a size: %r (e.g. 0.5G, 512M or 35%%)" % text)
    n, unit = float(m.group(1)), (m.group(2) or "g").lower()
    if unit == "%":
        return int(total * n / 100)
    return int(n * {"k": K, "m": MiB, "g": GiB, "t": GiB * K}[unit])


def idle_args(args, binary, w):
    """--idle (this minute is what idle looks like), --idle-vram X and
    --idle-ram X (say it outright), --idle-reset (measure again).
    True if any of them was given."""
    if "--idle-reset" in args:
        hw.set_idle()
        w("idle     forgotten; measured from what is open again\n")
        return True
    vals = {}
    for flag in ("--idle-vram", "--idle-ram"):
        if flag in args:
            i = args.index(flag)
            if i + 1 >= len(args):
                die("%s needs a size, e.g. 0.5G or 35%%" % flag)
            vals[flag[7:]] = args[i + 1]
    if not vals and "--idle" not in args:
        return False
    mach = hw.probe(binary, now=True)
    if vals:
        totals = {"vram": mach.vram_total, "ram": mach.ram_total}
        got = {k: size(v, totals[k]) for k, v in vals.items()}
    else:
        got = {"vram": mach.snap.vram_used, "ram": mach.snap.ram_used}
    hw.set_idle(**got)
    w("idle     booked: %s\n" % " · ".join(
        "%s %s G" % (k.upper(), g(v)) for k, v in got.items() if v is not None))
    return True


IDLE_HOW = {"set": "you set it", "seen at boot": "seen just after boot",
            "measured": "in use now, less open apps",
            "now": "in use now; no per-app figures",
            "typical": "typical for the OS"}


def idle_lines(mach):
    b, s = mach.idle, mach.snap
    if b is None or s is None:
        return []
    out = ["idle     VRAM %s G (%s) · RAM %s G (%s)" % (
        g(b.vram), IDLE_HOW[b.vram_how], g(b.ram), IDLE_HOW[b.ram_how])]
    if {b.vram_how, b.ram_how} & {"set", "seen at boot"}:
        meas = ["%s %s G" % (k, g(v)) for k, v in (
            ("VRAM", b.measured_vram), ("RAM", b.measured_ram)) if v is not None]
        if meas:
            out.append("         measured now: %s (system, startup apps, "
                       "this terminal)" % " · ".join(meas))
    apps = {}
    for p in s.apps:
        a = apps.setdefault(p.name, [0, 0])
        a[0] += p.vram
        a[1] += p.ram
    words = []
    for n, (v, r) in sorted(apps.items(), key=lambda x: -sum(x[1]))[:6]:
        bits = ["%s G %s" % (g(x), what) for x, what in ((v, "VRAM"),
                                                          (r, "RAM"))
                if x >= 64 * MiB]
        if bits:
            words.append("%s %s" % (n, " ".join(bits)))
    if words:
        out.append("open     " + " · ".join(words))
    if mach.cpu_load is not None:
        out.append("load     CPU %d%% busy now" % round(mach.cpu_load * 100))
    return out


def cmd_hw(args, out):
    models, binary = _config()
    w = out.write
    booked = idle_args(args, binary, w)
    mach = hw.probe(binary)
    w("machine  %s\n" % mach.summary())
    w("backend  %s · build %s\n" % (mach.backend, (mach.build or {}).get(
        "build", "?")))
    lg, ph, pc = mach.cores
    w("cpu      %s logical · %s physical%s\n" % (
        lg, ph or "?", " · %d performance" % pc if pc else ""))
    if mach.pcie:
        w("pcie     gen %s x%s (max gen %s x%s)\n" % (
            mach.pcie[0], mach.pcie[2], mach.pcie[1], mach.pcie[3]))
    for line in idle_lines(mach):
        w(line + "\n")
    for note in mach.notes:
        w("note     %s\n" % note)
    if "--probe" in args or booked:
        return
    bench_bin = calib.env_bench_bin(binary)
    if not bench_bin:
        die("llama-bench not found next to %s; install it to calibrate" %
            binary)
    busy = busy_words(mach)
    if busy:
        w("note     %s: calibrating now measures a slower machine than "
          "this one is at idle\n" % busy)
    invs = []
    for cfg in models.values():
        path = Path(str(cfg.get("model", "")))
        if path.is_file() and all(i.source != str(path) for i in invs):
            try:
                invs.append(gguf.load(path))
            except (gguf.NotGGUF, OSError):
                pass
    if not invs:
        die("calibration uses models from models.toml; add one first")
    p = perf.seeds(mach)
    res, done = {}, []
    ctx4 = model.Flags(ctx=4096, ub=512)

    def tg(inv, fl, label):
        w("%-8s tg128 on %s, %s...\n" % (label, inv.name, offload_words(
            fl, model.Shape(inv))))
        out.flush()
        r = calib.run_bench(bench_bin, inv.source, fl, n_prompt=0,
                            n_gen=128, reps=2)
        tps = r and r["tg"].get(0)
        if tps:
            w("         %.1f t/s\n" % tps)
        return tps

    # GPU bandwidth: the biggest model that fits whole, so the per-token
    # overhead is the smallest share of what is measured
    whole = [i for i in invs
             if model.memory(model.Shape(i), ctx4).gpu < mach.vram_usable]
    if whole:
        inv = max(whole, key=lambda i: perf.Placement(model.Shape(i),
                                                      ctx4).a_gpu)
        tps = tg(inv, ctx4, "bw_gpu")
        if tps:
            a = perf.Placement(model.Shape(inv), ctx4).a_gpu
            res["bw_gpu"] = a / max(1.0 / tps - p["t0"], 1e-4)
            done.append("bw_gpu %.0f GB/s" % (res["bw_gpu"] / 1e9))
    pp = dict(p, **res)
    # CPU bandwidth and hop cost: every expert on the CPU, on MoEs with
    # different bytes per layer. One model cannot separate the two -
    # its CPU bytes grow in lockstep with its host layers.
    moes = [i for i in invs if i.is_moe]
    pts = []
    for inv in sorted(moes, key=lambda i: i.file_size)[:3]:
        shape = model.Shape(inv)
        fl = model.Flags(ctx=4096, ub=512, ncmoe=shape.n_layer, mmap=False)
        if model.memory(shape, fl).gpu > mach.vram_usable:
            continue
        tps = tg(inv, fl, "bw_cpu")
        if tps:
            pl = perf.Placement(shape, fl)
            rest = 1.0 / tps - pl.a_gpu / pp["bw_gpu"] - pp["t0"]
            pts.append((pl.a_cpu, pl.host_layers, rest))
    solved = False
    if len(pts) >= 2:
        (a1, l1, t1), (a2, l2, t2) = sorted(pts)[0], sorted(pts)[-1]
        det = a1 * l2 - a2 * l1
        # well-conditioned only if bytes per layer differ by a margin
        if abs(det) > 0.2 * a2 * l1:
            inv_bw = (t1 * l2 - t2 * l1) / det
            hop = (a1 * t2 - a2 * t1) / det
            if inv_bw > 0 and hop >= 0:
                res["bw_cpu"], res["t_hop"] = 1.0 / inv_bw, hop
                solved = True
    if pts and not solved:
        a1, l1, t1 = max(pts)
        res["bw_cpu"] = a1 / max(t1 - l1 * pp["t_hop"], 1e-4)
    if res.get("bw_cpu"):
        done.append("bw_cpu %.0f GB/s" % (res["bw_cpu"] / 1e9))
        if res.get("t_hop") is not None and solved:
            done.append("hop %.2f ms" % (res["t_hop"] * 1e3))
    if moes:
        inv = min(moes, key=lambda i: i.file_size)
        shape = model.Shape(inv)
        # prefill with every expert on the CPU at two ubatches: the PCIe
        # trip per ubatch and the GPU's matmul rate fall out of the pair
        fl = model.Flags(ctx=4096, ncmoe=shape.n_layer, mmap=False)
        pl = perf.Placement(shape, fl)
        rows = []
        for ub in (512, 2048):
            w("bw_pcie  pp2048 at ubatch %d, experts on the CPU...\n" % ub)
            out.flush()
            r = calib.run_bench(bench_bin, inv.source, fl.replace(ub=ub),
                                n_prompt=2048, n_gen=0, reps=1)
            if r and r.get("pp"):
                rows.append((2048 // ub, 2048 / r["pp"]))
        if len(rows) == 2 and pl.w_host_touched:
            # t = n_ub x W/bw_pcie + 2 x A x 2048 / F. The matmul term is
            # the same at both ubatches (n_ub x ub = 2048 either way), so
            # the PCIe rate is the difference, and F is what is left.
            (n1, t1), (n2, t2) = rows
            if t1 > t2 and n1 != n2:
                x = (t1 - t2) / ((n1 - n2) * pl.w_host_touched)
                res["bw_pcie"] = 1.0 / x
                rest = t2 - n2 * pl.w_host_touched * x
                if rest > 0:
                    res["f_gpu"] = 2.0 * pl.active * 2048 / rest
        if res.get("bw_pcie"):
            done.append("bw_pcie %.1f GB/s" % (res["bw_pcie"] / 1e9))
        if res.get("f_gpu"):
            done.append("f_gpu %.1f TFLOPS" % (res["f_gpu"] / 1e12))
        threads = sorted({t for t in (mach.cores[2], mach.cores[1],
                                      mach.cores[0]) if t})
        best_t, best_tps = None, 0
        for t in threads:
            fl = model.Flags(ctx=4096, ub=512, ncmoe=shape.n_layer, mmap=False)
            w("threads  tg64 at -t %d...\n" % t)
            out.flush()
            r = calib.run_bench(bench_bin, inv.source, fl, n_prompt=0,
                                n_gen=64, reps=1, threads=t)
            if r and r["tg"].get(0, 0) > best_tps:
                best_t, best_tps = t, r["tg"][0]
        if best_t:
            res["threads"] = best_t
            done.append("threads %d" % best_t)
    if not res:
        die("no benchmark completed; is llama-bench working?")
    res["build"] = (mach.build or {}).get("build")
    if mach.cpu_load is not None:
        res["cpu_load"] = round(mach.cpu_load, 2)
    hw.record_bench(res)
    w("\ncalibrated: %s\nsaved to %s\n" % (" · ".join(done), hw.hw_path()))


def offload_words(flags, shape):
    label = search.offload_label(flags, shape)
    if label == "all":
        return "all on the GPU"
    if label.startswith("ncmoe"):
        return "experts of %d layers on the CPU" % min(flags.ncmoe,
                                                       shape.n_layer)
    return label


# ----------------------------------------------------------- calibrate --

def cmd_calibrate(args, out):
    if len(args) != 1:
        die("usage: mdl fit calibrate <model.gguf | name>")
    t = resolve(args[0])
    if t.kind == "hf":
        die("calibrate needs the file on disk; the oracle reads it")
    fit_bin = calib.env_fit_bin(t.binary)
    if not fit_bin:
        die("llama-fit-params not found next to %s" % t.binary)
    mach = hw.probe(t.binary, quick=True)
    shape = model.Shape(t.inv)
    build = (mach.build or {}).get("build")
    cap = t.inv.n_ctx_train or 32 * K
    mid = shape.n_layer // 3 if shape.is_moe else 0
    w = out.write
    w("oracle sweep for %s (%s)\n" % (t.inv.name, describe(t.inv)))
    w(" ub     ctx    predicted  oracle   miss\n")
    for ub in search.UBATCHES:
        for ctx in sorted({8 * K, min(64 * K, cap), cap}):
            for ncmoe in sorted({0, mid}):
                fl = model.Flags(ctx=ctx, ub=ub, b=max(2048, ub), ncmoe=ncmoe,
                                 ctk="q8_0", ctv="q8_0")
                e = calib.check(fit_bin, t.inv, shape, fl, build,
                                max_alloc=mach.max_alloc)
                if not e:
                    w(" %-5d  %-6s oracle declined\n" % (ub, kctx(ctx)))
                    continue
                p, a = sum(e["pred"]), sum(e["actual"])
                w(" %-5d  %-6s %6.0f M   %6.0f M  %+5.0f M%s\n" % (
                    ub, kctx(ctx), p / MiB, a / MiB, (a - p) / MiB,
                    "  (ncmoe %d)" % ncmoe if ncmoe else ""))
                out.flush()
    w("stored in %s\n" % calib.calib_path())


# ---------------------------------------------------------------- main --

def main(args, out=None):
    if out is None and not sys.stdout.isatty():
        try:                         # a pipe on Windows is cp1252, and · ≥ ✓
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    out = out or sys.stdout
    if not args or args[0] in ("-h", "--help"):
        out.write(USAGE)
        return
    if args[0] == "hw":
        return cmd_hw(args[1:], out)
    if args[0] == "inspect":
        return cmd_inspect(args[1:], out)
    if args[0] == "calibrate":
        return cmd_calibrate(args[1:], out)
    o = parse(args)
    if o["target"].startswith("hf:"):
        if (o.get("explain") or o.get("verify") or o.get("apply")
                or o.get("write")):
            # nothing is downloaded, so there is no file for a preset to
            # point at: saying "added" would leave eval nothing to run
            die("--explain, --verify, --apply and --write need a model on "
                "disk; download the quant, then mdl fit <file> --write NAME")
        return cmd_hf(o["target"], o, out)
    target = resolve(o["target"])
    if o.get("profiles"):
        return cmd_profiles(target, o, out)
    if o.get("explain"):
        return cmd_explain(target, o, out)
    if o.get("verify"):
        return cmd_verify(target, o, out)
    result, ctx_obj, opts = fit_one(target, o, out)
    if not result.picks:
        return
    if o.get("write"):
        f = result.best
        keys, _ = emit.table(f.flags, target.model_path, ctx_obj.shape.n_layer,
                             ctx_obj.machine.build or {}, target.mmproj,
                             keep_args=(target.cfg or {}).get("args", []))
        write_new(o["write"], keys, emit.stamp(
            opts.profile, f.gpu, ctx_obj.machine.vram_usable,
            f.speed.decode0, (ctx_obj.machine.build or {}).get("build")),
            dry_run=o.get("dry-run", False), out=out)
    elif o.get("apply"):
        if target.kind != "name":
            die("--apply writes into a models.toml entry; for a file use "
                "--write NAME")
        n = int(o["apply"])
        if not 1 <= n <= len(result.picks):
            die("no pick #%s" % o["apply"])
        apply_to(target, result.picks[n - 1].flags, ctx_obj, "pick #%d" % n,
                 dry_run=o.get("dry-run", False), out=out)
