"""The llama-server command and the models.toml block for a config.

Every flag the prediction depended on is written out, even where it
matches today's default. llama-server has changed defaults under people
before (-np went to auto, four slots, unified KV), and a prediction made
against one default is wrong the day it moves.
"""

import time

# Flags this module owns. Applying a fit strips these from a model's
# existing args before adding its own; everything else (sampling,
# --jinja, templates) is left exactly as it was.
VALUE_FLAGS = {"-c", "--ctx-size", "-ngl", "--n-gpu-layers", "--gpu-layers",
               "-ncmoe", "--n-cpu-moe", "-fa", "--flash-attn", "-ctk",
               "--cache-type-k", "-ctv", "--cache-type-v", "-b",
               "--batch-size", "-ub", "--ubatch-size", "-np", "--parallel",
               "-t", "--threads", "-lm", "--load-mode", "-fit", "--fit"}
BOOL_FLAGS = {"--no-mmap", "--mmap", "-cmoe", "--cpu-moe", "--swa-full",
              "-kvu", "--kv-unified"}


def ngl_value(flags, n_layer):
    """What to write for -ngl. 99 reads as 'all of it' to anyone who has
    tuned llama.cpp, so it is used whenever it still means that."""
    if flags.ngl >= n_layer + 1:
        return 99 if n_layer + 1 <= 99 else n_layer + 1
    return flags.ngl


def extra_args(flags, features):
    """Flags models.toml has no key for, as an args list."""
    out = ["-b", str(max(flags.b, flags.ub)), "-ub", str(flags.ub)]
    if flags.ctk != flags.ctv:
        out += ["--cache-type-k", flags.ctk, "--cache-type-v", flags.ctv]
    if not flags.mmap:
        out += (["--load-mode", "none"] if features.get("load_mode")
                else ["--no-mmap"])
    if flags.threads:
        out += ["-t", str(flags.threads)]
    if flags.swa_full:
        out.append("--swa-full")
    if features.get("fit_flag"):
        # What runs must be exactly what was predicted; llama.cpp's own
        # --fit would otherwise nudge whatever we left unset.
        out += ["-fit", "off"]
    return out


def server_argv(binary, model_path, flags, n_layer, features, mmproj=None):
    argv = [binary, "-m", str(model_path)]
    if mmproj:
        argv += ["--mmproj", str(mmproj)]
        if not flags.mmproj_offload:
            argv.append("--no-mmproj-offload")
    argv += ["-ngl", str(ngl_value(flags, n_layer))]
    if flags.ncmoe:
        argv += ["--n-cpu-moe", str(min(flags.ncmoe, n_layer))]
    argv += ["-c", str(flags.ctx), "-fa", "on" if flags.fa else "off",
             "-ctk", flags.ctk, "-ctv", flags.ctv, "-np", str(flags.np)]
    return argv + extra_args(flags, features) + ["--metrics"]


def strip_owned(args):
    """Existing args minus every flag this module writes itself."""
    out, i = [], 0
    while i < len(args):
        a = str(args[i])
        if a in VALUE_FLAGS:
            i += 2
            continue
        if a in BOOL_FLAGS:
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def table(flags, model_path, n_layer, features, mmproj=None, keep_args=(),
          port=None):
    """(keys, drop): the models.toml keys for this config, and the keys
    an existing entry should lose (kv_type when K and V now differ)."""
    keys = {"model": str(model_path).replace(chr(92), "/")}
    if mmproj:
        keys["mmproj"] = str(mmproj).replace(chr(92), "/")
    keys["ngl"] = ngl_value(flags, n_layer)
    drop = []
    if flags.ncmoe:
        keys["n_cpu_moe"] = min(flags.ncmoe, n_layer)
    else:
        drop.append("n_cpu_moe")
    keys["ctx"] = flags.ctx
    keys["flash_attn"] = bool(flags.fa)
    if flags.ctk == flags.ctv:
        keys["kv_type"] = flags.ctk
    else:
        drop.append("kv_type")
    keys["parallel"] = flags.np
    if port is not None:
        keys["port"] = port
    kept = strip_owned(list(keep_args))
    if "--metrics" not in kept:
        kept.insert(0, "--metrics")
    if mmproj and not flags.mmproj_offload and "--no-mmproj-offload" not in kept:
        kept.append("--no-mmproj-offload")
    keys["args"] = kept + extra_args(flags, features)
    return keys, drop


def stamp(profile, gpu_bytes, usable_bytes, tps, build):
    return "# mdl fit · %s · predicted %.1f/%.1f G · %.0f t/s · %s · build %s" % (
        profile, gpu_bytes / (1 << 30), usable_bytes / (1 << 30), tps,
        time.strftime("%Y-%m-%d"), build or "?")


def block(name, keys, comment=None):
    """A [name] table as TOML text, in the order mdl add writes them."""
    import mdl                      # toml_value lives with the config code
    lines = ["[%s]" % name]
    for key, value in keys.items():
        lines.append("%s = %s" % (key, mdl.toml_value(value)))
    if comment:
        lines.append(comment)
    return "\n".join(lines) + "\n"
