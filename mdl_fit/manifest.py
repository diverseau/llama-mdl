"""mdl manifest - what a server actually is, not what the config says now.

A result or a bug report is only as good as its description of what ran:
the command line, the llama.cpp build behind it, the exact model files,
the machine. The config describes what will run next time, which drifts
from what is running now the moment someone edits it. So this is read
from the running server's state (the argv spawn() recorded, the binary
it stat'ed at launch) and from the server itself, and only falls back to
the config - saying so - for a model that is not running.

identity() is the part that has to match for two results to be the same
measurement: the command minus its port, the build, the model's bytes.
"""

import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from . import gguf, hw, model

# Flags whose value is a secret, or a path to one: redact() blanks them.
SECRET_FLAGS = {"--api-key", "--api-key-file", "--ssl-key-file",
                "--ssl-cert-file", "-hft", "--hf-token"}
BUILD = re.compile(r"\bbuild[:=]?\s*(\d+)\s*\(([0-9a-f]{6,})\)", re.I)


def file_id(path):
    """A file named by what can be had from stat alone: cheap enough for
    every launch, and enough to notice the binary was rebuilt."""
    import mdl
    return mdl.file_id(path)


def shards(path):
    """Every file of a model, in order: one, or all N of a split GGUF."""
    p = Path(path)
    m = gguf.SPLIT_RE.search(p.name)
    if not m:
        return [p]
    total = int(m.group(2))
    stem = p.name[:m.start()]
    return [p.with_name("%s-%05d-of-%05d.gguf" % (stem, i, total))
            for i in range(1, total + 1)]


def model_files(path):
    """[{name, size, hash}] for each shard - the hash is evalrun's size
    plus first and last 8 MiB, which reads 16 MiB, not the whole file."""
    from . import evalrun
    out = []
    for s in shards(path):
        try:
            out.append({"name": s.name, "path": str(s),
                        "size": s.stat().st_size,
                        "hash": evalrun.file_hash(s)})
        except OSError:
            out.append({"name": s.name, "path": str(s), "missing": True})
    return out


def build_from_log(log):
    """(number, commit) from the load log's build line, or None. The
    server prints it at start; asking the binary would start it again."""
    try:
        with open(log, encoding="utf-8", errors="replace") as fh:
            head = fh.read(64 << 10)
    except (OSError, TypeError):
        return None
    m = BUILD.search(head)
    return (int(m.group(1)), m.group(2)) if m else None


def props(port, timeout=3):
    try:
        with urllib.request.urlopen(
                "http://127.0.0.1:%d/props" % port, timeout=timeout) as r:
            got = json.loads(r.read())
            return got if isinstance(got, dict) else {}
    except (urllib.error.URLError, OSError, ValueError):
        return {}


def machine(binary):
    m = hw.probe(binary, quick=True)
    return {"gpu": m.gpu_name, "backend": m.backend, "driver": m.driver,
            "vram_total": m.vram_total, "ram_total": m.ram_total,
            "cores": list(m.cores) if m.cores else None}


def build(name, models=None, binary=None, probe=True):
    """The manifest for <name>: its running server if it has one, else
    its preset, with "source" saying which."""
    import mdl
    if models is None:
        models, binary = mdl.load_config()
    state = mdl.read_state(name)
    if state and state.get("argv"):
        argv, source = list(state["argv"]), "running"
    else:
        if name not in models:
            mdl.die(f"no model named '{name}' in {mdl.CONFIG}")
        argv = mdl.build_argv(name, models[name], binary)
        # a server started by an older mdl did not record its command
        source = "config (running, but started before mdl recorded it)" \
            if state else "config"
    flags, _, path, mmproj = model.parse_argv(argv)
    exe = argv[0]
    out = {"mdl": mdl.VERSION, "name": name, "source": source,
           "argv": argv,
           "binary": (state or {}).get("binary") or file_id(
               mdl.shutil.which(exe) or exe),
           "build": None,
           "model": model_files(path) if path else [],
           "mmproj": model_files(mmproj) if mmproj else [],
           "flags": flags.as_dict()}
    if state:
        out["port"] = state.get("port")
        got = build_from_log(state.get("log"))
        server = props(state["port"]) if state.get("port") else {}
        if server.get("build_info"):
            out["build"] = server["build_info"]
        elif got:
            out["build"] = "b%d-%s" % got
        served = server.get("model_path")
        if served:
            out["served_model"] = served
        n_ctx = (server.get("default_generation_settings") or {}).get("n_ctx")
        if n_ctx:
            out["served_ctx"] = n_ctx
    if probe:
        out["machine"] = machine(exe)
    out["identity"] = identity(out)
    return out


def identity(man):
    """What has to match for two runs to be one measurement: the command
    minus its port, the build (or the binary's stat when the build is
    unknown), and the model's bytes. Not the machine: evals compare on
    one machine already, and a driver update is not a new model."""
    from . import evalrun
    key = {"argv": evalrun.sans_port(man["argv"][1:]),
           "build": man.get("build") or man.get("binary"),
           "model": [(f.get("name"), f.get("size"), f.get("hash"))
                     for f in man.get("model", [])],
           "mmproj": [(f.get("name"), f.get("hash"))
                      for f in man.get("mmproj", [])]}
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str)
                          .encode()).hexdigest()[:16]


def redact(man):
    """A copy fit for a bug report: secrets blanked, paths cut to their
    file names, the home directory out of anything left."""
    # compared with / and case folded: Windows writes the home directory
    # with either slash, and a config usually has forward ones
    home = str(Path.home()).replace("\\", "/").rstrip("/")
    out = json.loads(json.dumps(man, default=str))

    def scrub(s):
        if not isinstance(s, str):
            return s
        flat = s.replace("\\", "/")
        if "/" in flat and (Path(flat).suffix
                            or home.lower() in flat.lower()):
            return Path(flat).name or "<path>"
        at = flat.lower().find(home.lower())
        return flat[:at] + "~" + flat[at + len(home):] if at >= 0 else s

    argv, skip = [], False
    for i, a in enumerate(out.get("argv", [])):
        if skip:
            argv.append("<redacted>")
            skip = False
            continue
        flag = a.split("=", 1)[0]
        if flag in SECRET_FLAGS:
            argv.append(flag + "=<redacted>" if "=" in a else a)
            skip = "=" not in a
            continue
        if i == 0:
            argv.append(Path(a).name)
        elif a.startswith("-") and "=" in a:     # --slot-save-path=C:/...
            argv.append(flag + "=" + scrub(a.split("=", 1)[1]))
        else:
            argv.append(scrub(a))
    out["argv"] = argv
    for key in ("model", "mmproj"):
        for f in out.get(key, []):
            f.pop("path", None)
    if isinstance(out.get("binary"), dict):
        out["binary"]["path"] = Path(out["binary"].get("path", "")).name
    for key in ("served_model",):
        if key in out:
            out[key] = Path(out[key]).name
    return out


USAGE = """usage: mdl manifest <name> [--redact] [--no-probe]

What <name> is running as - its command line, the llama.cpp build, the
exact model files (size and a hash of each shard) and the machine - read
from the running server, or from its preset if it is not running.

  --redact     for a bug report: secrets blanked, paths cut to file names
  --no-probe   skip the machine (GPU, driver, RAM) section
"""


def main(args):
    import mdl
    if not args or args[0] in ("-h", "--help"):
        print(USAGE, end="")
        return None
    names = [a for a in args if not a.startswith("-")]
    unknown = [a for a in args if a.startswith("-")
               and a not in ("--redact", "--no-probe")]
    if unknown or len(names) != 1:
        mdl.die("usage: mdl manifest <name> [--redact] [--no-probe]")
    man = build(names[0], probe="--no-probe" not in args)
    if "--redact" in args:
        man = redact(man)
    print(json.dumps(man, indent=1, default=str))
    return man
