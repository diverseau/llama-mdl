"""mdl setup - the first minute, in one command.

Says whether llama.cpp is here and what it runs on, makes the config, and
adds the GGUFs already on this machine as presets fitted to it. With none
there, it offers `mdl find`, which ranks what fits and fetches the best.
Run again, it adds only what is new.
"""

import re
import shutil
import sys
from pathlib import Path

from . import scan

USAGE = """\
usage: mdl setup [--yes] [--dir PATH]...

  --yes       add every GGUF found without asking, and do not offer find
  --dir PATH  look in PATH too, before the usual places (repeatable)

It looks in the models folder (~/models, or $MDL_MODELS), the Hugging Face
cache, llama.cpp's -hf cache and LM Studio's models folder.
"""


def say(label, words):
    print("%-10s %s" % (label, words), flush=True)


def llama_line(binary):
    """(found, words): the llama-server this config runs, and what it
    runs on as llama.cpp lists it; how to install it when it is not
    there."""
    import mdl
    path = shutil.which(binary) or (binary if Path(binary).is_file() else None)
    if not path:
        return False, mdl.missing_binary(binary)
    from . import hw
    devices = hw.llama_devices(path)
    if devices:
        return True, "llama-server, on %s" % ", ".join(
            "%s (%s)" % (name, backend) for backend, _, name, _, _ in devices)
    gpus = [g["name"] for g in hw.nvidia()]
    if gpus:
        return True, ("llama-server, but a build that lists no GPU, on a "
                      "machine with %s: it runs many times slower than a GPU "
                      "build (%s)" % (", ".join(gpus), mdl.llama_hint()))
    return True, "llama-server, on the CPU"


def choose(count, answer):
    """The 0-based indexes an answer picks out of `count`: yes (or
    nothing) is all, no is none, and numbers are those: '1 3', '1,3',
    '2-4'. None when it is none of these."""
    answer = answer.strip().lower()
    if answer in ("", "y", "yes", "a", "all"):
        return list(range(count))
    if answer in ("n", "no", "none"):
        return []
    picked = []
    for part in re.split(r"[\s,]+", answer):
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            return None
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        if not 1 <= lo <= hi <= count:
            return None
        picked += [i - 1 for i in range(lo, hi + 1) if i - 1 not in picked]
    return picked


def ask(prompt):
    """A line from the terminal; None at end of input."""
    try:
        return input(prompt)
    except EOFError:
        return None


def add(found, name):
    """One found file as the preset `name`, fitted to this machine
    (pull's way: mdl fit --write, a free port, --metrics)."""
    import mdl

    from . import pull
    pull._add(name, found.path, found.mmproj,
              mdl.load_config(missing_ok=True)[0])


def describe(name):
    """What a preset came to, in a few words."""
    import mdl
    m = mdl.load_config(missing_ok=True)[0].get(name, {})
    words = []
    if m.get("ctx"):
        words.append("%dk context" % (int(m["ctx"]) // 1024))
    ngl, moe = m.get("ngl"), int(m.get("n_cpu_moe") or 0)
    if ngl is not None:
        words.append("on the CPU" if int(ngl) == 0
                     else "on the GPU but %d layers' experts, in RAM" % moe
                     if moe else "every layer on the GPU" if int(ngl) >= 99
                     else "%d layers on the GPU" % int(ngl))
    if m.get("mmproj"):
        words.append("vision")
    return ", ".join(words)


def main(args):
    import mdl
    yes, dirs = False, []
    rest = list(args)
    while rest:
        a = rest.pop(0)
        if a in ("-h", "--help"):
            print(USAGE, end="")
            return
        if a in ("-y", "--yes"):
            yes = True
        elif a == "--dir" and rest:
            dirs.append(Path(rest.pop(0)).expanduser())
        else:
            mdl.die(USAGE.splitlines()[0])
    interactive = sys.stdin.isatty() and sys.stdout.isatty() and not yes

    models, binary = mdl.load_config(missing_ok=True)
    have_llama, words = llama_line(binary)
    say("llama.cpp", words)
    made = mdl.ensure_config()
    say("config", mdl.short(mdl.CONFIG) + (" (new)" if made else ""))

    where = [("--dir", d) for d in dirs] + scan.places()
    found = scan.found(where, known=scan.configured(models))
    added = []
    if found:
        say("found", "%d GGUF%s on this machine, not in the config yet:"
            % (len(found), "" if len(found) == 1 else "s"))
        names = []
        for f in found:
            names.append(scan.name_for(f.path, set(models) | set(names)))
        # names as wide as the longest, and a path cut in its middle to
        # what is left of the line
        width = max(len(n) for n in names)
        room = max(30, shutil.get_terminal_size((100, 24)).columns
                   - width - 18)
        for i, (f, name) in enumerate(zip(found, names, strict=True), 1):
            print("  %2d  %-*s %7s  %s" % (
                i, width, name, mdl.human_size(f.size),
                mdl._fit_line(str(f.path), room)), flush=True)
        if yes:
            picked = list(range(len(found)))
        elif interactive:
            picked = None
            while picked is None:
                answer = ask("add them? [Y/n, or numbers like 1 3] ")
                picked = [] if answer is None else choose(len(found), answer)
        else:
            picked = []
            print("  (add them with: mdl setup --yes, or mdl add FILE)")
        for i in picked:
            f = found[i]
            name = names[i]
            if sys.stdout.isatty():     # the fit takes a few seconds
                print("adding     %s ..." % f.path.name, end="\r", flush=True)
            add(f, name)
            added.append(name)
            print("added      %-*s %s" % (width, name, describe(name)),
                  flush=True)
    else:
        say("found", "no GGUFs on this machine that the config does not "
            "already run")

    models, _ = mdl.load_config(missing_ok=True)
    if not models and interactive and have_llama:
        answer = ask("look for the best model this machine can run? it reads "
                     "the model catalog, a minute the first time [Y/n] ")
        if answer is not None and choose(1, answer) == [0]:
            from . import find
            find.main([])
            return
    print()
    if not have_llama:
        say("next", "install llama.cpp (%s), then mdl run NAME"
            % mdl.llama_hint())
    elif added or models:
        say("next", "mdl run %s   (or mdl ui, the dashboard)"
            % (added[0] if added else sorted(models)[0]))
    else:
        say("next", "mdl find shows what fits this machine; mdl find --run 1 "
            "fetches the best of it and starts it")
