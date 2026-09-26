"""GGUFs already on this machine, where people and other tools keep them.

`mdl setup` offers them, and the web page does while the config has no
models: a first model that is already downloaded should not be fetched
again, or found by hand and typed into `mdl add`.
"""

import os
import re
import sys
from pathlib import Path

MIN_BYTES = 50_000_000       # smaller is a vocab or a template, not weights
DEPTH = 6                    # HF's cache is models--o--r/snapshots/rev/dir/f
SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


class Found:
    """One model file (its first shard, if split), and what it came to."""

    def __init__(self, path, size, place, mmproj=None):
        self.path, self.size, self.place, self.mmproj = path, size, place, mmproj

    def __repr__(self):
        return "Found(%r, %d, %r)" % (str(self.path), self.size, self.place)


def places():
    """[(label, folder)] to look in, most likely first. Where a tool keeps
    its downloads, as that tool documents it; the ones not here are
    skipped quietly."""
    from . import pull
    home = Path.home()
    out = [("models folder", pull.models_dir()),
           ("Hugging Face cache", pull.hub_cache())]
    # llama-server -hf: $LLAMA_CACHE, else the platform's cache directory
    if os.environ.get("LLAMA_CACHE"):
        out.append(("llama.cpp cache", Path(os.environ["LLAMA_CACHE"])))
    elif os.name == "nt":
        out.append(("llama.cpp cache", Path(os.environ.get(
            "LOCALAPPDATA", home / "AppData" / "Local")) / "llama.cpp"))
    elif sys.platform == "darwin":
        out.append(("llama.cpp cache", home / "Library" / "Caches" /
                    "llama.cpp"))
    else:
        out.append(("llama.cpp cache", Path(os.environ.get(
            "XDG_CACHE_HOME", home / ".cache")) / "llama.cpp"))
    out += [("LM Studio", home / ".lmstudio" / "models"),
            ("LM Studio", home / ".cache" / "lm-studio" / "models")]
    return out


def _identity(path):
    """What makes two paths one file: a hard link, or the HF cache's
    symlink to its blob, is the same model."""
    try:
        st = path.stat()
        return (st.st_dev, st.st_ino) if st.st_ino else os.path.realpath(path)
    except OSError:
        return os.path.realpath(path)


def _size(path):
    """A file's size, or every shard's for a split one; None if one of
    them is missing."""
    m = SHARD.search(path.name)
    if not m:
        return path.stat().st_size
    total = 0
    for i in range(1, int(m.group(2)) + 1):
        shard = path.with_name(SHARD.sub("-%05d-of-%s.gguf" % (i, m.group(2)),
                                         path.name))
        if not shard.is_file():
            return None
        total += shard.stat().st_size
    return total


def found(where=None, known=()):
    """[Found], biggest first, in the folders of `where` (default:
    places()), leaving out projectors, later shards, half downloads, and
    the files in `known` (paths the config already runs)."""
    import mdl
    seen = {_identity(Path(p)) for p in known if p}
    out = []
    for place, root in (places() if where is None else where):
        root = Path(root)
        if not root.is_dir():
            continue
        base = len(root.parts)
        for folder, dirs, files in os.walk(root):
            if len(Path(folder).parts) - base >= DEPTH:
                dirs[:] = []
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in files:
                low = name.lower()
                if not low.endswith(".gguf") or "mmproj" in low:
                    continue
                m = SHARD.search(name)
                if m and int(m.group(1)) != 1:
                    continue
                path = Path(folder) / name
                try:
                    size = _size(path)
                except OSError:
                    continue
                key = _identity(path)
                if size is None or size < MIN_BYTES or key in seen:
                    continue
                seen.add(key)
                out.append(Found(path, size, place, mdl.find_mmproj(path)))
    out.sort(key=lambda f: -f.size)
    return out


def name_for(path, taken=()):
    """A preset name from a file name, as mdl add makes one, made unique
    against `taken`: Qwen3-8B-Q4_K_M.gguf -> qwen3-8b."""
    stem = SHARD.sub("", path.name)
    stem = re.sub(r"\.gguf$", "", stem, flags=re.I)
    stem = re.sub(r"[-_.]?(ud-)?([ipt]?q\d+[_0-9a-z]*|f16|f32|bf16)$", "", stem,
                  flags=re.I)
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-").lower()[:60]
    name = name.strip("-") or "model"
    out, n = name, 2
    while out in taken:
        out, n = "%s-%d" % (name, n), n + 1
    return out


def configured(models):
    """The model files a config's presets already run."""
    return [Path(m["model"]).expanduser() for m in models.values()
            if isinstance(m, dict) and m.get("model")]
