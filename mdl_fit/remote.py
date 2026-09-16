"""hf:org/repo - fit a model you have not downloaded.

The HF API lists the repo's files and sizes; an HTTP Range request on each
GGUF's resolve URL fetches only its header. Header length is not known up
front (a 250k-token vocab is megabytes of strings), so it asks for 4 MiB,
parses, and asks for more if the parse runs off the end. Parsed
inventories are cached by repo, file and content hash, so the second look
at a repo costs nothing.
"""

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import gguf, hw

FIRST = 4 << 20
LIMIT = 256 << 20
UA = "mdl-fit (+https://github.com/diverseau/llama-mdl)"


class RemoteError(Exception):
    pass


def endpoint():
    return os.environ.get("MDL_HF_ENDPOINT", os.environ.get(
        "HF_ENDPOINT", "https://huggingface.co")).rstrip("/")


def token():
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" /
                               "huggingface")) / "token"
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def parse_spec(spec):
    """'hf:org/repo[:selector]' -> (repo, selector or None)."""
    body = spec[3:] if spec.startswith("hf:") else spec
    repo, _, selector = body.partition(":")
    if repo.count("/") != 1:
        raise RemoteError("expected hf:org/repo, got %r" % spec)
    return repo, (selector or None)


def _request(url, headers=None, timeout=30):
    headers = dict(headers or {})
    headers.setdefault("User-Agent", UA)
    tok = token()
    if tok:
        headers.setdefault("Authorization", "Bearer " + tok)
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RemoteError("%s: access denied - a gated repo needs "
                              "HF_TOKEN" % url) from None
        if e.code == 404:
            raise RemoteError("%s: not found" % url) from None
        raise RemoteError("%s: HTTP %d" % (url, e.code)) from None
    except (urllib.error.URLError, OSError) as e:
        raise RemoteError("%s: %s" % (url, getattr(e, "reason", e))) from None


def list_files(repo, revision="main"):
    """[{'path', 'size', 'oid'}] for every file in the repo."""
    url = "%s/api/models/%s/tree/%s?recursive=true" % (
        endpoint(), repo, urllib.parse.quote(revision, safe=""))
    out = []
    while url:
        _, headers, body = _request(url)
        for item in json.loads(body):
            if item.get("type") != "file":
                continue
            lfs = item.get("lfs") or {}
            out.append({"path": item["path"],
                        "size": lfs.get("size", item.get("size", 0)),
                        "oid": lfs.get("oid") or item.get("oid", "")})
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", ""))
        url = m.group(1) if m else None
    return out


AUX_WORDS = {"mmproj", "projector", "draft", "dflash", "fastmtp", "lora",
             "adapter"}


def auxiliary(name):
    """True for a GGUF that ships beside a model rather than being one.

    Vision projectors, LoRA adapters, speculative-decoding drafts and MTP
    heads all carry a quant in their names, so left in they read as a
    1.4 GB "Q4_0" of a 27B model: the smallest quant it has, and the one
    `find` settles on when nothing else fits. Sizes cannot tell them
    apart - a real Q1_0 is 1.1 bits a weight, the same as a draft - so
    this goes by name, one word at a time.
    """
    parts = [p.lower() for p in re.split(r"[/\\]", name) if p]
    words = [re.split(r"[-_. ]+", p) for p in parts]
    flat = {w.rstrip("0123456789") for ws in words for w in ws}  # DFlash2
    if "lora" in flat and flat & {"merged", "merge"}:
        flat.discard("lora")                    # a merged LoRA is a model
    if flat & AUX_WORDS:
        return True
    # "mtp-X-Q4_0.gguf" and "MTP/..." are the head alone; "X-Q4_0-mtp.gguf"
    # is the whole model with its MTP layers kept, and is a quant.
    return any(ws[0] == "mtp" for ws in words)


def gguf_groups(files):
    """{label: [file dicts, in shard order]} for the model GGUFs in a repo.

    Projectors, adapters, drafts and MTP heads are left out - they are
    add-ons, not quants (see auxiliary). Split shards group under the name
    they share.
    """
    groups = {}
    for f in files:
        name = f["path"]
        if not name.lower().endswith(".gguf"):
            continue
        if auxiliary(name):
            continue
        key = gguf.SPLIT_RE.sub(".gguf", name)
        groups.setdefault(key, []).append(f)
    for key in groups:
        groups[key].sort(key=lambda f: f["path"])
    return groups


def mmproj_files(files):
    return [f for f in files if Path(f["path"]).name.lower().startswith(
        "mmproj") and f["path"].lower().endswith(".gguf")]


def fetch_header(repo, path, size, revision="main"):
    """Just enough of a remote GGUF to parse its header."""
    url = "%s/%s/resolve/%s/%s" % (endpoint(), repo, revision,
                                   urllib.parse.quote(path))
    want = min(FIRST, size) if size else FIRST
    while True:
        status, _, body = _request(url, {"Range": "bytes=0-%d" % (want - 1)})
        if status == 200 and size and len(body) > want:
            body = body[:want]                    # server ignored Range
        try:
            gguf.parse_header(body)
            return body
        except gguf.Truncated as e:
            if (size and want >= size) or want >= LIMIT:
                raise RemoteError("%s: header larger than %d MiB"
                                  % (path, want >> 20)) from None
            want = max(want * 2, int(e.args[0]) + (1 << 20))
            if size:
                want = min(want, size)


def _cache_path(repo, key, oid):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", "%s__%s__%s" % (
        repo, key, (oid or "")[:16]))
    return hw.cache_dir() / "gguf" / (safe + ".json")


def inventory(repo, key, shards, revision="main", cache=True):
    """Inventory for one quant (all its shards), cached by content hash."""
    oid = "+".join(s.get("oid", "") for s in shards)
    path = _cache_path(repo, key, oid)
    if cache and oid:
        try:
            return gguf.Inventory.from_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError):
            pass
    pieces = [(fetch_header(repo, s["path"], s["size"], revision), s["size"])
              for s in shards]
    inv = gguf.merge("hf:%s/%s" % (repo, key), pieces)
    if cache and oid:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(inv.to_json(), encoding="utf-8")
        except OSError:
            pass
    return inv


def select(groups, selector):
    """Narrow the quant groups to one file or quant name, if asked."""
    if not selector:
        return groups
    sel = selector.lower()
    hit = {k: v for k, v in groups.items()
           if sel == k.lower() or sel in Path(k).name.lower()}
    if not hit:
        raise RemoteError("no GGUF in the repo matches %r; have: %s" % (
            selector, ", ".join(sorted(Path(k).name for k in groups))))
    return hit
