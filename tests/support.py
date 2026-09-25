"""Shared test helpers: a sandboxed mdl, and a tiny result tally.

Every test runs against a temp config and temp state dir. Nothing here may
touch ~/.config/mdl or ~/.local/state/mdl - the UI tests write to the config
on purpose, and doing that to a real one would be unforgivable.
"""
import os
import shutil
import socket
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# No suite may ask the real PyPI whether mdl is out of date, which the
# dashboard and doctor would otherwise do. test_update turns it back on
# against a fake index of its own.
os.environ["MDL_NO_UPDATE_CHECK"] = "1"
FAKE = Path(__file__).resolve().parent / "fake_llama_server.py"
sys.path.insert(0, str(ROOT))

import mdl  # noqa: E402
from mdl_fit import gguf  # noqa: E402


class Tally:
    def __init__(self, name):
        self.name, self.fails = name, []

    def check(self, label, got, want):
        ok = got == want
        print(("PASS " if ok else "FAIL ") + label)
        if not ok:
            print("      got:  %r" % (got,))
            print("      want: %r" % (want,))
            self.fails.append(label)

    def done(self):
        print("\n%s: %d failure(s)" % (self.name, len(self.fails)))
        return 1 if self.fails else 0


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _launcher(root):
    """A runnable stand-in for llama-server that mdl can exec directly."""
    if os.name == "nt":
        path = root / "fake-llama-server.cmd"
        path.write_text('@echo off\r\n"%s" "%s" %%*\r\n' % (sys.executable, FAKE),
                        encoding="ascii")
    else:
        path = root / "fake-llama-server"
        path.write_text('#!/bin/sh\nexec "%s" "%s" "$@"\n' % (sys.executable, FAKE),
                        encoding="ascii")
        path.chmod(0o755)
    return path


def sandbox(port=None, extra="", model=None, binary=None):
    """Point mdl at a throwaway config and state dir. Returns (root, port)."""
    root = Path(tempfile.mkdtemp(prefix="mdl-test-"))
    (root / "config").mkdir()
    (root / "state").mkdir()
    mdl.CONFIG = root / "config" / "models.toml"
    mdl.STATE_DIR = root / "state"
    port = port or free_port()
    binary = binary or _launcher(root)
    model = model or FAKE                # any real file works as the "model"
    mdl.CONFIG.write_text(
        'llama_server = "%s"\n\n'
        "[demo]\n"
        'model = "%s"\n'
        "ngl = 99\n"
        "ctx = 4096\n"
        "flash_attn = true\n"
        'kv_type = "q8_0"\n'
        "parallel = 1\n"
        "port = %d\n%s" % (str(binary).replace("\\", "/"),
                           str(model).replace("\\", "/"), port, extra),
        encoding="utf-8")
    return root, port


def teardown(root):
    for name in ("MDL_FAKE_MODE", "MDL_LLAMA_SERVER"):
        os.environ.pop(name, None)
    shutil.rmtree(root, ignore_errors=True)


def run(fn, *args):
    """Call a cmd_* function, capturing output the way main() reports it."""
    import contextlib
    import io
    out, err, code = io.StringIO(), io.StringIO(), 0
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            fn(*args)
    except SystemExit as e:
        code = e.code
    except mdl.MdlError as e:
        err.write("mdl: " + str(e) + "\n")
        code = 1
    return out.getvalue(), err.getvalue(), code


# ------------------------------------------------------- a GGUF writer --

def _val(v):
    """(type id, packed bytes) for a metadata value."""
    if isinstance(v, bool):
        return 7, struct.pack("<?", v)
    if isinstance(v, int):
        return 4, struct.pack("<I", v)
    if isinstance(v, float):
        return 6, struct.pack("<f", v)
    if isinstance(v, str):
        b = v.encode()
        return 8, struct.pack("<Q", len(b)) + b
    if isinstance(v, list):
        etype = _val(v[0])[0] if v else 4
        body = b"".join(_val(x)[1] for x in v)
        return 9, struct.pack("<IQ", etype, len(v)) + body
    raise TypeError(v)


def write_gguf(path, meta, tensors, align=32):
    """tensors: [(name, dims, ggml_type)]; data is zeros, laid out the way
    llama.cpp's writer does it (each tensor padded to the alignment)."""
    head = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(meta))]
    for k, v in meta.items():
        kb = k.encode()
        vt, vb = _val(v)
        head.append(struct.pack("<Q", len(kb)) + kb + struct.pack("<I", vt)
                    + vb)
    offset, sizes = 0, []
    for name, dims, ty in tensors:
        n = 1
        for d in dims:
            n *= d
        size = gguf.type_bytes(ty, n)
        nb = name.encode()
        head.append(struct.pack("<Q", len(nb)) + nb
                    + struct.pack("<I", len(dims))
                    + b"".join(struct.pack("<Q", d) for d in dims)
                    + struct.pack("<IQ", ty, offset))
        sizes.append(size)
        offset += -(-size // align) * align
    raw = b"".join(head)
    raw += bytes(-(-len(raw) // align) * align - len(raw))
    with open(path, "wb") as fh:
        fh.write(raw)
        fh.write(bytes(offset))
    return path


F32, F16, Q8_0, Q4_K = 0, 1, 8, 12


def llama(path, n_layer=4, embd=256, heads=4, kv_heads=2, head=64, ff=512,
          vocab=1000, tied=False, arch="llama", extra_meta=None,
          layer_extra=None, ctx=8192, experts=0, used=0):
    """A small model of any of the shapes the engine knows about."""
    meta = {"general.architecture": arch, "general.name": "t",
            "%s.block_count" % arch: n_layer,
            "%s.context_length" % arch: ctx,
            "%s.embedding_length" % arch: embd,
            "%s.attention.head_count" % arch: heads,
            "%s.attention.head_count_kv" % arch: kv_heads,
            "tokenizer.ggml.tokens": ["t%d" % i for i in range(vocab)]}
    if experts:
        meta["%s.expert_count" % arch] = experts
        meta["%s.expert_used_count" % arch] = used
    meta.update(extra_meta or {})
    ts = [("token_embd.weight", [embd, vocab], Q8_0),
          ("output_norm.weight", [embd], F32)]
    if not tied:
        ts.append(("output.weight", [embd, vocab], Q8_0))
    for il in range(n_layer):
        b = "blk.%d." % il
        ts += [(b + "attn_norm.weight", [embd], F32),
               (b + "attn_q.weight", [embd, heads * head], Q8_0),
               (b + "attn_k.weight", [embd, kv_heads * head], Q8_0),
               (b + "attn_v.weight", [embd, kv_heads * head], Q8_0),
               (b + "attn_output.weight", [heads * head, embd], Q8_0),
               (b + "ffn_norm.weight", [embd], F32)]
        if experts:
            ts += [(b + "ffn_gate_inp.weight", [embd, experts], F32),
                   (b + "ffn_gate_exps.weight", [embd, ff, experts], Q4_K),
                   (b + "ffn_up_exps.weight", [embd, ff, experts], Q4_K),
                   (b + "ffn_down_exps.weight", [ff, embd, experts], Q4_K),
                   (b + "ffn_up_shexp.weight", [embd, ff], Q8_0)]
        else:
            ts += [(b + "ffn_gate.weight", [embd, ff], Q8_0),
                   (b + "ffn_up.weight", [embd, ff], Q8_0),
                   (b + "ffn_down.weight", [ff, embd], Q8_0)]
        ts += (layer_extra or (lambda il: []))(il)
    return write_gguf(path, meta, ts)
