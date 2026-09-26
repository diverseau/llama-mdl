"""GGUF header parser and tensor inventory.

Reads the metadata and the tensor table, never the weights. Tensor sizes
come from the gaps between offsets, which is exact and needs no knowledge
of any quant's block layout; the type table is kept as a cross-check and
for the cases where adjacency breaks.
"""

import json
import re
import struct
from pathlib import Path

MAGIC = b"GGUF"
ALIGNMENT = 32

# ggml type id -> (name, elements per block, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
}

# value type id -> struct format; 8 is string, 9 is array
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
# Arrays longer than this are kept as a length only. Per-layer arrays
# (head counts, SWA patterns) are well under it; token lists are not.
KEEP_ARRAY = 4096

SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


class Truncated(Exception):
    """The buffer ended mid-header. Fetch more and parse again."""


class NotGGUF(ValueError):
    pass


_STRUCTS = {}
_U64 = struct.Struct("<Q")


class _Reader:
    def __init__(self, buf):
        self.buf, self.pos = buf, 0

    def take(self, fmt):
        s = _STRUCTS.get(fmt)
        if s is None:
            s = _STRUCTS[fmt] = struct.Struct(fmt)
        if self.pos + s.size > len(self.buf):
            raise Truncated(self.pos + s.size)
        (value,) = s.unpack_from(self.buf, self.pos)
        self.pos += s.size
        return value

    def skip(self, n):
        if self.pos + n > len(self.buf):
            raise Truncated(self.pos + n)
        self.pos += n

    def string(self):
        n = self.take("<Q")
        if self.pos + n > len(self.buf):
            raise Truncated(self.pos + n)
        raw = bytes(self.buf[self.pos:self.pos + n])
        self.pos += n
        return raw.decode("utf-8", "replace")

    def value(self, vtype):
        if vtype in _SCALAR:
            return self.take(_SCALAR[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            etype, count = self.take("<I"), self.take("<Q")
            if count > KEEP_ARRAY:
                self._skip_array(etype, count)
                return {"_len": count}
            return [self.value(etype) for _ in range(count)]
        raise NotGGUF("unknown metadata value type %d" % vtype)

    def _skip_array(self, etype, count):
        if etype in _SCALAR:
            self.skip(struct.calcsize(_SCALAR[etype]) * count)
        elif etype == _STRING:
            # a vocabulary: 150k-260k strings, walked on every header
            # parse; in locals, not a call per string, gemma-3's header
            # parses in a third of the time (0.16 s to 0.055 s)
            buf, pos, end = self.buf, self.pos, len(self.buf)
            unpack = _U64.unpack_from
            for _ in range(count):
                if pos + 8 > end:
                    raise Truncated(pos + 8)
                pos += 8 + unpack(buf, pos)[0]
                if pos > end:
                    raise Truncated(pos)
            self.pos = pos
        else:
            for _ in range(count):
                self.value(etype)


def parse_header(buf):
    """(meta, tensor infos, data_start) from the first bytes of a GGUF.

    Raises Truncated with the byte count it needed when the buffer is too
    short, so a remote caller knows how much more to fetch.
    """
    r = _Reader(buf)
    if len(buf) < 4:
        raise Truncated(4)
    if bytes(buf[:4]) != MAGIC:
        raise NotGGUF("not a GGUF file")
    r.pos = 4
    version = r.take("<I")
    if version < 2:
        raise NotGGUF("GGUF v%d is too old; convert it again" % version)
    n_tensors, n_kv = r.take("<Q"), r.take("<Q")
    meta = {}
    for _ in range(n_kv):
        key = r.string()
        meta[key] = r.value(r.take("<I"))
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.take("<I")
        dims = [r.take("<Q") for _ in range(n_dims)]
        infos.append((name, dims, r.take("<I"), r.take("<Q")))
    align = meta.get("general.alignment", ALIGNMENT) or ALIGNMENT
    data_start = -(-r.pos // align) * align
    return meta, infos, data_start


def type_bytes(ggml_type, n_elements):
    info = GGML_TYPES.get(ggml_type)
    if not info:
        return None
    _, block, size = info
    return -(-n_elements // block) * size


def type_name(ggml_type):
    info = GGML_TYPES.get(ggml_type)
    return info[0] if info else "type%d" % ggml_type


# ------------------------------------------------------------------ roles --

_EXPS = re.compile(r"^ffn_(gate|up|down|gate_up)_(ch)?exps(\.|$)")
_SHEXP = re.compile(r"^ffn_(gate|up|down|gate_up)_(ch)?shexp(\.|$)")
_ROUTER = re.compile(r"^(ffn_gate_inp|ffn_gate_inp_shexp|exp_probs_b)(\.|$)")
_FFN = re.compile(r"^ffn_(gate|up|down|gate_up)(\.|$)")
_RECURRENT = re.compile(r"^(ssm_|time_mix_|channel_mix_|shortconv|conv1d)")
_NORM = re.compile(r"_norm(_\w+)?(\.|$)|^layer_output_scale|^norm(\.|$)")
_LAYER = re.compile(r"^blk\.(\d+)\.(.+)$")

# Tensors outside the blocks that llama.cpp keeps on the input device.
HOST_GLOBALS = ("token_embd.", "per_layer_token_embd.", "token_types.",
                "position_embd.")


def classify(name):
    """(layer or None, role). Unknown names are 'other' - never a crash."""
    m = _LAYER.match(name)
    if not m:
        if name.startswith(("token_embd.", "per_layer_token_embd.")):
            return None, "embd"
        if name.startswith(("token_types.", "position_embd.")):
            return None, "embd"
        if name.startswith("output_norm"):
            return None, "output_norm"
        if name.startswith("output."):
            return None, "output"
        return None, "global"
    layer, rest = int(m.group(1)), m.group(2)
    if _EXPS.match(rest):
        role = "exps"
    elif _SHEXP.match(rest):
        role = "shexp"
    elif _ROUTER.match(rest):
        role = "router"
    elif _FFN.match(rest):
        role = "ffn"
    elif rest.startswith("nextn."):
        role = "mtp"
    elif rest.startswith(("inp_gate.", "proj.")):
        role = "ple"                 # Gemma 3n/4 per-layer embedding path
    elif _RECURRENT.match(rest):
        role = "recurrent"
    elif rest.startswith("attn_") and not _NORM.search(rest):
        role = "attn"
    elif _NORM.search(rest):
        role = "norm"
    else:
        role = "other"
    return layer, role


class Tensor:
    __slots__ = ("name", "layer", "role", "type", "dims", "offset", "nbytes",
                 "shard")

    def __init__(self, name, dims, ggml_type, offset, shard=0):
        self.name, self.dims, self.type = name, list(dims), ggml_type
        self.offset, self.shard, self.nbytes = offset, shard, 0
        self.layer, self.role = classify(name)

    @property
    def n_elements(self):
        n = 1
        for d in self.dims:
            n *= d
        return n

    def to_list(self):
        return [self.name, self.dims, self.type, self.offset, self.nbytes,
                self.shard]

    @classmethod
    def from_list(cls, row):
        t = cls(row[0], row[1], row[2], row[3], row[5])
        t.nbytes = row[4]
        return t


class Inventory:
    """Every tensor in a model, with its layer, role and exact byte size."""

    def __init__(self, source, meta, tensors, file_sizes, data_starts,
                 warnings=None):
        self.source = source              # a path, or hf:org/repo/file
        self.meta = meta
        self.tensors = tensors
        self.file_sizes = file_sizes      # one per shard
        self.data_starts = data_starts
        self.warnings = list(warnings or [])
        self.arch = meta.get("general.architecture", "unknown")
        others = sorted({t.name.split(".", 2)[2] if t.layer is not None
                         else t.name for t in tensors if t.role == "other"})
        if others:
            self.warnings.append("unrecognised tensors travel with their "
                                 "layer: " + ", ".join(others[:6])
                                 + (" ..." if len(others) > 6 else ""))

    # -- metadata helpers ---------------------------------------------------
    def hp(self, key, default=None):
        """{arch}.<key> from the metadata."""
        return self.meta.get("%s.%s" % (self.arch, key), default)

    @property
    def name(self):
        return Path(str(self.source)).name

    @property
    def file_size(self):
        return sum(self.file_sizes)

    @property
    def n_layer(self):
        n = self.hp("block_count")
        if n:
            return int(n)
        layers = [t.layer for t in self.tensors if t.layer is not None]
        return max(layers) + 1 if layers else 0

    @property
    def n_embd(self):
        n = self.hp("embedding_length")
        if n:
            return int(n)
        embd = self.find("token_embd.weight")
        return embd.dims[0] if embd else 0

    @property
    def n_vocab(self):
        tokens = self.meta.get("tokenizer.ggml.tokens")
        if isinstance(tokens, dict):
            return tokens["_len"]
        if isinstance(tokens, list):
            return len(tokens)
        if self.hp("vocab_size"):
            return int(self.hp("vocab_size"))
        embd = self.find("token_embd.weight") or self.find("output.weight")
        return embd.dims[1] if embd and len(embd.dims) > 1 else 0

    @property
    def n_ctx_train(self):
        return int(self.hp("context_length", 0) or 0)

    @property
    def n_expert(self):
        return int(self.hp("expert_count", 0) or 0)

    @property
    def n_expert_used(self):
        return int(self.hp("expert_used_count", 0) or 0)

    @property
    def is_moe(self):
        return any(t.role == "exps" for t in self.tensors)

    @property
    def tied_embeddings(self):
        return self.find("output.weight") is None

    def per_layer(self, key, il, default=None):
        """A hyperparameter that may be a scalar or a per-layer array."""
        v = self.hp(key, default)
        if isinstance(v, list):
            return v[il] if il < len(v) else default
        return v

    def find(self, name):
        for t in self.tensors:
            if t.name == name:
                return t
        return None

    def layer_tensors(self):
        """{layer: [tensors]}"""
        out = {}
        for t in self.tensors:
            if t.layer is not None:
                out.setdefault(t.layer, []).append(t)
        return out

    @property
    def n_params(self):
        return sum(t.n_elements for t in self.tensors)

    @property
    def bpw(self):
        """Bits per weight across the whole file - the real quant tier."""
        n = self.n_params
        return 8.0 * sum(t.nbytes for t in self.tensors) / n if n else 0.0

    @property
    def quant_label(self):
        """Q4_K_M-style label from the filename, else the dominant type."""
        # the catalog's reading, so a PQ2_0 file is not called Q2_0 here
        from .catalog import quant_of
        q = quant_of(str(self.source))
        if q != "?":
            return q
        sizes = {}
        for t in self.tensors:
            sizes[t.type] = sizes.get(t.type, 0) + t.nbytes
        return type_name(max(sizes, key=sizes.get)) if sizes else "?"

    # -- persistence -------------------------------------------------------
    def to_json(self):
        return json.dumps({
            "v": 1, "source": str(self.source), "meta": self.meta,
            "tensors": [t.to_list() for t in self.tensors],
            "file_sizes": self.file_sizes, "data_starts": self.data_starts,
            "warnings": self.warnings})

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        inv = cls(d["source"], d["meta"],
                  [Tensor.from_list(r) for r in d["tensors"]],
                  d["file_sizes"], d["data_starts"])
        inv.warnings = d.get("warnings", [])
        return inv


def assign_sizes(tensors, data_start, file_size, warnings):
    """Size each tensor from the gap to the next offset.

    The last tensor runs to the end of the file. Without a file size (a
    remote header with no listing) the type table has to do.
    """
    order = sorted(tensors, key=lambda t: t.offset)
    for i, t in enumerate(order):
        if i + 1 < len(order):
            gap = order[i + 1].offset - t.offset
        elif file_size:
            gap = file_size - data_start - t.offset
        else:
            gap = None
        table = type_bytes(t.type, t.n_elements)
        if gap is None or gap < 0 or (table is not None and gap < table):
            if table is None:
                warnings.append("cannot size %s (%s)" % (t.name,
                                                         type_name(t.type)))
                gap = 0
            else:
                gap = table
        t.nbytes = gap


def from_buffer(source, buf, file_size, shard=0, parsed=None):
    """Inventory of one GGUF (or one shard) from its header bytes, or
    from `parsed`, parse_header(buf) when the caller already has it."""
    meta, infos, data_start = parsed or parse_header(buf)
    tensors = [Tensor(n, d, ty, off, shard) for n, d, ty, off in infos]
    warnings = []
    assign_sizes(tensors, data_start, file_size, warnings)
    return meta, tensors, data_start, warnings


def read_local(path, chunk=8 << 20):
    """(header bytes, file size, parse_header of them) of a local file,
    growing the read until it parses."""
    path = Path(path)
    size = path.stat().st_size
    want = min(chunk, size)
    with open(path, "rb") as fh:
        while True:
            fh.seek(0)
            buf = fh.read(want)
            try:
                return buf, size, parse_header(buf)
            except Truncated as e:
                if want >= size:
                    raise NotGGUF("header runs past the end of the file") \
                        from None
                want = min(size, max(want * 2, int(e.args[0]) + (1 << 20)))


def shard_paths(path):
    """Every shard of a split GGUF, in order, or just [path]."""
    path = Path(path)
    m = SPLIT_RE.search(path.name)
    if not m:
        return [path]
    count = int(m.group(2))
    stem = path.name[:m.start()]
    return [path.with_name("%s-%05d-of-%05d.gguf" % (stem, i, count))
            for i in range(1, count + 1)]


def load(path):
    """Inventory of a local GGUF, split or not."""
    shards = shard_paths(path)
    missing = [p for p in shards if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing shard %s" % missing[0])
    return merge(str(path), [(read_local(p)) for p in shards])


def merge(source, pieces):
    """Union the tensor tables of each shard; metadata from the first.
    A piece is (header bytes, file size), and its parse if there is one."""
    meta, tensors, sizes, starts, warnings = None, [], [], [], []
    for i, (buf, size, *parsed) in enumerate(pieces):
        m, ts, start, w = from_buffer(source, buf, size, shard=i,
                                      parsed=parsed[0] if parsed else None)
        meta = meta if meta is not None else m
        tensors += ts
        sizes.append(size)
        starts.append(start)
        warnings += w
    return Inventory(source, meta, tensors, sizes, starts, warnings)
