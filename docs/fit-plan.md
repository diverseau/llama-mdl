# mdl fit - implementation plan

The spec is `mdl-fit-mvp.md`. This file is how it gets built, and what is
built so far.

## Where it lives

`mdl.py` is a single-file module, so the spec's `mdl/fit/` would shadow it.
The engine is its own package, `mdl_fit/`, stdlib only, and never imports
textual. `mdl fit ...` in `mdl.py` hands its arguments to `mdl_fit.cli`.
Splitting it into its own PyPI package later is a rename.

```
mdl_fit/
  gguf.py     header parser, tensor inventory, roles, hyperparams    (M0)
  remote.py   hf:org/repo - file list, Range fetch, inventory cache  (M4)
  hw.py       nvidia-smi / RAM / CPU / llama-server build probe       (M3)
  model.py    place(), memory model, perf model                      (M1, M3)
  search.py   enumerate, max-ctx per combo, Pareto, profiles         (M2)
  explain.py  predicted and real (log) failures, ranked fixes        (M2)
  emit.py     llama-server command, models.toml block                (M2)
  calib.py    oracle (llama-fit-params), llama-bench, calib.jsonl    (M1, M3)
  cli.py      mdl fit {<gguf>|hf:repo|<name>} [--profile ...] etc.
```

## Build order

The spec's own rule: memory truth first, because everything stands on it.

| | Piece | Exit test here |
|---|---|---|
| M0 | parser, `mdl fit inspect` | tensor bytes + data start == file size, every local model |
| M1 | placement + memory, oracle harness | predicted vs `llama-fit-params -fitp on` per component, every model in models.toml |
| M2 | solver, profiles, emit, explainer | tiel-coder-Fast gets a correct, ranked fix |
| M3 | `mdl fit hw`, perf model, `--verify`, calib store | measured bandwidths replace the seeds |
| M4 | `hf:` pre-download mode | quant table for a repo with no weights fetched |
| M5 | TUI: the est-VRAM bar reads place() | the fake bar is gone |
| M6 | catalog crawler + lineage graph | not in this pass |
| M7 | `mdl eval` | not in this pass |
| M8 | quality model, `mdl find` | not in this pass |

## Decisions that are not in the spec

- **The oracle is `llama-fit-params -fitp on`.** On build 10424 it prints
  `<device> <model> <context> <compute>` in MiB per device, to stdout, with
  no GPU allocation. That is the ground truth for M1 and for `--verify`.
- **KV sizes come from tensor shapes, per layer.** `attn_k`'s output dim is
  `n_head_kv x d_k` for that layer, which covers GQA, per-layer head counts,
  SWA layers with a different head size, and hybrid models (a recurrent layer
  has no `attn_k`) with one rule. Metadata is the fallback for fused `attn_qkv`,
  MLA has an adapter.
- **The compute buffer is seeded, then fitted.** Seed:
  `4 x ub x n_vocab` (logits, when the output head is on the GPU) plus
  activation terms, plus one layer's host experts when op offload has to
  copy them up for prefill. Per-arch least-squares corrections live in
  `calib.jsonl` and are applied on top.
- **The models.toml block matches today's schema**: `kv_type` when K and V
  agree, otherwise `--cache-type-k/-v` in `args`; `-b/-ub`, `-np`-free
  keys and mmap choice go through `args` (`parallel` is already a key).
  `--load-mode none` replaces `--no-mmap` on builds that have it.
- **Search is brute force with a closed-form memory function.** Per
  (kv pair, ubatch, n-cpu-moe or ngl, mmap) the memory is O(1) from
  prefix sums, and the max context is a binary search over 256-cell steps.
- **Threads are not searched until `mdl fit hw` has measured them**; until
  then the emitted command leaves `-t` to llama.cpp.

## Where it stands (0.6.0)

M0-M5 are in. Measured on the box it was built on (RTX 3060, Vulkan
llama.cpp build 10424, 16 GB RAM), against `llama-fit-params`:

- **Parser**: tensor bytes + header equal the file size exactly for all
  nine GGUFs in models.toml; about 0.3 s per file.
- **Weights, KV and recurrent state**: within 2 MiB of the oracle on all
  208 oracle points (10 configs x a grid of ubatch, ctx, n-cpu-moe, -ngl,
  -np and --swa-full), across dense, MoE, qwen35 hybrid delta-net,
  LFM2 shortconv, and Gemma 4 SWA with shared-KV layers.
- **Device total, uncalibrated**: 9 of the 10 current configs within
  ±256 MiB. The tenth, a UD-Q3_K_XL MoE at ubatch 1024 with experts on
  the CPU, is 507 MiB under: its compute buffer grows with ubatch in a
  way no other quant here does, and only the oracle can see it. One
  oracle check books that against the file and closes it.
- **Latency**: about 1 s analytic, about 3 s with the oracle checking the
  picks, the first time; the oracle takes ~0.65 s even on a 35B MoE.

What turned out to be true on this build, and is now encoded:

- `-ngl L` puts the last L-1 blocks *and the output head* on the GPU
  (the output is layer n_layer in llama.cpp's own count).
- SWA layers hold `window x n_seq + ubatch` cells, padded to 256.
- Gemma 4's `shared_kv_layers` hold no cache; its `per_layer_model_proj`
  follows layer 0's device.
- An unused MTP head is not loaded at all - no weights, no KV.
- The GPU compute buffer is the largest of three phases (logits; one
  layer's host weights copied up; from ubatch 2048, or 1024 on SWA
  models, logits + a per-token term + the KQ mask), and a logits tensor
  of exactly 4 GiB or more is computed on the CPU under Vulkan.
- `-b` does not change any buffer; only `-ub` does.
- Flash attention off crashes the oracle on this build, so the engine
  always emits `-fa on`.

Known gaps: the speed model is bytes over bandwidth and is only as good
as `mdl fit hw` makes it; per-arch efficiency comes from `--verify` runs.
MLA and fused-QKV KV sizes follow the metadata but were not oracle-checked
here (no such model on the box). Multi-GPU is out of scope for the MVP.

## Part 2, when it comes

M6 needs a GitHub Action and an HF dataset under `diverseau` - both
outward-facing, so they wait for an explicit go. M7 (`mdl eval`) only needs
a running server and can be built in parallel with anything. M8 consumes
M6 + M7 + `search.best()` per quant, which is why it comes last.
