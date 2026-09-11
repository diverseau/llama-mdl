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
  usage.py    idle-aware VRAM/RAM: OS floors, startup apps, boot baseline
  cli.py      mdl fit {<gguf>|hf:repo|<name>} [--profile ...] etc.
  catalog.py  hub crawler, lineage graph, snapshot pull, reader      (M6)
  evalsuite.py  seeded item generators and graders                   (M7)
  evalrun.py  mdl eval: runner, client, sandbox, bootstrap, store    (M7)
  quality.py  public z-scores, lineage prior, penalties, local map   (M8)
  find.py     mdl find: shortlist, fit per quant, rank, explore      (M8)
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
| M6 | catalog crawler + lineage graph | Qwen3-8B tree holds only its own quant repos; built, not published |
| M7 | `mdl eval` | a live model runs every suite, scores stored and shown |
| M8 | quality model, `mdl find` | ranked table from a local catalog plus models.toml |

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

## Where it stands (0.6.2)

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

## Part 2 (M6-M8)

Built in the order M7, M6, M8: eval only needs a server, and find needs
the other two.

**M7, `mdl eval`.** Items are generated, not stored: a per-install secret
in `eval-seed` seeds every generator, so the suite exists on this machine
only and cannot leak into anyone's training set. Code is graded by
hidden unit tests run in a subprocess (`python -E -s` in a temp dir, with
a timeout), or with `--sandbox` in a podman/docker container with no
network. Tool items run against small mock worlds (orders, prices,
files, a calendar) for up to eight turns. Long-context documents are
sized with the server's own `/tokenize`, and each is sent once so the
prompt cache carries the rest of its questions. Scores carry a bootstrap
95% interval and go to `evals.jsonl`, keyed by a cheap file hash (size
plus the first and last 8 MiB). Checked live on LFM2-8B-A1B.

**M6, the catalog.** The crawler walks the hub's model tree
(`base_model:finetune|merge:<id>`, and `base_model:quantized:<id>` +
`gguf` for quant repos), keeps what clears a noise gate (500 downloads
in 30 days or 20 likes), and stores nodes, GGUF files and reported evals
in one SQLite file. A new build reuses the file lists of repos that have
not changed since the last one. `.github/workflows/catalog.yml` rebuilds
it nightly and uploads it to `$MDL_CATALOG_REPO`; `mdl catalog pull`
fetches it with an ETag. Neither the dataset nor the Action's `HF_TOKEN`
exists yet: publishing is outward-facing and waits for an explicit go,
so for now `mdl catalog build` makes a local snapshot.

**M8, quality and `mdl find`.** Public results become z-scores per
benchmark (scale 50 + 15z), weighted by inverse variance. A score far
above what a model's other scores predict (a leave-one-out regression
per benchmark), or one on a benchmark whose dataset the model card lists
for training, is down-weighted and flagged. A model with no results
borrows its parent's, with variance growing per hop. Quant and KV
penalties come off the estimate; the quant penalty's scale is relearned
once the same model has local evals at two quants. After three models
have both public and local results, the public scale is mapped onto the
local one per domain. `find` fits the best-scoring models from the
shortlist at up to three quants each, fetching one header per model and
rescaling it for the siblings, and keeps what clears the profile's floors.
Models with local evals rank on those; unrated ones appear only under
"worth testing", with the mdl command that would rate them.

Deviations from the spec:

- Custom eval tasks are TOML (`~/.config/mdl/evals/*.toml`, `[[task]]`),
  not YAML: stdlib has a TOML reader and no YAML one.
- The hub ignores `other=` on the quantized-children query and returns
  the most popular GGUF repos of all; both conditions go in `filter=`.
- Of a model's quants, the fastest one within half a point of the best
  is the one shown, so F16 does not win on a hundredth of a point over
  Q8_0.
