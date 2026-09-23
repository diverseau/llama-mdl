# mdl-lab — a comparison harness for models and configs

**Status:** built as `mdl lab` (`mdl_fit/lab.py`). Section 16 says where
it departs from this design and what is not built yet.
**Written:** 2026-09-10, against mdl 0.5.2 on this machine.

## 1. Why

The evidence this is needed is already in `~/.config/mdl/`:

```
models.toml.qwen27b-baseline-ngl56-20260904.bak
models.toml.qwen27b-best-ngl57-mtpoff-20260904.bak
models.toml.qwen27b-best-ngl55-ctx32k-mtpoff-20260905.bak
models.toml.qwen27b-best-ngl51-ctx64k-mtpoff-20260905.bak
models.toml.qwen27b-best-ngl48-ctx96k-mtpoff-20260905.bak
models.toml.qwen27b-best-ngl45-ctx128k-mtpoff-20260905.bak
```

That is a parameter sweep encoded in filenames. The *findings* live in the
names, the *numbers* live nowhere, and the only way to compare two of them is
to remember. `llama-bench` covers synthetic pp/tg sweeps, but it does not
answer the actual question — "with **my** prompt, on **my** config, what does
this cost and how fast is it?" — and it cannot compare CUDA against Vulkan
against a different quant against a different `n_cpu_moe` in one table.

`mdl-lab` makes a run a **record**, not a memory.

## 2. What it does

One sentence: *runs the same prompt against N model configs, one at a time,
measures speed and system cost throughout, and prints a table.*

```console
$ mdl-lab run --suite ngl-sweep
  qwen38-fast/ngl45-cuda   rep 1/3  ########..  1,204 tok   38.2 tok/s  9.1G VRAM
  ...

$ mdl-lab report ngl-sweep

variant           backend  decode tok/s        ttft   prefill   VRAM peak  RAM d
                           avg   p95   floor
ngl45-cuda        cuda     38.2  41.0   31.4   0.42s   612/s     9,140M   1.2G
ngl45-vulkan      vulkan   24.9  26.1   22.5   0.88s   422/s     9,020M   1.2G
ngl38-cuda        cuda     33.1  35.8   28.0   0.51s   598/s     8,110M   2.4G
```

## 3. Machine facts that shape the design

Measured on this box today. **These are the constraints; do not design past them.**

| Fact | Consequence for the tool |
|---|---|
| `nvidia-smi --query-compute-apps=used_memory` returns `[N/A]` for every process — Windows WDDM does not do per-process GPU accounting | **Per-process VRAM is impossible.** Attribute VRAM by *total-used delta* against a pre-load baseline, and cross-check against llama.cpp's own buffer-size log lines. Never claim a per-process figure. |
| CUDA build is `b10798 (c390d0abb)`, Vulkan build is `b10424 (2bacf9ea5)` — 374 builds apart | `mdl` vs `mdl-cuda` is **not** a clean backend A/B. Every record must store build id + commit, and any backend comparison must print a confound warning. |
| `mdl-cuda` is a 3-line shim setting `MDL_LLAMA_SERVER`, then calling `mdl` — same config, same state dir, same port | Do not shell out to two commands. The runner sets `MDL_LLAMA_SERVER` itself and treats **backend as an axis of the matrix**. |
| Every model in `models.toml` shares port 8080 | Runs are **strictly serial**. No parallelism, ever. Guard on `mdl ps` before starting. |
| `XDG_CONFIG_HOME`, `XDG_STATE_HOME`, `MDL_LLAMA_SERVER` all override mdl's paths (`mdl.py:30-33`, `mdl.py:98`) | Variants run against a **generated temp config in a temp state dir**. `models.toml` is never written to. This replaces the `.bak` workflow outright. |
| Free RAM swings — 8.13 GB now, has been seen at 4.8 GB | Re-measure before every run; abort a run whose offload no longer fits rather than paging to disk. |
| `llamacpp:n_decode_total` is the only metric that moves during streaming; `predicted_tokens_seconds` sits at 0 and publishes one average at the end (`mdl_ui.py:1424-1439`) | Do not poll `/metrics` for a live rate curve. Derive the curve from **SSE token arrival times** client-side. |
| `parallel = 1` on every model | One slot; no concurrent-slot accounting needed in v1. |

## 4. Core model

```
Variant   = base model name + config overrides + backend
Workload  = prompt + depth + generation cap + sampler params (fixed)
Run       = Variant x Workload x repetition  ->  Record
Suite     = a named matrix of Variants sharing one Workload
```

A **Record** is one JSON object with everything needed to reproduce and
compare it. Records are append-only.

## 5. Architecture

Separate tool, not a patch to `mdl.py`. `CONTRIBUTING.md` requires `mdl.py`
stay dependency-free, and this needs `psutil`. It *drives* mdl through its
documented CLI and env vars rather than importing it.

```
mdl-lab/
  lab.py          CLI entry, subcommand dispatch       (stdlib)
  runner.py       lifecycle: materialize config, start, warm, measure, stop
  sampler.py      background system sampler            (psutil, nvml)
  client.py       SSE streaming client + timings parse (stdlib http.client)
  metrics.py      statistics over a Record
  store.py        JSONL read/write, schema versioning
  report.py       tables, markdown, regression diff
  suites/         suite definitions (TOML)
```

**Dependencies:** `psutil` (5.9.8, already installed) for RAM/CPU.
`nvidia-ml-py` **recommended** — NVML sampling is ~1 ms versus ~150 ms for an
`nvidia-smi` subprocess, which matters at a 250 ms cadence. Fall back to
`nvidia-smi` when absent, at a slower cadence, and say so in the record.
`textual` only for the UI pane (already installed).

### Isolation

Per run, the runner:

1. Writes a temp config containing **only** the variant, into `$TMP/lab-<id>/mdl/models.toml`.
2. Sets `XDG_CONFIG_HOME=$TMP/lab-<id>`, `XDG_STATE_HOME=$TMP/lab-<id>/state`,
   and `MDL_LLAMA_SERVER` to the chosen backend's binary.
3. Calls `mdl run <name>`, waits for ready.
4. Measures.
5. Calls `mdl stop --all`, then verifies the port is free and VRAM returned to
   baseline before the next run.

The user's `models.toml`, state dir and logs are untouched. A crashed harness
leaves a temp dir, not a mangled config.

## 6. Measurement

### Sources, and what each is authoritative for

| Source | Cadence | Authoritative for | Notes |
|---|---|---|---|
| **SSE stream** from `/v1/chat/completions` | per token | TTFT, the inter-token curve, avg/peak/floor tok/s | The primary source. Timestamp each chunk on arrival with `time.perf_counter()`. |
| **`timings` object** in the final chunk | once | prompt-eval tok/s, decode tok/s as llama.cpp sees it | Send `"timings_per_token": true`. This is the number to reconcile against; if it disagrees with the SSE curve by >5%, record both and flag. |
| **`/metrics`** Prometheus | 1 s | KV-cache ratio, slot state, decode counter | Needs `--metrics` in args — every current model has it. Not used for the rate curve (see §3). |
| **NVML / nvidia-smi** | 250 ms | total VRAM used, GPU util, temperature, clocks | Total-device only. Baseline-delta for attribution. |
| **psutil** on the llama-server pid **+ children** | 250 ms | RSS, CPU %, page faults | Pid comes from `$XDG_STATE_HOME/mdl/run/<name>.json`. Must walk children — mdl notes the binary may be a wrapper. |
| **server log** | on completion | layer offload split, buffer sizes, backend banner | Parse `load_tensors:` / buffer lines for the authoritative per-buffer VRAM claim, which beats an nvidia-smi delta. |

### Phases

Every run is segmented, and every sample is tagged with its phase:

```
baseline -> load -> ready -> [warmup rep] -> prefill -> decode -> settle
```

`baseline` is sampled for ~3 s with nothing running; it is what VRAM and RAM
deltas are measured against. `settle` is sampled after stop, to confirm memory
was actually released before the next variant starts.

### Token-indexed sampling

This is the feature asked for — *"system usage 500 tokens into generation"*.

Each sample carries the generated-token count at the moment it was taken:

```json
{"t": 12.478, "phase": "decode", "tok_idx": 500, "vram_mb": 9140,
 "rss_mb": 2210, "cpu_pct": 38.5, "gpu_util": 96, "kv_ratio": 0.21}
```

That makes every question a slice: usage *at* token 500, usage *between*
tokens 400–600, VRAM growth per 1k tokens of KV. Report a `@500` column by
default and let `--at N` change it.

## 7. Metrics computed

**Throughput** — the headline set:

- `decode.avg` — total generated tokens ÷ total decode wall time.
- `decode.p50 / p95 / p05`
- `decode.peak` — max of the **windowed** rate, not the fastest single gap.
- `decode.floor` — min of the windowed rate. See the note below.
- `decode.stddev` and `decode.cv` — the stability number. A config with a high
  average and a high CV is worse to *use* than a slightly slower steady one.
- `prefill.tps`, `ttft`, `wall`

> **Defining "low" honestly.** The min single inter-token gap is meaningless —
> one scheduler hiccup or a GC pause sets it. Compute the rate over a sliding
> **1-second (or 20-token) window**, then take the minimum of that series,
> after discarding the first 2% of tokens. Report `p05` alongside it so a
> single stall is visible as a gap between the two rather than hidden in one.

**System cost:**

- `vram.peak`, `vram.steady` (median over decode), `vram.delta` vs baseline
- `vram.load` — from the server log's buffer lines, the honest attribution
- `ram.peak_rss`, `ram.delta`, `ram.pagefaults` — page faults are the early
  warning that an offload is about to make the machine unusable
- `cpu.mean`, `cpu.peak` — expected to be high for `n_cpu_moe` configs; this is
  the number that says *how* high
- `gpu.util_mean`, `gpu.temp_peak`, `gpu.clock_min` — a thermally throttled run
  is not a slow config, and without clocks you cannot tell them apart
- `kv.ratio_at_end`

**Derived / comparative:**

- `tok_per_gb` — throughput per GB of VRAM. The efficiency column that decides
  what actually earns its place on a 12 GB card.
- `time_to_1k` — wall seconds to 1,000 output tokens, TTFT included. Closer to
  felt speed than tok/s.

## 8. Rigor — the part that makes this trustworthy

Without these it is a random-number generator with a nice table.

- **Fixed sampler params across variants.** Temperature, top-p, top-k, seed and
  max tokens come from the *workload*, and override whatever the model's `args`
  say. Record both, warn on override. Note several models pin `--temp`/`--top-p`
  in `args` today; the harness must win.
- **Seeded.** `"seed": N` per repetition, same across variants.
- **Warmup discarded.** First repetition is thrown away by default
  (`--warmup 1`); first-token paths and file cache are cold.
- **N ≥ 3 repetitions**, report median and spread. A single run is a datapoint,
  not a measurement. Print the spread; refuse to declare a winner when
  intervals overlap — say "indistinguishable".
- **Cooldown between variants** (`--cooldown 20s`, default) so VRAM is released
  and the card is off boost residue. Verify baseline is re-reached before
  proceeding; abort with a clear message if it is not.
- **Pre-flight guards**, all fatal by default:
  - something already on port 8080 → refuse (never auto-kill; that is the
    user's running server)
  - free RAM below `offload estimate + 2.5 GB` → refuse this variant, continue
    the suite, mark it `skipped: insufficient RAM`
  - `mdl check` fails → refuse
- **Depth sweeps.** `--depth 0,8k,32k` prefills a synthetic context before
  measuring decode. This is where configs actually diverge — visible in this
  machine's own logs, where `nex-n25-fast` runs 24.95 tok/s at ~7k context and
  22.55 tok/s at 32k. A benchmark at depth 0 picks configs that degrade in use.
- **Environment capture** per record: both build ids and commits, driver
  version, free RAM/VRAM at start, OS build. Cheap to collect, and the only way
  to explain an outlier later.

## 9. CLI surface

```
mdl-lab run <variant|--suite NAME>   run and record
    --prompt FILE|-              workload text
    --depth 0,8k,32k             prefill depths to sweep
    --reps N  --warmup N         repetitions, discarded warmups
    --max-tokens N               generation cap (default 512)
    --backend cuda,vulkan        backend axis
    --at N                       token index for the @N snapshot
    --cooldown S                 seconds between variants
    --dry-run                    print the matrix and the commands, run nothing
mdl-lab report [suite]           table; --format md|csv|json
mdl-lab compare A B              two records side by side, deltas and verdict
mdl-lab baseline set|diff        pin a record; diff current against it
mdl-lab ls                       recorded runs
mdl-lab export <id>              one record as JSON
mdl-lab apply <id>               print the models.toml block for a winner
```

`apply` prints; it does not write. Editing `models.toml` stays a decision the
user makes, with a diff in front of them.

`--dry-run` matters more than it looks: a 6-variant × 3-rep × 3-depth suite is
54 model loads. Seeing that before committing an hour to it is the difference
between using the tool and abandoning it.

## 10. UI integration

The ask was "do a prompt in the ui" — so the entry point is the dashboard.

- **`b`** on a selected model: prompt for workload, run a single measured run
  against the current config, show the live pane, append the record.
- **Live pane** during a run: the existing tok/s sparkline, plus VRAM/RAM/CPU
  meters, current token index, and the running avg/peak/floor.
- **`B`**: pick a suite and run the whole matrix, with a progress list.
- **Compare view**: select two or more records, get the table inline.
- The existing chat pane (`p`) already measures tok/s and TTFT — reuse its
  streaming client rather than writing a second one, and have it write a record
  when the user asks for one.

Everything remains available headless; the UI is a front end to `runner.py`,
never the only path.

## 11. Storage

`~/.local/state/mdl-lab/records.jsonl` — append-only, one Record per line,
`schema: 1`. JSONL because it survives a crash mid-suite and is trivially
greppable. Raw per-sample series go beside it as `samples/<record-id>.jsonl`
so the index stays small and readable.

```json
{"schema": 1, "id": "2026-09-10T01-42-11Z-a3f1", "suite": "ngl-sweep",
 "variant": {"base": "qwen38-fast", "backend": "cuda",
             "overrides": {"ngl": 45, "ctx": 131072}},
 "build": {"backend": "cuda", "build": 10798, "commit": "c390d0abb"},
 "workload": {"prompt_sha": "9c1f", "tokens_in": 705, "max_tokens": 512,
              "seed": 42, "temp": 0.6, "depth": 0},
 "rep": 2, "warmup": false,
 "metrics": {"decode": {"avg": 38.2, "p95": 41.0, "p05": 33.9,
                        "floor": 31.4, "peak": 42.6, "cv": 0.08},
             "prefill": {"tps": 612.4}, "ttft": 0.42,
             "vram": {"peak": 9140, "steady": 9080, "delta": 7855},
             "ram": {"peak_rss": 2210, "delta": 1180},
             "cpu": {"mean": 38.5, "peak": 61.2},
             "at": {"500": {"vram": 9120, "rss": 2205, "cpu": 39.1}}},
 "env": {"ram_free_start": 8320, "vram_free_start": 10831, "driver": "..."},
 "flags": ["timings_agree"]}
```

## 12. Reporting

- Default: a terminal table, one row per variant, medians across reps, spread
  shown as `38.2 ±1.1`.
- `--format md` for pasting into notes; `csv` for a spreadsheet.
- `compare` prints deltas as percentages with a plain-language verdict —
  including **"indistinguishable"** when the intervals overlap.
- Any comparison spanning backends prints the build-mismatch warning from §3.
- A `flags` column surfaces `thermal_throttle`, `ram_pressure`,
  `timings_disagree`, `high_variance` rather than burying them.

## 13. Build order

| Phase | Delivers | Worth it because |
|---|---|---|
| **1** | `runner.py` + `client.py`: one variant, one prompt, SSE timing, JSONL record. No sampler. | Already beats the `.bak` workflow — real numbers, stored. |
| **2** | `sampler.py`: VRAM/RAM/CPU at 250 ms, phases, token-indexed samples, `@N` snapshot. | This is the actual ask. |
| **3** | Suites, matrix expansion, reps/warmup/cooldown, guards, `--dry-run`. | Turns single runs into comparisons. |
| **4** | `report.py`: tables, compare, baseline diff. | Makes the records answer questions. |
| **5** | UI pane (`b`), live meters, inline compare. | The requested entry point; needs 1–4 underneath it. |
| **6** | Depth sweeps, thermal/clock capture, regression mode. | Rigor, once the loop is proven. |

Phase 1+2 is the minimum that answers the original question. Ship that, use it
for a week, then decide whether 3–6 survive contact.

## 14. Non-goals

- **Not a replacement for `llama-bench`.** That tool sweeps synthetic pp/tg
  faster and without HTTP in the path. `mdl-lab` measures *your prompt through
  a real server*. Use both; cite which one a number came from.
- **No auto-tuning.** It measures and reports. It does not search the parameter
  space on its own or edit `models.toml`.
- **No quality evaluation.** Speed and system cost only. Whether the output is
  *good* is a different tool and a much harder problem — do not let a tok/s
  table imply otherwise.
- **No daemon, no scheduling.** Matches mdl's own non-goals.
- **Not concurrent.** One port, one server, serial by physics.

## 15. Known risks

- **Backend/build confound** (§3) is the biggest threat to the tool's
  credibility. Mitigate by warning loudly, and ideally by building a
  matched-commit Vulkan binary before trusting any cuda-vs-vulkan number.
- **VRAM attribution is a delta, not a measurement.** Anything else running on
  the GPU during a run corrupts it. Sample the baseline, re-check at `settle`,
  and flag drift.
- **Windows timer resolution** is ~15.6 ms by default, which is coarse against
  a 40 ms/token decode. Use `time.perf_counter()` and note that sub-10 ms TTFT
  differences are not real.
- **A 54-load suite takes hours** and holds the GPU the whole time. `--dry-run`
  and a printed time estimate are not polish, they are what stops the tool from
  being abandoned after one accidental overnight run.

## 16. As built

`mdl lab` is this design as a subcommand of mdl, not a separate tool.

**Where it departs, and why.**

- **Standard library only.** `mdl_fit` takes no dependencies, so there is
  no psutil and no NVML. VRAM, GPU load, temperature and SM clock come
  from one `nvidia-smi --query-gpu` per sample; RAM is the resident size
  of the server's whole process tree, from mdl's own process table
  (`usage.processes`); CPU is the OS's system-wide load. That costs the
  250 ms cadence: `--interval` defaults to 1 s, which is what a sample
  takes on Windows. There is no per-process CPU figure and no page-fault
  count.
- **A build, not a backend name.** `--server PATH[,PATH]` runs each
  variant on each llama-server given; the record carries that build's
  number, commit and backend (`hw.llama_build`, `hw.backend_for`), and
  any report or comparison across builds says the difference is the
  build's as much as the config's.
- **Records live with mdl's state**, in `<state>/lab/records.jsonl` and
  `<state>/lab/samples/`, not in a directory of their own.
- **Pre-flight** is `check_cfg` on each variant's table rather than a
  whole `mdl check`, so one bad variant does not stop the others.
- **The final chunk's `timings`** are what the client's rate is checked
  against; `timings_per_token` is not requested.
- **Every repetition prefills**: requests send `cache_prompt: false`, and
  `ignore_eos` so each reply runs to `--max-tokens`.

**Built:** `run` (names by `--set` sweeps by `--server`, or `--suite`),
`--prompt`, `--depth`, `--reps`, `--warmup`, `--max-tokens`, `--at`,
`--seed`, `--temp`, `--cooldown`, `--interval`, `--dry-run` with an
estimate from mdl fit's predicted speeds; the guards (a server of yours
on the port refuses the run, too little free RAM skips a variant, VRAM
not returned to baseline ends the run); phase- and token-tagged samples
and the `@N` snapshot; the throughput set (avg, p05/p50/p95, windowed
floor and peak, cv), prefill, TTFT, time to 1k, VRAM peak, steady and
delta, the load log's claimed VRAM, RSS, RAM delta, CPU, GPU load,
temperature and clock; the `timings_disagree`, `high_variance` and
`clock_drop` flags; `report` as table, md, csv or json, `compare` with
its "indistinguishable" verdict, `ls`, `export`, `apply`; `baseline set`
and `baseline diff`, which pin a whole run rather than one record and set
a later run against it variant by variant, with `--fail` as the
regression mode (a decode, TTFT or VRAM change the spread cannot explain
is the exit code); `tok_per_gb` as the report's "t/s per G", over the
VRAM the model took (its rise over the idle card, else the load log's
claim) rather than the card's peak, which counts the desktop too; and
`b` in the dashboard for one measured repetition of the selected config.

**Not built yet:** the dashboard's live meters pane, its `B` suite
picker, and inline compare in the dashboard.
