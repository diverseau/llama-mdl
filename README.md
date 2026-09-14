# mdl

A small CLI for running local [llama.cpp](https://github.com/ggml-org/llama.cpp)
servers from a config file, instead of pasting flag soup into your shell.

```sh
# without mdl
llama-server -m /srv/models/Ornith-1.5-35B-A3B-Q4_K_M.gguf -ngl 99 \
  --n-cpu-moe 24 -c 65536 -fa on --cache-type-k q8_0 \
  --cache-type-v q8_0 -np 1 --port 8080

# with mdl
mdl run ornith
```

![the mdl dashboard](docs/screenshot.svg)

`mdl.py` is a single file, Python 3.11+ (needs `tomllib`), standard library
only. Runs on Linux, macOS and Windows.

`mdl_ui.py` adds an optional terminal dashboard (`mdl ui`). It is the only
part that needs a dependency — [Textual](https://textual.textualize.io/) —
and the CLI never imports it, so every command but `ui` stays
dependency-free.

In practice, it lets you manage llama.cpp servers through named model presets:
start llama-server from a config file, switch between GGUF models, or run
multiple llama-server instances at once.

## Install

```sh
pipx install llama-mdl          # or: pip install llama-mdl
pipx install "llama-mdl[ui]"    # with the terminal dashboard
```

The package is `llama-mdl`; the command it installs is `mdl`. (Plain `mdl`
on PyPI is an unrelated project.) Nothing but the dashboard has a
dependency, and that is [Textual](https://textual.textualize.io/).

Or run it straight from a clone - it is two files and a standard library:

```sh
git clone https://github.com/diverseau/llama-mdl ~/src/mdl
python ~/src/mdl/mdl.py --help
```

Then create a starter config:

```sh
mdl init
```

That writes `~/.config/mdl/models.toml`, finds `llama-server` on your PATH if
it is there, and tells you what to edit.

## Config

`~/.config/mdl/models.toml`. One table per model; the table name is what you
pass to `mdl run`.

```toml
# Optional. Defaults to "llama-server" on $PATH.
# $MDL_LLAMA_SERVER overrides this.
llama_server = "/opt/llama.cpp/build/bin/llama-server"

[ornith]
model = "/srv/models/Ornith-1.5-35B-A3B-Q4_K_M.gguf"
ngl = 99
n_cpu_moe = 24
ctx = 65536
flash_attn = true
kv_type = "q8_0"
parallel = 1
port = 8080

[qwen-small]
model = "/srv/models/Qwen3-8B-Q5_K_M.gguf"
ngl = 99
ctx = 16384
port = 8081
```

On Windows, write paths with forward slashes (`C:/models/foo.gguf`) or double
the backslashes, since TOML treats `\` as an escape character.

### Ports

Models may share a port, and it is reasonable for them to: only one can be
running on it at a time, and `mdl` will say so rather than let you find out
from llama.cpp. To run two at once, give each its own port - or pass
`mdl run <name> --port N` for a one-off. `mdl check` lists which models
share one.

### Keys

| Key | llama-server flag | Notes |
| --- | --- | --- |
| `model` | `-m` | Required. |
| `mmproj` | `--mmproj` | The vision projector, for a multimodal model. |
| `ngl` | `-ngl` | |
| `n_cpu_moe` | `--n-cpu-moe` | |
| `ctx` | `-c` | |
| `flash_attn` | `-fa on` | Only emitted when `true`. |
| `kv_type` | `--cache-type-k` and `--cache-type-v` | Both get the same value. |
| `parallel` | `-np` | |
| `port` | `--port` | Defaults to 8080. |
| `group` | none | Folds models together in the dashboard list. |
| `args` | passed through verbatim | Array of strings, appended last. |

Two top-level keys sit outside the model tables: `llama_server` (above) and
`ready_timeout`, the seconds `run` waits for `/health` before giving up.
It defaults to 300, which a 70B on a slow disk can exceed.

Anything else in a model table is an error, so a typo like `flash_atn` tells you
instead of silently doing nothing.

### Grouping

Give several models the same `group` and the dashboard folds them under
one row - useful for the six context-length variants of one model, or
everything from one lab. `enter` on a group opens and closes it, and a
closed group still shows a marker when something inside it is running.
The fold is remembered between sessions.

There is nothing to create or delete: a group is a name two models share.
It changes nothing about how they run, and `mdl run <name>` is unaffected.

### Vision models

A multimodal model is two files: the weights and a projector, shipped
alongside them as `mmproj-*.gguf`. Point `mmproj` at it and llama-server
accepts images; leave it out and you get a text-only server that says
nothing about the eyes it is missing. `mdl add` fills it in when it finds
exactly one beside the weights, and `mdl check` tells you if it later goes
away. To keep it off a full GPU, add `--no-mmproj-offload` to `args`.

## Commands

```
mdl run <name>   Start <name> in the background, tail its log until the
                 server answers /health, and exit. The server keeps running
                 after mdl exits. --port N overrides the config for one run.
mdl stop [name]  SIGTERM the server, SIGKILL after 10s, clean up. Takes a
                 name when more than one is up, or --all for every one.
mdl ps [--json]  name, pid, port and uptime per server, or "nothing
                 running". --json prints a JSON list ([] when idle) for
                 scripts and status bars.
mdl list         The models defined in the config.
mdl add <gguf>   Append an entry for a .gguf to the config, with sane
                 defaults. Takes an optional name and port.
mdl check        Validate every model in the config without launching
                 anything. Exits non-zero if it finds a problem.
mdl init         Write a starter config, if you do not have one.
mdl --version    The version, for bug reports.
mdl ui           The dashboard. Bare `mdl` opens it too.
mdl logs [-f] [name]
                 Print or follow a server's log. Takes a name when more than
                 one is up, or to read a stopped one's log.
mdl fit ...      What a GGUF will do on this machine, and the flags for it.
mdl eval <name>  Score a model on a private, auto-graded suite.
mdl catalog ...  The hub's models, fine-tunes and GGUF quants, offline.
mdl find         The best model this machine can run, and how to run it.
```

Without textual installed, `mdl ui` fails with one line and bare `mdl` prints
the usage string, exactly as it always did.

```console
$ mdl list
ornith      /srv/models/Ornith-1.5-35B-A3B-Q4_K_M.gguf
qwen-small  /srv/models/Qwen3-8B-Q5_K_M.gguf

$ mdl run ornith
starting ornith (pid 48812), log /home/leon/.local/state/mdl/ornith.log
load_tensors: offloaded 43/43 layers to GPU
llama_context: n_ctx = 65536
main: server is listening on http://127.0.0.1:8080
ready: ornith on http://127.0.0.1:8080 (pid 48812)

$ mdl ps
ornith      pid 48812  port 8080  up 1h04m
qwen-small  pid 49107  port 8081  up 12m

$ mdl stop
stopped ornith (pid 48812)
```

`add` and `check` are the two that save the most time:

```console
$ mdl add ~/models/Qwen3-8B-Q5_K_M.gguf
added [qwen3-8b-q5-k-m] to /home/leon/.config/mdl/models.toml
  Qwen3-8B-Q5_K_M.gguf (5.4G, 37 layers)
  run it with: mdl run qwen3-8b-q5-k-m

$ mdl check
ornith      ok
qwen-small  model file not found
mdl: 1 problem(s) found
```

`add` only appends, and `check` never launches anything, so both are safe
to run against a config you care about.

## Fitting a model: `mdl fit`

`mdl fit` answers "what will this do on my machine, and with which flags"
before you run anything. It reads the exact tensor table out of the GGUF,
replays where llama.cpp puts every tensor for a given `-ngl`,
`--n-cpu-moe`, KV type and ubatch, costs the KV cache layer by layer
(sliding-window, hybrid recurrent and shared-KV layers included), and
searches every combination for the one that suits the profile.

```
mdl fit ./Tiel-Coder-35B-A3B-UD-IQ3_XXS.gguf    a file
mdl fit tiel-coder-Fast                          a models.toml entry
mdl fit hf:someorg/Some-Model-GGUF               every quant in a repo,
                                                 headers only, nothing downloaded
mdl fit tiel-coder-Fast --explain                why it does not fit, and fixes
mdl fit tiel-coder-Fast --verify                 run it, measure, learn
mdl fit hw                                       measure this machine (~3 min)
mdl fit hw --idle                                book now as what idle looks like
mdl fit inspect <gguf>                           per-layer tensor inventory
```

```console
$ mdl fit tiel-coder-Fast --explain
tiel-coder-Fast  ✗  over by 0.5 G   (weights on the GPU is 8.6 G of the total)

  fix                                 VRAM     decode @0   s/turn
  1  n-cpu-moe 13 → 15                -0.5 G   -4 t/s      +4%
  2  ubatch 1024 → 256                -0.6 G   ±0          +34%
  3  ctx 256k → 166k                  -0.5 G   ±0          ctx cap
```

Profiles decide what "best" means: `agent` (the default) minimises the
seconds for a 16k-token prompt plus an 800-token reply at 48k deep, with
at least 128k of context; `chat` maximises decode at 8k; `max-ctx` takes
all the context it can at 15 t/s or better; `speed` maximises decode.
`--min-ctx`, `--min-tps`, `--kv-floor` and `--np` override the floors.
The KV cache never goes below q8_0 unless you say `--kv-floor q4_0`, or
the config being fitted already runs 4-bit KV; either way the output
says what 4-bit KV would buy in context and speed.

Memory is checked against llama.cpp itself rather than trusted:
`llama-fit-params`, which ships with llama.cpp, reports what a set of
flags would allocate on each device without allocating it, and `mdl fit`
puts its picks in front of it before showing them. A `✓` next to a VRAM
figure means the oracle agreed. What it gets wrong is stored against the
file in `~/.config/mdl/calib.jsonl`, so the next fit starts from it.
Speed is bytes over bandwidth per side, and the bandwidths are seeds
until `mdl fit hw` has measured them; the confidence line says which.

VRAM and RAM are planned for the machine at idle, not as the scan finds
it: a game or a browser full of tabs open while you run `mdl fit` is
taken back off. The idle figure is, best first:

- what you told it: `mdl fit hw --idle` books the machine as it is now,
  `mdl fit hw --idle-vram 0.5G --idle-ram 35%` says it outright, and
  `--idle-reset` forgets;
- what it saw in the first ten minutes after a boot, booked by any probe;
- what is in use now, less what the apps opened since boot hold. An app
  is anything that is not the system's, a startup entry (Run keys and
  the Startup folder on Windows, XDG autostart on Linux, LaunchAgents on
  macOS) or the terminal you ran mdl from; browsers always count as
  apps. Totals come from the adapter and the OS, per-app figures only
  say what to take back off;
- never less than a typical floor for the OS and desktop.

When a pick needs more VRAM than is free this minute, it names the apps
to close; `--now` plans for the machine as it is instead. The card keeps
a 256 MiB margin; a config inside the free VRAM but eating the margin is
"fits, tight". RAM has a hard limit, the total less 3 G for the system:
past what is free at idle the OS pages idle programs out, which is a
note, not a failure, but a model whose CPU-side weights do not fit under
the limit would page from disk on every token. A busy CPU is noted, and
`mdl fit hw` warns before calibrating on a machine that is not idle.

`--write NAME` appends the winner to models.toml; `--apply N` rewrites an
existing entry with pick or fix N, leaving its sampling flags, comments
and a `.bak` behind. Every flag the prediction rests on is written out,
including `-np` and `-fit off`, so what runs is what was predicted.

## Scoring a model: `mdl eval`

Leaderboards say how a model does in someone else's setup. `mdl eval`
says how it does here, at your quant, KV type and context:

```
mdl eval qwen-small                    every suite (~15-90 min, see --estimate)
mdl eval qwen-small --suite code,tools --limit 5
mdl eval qwen-small --estimate         how long it would take
mdl eval --results                     past runs, with 95% intervals
mdl eval --compare qwen-small ornith   which of two is actually better
```

It starts the model as models.toml runs it (or uses it if it is already
up) and runs five suites: code (40 functions, graded by hidden unit
tests that are actually executed), tools (30 tool-calling tasks, single
and multi-step, against mock worlds, one of which freezes partway
through so that checking the rules once is not enough), long-context (24 retrieval,
multi-hop, counting and exhaustive-recall questions at 32k, 64k and
128k, those beyond the configured context skipped), instruct (20
checkable format rules) and reason (20 exact-answer problems).

One long-context question per document asks for every match rather than
one: three of the six people share a floor, each by way of a room named
somewhere else again, and it is scored by overlap, so stopping after two
is worth more than nothing and less than finishing.

Another has no answer in the document at all.

Every grader is checked against replies that contain no knowledge -
nothing, a refusal, a guessed number, the instructions read back - and
none of them may be paid for. That check runs with the tests, because a
grader that can be satisfied without doing the work is worse than no
grader: it still reads as evidence. The project is real and nine other projects do list a code, so the
pull towards writing one down is strong; saying it is not recorded is
the only reply that scores. Nothing else in the suite measures making
things up. Add your own as `[[task]]` entries in
`~/.config/mdl/evals/*.toml`, checked by `contains`, `regex`, `exact` or
a Python snippet.

Three of every five items are the harder tier, and the report scores the
tiers separately. A suite everything passes ranks nothing, so the questions
are built to be failed: reasoning that takes several steps with a
plausible wrong turn at each, tool calls whose arguments have to be
worked out rather than copied out of the request, requests that are
missing something the tool needs and should be asked about instead of
guessed, and documents where the answer is three hops apart or is
superseded further down. Code is marked per hidden test, so a function
that handles the ordinary cases and trips on one edge does not score
the same as one that does not run.

The items are generated from a seed in `~/.config/mdl/eval-seed`, so
they exist on this machine and nowhere else, and no model can have
trained on them. Model-written code runs in a subprocess in a temp
directory with a timeout; `--sandbox` runs it in a throwaway podman or
docker container with no network instead. Results go to
`~/.config/mdl/evals.jsonl`, keyed by the file, quant, KV type and
context they were run at.

Every run also records what it cost - reply tokens spent, and tokens per
right answer - because a model that thinks four times as long for the
same score is not as good on a machine you own. The second run of a
model estimates itself from what the first one actually spent, instead
of guessing.

`mdl eval --compare A B` puts two runs side by side. Both answered the
same generated items, so it pairs them item by item and bootstraps the
difference: the verdict is "ahead" only when the interval clears zero,
and "too close to call" otherwise. Runs whose item sets differ - a
changed template, a different machine - are refused rather than
compared, and every result carries a fingerprint of the questions
themselves so that can be checked rather than assumed.

## Finding a model: `mdl catalog` and `mdl find`

`mdl find` answers "what is the best thing I can run here":

```
mdl find                               agent profile: ctx ≥ 128k, ≤ 90 s a turn
mdl find --profile chat                ≥ 20 t/s decode
mdl find --license apache --tag code
mdl find --new                         only what has appeared since last time
mdl find --no-fetch                    use cached GGUF headers only
```

It looks at everything in models.toml and at the catalog: the hub's
model tree, with each model's fine-tunes and merges, every GGUF quant of
each, and the eval results they report. For the best-scoring candidates
it runs `mdl fit` on up to three quants each (headers only, nothing
downloaded), keeps what clears the profile's floors, and ranks by
expected quality:

- Public results are put on one scale (50 + 15 z against the catalog).
  A score far above what the model's other scores predict, or one on a
  benchmark the model card says it trained on, is down-weighted and
  flagged with ⚑.
- A fine-tune with no results of its own borrows its parent's, with
  wider error bars per generation.
- Lower-bit quants and 4-bit KV cost points; how many is relearned once
  you have `mdl eval` results for one model at two quants.
- Once three models have both public results and your own, scores are
  shown on your local scale instead.

Models no one has rated show up under "worth testing" when they could
beat the #1, with the `mdl fit --write` and `mdl eval` commands that
would rate them.

The catalog is one SQLite file in `~/.config/mdl/cache/`. `mdl catalog
pull` fetches the nightly snapshot from
[diversemate/mdl-catalog](https://huggingface.co/datasets/diversemate/mdl-catalog)
(`$MDL_CATALOG_REPO` points it elsewhere). `mdl catalog build` crawls the
hub yourself instead (`--org LiquidAI`, `--base Qwen/Qwen3-8B`; a few
minutes for a few orgs). `mdl catalog tree <org/repo>` lists every quant of every
fine-tune of a model, and `mdl catalog search` finds one by name.

## The UI

`mdl ui` (or just `mdl`) opens a dashboard over the same config and the same
state file. Anything you do in it is visible to the CLI and vice versa.

Idle, it lists your models with a status dot, shows the selected model's
parameters, and previews the exact `llama-server` command it would run.
`e` edits those parameters and saves them back to `models.toml`, leaving
your comments and layout alone.
Running, it swaps in live telemetry: VRAM, KV-cache use, a tokens/sec
sparkline, busy slots, and a colour-coded log tail.

```
 key          does
 up/down, j k select a model
 enter, r     run the selected model
 s            stop the selected model
 R            restart
 e            edit ngl / ctx / kv_type / port, saved to models.toml
 c            copy the llama-server command
 p            prompt the running model without leaving the UI
 l            focus the log, / filters it
 g            reload the config
 ?            help
 q            quit the UI - the server keeps running
```

The telemetry panels need llama.cpp's metrics endpoint, so add `--metrics`
to a model's `args` to light them up:

```toml
args = ["--metrics"]
```

Without it the dashboard still works, and those panels say `metrics off`
rather than failing. While a model is loading they say `loading` instead,
since nothing is listening yet.

### Talking to the model

`p` opens a chat with whatever is running, without leaving the UI.

![the chat pane](docs/chat.svg)

It keeps the conversation, so follow-up questions have context; `ctrl+l`
starts a fresh one. Reasoning is shown dimmed and timed
separately, whether the server hands it back in its own field or inline
as `<think>` tags. `esc` interrupts a running reply - it closes the
socket rather than waiting for the next token - and closes the pane once
nothing is streaming.

The rate is the server's own `tok/s` when it reports timings, and ours
otherwise. `ttft` is time to first token, which is the number that tells
you whether a long context is hurting.

### Animation

The wordmark drifts its gradient by default. Set `ui_fx = "off"` at the
top level of the config to paint it flat, or pass `mdl ui --no-fx` for a
one-off.

## Files

```
~/.config/mdl/models.toml        your config
~/.local/state/mdl/run/<name>.json  pid, port and start time of each server
~/.local/state/mdl/<name>.log    server stdout+stderr, rotated on each run
~/.local/state/mdl/<name>.log.1  the previous run, and .2 before that
~/.local/state/mdl/ui-marks.json which models the UI has seen start or fail
```

`$XDG_CONFIG_HOME` and `$XDG_STATE_HOME` are honoured if set. On Windows the
same layout lives under `%USERPROFILE%`.

## Behaviour notes

- **As many servers as you have ports and VRAM for.** Each needs its own
  port; `run` on a port already serving something says which model has it.
  Commands that took no argument still take none while one server is up,
  and ask which only when there is a real choice.
- **Readiness is an HTTP probe, not log scraping.** `run` polls `/health` on
  the configured port. llama.cpp has reworded its startup line between builds;
  this contract has not.
- **Obvious mistakes fail before launch.** A missing model file, a missing
  binary or a busy port is one line in milliseconds, not a failed model load.
- **Stale state self-heals.** If the pid in `run/<name>.json` is gone (crash,
  reboot, `kill -9`) the file is removed and `ps` no longer lists that server.
- **If the server exits during startup,** `run` reports its exit status, removes
  the state file, and exits 1. The log has the reason.
- **If it does not report ready in time,** `run` exits 1 but leaves the server
  running, since it may still be loading. Check the log, or `mdl stop`. Raise
  `ready_timeout` if 300s is genuinely not enough.
- **Pid reuse is guarded against.** The state file records the OS process
  creation time, so a recycled pid is not mistaken for your server. macOS has
  no cheap way to read that, so it falls back to the pid alone.
- **`stop` signals the process tree, not just the pid.** If your `llama_server`
  is a wrapper script, killing the wrapper alone would orphan the real server
  and leave the port held.
- **The last few logs are kept.** `<name>.log` shuffles along to `.1` and `.2`
  on each run, so the crash you were not watching is still there.
- Errors are one line on stderr and a non-zero exit. No tracebacks.

## Tests

```sh
python tests/run.py           # fast: no real model needed
python tests/run.py --live    # also drives a real model through the UI
```

The fast suites run against a temp config and a fake `llama-server`, so they
never touch `~/.config/mdl`. The POSIX process semantics (detaching, orphan
self-heal, SIGTERM escalating to SIGKILL) need Linux:

```sh
docker run --rm -v "$PWD:/repo:ro" python:3.12-slim \
    sh -c 'cp -r /repo /w && cd /w && python3 tests/integration_posix.py'
```

## How this compares

[llama-swap](https://github.com/mostlygeek/llama-swap) is automatic and proxied: it swaps models on demand behind one endpoint. Pick it when you want requests to choose and load models for you.
[Ollama](https://ollama.com) uses its own model format and registry, so it is a different thing entirely. Pick it when you want that managed model ecosystem instead of running GGUF files through llama-server yourself.

## Non-goals

These are deliberate, and issues asking for them will be closed with a link
here. `mdl` starts servers, stops them, and tells you what is running.

- **No daemon.** Nothing runs in the background except the server itself.
- **No model downloading.** Use `huggingface-cli`, or your browser.
- **No hot-swap or auto-unload.** Nothing is unloaded to make room for
  something else; what you started stays started until you stop it.
- **No web UI.** llama-server already ships one.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: open an issue first,
keep `mdl.py` free of dependencies, and run the tests.

```sh
python tests/run.py
ruff check .
```

CI runs the suites on Linux, macOS and Windows across Python 3.11-3.13, the
pinned Textual floor and the current release, Ruff, the POSIX process suite,
and a packaging check on both the wheel and the sdist.

Security issues go through [SECURITY.md](SECURITY.md), privately, rather
than the public tracker.

## License

MIT. See [LICENSE](LICENSE).
