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
only, and on its own it is everything you need to run servers. Linux, macOS
and Windows.

The `mdl_fit` package behind `mdl fit`, `mdl eval`, `mdl catalog`,
`mdl find` and `mdl pull` is standard library too, and `mdl.py` imports it
only when you call one of those, so the commands you use every day load
nothing extra.

`mdl ui` opens the same thing in a window: your models, what is running
and how fast, what each has served, run and stop at a click, and a coding
agent opened on any of them. It is `mdl_web/`, standard library
too, serving a page that has no build step and loads nothing from the
network. `mdl tui` (or just `mdl`) is the terminal dashboard, `mdl_ui.py`.
It is the one part with a dependency - [Textual](https://textual.textualize.io/),
which installs with mdl - and the CLI imports it only for `tui`, so the
other commands never load it.

In practice it does two jobs. It manages llama.cpp servers through named
model presets: start one from a config file, switch between GGUF models, or
run several at once. And it answers the questions that come before that -
what this GGUF will do on your hardware and with which flags (`mdl fit`),
whether the quant you picked is measurably worse than the one above it
(`mdl eval`), and what the best thing you can run here is at all
(`mdl catalog` and `mdl find`).

## Install

mdl runs llama.cpp's `llama-server`, so that comes first:

```sh
winget install ggml.llamacpp    # Windows
brew install llama.cpp          # macOS, and Linux with Homebrew
```

or a build for your GPU from
[llama.cpp's releases](https://github.com/ggml-org/llama.cpp/releases).
`mdl doctor` says which GPU the `llama-server` on your PATH runs on, and
warns if it is a CPU-only build on a machine with a GPU.

Then mdl itself, with any of:

```sh
uv tool install llama-mdl       # the quickest
pipx install llama-mdl
pip install llama-mdl
```

To update:

```sh
mdl update                      # the newest release, the way you installed it
```

It works out whether this copy came from pipx, `uv tool` or pip and has
that tool upgrade it, so nothing changes but the version. See
[Updating](#updating) for what it checks and what it refuses.
`mdl --version` says which one you are running.

The package is `llama-mdl`; the command it installs is `mdl`. (Plain `mdl`
on PyPI is an unrelated project.) Its one dependency is
[Textual](https://textual.textualize.io/), for the terminal dashboard.
`pip install "llama-mdl[ui]"`, the way to get the dashboard before 0.13,
still works and installs the same thing.

Or run it straight from a clone - it is Python, nothing to build, and
only `mdl tui` needs Textual (`pip install textual`):

```sh
git clone https://github.com/diverseau/llama-mdl ~/src/mdl
python ~/src/mdl/mdl.py --help
```

Then get a model and start it:

```sh
mdl find                        # what fits this machine, best first
mdl find --run 1                # fetch #1, fit a preset to it, start it
mdl pull unsloth/Qwen3-8B-GGUF --run   # or a repo you already know
```

There is no config to write first: the first model you pull or add
creates `~/.config/mdl/models.toml`, with `llama-server` set to the one on
your PATH. `mdl add file.gguf` adds a GGUF you already have. `mdl init`
writes the same starter config on its own, with a commented example to
copy from.

For bulk edits, run `mdl config` to open models.toml in `$VISUAL` or
`$EDITOR` (Notepad on Windows or vi elsewhere if neither is set).
Editor arguments work too, for example `EDITOR="code --wait"`.
`mdl config --path` prints the active config's location.
Writes by mdl keep five previous versions: `.bak` (newest), then `.bak.1`
through `.bak.4`. `mdl config --history` lists their dates, sizes and changed
tables. `mdl config --undo` swaps the current file with `.bak`, leaving the
older backups in place; undo twice to get back where you started.
Edits made directly in your editor do not create these backups.

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
| `flash_attn` | `-fa on` / `-fa off` | Left out, the build's default applies. |
| `kv_type` | `--cache-type-k` and `--cache-type-v` | Both get the same value. |
| `parallel` | `-np` | |
| `port` | `--port` | Defaults to 8080. |
| `group` | none | Folds models together in the dashboard list. |
| `llama_server` | none | This model's own llama-server, for one that needs a different build (a fork with a quant type upstream cannot load yet). Beats `$MDL_LLAMA_SERVER` and the top-level setting. |
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
mdl run <name>   Start <name> in the background, follow its log until the
                 server answers /health, and exit. The server keeps running
                 after mdl exits. --port N overrides the config for one run.
                 On a terminal the load is one line; -v prints the log.
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
mdl doctor [--json] [name]
                 Diagnose the environment and all presets, or just one.
mdl init         Write a starter config, if you do not have one. (The
                 first pull or add writes it too.)
mdl config       Open models.toml in your editor; --path prints its location.
mdl --version    The version, for bug reports.
mdl update [--check]
                 Upgrade to the newest release, the way mdl was installed.
                 --check only says whether there is one.
mdl ui           The web UI, in an app window. --no-open prints its URL
                 instead; --port N picks the port.
mdl tui          The terminal dashboard. Bare `mdl` opens it too.
mdl snapshot     What the web UI shows, as JSON.
mdl logs [-f] [name]
                 Print or follow a server's log. Takes a name when more than
                 one is up, or to read a stopped one's log.
mdl fit ...      What a GGUF will do on this machine, and the flags for it.
mdl eval <name>  Score a model on a private, auto-graded suite.
mdl catalog ...  The hub's models, fine-tunes and GGUF quants, offline.
mdl find         The best model this machine can run, and how to run it.
mdl pull <org/repo[:quant]> [--run]
                 Download a GGUF, check it, and add a preset fitted to it.
                 Without a quant, the one that suits this machine.
mdl lab ...      The same prompt through variants of a config, measured.
mdl manifest <name>
                 What <name> is running as: its command line, llama.cpp
                 build, model files (size and hash per shard) and machine,
                 as JSON. --redact cuts paths and secrets for a bug report.
```

In a clone without Textual, `mdl tui` fails with one line saying how to
add it, and bare `mdl` prints the usage string.

```console
$ mdl list
ornith      /srv/models/Ornith-1.5-35B-A3B-Q4_K_M.gguf
qwen-small  /srv/models/Qwen3-8B-Q5_K_M.gguf

$ mdl run ornith
starting ornith (pid 48812), log ~/.local/state/mdl/ornith.log
ready: ornith on http://127.0.0.1:8080 (pid 48812)
  chat in a browser: http://127.0.0.1:8080  ·  OpenAI API: http://127.0.0.1:8080/v1
  dashboard: mdl ui  ·  stop it: mdl stop ornith

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

`mdl doctor` brings config, binary build, supported flags, GGUF headers and
runtime checks together. It warns about shared ports and leftover launch locks,
without launching a server or changing config or state. Pass a name to check one
preset, or `--json` for scripts. Findings are `ok`, `warn` or `fail`; only failures
make it exit non-zero.

`mdl manifest` describes the server that is running, not the config: the
command spawn launched is kept in its state file, so editing models.toml
after a start does not change what the manifest - or an `mdl eval` result,
which carries one - says ran. A model that is not running is described
from its preset, and the manifest says so. Each model file is named by a
sha256 of all of it: read once - a 20 GB model takes a while the first
time - and kept against the file's size and modification time after
that. A file replaced under a running server is reported as replaced,
since the server still has the old one loaded, and `mdl eval` refuses
it.

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
mdl fit tiel-coder-Fast --profiles               what each config measured here
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
Past 2 MiB that file drops what nothing reads any more - an observation
a later one of the same configuration replaced, and load logs beyond
the last 20 per model - and keeps every failure.
Speed is bytes over bandwidth per side, and the bandwidths are seeds
until `mdl fit hw` has measured them; the confidence line says which.

A prediction is replaced by a measurement where there is one. `--verify`
runs llama-bench at the config, and every `mdl eval` keeps the speed the
server itself reported for each reply at the depth it was made; both are
filed as a measured profile of that exact configuration - the same model
bytes (every shard and the projector), the same llama.cpp build and
commit from the same binary, the same command. A `--verify` of a preset
with flags llama-bench does not take, such as `-ot`, is kept but not
booked to the preset's command, and says so.
`mdl fit <name>` shows the current config's profile under its prediction,
with the difference, and `--profiles` lists every configuration of the
model measured here. A profile is speed only: it says nothing about
quality.

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
"fits, tight". When `--verify` runs out of memory on a config predicted
to fit, that architecture's margin grows by 256 MiB and later fits of it
say so. RAM has a hard limit, the total less 3 G for the system:
past what is free at idle the OS pages idle programs out, which is a
note, not a failure, but a model whose CPU-side weights do not fit under
the limit would page from disk on every token. A busy CPU is noted, and
`mdl fit hw` warns before calibrating on a machine that is not idle.

`--write NAME` appends the winner to models.toml; `--apply N` rewrites an
existing entry with pick or fix N, leaving its sampling flags, comments
and a `.bak` behind. Every flag the prediction rests on is written out,
including `-np` and `-fit off`, so what runs is what was predicted.
Add `--dry-run` to `--apply N` or `--write NAME` to see the config diff and
server command before and after. Nothing is written, including backups.

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
and multi-step, against mock worlds), long-context (24 questions over
documents at 32k, 64k and 128k, those beyond the configured context
skipped), instruct (20 checkable format rules) and reason (20
exact-answer problems). Add your own as `[[task]]` entries in
`~/.config/mdl/evals/*.toml`, checked by `contains`, `regex`, `exact`
or a Python snippet. A task that cannot run - no prompt, a regex that
does not compile, a domain the report has no row for, an id used twice -
is refused when the suite is built, naming the file and the task, not
found an hour into the run.

Three of every five items are the harder tier, and the report scores
the tiers separately. A suite everything passes ranks nothing, so the
hard items are built to be failed - and, more importantly, built so
that recognising them does not help. Half of the hard code and reason
items are generated rather than named: a price with five clauses that
interact, a validator with a stated precedence over its rules, a stack
machine whose opcodes come from the seed, a line splitter whose
separator, quote, escape and comment characters come from the seed, a
seating puzzle pruned until exactly one arrangement fits, a counter
machine that has to be simulated, a set of rules to chain to a fixed
point. There is no prior solution to recall, because the constants, the
rules and the names are all made here.

The other half are the classics - longest increasing subsequence,
topological sort, an LRU cache, work rates, mixtures, lattice paths -
and they are kept because they still catch arithmetic slips, but on
their own they measured memory rather than reasoning: a 35B coder in a
3-bit quant answered 36 of 40 of them.

The tool worlds take four to a dozen dependent calls. One forbids the
action the request asks for, and doing it anyway is the failure. One
fails a call the first time. One offers eight tools, four of them
beside the point, and a transaction list that only arrives a page at a
time, so an agent that reads page one and stops looks exactly like one
that finished. One freezes deploys partway through the job, so checking
the rules once at the start is not enough. One has four customers with
the same name, and only reading each one says which to cancel. One has
another writer change a counter between the read and the write, so
retrying the value already worked out erases their change - a 35B coder
in a 3-bit quant did exactly that. Eighteen of the thirty tools items
are worlds like these, and two in three of those are the hard ones.

The long-context questions are not all needles. One per document asks
for every match rather than one - three of six people share a floor,
each by way of a room named somewhere else again - and is scored by
overlap, so stopping after two is worth more than nothing and less than
finishing. One counts something that has to be read for in full. And
one has no answer in the document at all: the project is real and nine
others do list a code, so the pull towards writing one down is strong,
and saying it is not recorded is the only reply that scores. Nothing
else in the suite measures making things up.

Code and format items are marked in parts, so a function that handles
the ordinary cases and trips on one edge does not score the same as one
that does not run.

Every grader is checked against replies that contain no knowledge -
nothing, a refusal, a guessed number, the instructions read back - and
none of them may be paid for. That check runs with the tests, because a
grader that can be satisfied without doing the work is worse than no
grader: it still reads as evidence.

The items are generated from a seed in `~/.config/mdl/eval-seed`, so
these exact questions exist on this machine and nowhere else, and a model
cannot have memorised them. That is narrower than "never seen anything
like them": the templates, the problem families and the algorithms behind
the reference solutions are public, and familiar to any model trained on
code. Model-written code runs in a throwaway podman or docker container
with no network, 512 MB and 128 processes whenever one is available and
answering. Without one - or with `--no-sandbox` - it runs in a subprocess
in a temp directory with a timeout, which is not a sandbox: it runs as
you, and can read your files and reach the network, and the run says so.
`--sandbox` insists on the container. The expected answers never enter
either: the code writes its results, and mdl compares them outside.

Every item is written to a checkpoint as it finishes. An interrupted run
continues with `mdl eval <name> --resume`, but only onto the same items
and the same server - the same command bar its port, the same build, the
same model bytes (see `mdl manifest`); otherwise it starts over and says
why. An item the server failed on, rather than the model, is retried
twice and then left unscored, and `--resume` tries it again. Results go to
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

Nothing about the model goes into making the questions - only the seed
on this machine - so two entries in `models.toml` always get the same
items. That is what makes the question people actually have answerable:
run the same model at Q4_K_M and at IQ3_XXS, compare the two, and the
answer is either a number with an interval that clears zero or "too
close to call", rather than a feeling about whether the smaller one
seems worse.

## Finding a model: `mdl catalog` and `mdl find`

`mdl find` answers "what is the best thing I can run here":

```
mdl find                               agent profile: ctx ≥ 128k, ≤ 90 s a turn
mdl find --profile chat                ≥ 20 t/s decode
mdl find --license apache --tag code
mdl find --new                         only what has appeared since last time
mdl find --no-fetch                    use cached GGUF headers only
mdl find --run 1                       fetch the top row and start it
mdl find --pull 3                      fetch row 3, add a preset, no start
```

The table ends with the command that gets #1 running. The first `find`
fetches the catalog (`mdl catalog pull`) if there is none, and asks for a
newer one when the last check is a week old; `--no-fetch` does neither.

The ranking is a heuristic for a shortlist, not a measured quality
scale: it combines public scores that were run by different people with
different harnesses, assumed quant and KV penalties, and lineage priors.
The bands around each estimate are the same heuristic's, not validated
confidence intervals. Your own `mdl eval` results are the measurement.

`mdl find --why MODEL` explains a catalog id or models.toml name (a unique,
case-insensitive substring works too): its rank or why it is absent, every
considered quant and rejection, and the heuristic's evidence, parent priors,
flags and quant/KV penalties. It uses the same profile and filters as `find`;
add `--json` for the explanation as data.

It looks at everything in models.toml and at the catalog: a mix of popular
and newly created GGUF repositories, grouped by their source models, with
the eval results those models report. For the best-scoring candidates
it runs `mdl fit` on up to three quants each (headers only, nothing
downloaded), keeps what clears the profile's floors, and ranks by
expected quality:

- Public results are put on one scale (50 + 15 z against the catalog).
  A score far above what the model's other scores predict, or one on a
  benchmark the model card says it trained on, is down-weighted and
  flagged with ⚑ - inconsistent evidence, not proof of gaming.
- A fine-tune with no results of its own borrows its parent's, with
  wider error bars per generation.
- Lower-bit quants and 4-bit KV cost points; how many is relearned once
  you have `mdl eval` results for one model at two quants.
- Once three models have both public results and your own, scores are
  shown on your local scale instead.

Models no one has rated show up under "worth testing" when they could
beat the #1, with the steps that would rate them: `mdl pull` the quant,
then `mdl eval NAME`.

## Downloading a model: `mdl pull`

```
mdl pull unsloth/Qwen3-8B-GGUF                 the quant that suits this machine
mdl pull unsloth/Qwen3-8B-GGUF:Q4_K_M          this quant: download, check, add a preset
mdl pull unsloth/Qwen3-8B-GGUF:Q4_K_M --run    and start it
mdl pull org/repo:Q8_0 --name mine             under a name of your own
```

Without a quant, it picks the one `mdl find` would: it reads one header
(nothing is downloaded yet), sizes the other quants from it, leaves out
F16 and above, and takes the best quant that clears the agent profile's
floors here, then the fastest within half a point of it. When none
clears them - a model trained for less than 128k of context never can -
it takes the best that runs, and says so in the line naming its pick.

`mdl pull` pins the repo at its current commit and takes every shard of
the quant, plus its vision projector when the repo ships one (the full
precision one, when there is a choice). A file already in the Hugging
Face cache is used where it is, once its size and sha256 match the
Hub's. Anything else is downloaded to `$MDL_MODELS` (else `~/models`),
in a folder per repo with a `.mdl-pull.json` naming the repo and commit.
Every file is checked against the Hub's sha256 before it is kept. A
download that stops keeps its `.part`, and the next `mdl pull` resumes
it. Then it adds a preset fitted to this machine (`mdl fit --write`),
with `--metrics`, on a port no other preset uses. A gated repo needs
`HF_TOKEN`.

The catalog is one SQLite file in `~/.cache/mdl/` (`$XDG_CACHE_HOME` is
honoured). `mdl catalog
pull` fetches a published snapshot from
[diversemate/mdl-catalog](https://huggingface.co/datasets/diversemate/mdl-catalog)
(`$MDL_CATALOG_REPO` points it elsewhere).

A scheduled GitHub Actions job rebuilds and publishes the snapshot; it
runs only when the repository enables it (`CATALOG_ENABLED=true`, with a
write-capable `HF_TOKEN`), and a skipped run does not refresh it. If the
dataset has no snapshot, `mdl catalog pull` says so; `mdl catalog build`
makes one locally.

The default build takes the top 3,000 GGUF repositories by downloads and
the newest 500 by creation date, deduplicates the overlap, and reads their
file lists. It fetches each distinct quantization source's metadata once
to retain benchmark evidence, without walking descendants. This is at most
3,500 repositories, not 3,500 distinct models; it deliberately omits much
of the long tail and does not promise complete model family trees.

```sh
mdl catalog build --popular 800 --recent 200 --budget-minutes 40
```

At the time budget, the builder saves a usable partial, including its
pending work and pagination cursors in the same SQLite file. Run it again
to resume; `--from snapshot.sqlite` resumes a downloaded copy. Once a cycle
finishes, the next build refreshes both seed lists. During a partial
refresh the previous rows remain available; obsolete rows are removed only
after completion. Unchanged repositories reuse their file inventories.
`pull`, `stats` and `find` label partial snapshots. A killed local process
can resume its committed pages from the adjacent `.building` file, but a
cancelled Actions run cannot upload work that never reached Publish.

**Headers.** Sizing a model needs its GGUF header, and a header is 4-8
MiB, mostly vocabulary, for the few kilobytes `find` keeps of it. So
`mdl catalog headers` stores, in the snapshot, the header of the quant
`find` reads first for each model (the biggest under 8.6 bpw, in the
repository with the most downloads), compressed, keyed by the file's
content hash so a new upload is never mistaken for the old one. `find`
takes a header from there before asking the Hub; one of a quant too big
for this machine still sizes the smaller quants of that model. Measured
here, with them, a first `find` took 14 s instead of 40. They average
8 KB each, so a header for all 1,712 models with GGUFs in today's
snapshot adds about 14 MB to its 25. It is incremental, most downloaded
first, and drops headers no quant names any more.

```sh
mdl catalog headers --in catalog.sqlite --budget-minutes 10
```

The workflow gives the crawl 40 minutes, the headers 10, and the job 60,
leaving time to publish a partial. A manual dispatch can run while the nightly switch
stays off, with separate inputs for both seed sizes and the time budget.
It reports the publishing account and API quota without printing the token.

`mdl catalog tree <org/repo>` lists the relationships and quants present
in the catalog, and `mdl catalog search` finds models by name. Explicit
`build --org LiquidAI` or `--base Qwen/Qwen3-8B` retains the original,
unbudgeted lineage traversal for targeted exploration; it does not use the
mixed-seed checkpoint engine.

## Measuring configs: `mdl lab`

`mdl fit` predicts and `mdl eval` scores; `mdl lab` measures. It runs the
same prompt through each variant of a model's config, one at a time, and
keeps what each did: time to first token, decode speed and how steady it
was, prefill, and the VRAM, RAM and CPU it cost - sampled against the
token count, so "usage 500 tokens in" is a column, not a guess.

```
mdl lab run qwen27b --set ngl=56,51,45 --set ctx=32k,64k --dry-run
mdl lab run qwen27b --set ngl=56,51,45 --set ctx=32k,64k
mdl lab report                   the last run: a row per variant and depth
mdl lab compare qwen27b/ngl56/ctx32768 qwen27b/ngl51/ctx65536
mdl lab apply qwen27b/ngl51/ctx65536    its models.toml table, printed
mdl lab baseline set             pin the last run
mdl lab baseline diff --fail     a later run against it; exit 1 if it regressed
```

Each variant is measured twice over by default: with an empty context,
and with its context full - the prompt filled to just short of the top,
so the reply ends there, as it does late in a long task. That is where
configs part ways; a benchmark at depth 0 picks the one that degrades in
use. `--depth` takes tokens (`8k`) or shares of each variant's own
context (`0,50%,100%`), so a 32k variant and a 64k one are each filled
to their own top; the report shows what a share came to in tokens, and
`compare` sets two variants side by side depth by depth.

A `--set` with a list sweeps it, and every combination is a variant;
`--server a,b` runs each on more than one llama.cpp build. A suite file in
`~/.config/mdl/lab/<name>.toml` holds the same as `[suite]` options and
`[[variant]]` tables (`base`, `label`, `set`, `server`), for
`mdl lab run --suite <name>`. `--dry-run` prints the matrix and a time
estimate from `mdl fit`'s predictions before an hour goes into it.

What makes the numbers comparable:

- Each variant runs from a config written for it alone, in a temp dir.
  `models.toml`, your state and your logs are never touched, and a server
  of yours on the port is never stopped - the run refuses instead.
- Sampling is the workload's (`--seed`, `--temp`), sent with every
  request, so a preset that pins `--temp` in its args is measured on the
  same terms. Replies run to `--max-tokens` and the prompt cache is off,
  so every repetition prefills.
- One warmup repetition is not counted (`--warmup`); the rest (`--reps`,
  3) are a median with its spread, and `compare` calls two variants
  indistinguishable when their spreads overlap rather than ranking noise.
- Decode's floor and peak are over a one-second window, the first 2% of
  tokens left out: a single slow gap is a hiccup, not a speed.
- A variant that would leave under 2.5 G of RAM free is skipped, not
  paged to disk; VRAM that does not come back to its baseline after a
  stop ends the run, since the rest would not compare. A run across
  builds says it is one.

VRAM, GPU load, clocks and temperature come from `nvidia-smi` - a whole
card's figure, not one process's, which Windows does not keep. Without
`nvidia-smi` the VRAM column is what the server's load log claims,
marked `(log)`. "t/s per G" is decode speed over the VRAM the model
itself took - what earns a place on a small card.

RAM is the machine's rise over its level before the load, which is what
a model with its experts in RAM (`n_cpu_moe`) really costs; RSS is the
server's own. "RAM free idle" and "VRAM free idle" are what a config
leaves on the machine as you sit down to it: its total, less what the
OS and the programs that start with it hold - measured the way `mdl fit`
plans for the idle machine, not counting the browser and the rest you
opened since - less what the model took. Below zero, it does not fit.

The flags column says what to distrust: `warming_up` or `slowing` when
the last third of a reply ran more than 10% off its first,
`rep_drift` when each repetition was faster than the last (the warmup
was not enough), `clock_drop` when the card throttled, `high_variance`,
and `timings_disagree` when the server's rate and the stream's differ.

`b` in the dashboard measures the selected config the same way, with one
repetition, empty and full; the log pane follows the run request by
request, with what is left. `b` again stops it - at once, mid-reply or
mid-load - as does quitting; Ctrl-C does the same to `mdl lab run`. Its
server is stopped, its temp files removed, and what it measured is kept.
The design, and what is left of it, is in
[docs/mdl-lab.md](docs/mdl-lab.md).

## The web UI

`mdl ui` opens a window over the same config and the same state file as
the CLI. Anything you do in it is visible to the CLI and vice versa.

The page is [0xSero's Local AI panel for
Omarchy](https://github.com/0xSero/omarchy-local-ai) (MIT), drawn the
same, row for row, over mdl's models instead of his recipes. At the top,
your lifetime: tokens and requests, and a grid of the last 20 weeks, a
column a week. Then each running model as a card: its tokens as a line,
its decode speed, what it has served, and Open. Then every card that is
free, a click from running the first model in your config, and any start
that failed, to run again or dismiss. Each card's Config lists every
model in your config to pick from, then up to six of `mdl find`'s picks
from the Hub. Run on one of those pulls it first (`mdl pull --run`): its
card shows how much of it is down, then it loads like any other. The
picks come from a `mdl find` the page runs in the background, again
every 12 hours or when your config changes.

A running model's page opens with how fast it runs as its context fills:
decode speed against context depth, from 0 to the whole context, so you
can see how much slower it gets as a session grows and how much of the
context your use has reached. A click shows prefill instead. The points
come from your own requests. When `mdl lab` has measured the same config,
its results are dots on the same chart. A different config (context, KV
type, layers, any flag but the port and key) starts a new chart. Until
there are a few requests, the chart is the panel's token line.

Below it: averages for decode, prefill and the wait for a first token,
what it served this session and this week, how long it has been up,
which agent and folder Open uses, its weights on Hugging Face when the
path says where they came from, and where it answers.

Open starts a coding agent on the model, in a terminal, in the folder you
picked: pi, claude, codex, opencode, omp, crush, grok, copilot or hermes,
whichever are installed. What the agent needs to find the server is
written under mdl's state directory, readable only by you; nothing in
the agent's own config changes.

The history comes from a recorder that `mdl run` starts alongside the
first server and that goes 30 seconds after the last one stops, whether
or not the page is open. Every 2 seconds it reads each server's
`/metrics` and the new lines of its log, and books what moved, per model
and per hour, in `usage/` under mdl's state directory. It needs
`--metrics` in the model's `args`; a server without it runs as usual and
the page says what to add. Tokens and seconds are llama-server's own
counters. Requests, and the speed of each at the depth it ran at, come
from the log: llama-server logs every request's prompt batch by batch,
its generation every few seconds, and how full the context was when it
finished, cached prompt included. The recorder keeps its place in the
usage file, so if it restarts it carries on without counting anything
twice. `MDL_RECORD=off` stops the recorder starting.

A model can be shared on your tailnet with `tailscale serve`, but only
when its server has an `--api-key`: anyone on the tailnet could use it
otherwise. The page offers the share when both are there.

It opens as an app window of its own when there is a Chromium browser
(Edge, Chrome, Brave, Chromium), with a profile of its own under mdl's
state directory, and in your default browser otherwise. It stops a
little after its last window closes, or at Ctrl-C.

Only this machine can reach it, and only the window mdl opened. It
listens on 127.0.0.1, makes a new token each start, and refuses any
request without it, any that names another host (so a page elsewhere
cannot reach it by pointing a name at 127.0.0.1), and any change from
another origin. The page updates as the servers do, pushed over one
connection rather than polled.

`mdl snapshot` prints what the page is drawn from, as JSON.

## The terminal dashboard

`mdl tui` (or just `mdl`) opens a dashboard over the same config and the
same state file. `mdl ui --tui` still works for one more release.

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
 b            measure the selected config with mdl lab; b again stops it
 l            focus the log, / filters it
 g            reload the config
 u            update mdl, when a newer release is out
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
top level of the config to paint it flat, or pass `mdl tui --no-fx` for a
one-off.

## Updating

`mdl update` asks PyPI for the newest release this Python can run, then
has the tool that installed mdl upgrade it:

| installed with | runs |
|---|---|
| pipx | `pipx upgrade llama-mdl` |
| `uv tool` | `uv tool upgrade llama-mdl` |
| pip, into any environment | `python -m pip install -U "llama-mdl==<newest>"`, with `--user` if mdl is |

While the installer runs, one line says what it is doing; its own output
is kept and shown only if it fails (`-v` shows it as it goes). It then
starts a fresh interpreter to check that the new version is the one that
imports, says how long it took, and lists what is new: the first line of
each changelog entry between the two versions, read from the changelog at
the new version's tag. It says so if the `mdl` first on your `PATH` is a
different install it did not touch. It refuses, and changes nothing:

- **In a source checkout** (`python mdl.py`, or `pip install -e`). Update
  that with `git pull`.
- **While `mdl eval` is running, or a server is starting.** Replacing the
  files under a running mdl mixes two versions in one process. `--force`
  goes ahead anyway. (`mdl lab` holds no lock, so it is not seen: do not
  update in the middle of one.)

If the installer fails, you are still on the version you had, and its
last lines are printed under the error. On Windows the running `mdl.exe` is moved aside
first, since `uv` cannot replace an exe that is running; a leftover
`mdl.exe.old-<pid>` is removed by the next update.

**The terminal dashboard asks once a day.** When it opens, `mdl tui`
checks whether a newer release is out - at most once a day, cached in
`~/.cache/mdl/update.json` - and if there is one, offers it: update and
restart, later, or skip that version. Servers keep running through the
restart, and `u` brings the offer back; after the restart it shows the
first few things that are new. **The web page offers it too**: a row on
the page says the new version is out, and one click installs it, shows
the installer's progress, and restarts `mdl ui` on the same address - the
open window reconnects and reloads itself. `mdl doctor` reports the same
check. The request is a plain GET of PyPI's JSON for `llama-mdl`; nothing
about you or your models is sent. To turn it off, set
`MDL_NO_UPDATE_CHECK=1`, or put this at the top of the config:

```toml
update_check = false
```

`mdl update` itself always asks, whatever that says.

## Files

```
~/.config/mdl/models.toml        your config
~/.local/state/mdl/run/<name>.json  pid, port and start time of each server
~/.local/state/mdl/<name>.log    server stdout+stderr, rotated on each run
~/.local/state/mdl/<name>.log.1  the previous run, and .2 before that
~/.local/state/mdl/ui-marks.json which models the UI has seen start or fail
~/.local/state/mdl/lab/          mdl lab's records, and the samples behind them
~/.cache/mdl/update.json         when mdl last asked PyPI, and a skipped version
```

`$XDG_CONFIG_HOME`, `$XDG_STATE_HOME` and `$XDG_CACHE_HOME` are honoured
if set. On Windows the
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
- **The load is one line on a terminal.** While it loads, `run` redraws
  `loading ornith · 43/43 layers on the GPU · 12s`, and prints only the log
  lines that report a problem. `-v` prints the whole log, and so does any
  `run` whose output is not a terminal - a script sees the log as before,
  and the `ready:` line stays last.
- **If the server exits during startup,** `run` reports its exit status, removes
  the state file, and exits 1. On a terminal it shows the last 30 lines of
  the log, which is where the reason is.
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

Neither answers the question that costs the most time: whether a model will
fit this machine before you download it, what it will actually run at, and
whether the quant you chose is measurably worse than the one above it. That
is what `mdl fit`, `mdl find` and `mdl eval` are for.

## Non-goals

These are deliberate, and issues asking for them will be closed with a link
here. `mdl` starts servers, stops them, says what is running, works out
what to run and how, and fetches the model you chose. What it does not do
is run things behind your back or decide for you what stays loaded.

- **No daemon.** Nothing runs in the background but the servers and the
  usage recorder `mdl run` starts beside them, which goes 30 seconds after
  the last one stops (`MDL_RECORD=off` keeps it from starting at all).
  `mdl catalog pull` fetches the published snapshot and exits; the nightly
  crawl that builds that snapshot runs in CI, not on your machine.
- **No hot-swap.** Nothing is unloaded to make room for something else; what
  you started stays started until you stop it. `mdl eval` is the single
  exception and a narrow one: it starts a server for the run and stops it
  afterwards, but only one it started itself - a server that was already up
  is used as it stands and left running.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: open an issue first,
keep `mdl.py` to the standard library, and run the tests.

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
