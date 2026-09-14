# Changelog

Notable changes. Dates are ISO; versions follow [semver](https://semver.org/).

## [0.6.2] - 2026-09-12

0.6.0 and 0.6.1 were never released; everything since 0.5.2 is here.

### Added

- `mdl fit`: what a GGUF will do on this machine before you run it, and
  the flags to run it with. It reads the tensor table out of the file
  (or, for `hf:org/repo`, out of a Range request on each quant's header,
  nothing downloaded), replays llama.cpp's placement for every
  combination of KV type, ubatch and n-cpu-moe or -ngl, finds the most
  context each one holds, predicts decode, prefill and seconds per agent
  turn, and hands back the winner and two runner-ups with the
  llama-server command and a models.toml block. Profiles: agent (the
  default: least time per 16k-in, 800-out turn at 48k deep, 128k
  context floor), chat, max-ctx and speed.
- Memory is checked against llama.cpp itself. `llama-fit-params -fitp on`
  reports model, context and compute bytes per device without
  allocating, in well under a second, so the picks are put in front of
  it before they are shown and anything it disagrees with is learned
  and searched again. Weights, KV and recurrent state were exact on all
  ten configs in the test set; the compute buffer is a seeded model
  plus what the oracle teaches it, stored per file in `calib.jsonl`.
- `mdl fit <name> --explain`: why a config does not fit - the component
  that blew, and by how much - and the cheapest fixes, ranked by what
  they cost you, context cuts last. `--apply N` writes one back into
  models.toml, keeping comments, sampling flags and a `.bak`.
- `mdl fit hw` measures this box with llama-bench (GPU and CPU
  bandwidth, PCIe, matmul rate, best thread count) and `--verify` runs
  one config for real, so speed predictions stop being spec-sheet
  guesses. `mdl fit inspect` prints the tensor inventory per layer.
- `mdl run` books any buffer sizes a load log prints against its config,
  so fits improve from ordinary use.
- Budgets are the ones that decide whether a load works. Free VRAM is
  llama.cpp's own figure (under Vulkan it counts what idle desktop apps
  give back; nvidia-smi does not, and reads a gigabyte short). The RAM
  limit is the total less 3 G for the system; going past what is free
  right now is a note, not a failure, since the OS pages idle programs
  out. A config inside the free VRAM but eating the safety margin is
  "fits, tight", not over. A config already on 4-bit KV lets the search
  use it, and otherwise every fit says what q4_0 would buy - context,
  speed and turn time over the #1 - without picking it for you.
- Fits are planned for the machine at idle, so a scan taken mid-game or
  with a browser full of tabs gives the same answer as one taken on a
  quiet desktop. What the OS and resident programs hold is, best first:
  what `mdl fit hw --idle` (or `--idle-vram/--idle-ram`) was told, what a
  probe saw in the first ten minutes after boot, or what is in use now
  less what apps opened since boot hold - anything not the system's, not
  a startup entry and not the terminal mdl runs in - floored at a typical
  figure per OS and desktop. When this minute's machine is short, the
  output names the apps to close; `--now` plans for it as it is. A busy
  CPU is noted, and `mdl fit hw` warns before calibrating on one.
- `mdl eval <name>`: a private, auto-graded suite run against a model
  the way models.toml runs it. Code (40, graded by hidden unit tests that
  are executed), tool calls (30, single and multi-step against mock
  tools), long-context retrieval at 32k, 64k and 128k (15), format rules
  (20) and exact-answer reasoning (20), plus your own tasks from
  `~/.config/mdl/evals/*.toml`. Items come from a seed that never leaves
  the machine. `--sandbox` runs model code in a podman or docker
  container; `--estimate` says how long a run takes; `--results` shows
  past runs with 95% intervals.
- `mdl catalog`: the hub's lineage graph - models, their fine-tunes and
  merges, every GGUF quant of each, and reported eval results - in one
  SQLite file. `build` crawls it locally, `pull` fetches the published
  snapshot, `tree` and `search` read it.
- `mdl find`: the best model this machine can run for a profile, across
  the catalog and models.toml, at the quant and config it would run at.
  Ranked by expected quality with speed as a floor. Public results
  are down-weighted when they look trained-for, and your own `mdl eval`
  scores take over once there are enough of them. Models nobody has rated
  are listed under "worth testing" with the commands to rate them.

### Fixed

- `mdl eval` marked every multi-hop long-context answer wrong: the
  question asks for just the room, the reply gave just the room, and the
  grader wanted the word "room" in front of it.
- The dashboard's CPU line left the embeddings out, so what sat on the
  card plus what sat in RAM did not add up to the size of the file.
- Running `py mdl.py` directly, an error from `mdl fit` (and now
  `eval`, `catalog`, `find`) printed a traceback instead of one line: the
  module was loaded twice, and the second copy's error went uncaught.

### Changed

- Eval suite v2, because v1 could not tell models apart: a good 35B at
  3 bits scored 1.00 on three suites of five. Most items are now a
  harder tier, scored separately - multi-step reasoning, tool arguments
  that must be derived, requests with something missing that should be
  asked about, three-hop and superseded facts in long documents - and
  code is marked per hidden test instead of all-or-nothing. The code
  suite gained ten algorithmic tasks - a parser, a binary search on the
  answer, a monotonic deque - each with hidden tests for the edge cases
  a first draft gets wrong and one input big enough that a quadratic
  answer runs out of time. Three more are not algorithms with names
  at all: a price with five clauses that interact, a validator with a
  stated precedence over its rules, and a stack machine whose opcodes
  are named by the seed. A model cannot recall a solution to those,
  because the constants, the rules and the names are generated - it has
  to read the specification. That was the real ceiling: a 35B coder in
  a 3-bit quant solved 36 of 40 of the named ones. A fourth splits a
  line whose separator, quote, escape and comment characters all come
  from the seed, so it is nobody's CSV parser. The hard pool is drawn
  from in halves - one named task, one generated - because pooled
  together the generated ones were four templates in fifteen and
  reached only a fifth of the hard items, which is not enough to move
  a score. The tool suite gained four worlds that take
  four or five dependent calls, including one where the rules forbid the
  action and doing it anyway is the failure, and one where a call fails
  the first time. A sixth asks for something none of the others do:
  looking again. Every other world can be solved by gathering the facts
  once and then acting on them, which is the habit that goes wrong
  against a real system, so in this one a deploy freeze begins in the
  middle of the job and is only visible to an agent that checks before
  each step rather than once at the start. Results from v1 are kept but no longer read as
  evidence about a model. One long-context question per document now
  has no answer in the document, so making one up is measured directly,
  and another asks for every match rather than one. The tool suite
  gained a world with eight tools, four of them beside the point, a
  transaction list that only arrives a page at a time, and a dozen
  dependent calls to get through - an agent that reads page one and
  stops looks exactly like one that finished, so the score is overlap
  rather than all-or-nothing. The reason suite gained a seating puzzle
  generated and then pruned until exactly one arrangement fits, and a
  two-counter machine that has to be simulated round by round, and a
  set of rules to chain to a fixed point where stopping after the
  obvious step gives a wrong answer that looks finished; the eight it
  had were all textbook shapes a model recognises on sight. Its hard
  pool is drawn in halves too, one named and one generated.
- Four graders could be satisfied without doing the work, which means
  every score they ever produced carried some free credit. Found by
  walking every item with replies that contain no knowledge at all -
  nothing, a refusal, a guessed number, the instructions read back.
  "No tool fits this request" only checked that no tool was called, so
  "I don't know" scored full marks; it now has to answer as well, and
  the questions were changed to ones with a checkable answer. Format
  rules were scored by the fraction kept, and an empty reply keeps "no
  commas" and "do not use the letter t" for free - up to two thirds of
  an item for saying nothing; part marks now need a real attempt, and
  a reply that repeats its own instructions scores nothing, which is
  the failure a model with the wrong chat template actually has. The
  long-context questions say to reply with just the room, just the
  code, just the number, and are now held to it, because grading the
  whole reply by substring paid a model that quoted a slab of the
  document back for whatever happened to be inside it. And the count of
  hidden tests a solution passed is now carried by a word the solution
  could not have known, because the code under test writes to the same
  stdout the score is read from, and `print("passed 9 of 9")` is four
  keystrokes. The check runs as part of the test suite from now on.
- Every run records what it cost: reply tokens, and tokens per right
  answer. A model that thinks four times as long for the same score is
  not as good on hardware you own, and nothing else was measuring that.
- A run is fingerprinted by the questions it actually asked, not by the
  seed they came from, so two runs that claim to be comparable can be
  checked. `mdl eval --compare A B` pairs two runs item by item and
  bootstraps the difference, and says "too close to call" when the
  interval does not clear zero.
- The time estimate learns: after a model has run once, the next
  estimate uses what it actually spent per suite instead of a constant
  multiplied by four for anything that thinks.
- The config writer the dashboard uses moved into `mdl.py`, so `mdl fit`
  can edit models.toml without textual installed.

## [0.5.2] - 2026-09-05

### Fixed

- Quitting while the dashboard was mid-poll could still raise, this
  time on the status line. 0.3.5, 0.4.1 and this release each guarded
  one more widget the poll touches; the guard now sits on the tick
  itself, which is what all of them had in common.

## [0.5.1] - 2026-09-05

### Fixed

- A group's size added up its members, so six context presets of one
  11.3G gguf claimed 67.7G. Presets of a single model are the usual
  reason to group, so the figure was wrong in the common case. Each
  file is counted once now, and a member pointing somewhere else
  still adds its own.

## [0.5.0] - 2026-09-05

### Added

- `group`, per model: models sharing a name fold under one row in the
  dashboard. `enter` opens and closes a group, the fold is remembered
  between sessions, and a closed group still shows a marker when
  something inside it is running - otherwise folding would hide the one
  thing the pane is for.
  There is no command to make a group and none to delete one, because a
  group is not a thing: it is a name two models share. Set it in the
  editor beside every other key.
- `y` copies the log. Textual's own selection reads nothing out of a
  `RichLog` - `Widget.get_selection` only handles a widget that renders
  to a single `Text`, so every scrolling widget copies as nothing - and
  the log now hands its lines over, so drag-and-`ctrl+c` works too.
  A drag marks the whole widget rather than a span, so both copy the
  whole buffer; for part of it, hold shift and use the terminal's own
  selection.

## [0.4.1] - 2026-09-03

### Fixed

- Quitting the dashboard while a model was still loading could leave
  the startup watcher drawing into widgets that had gone. 0.3.5
  guarded the poll; the watcher reaches the table by its own path,
  and that one was still open.

## [0.4.0] - 2026-09-03

### Added

- `mmproj`, per model: the vision projector a multimodal model needs
  alongside its weights, emitted as `--mmproj`. Without it llama-server
  loads the text half and says nothing about the missing eyes.
- `mdl add` writes it in when it finds exactly one `mmproj-*.gguf`
  beside the weights, and names it in the output. Two candidates is a
  choice rather than a default, so it leaves that to you.
- `mdl check` reports a projector whose file has gone, the same way it
  reports missing weights, and the params pane marks it `missing`.
- The projector counts towards the estimated VRAM. It loads onto the
  card with everything else, and on a small one it is not a rounding
  error.

### Fixed

- A test added in 0.3.5 removed widgets from a running app to check
  the poll survives them, then asserted the app was still running -
  which removing them is itself enough to change. It now asks the
  question it meant to: does the poll raise?

## [0.3.5] - 2026-09-02

### Fixed

- The dashboard decided a server was ready by matching a line in
  its log, while `mdl run` asked `/health`. The log wording has
  already changed once between llama.cpp builds; the endpoint is
  the contract, and both use it now.
- The dashboard ignored `ready_timeout` and always gave up after
  300 seconds, so a slow model that `mdl run` waited for happily
  was reported as failing to start.
- `R` waited a fixed 1.5 seconds between the stop and the run, and
  a slower stop left the restart colliding with its own old port.
  It now waits for the stop to finish. On a model that is not up,
  it starts it instead of first saying it is not running.
- `R` also picked its row by the displayed name, so restarting a
  model whose name is too long for the pane started whichever one
  the cursor happened to be on.
- VRAM was read from the first card `nvidia-smi` listed rather than
  summed across them, so a second GPU was invisible.
- The poll could outlive the widgets it draws into and raise on the
  way out of the app.

### Changed

- A model whose file is not on disk reads `missing` in the size
  column instead of `0B`, which was indistinguishable from a file
  that is genuinely empty.
- Help says `s` stops the *selected* server, which has been true
  since 0.3.0, and that `g` reloads `models.toml` as well as the
  telemetry - useful, and previously undocumented.
- One `human_size`, in `mdl`. There were two, with different units:
  the same byte count printed differently depending on the caller.

## [0.3.4] - 2026-09-02

### Changed

- The estimated VRAM bar is split into weights and kv cache, with
  each figure spelled out beneath it. Past the card's capacity it
  scales to the total rather than clipping, so the parts stay
  visible: once a bar is full, what is in it is the question.
- A config that keeps weights off the gpu - `n_cpu_moe`, `-ot`, a
  partial `-ngl` - now reads `<= 16.3 / 12.0 G` in ordinary text
  rather than red. The estimate counts the whole file as resident,
  which for a MoE model with experts on the cpu is far too high, so
  it was calling working configurations broken.

## [0.3.3] - 2026-09-02

### Fixed

- Saving an argument containing a double quote corrupted the config.
  The TOML writer escaped backslashes but not quotes, so a value like
  the JSON `--chat-template-kwargs` takes closed its string early and
  left the `args` array open - the next read failed to parse, and the
  edit was lost. Quotes are escaped now, after the backslashes rather
  than before, so the escape characters are not themselves escaped.
- `mdl add` had the same hole: a quote in a model's path, legal on
  POSIX, produced a config that would not parse.

## [0.3.2] - 2026-09-02

### Added

- An `args` row in the edit screen. The config key already passed
  flags through verbatim; now you can set it without leaving the TUI,
  which is the answer to every llama-server flag mdl has no key for.

### Fixed

- The VRAM estimate ignored `args`, so moving `--cache-type-k` out of
  `kv_type` and into `args` added two gigabytes to the figure without
  changing the command. It now reads the context size and cache type
  off the built command line, where every flag ends up whichever
  field it came from.

## [0.3.1] - 2026-09-02

### Fixed

- The ctx meter read `idle` and drew an empty bar on a server that had
  a whole conversation in its cache. Recent llama.cpp builds no longer
  export `llamacpp:kv_cache_usage_ratio` at all; where it is missing,
  the occupancy now comes from `/slots`, which keeps reporting what is
  resident after the slot goes idle.
- Peak tok/s decayed to 0 about forty seconds after a reply finished.
  It was the maximum of the sparkline's rolling window, not a peak. It
  is now the high-water mark for that server, and resets when it does.
- A long model name pushed the size and ctx columns off the right edge
  of the models pane, so `65536` was drawn as `65` with nothing to say
  it had been cut. Names longer than the column are now truncated with
  an ellipsis instead; selection still follows the real name.

## [0.3.0] - 2026-09-01

### Added

- More than one server at a time. `run` no longer refuses when something
  else is up; each model needs its own port, and a clash names the model
  holding it rather than failing vaguely.
- `mdl run <name> --port N` overrides the config for one run, for a
  config whose models share a port.
- `mdl stop <name>` and `mdl stop --all`. `--all` tries every server,
  reports each, and exits non-zero if any survived.
- `mdl check` lists models that share a port. It is a note, not a
  problem: sharing is fine until you want both at once.
- The dashboard follows the selected model rather than "the" running
  one, and the model list marks everything that is up.

### Fixed

- The tok/s reading was zero for a whole generation and then one
  spike. It was taken from `llamacpp:predicted_tokens_seconds`, which
  llama.cpp holds at 0 until a request finishes and then publishes as
  a single average. `llamacpp:n_decode_total` is the counter that
  actually moves while a reply streams, so the rate now comes from
  its delta over the measured interval.
- The log pane could be squeezed to nothing. The dashboard and command
  panes had fixed heights and the log took what was left, which on a
  30-row terminal was zero rows. The command pane now sizes to its
  content and the log keeps a floor.

### Changed

- **Breaking: `mdl ps --json` always prints a list**, `[]` when nothing
  is running. It used to print a bare object, or `null`. A shape that
  changes with the number of results breaks the status-bar scripts this
  flag exists for, and it breaks them silently, the first time someone
  starts a second server.
- State moved from `state.json` to one file per server under
  `~/.local/state/mdl/run/`. Two `mdl run` calls at the same moment would
  otherwise read, modify and write one file, and one would lose. An
  existing `state.json` is migrated on first read, so upgrading with a
  server running keeps control of it.
- Commands that took no argument still take none while one server is up.
  `stop` and `logs` ask which only when there is a genuine choice.

## [0.2.0] - 2026-09-01

### Changed

- `mdl check` no longer counts the model path in the `[example]` table
  `mdl init` writes as a problem. It is a to-do, and it says so.
  A missing `llama_server` is still reported, because that one you
  cannot run anything without.
- The Textual requirement is `>=3,<9`. The old `>=0.80` was a guess:
  0.80, 1.0 and 2.0 all fail the suite and 3.0 is the first that does
  not, and CI now runs the floor as well as the current release.

### Fixed

- The POSIX suite sandboxed itself with `HOME` alone, so on a runner
  that sets `XDG_CONFIG_HOME` it read the real config instead of its
  own. It clears both XDG variables now.
- A failing check in that suite reported only the exit status, not the
  line `mdl` printed to say why.

### Project

- CI covers Linux, macOS and Windows across Python 3.11-3.13, both ends
  of the supported Textual range, Ruff, and a packaging check that
  builds the wheel and the sdist and installs each into a clean venv.
- Tagging `v*` publishes to PyPI through Trusted Publishing, with no
  API token stored in the repository. It refuses to publish when the
  tag and `mdl.VERSION` disagree.
- `SECURITY.md`, `CONTRIBUTING.md` and issue templates.

## [0.1.0]

First release.

### Commands

- `mdl run`, `stop`, `ps`, `list` - one llama.cpp server at a time, driven
  from `~/.config/mdl/models.toml`.
- `mdl init` writes a commented starter config and finds `llama-server` on
  your PATH if it is there.
- `mdl add <model.gguf> [name] [port]` appends a config entry with sane
  defaults, naming it after the file and reporting its layer count.
- `mdl check` validates every model - binary, model files, unknown keys,
  `ngl` below the layer count - without launching anything.
- `mdl logs [-f] [name]` prints or follows a server's log.
- `mdl ps --json` for scripts and status bars.
- `mdl --version`.

### The dashboard

- `mdl ui`, and bare `mdl`, open a terminal dashboard over the same config
  and state file: model list, live `llama-server` command preview, VRAM and
  KV-cache meters, a tokens/sec sparkline, a colour-coded log tail, and
  inline parameter editing that saves back to the config, comments intact.
- `p` opens a conversation with the running model. It keeps context across
  turns, and `ctrl+l` clears it.
- Reasoning is shown dimmed and inset and timed on its own, from either a
  `reasoning_content` delta or inline `<think>` tags, which can straddle a
  chunk boundary.
- While it works it says which part it is on - reading the prompt,
  reasoning, generating - with a spinner, a running token count and tok/s,
  then a final line with tok/s, time to first token and time spent
  reasoning. Server-reported timings win over our own count.
- `esc` interrupts a reply by closing the socket, so it stops now rather
  than at the next token, and reports it as interrupted, not failed.
- Needs `pip install "llama-mdl[ui]"`. Every other command needs nothing.

### Behaviour

- Config writes are atomic and keep a `.bak`, so a crash, a full disk or a
  Ctrl-C cannot leave `models.toml` truncated.
- Pre-flight checks: a missing binary, a missing model file or a busy port
  fails in milliseconds with one line, rather than after a model load.
- `ready_timeout` in the config, for a model that takes longer than 300s.
- Logs rotate: `<name>.log` shuffles to `.1` and `.2` on each run.
- `$XDG_CONFIG_HOME` and `$XDG_STATE_HOME` are honoured.
- Test suites in `tests/`, including POSIX process semantics and an opt-in
  suite that drives a real model.

### Notes

- Readiness is an HTTP probe of `/health`, not a regex over the log.
  llama.cpp has reworded that line between builds; the endpoint is stable.
- `stop` signals the whole process tree. If `llama_server` is a wrapper
  script, signalling only the recorded pid orphans the real server and
  leaves the port held.
- The state file records the OS process creation time, so a recycled pid
  is not mistaken for a running server. macOS has no cheap way to read
  that, so it falls back to the pid alone.
- On POSIX a server stopped from inside `mdl ui` is reaped rather than
  left a zombie, which `kill(pid, 0)` reports as still alive.
- Python 3.11+ (for `tomllib`). Older versions exit with one line.
