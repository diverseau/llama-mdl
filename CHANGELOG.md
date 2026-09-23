# Changelog

Notable changes. Dates are ISO; versions follow [semver](https://semver.org/).

## [Unreleased]

### Fixed

- Two threads saving the same file at once - the dashboard's workers,
  `mdl find`'s header fetches - shared one temp file: one removed the
  other's, and the second save failed. Each save has a temp file of its
  own, and on Windows a rename that lands at the same instant as another
  is tried again rather than refused.
- `mdl ui`'s edit modal saved values `mdl run` then refused - a port
  over 65535, `parallel = 0`, a negative `ctx`, a cache type with a
  space, `--port` in args - leaving a config nothing would start from.
  They are checked as `mdl run` checks them, and the modal stays open to
  fix them.
- Two models sharing a port, started at once or one while the other was
  still loading, both found the port free - a server takes seconds to
  bind it - and the second died at load. A server still loading now
  holds its port, and launches onto one port take turns.
- `mdl eval NAME --port N` with NAME already running on another port
  ignored `--port` and said nothing. It is refused, saying which port the
  server is on.

## [0.8.0] - 2026-09-23

From an audit of the whole project: what the dashboard's edit modal can
set, what `mdl fit` holds back after `--verify`, races between mdl
processes sharing a file, and the tests that were missing around them.

### Changed

- `mdl ui`'s edit modal sets `flash_attn` to on, off, or empty for the
  build's default, and shows empty when the key is not there. It used to
  offer only on and off, and dropped `flash_attn = false` on save, so
  the server started at the build's default instead of `-fa off`. It
  also takes `ngl = "all"` and `"auto"`, as the config does; a model
  with either could not be saved from it at all.
- `mdl fit` holds back, for a model's architecture, the bigger margin
  `mdl fit --verify` booked when that architecture ran out of memory,
  and says so ("N MiB held back on the card for ARCH"). `--verify` said
  the next fit would use it, but nothing read it back: the next fit made
  the same prediction. A second out-of-memory run grows the margin from
  the one the fit used, not from the default again.
- A manual run of the release workflow builds and checks the package
  but no longer publishes it; only a `v*` tag does.

### Removed

- The move of a pre-0.3 `state.json` to one file per server, kept for
  upgrades from 0.2. A server started by mdl 0.2 and still running is
  not listed; stop it by hand.

### Fixed

- A mistyped value ended in a traceback instead of one line: an `hf:`
  spec that is not `org/repo` (`mdl fit`, `mdl fit inspect`), and a
  number that is not one (`mdl fit --apply`, `mdl eval --port`,
  `mdl find --top`). `mdl eval --port` is also checked for a port
  number, and the error names the flag rather than the config.
- Saving from `mdl ui`'s edit modal took the dashboard down with a
  traceback when the model's table had been renamed or removed on disk
  while it was open. The save is refused with a message.
- The README said `flash_attn = false` was never passed on. It is,
  as `-fa off`.
- Every `mdl fit`, `find` and `eval` rewrote `hw.json` from the copy it
  read when it started, so one running during `mdl fit hw` could undo
  the calibration. Each write now takes a lock, re-reads the file and
  adds only what it learned. A probe that finds the file busy leaves it,
  and still answers.
- Two mdl processes waiting on a lock whose holder had died could both
  take it: the second removed the lock the first had just made. Taking
  over a dead holder's lock is now done by one waiter at a time, and only
  while the lock still names the dead pid.
- Two `mdl eval`s started at once on a fresh install could each make
  their own eval seed, so one ran items no later run would ask again.
  The seed is now made by whichever gets there first, and the other
  waits for it. The `hf:` header cache and `mdl find --new`'s mark are
  written whole or not at all, like the rest of mdl's files.

## [0.7.1] - 2026-09-19

Fixes from a peer review of 0.7.0's manifest, eval resume and profiles.

### Fixed

- A second `mdl eval` of the same items on the same server deleted the
  first one's checkpoint before it was refused. The run's lock is now
  taken before the checkpoint is read, and held until the result is
  saved.
- A finished eval deleted its checkpoint before saving its result, so a
  failed save - a full disk - lost the whole run. It saves first.
- An item appended after a line an interrupt cut off was lost on the
  next resume along with it; the torn line is cut off first.
- A server reply cut off before its Content-Length escaped as a
  traceback instead of being retried and left unscored.
- A podman that is installed but not answering kept a working docker
  from being used, and model code ran unsandboxed. Each runtime is tried
  in turn.
- Code run in the container lost `DOCKER_HOST` and the other settings
  the runtime was checked with, so a non-default daemon passed the check
  and then failed every item.
- Model files were named by their size and first and last 8 MiB, so a
  change in the middle went unnoticed by `mdl manifest`, `--resume` and
  profiles. They are named by a sha256 of every byte, read once and kept
  against size and mtime. A profile covers every shard and the
  projector.
- A model file replaced under a running server was described as if the
  server had loaded it. The files are stat'ed at launch; `mdl manifest`
  reports a replaced one and `mdl eval` refuses it.
- The runtime identity took the build string or the binary, and profiles
  only the build number and the binary's path: a rebuild with other
  settings, or another commit under one build number, matched. Both
  now take the build, its commit and the binary's stat.
- `mdl fit --verify` booked its measurement to the preset's whole
  command, flags llama-bench never ran (`-ot`, a split mode) included.
  Such a measurement is kept, but not as the preset's.
- An `mdl eval` of a preset with a projector booked its speeds under
  flags `mdl fit` never looks up, so they were never shown as the
  current config's.
- `mdl manifest --redact` left an absolute path outside the home
  directory with no extension, like `D:/private/alice/slots`, whole.

## [0.7.0] - 2026-09-19

Everything since 0.6.6. (0.6.7 to 0.6.20 were development builds on
`main`, numbered a patch at a time by mistake and never released.)

### Added

- `mdl doctor [--json] [name]`: one diagnosis of the environment and
  every preset - the config's checks, the llama-server each one runs and
  its build, whether that build lists every flag the preset passes, GGUF
  headers on the model and projector, and each server's health against
  its state (a stale state, a server whose launcher is gone, something
  else on its port, a leftover launch lock). It never launches a server
  and never changes the config or the state files; only failures make
  it exit non-zero.
- `mdl manifest <name>`: what a model is running as - the command line
  it was launched with, the llama.cpp build, each model shard's size and
  hash, and the machine - read from the running server and its state,
  or from the preset (and saying so) when it is not running.
  `--redact` blanks secret flags and cuts paths, and the home directory,
  out of it for a bug report. Every `mdl eval` result carries the
  manifest of the server that answered.
- `mdl eval --resume`: every finished item is kept in a checkpoint as it
  finishes, and an interrupted run continues where it stopped - only onto
  the same items and the same server (same command bar the port, build
  and model bytes, per `mdl manifest`); otherwise it starts over and says
  why. A finished run replaces the partial records its interruptions
  left, and a second eval of the same items on the same server is
  refused rather than interleaved.
- `mdl find --why MODEL` explains one model instead of printing the
  table: where it ranks, or why it is not shown; every quant considered,
  each with its fit or the reason it was turned away (too big, an
  architecture this build does not load, a header that is not the model
  it claims to be, the floor it misses, a header that could not be had,
  a quant not kept); the public benchmarks its estimate rests on, the
  parent it borrows from, the quant and KV penalty, and what beat it.
  MODEL is a catalog id, a models.toml name or a unique part of either;
  `--json` for scripts. The estimate is labelled for what it is: a
  heuristic for a shortlist.
- Measured fit profiles: what one exact configuration - model bytes,
  llama.cpp build, binary and the whole command - did on this machine,
  as decode speed at the depths it reached and prefill speed.
  `mdl fit --verify` books its llama-bench run as one, and `mdl eval`
  books the timings the server reported for every reply. `mdl fit
  <name>` shows the current config's measurements beside the prediction,
  and `mdl fit <name> --profiles` lists every configuration measured.
  Speed only - a profile claims nothing about quality.
- `mdl fit --apply N --dry-run` and `--write NAME --dry-run` show the
  exact change first: a diff of models.toml and the llama-server command
  before and after, and write nothing. A dry run fails the same way the
  real one would.
- Every write to models.toml keeps the last five versions
  (`models.toml.bak`, then `.bak.1` to `.bak.4`); `mdl config --history`
  lists them with the tables each differs in, and `mdl config --undo`
  swaps the config with the newest backup - a second `--undo` puts it
  back.
- A model may name its own `llama_server`, for one that needs a
  different build - a fork that loads a quant type upstream does not,
  like PrismML's for Bonsai's PQ2_0. It beats `$MDL_LLAMA_SERVER` and the
  top-level setting for that model only, in `run`, the dashboard, `fit`,
  `eval` and `find`; `mdl check` and `mdl doctor` say when it is missing.

### Changed

These can refuse a config, or a comparison, that 0.6.6 accepted.

- Every model's settings are checked before anything runs, each fault
  one line: `port` a number from 1 to 65535, counts not negative,
  `parallel` at least 1, `flash_attn` a real true/false, `args` a list
  of strings. A `--port` or `-m` in `args` is refused: it would override
  the key mdl reads for its state, its pre-flight and its health check.
- Model names are letters, digits, `-`, `_` and `.`, starting with a
  letter or digit - they name a TOML table and the state and log files.
  A name with a dot is written quoted.
- `flash_attn = false` now runs `-fa off` rather than whatever the build
  defaults to.
- `mdl eval` results carry a new fingerprint that covers each item's
  system prompt, full tool schemas, reply cap and document content, and
  the grader version. Runs from 0.6.6 are not comparable with 0.7.0
  ones, and `--compare` says so rather than subtracting them.
- Model-written code runs in the podman or docker sandbox by default
  whenever one is available and answering; `--no-sandbox` opts out, and
  a run without one says plainly that the code runs as you.
- An eval item the server failed on - a dropped connection, an HTTP
  error - is retried twice, and one that still fails is left unscored
  and reported, rather than counted as a wrong answer. A run with such
  items is marked partial.
- `mdl eval` on a server that was already running refuses one whose
  settings have changed since it started, and checks the server is
  serving the file it is about to credit.

### Fixed

- `mdl fit --apply` kept only the keys it tuned and deleted the rest of
  the model's table - `model`, `port`, `group`, `llama_server` - so the
  next `mdl run` had nothing to load. It now changes what it tuned and
  leaves everything else alone.
- Editing `models.toml` from `fit` or the dashboard handles tables the
  way people write them by hand: a comment after `[name]`, indentation,
  a quoted name, an `args` array over several lines. The result is
  parsed before it replaces the file, and a missing table is one line,
  not a traceback.
- `mdl eval` code suite: the score no longer comes from the harness's own
  output and exit code, which the code under test shares. A submission
  that exited cleanly at import, printed a pass line or wrote its own
  results scored full marks. The expected answers now stay in the
  grading process; anything short of one well-formed result per case
  scores nothing. Output is kept to a bounded tail, and a run that times
  out is killed with everything it started.
- `mdl stop` reported success when a `llama_server` wrapper exited and
  left the real server running - still holding the port and the GPU,
  with `ps` showing nothing. The process group (POSIX) or process tree
  (Windows) is now tracked after the wrapper exits, signalled, and
  checked; stop succeeds only when it is gone and the port is free.
- Two launches of one model at the same moment could both start a
  server, and the second state file hid the first for good. A launch now
  holds a per-model lock and checks again inside it; a server whose
  state cannot be written is stopped instead of left untracked.
- `mdl eval`: Ctrl-C while the model was still loading left the server
  it had started running.
- `mdl eval` on a running server recorded settings from `models.toml` as
  it is now, which need not be what the server was started with; it
  records the command that actually ran.
- `fit hf:` header fetches read at most what they asked for, even from a
  server that ignores Range; a wrong Content-Range or a short body is
  refused; and a header whose length fields claim more than the 256 MiB
  limit is refused instead of fetched.
- The remote header cache is keyed by every shard's path, size and
  content id and the hub it came from, not the first 16 characters of
  the first shard; it is written atomically.
- `mdl find` could crash when a row sized from a sibling's header failed
  on its own. A re-evaluation clears everything derived, only rows with
  a fit, a score and no rejection are ranked, and refinement repeats
  until the rows shown have their own headers.
- `mdl find` judged every model by the default llama-server, so a model
  that names its own fork for an architecture upstream lacks was turned
  away. Each is judged by the build it runs on.
- `mdl add` stored a relative path as given, so the model only ran from
  the directory it was added in; it stores it absolute now. A name like
  `bad name` wrote a table the next load could not parse; it is refused.
- `mdl fit hf:... --write NAME` said nothing and wrote nothing; it is
  refused with the step that works (download, then fit the file), and
  `mdl find` suggests that step instead.
- Unified KV (`--kv-unified`) survives a fit, a projector's placement
  changes cleanly both ways, and `--flag=value` forms are read and
  replaced like the spaced ones.
- `mdl fit --verify` benchmarked at llama-bench's default thread count
  rather than the config's, and took any failed benchmark - a timeout, a
  flag the build does not take - as out of memory and widened the
  memory margin. Only an out-of-memory failure moves the margin now.
- Piped or redirected output on Windows (`mdl find | more`, `> out.txt`)
  crashed with a UnicodeEncodeError on the tables' → · ✓ ⚑; what the
  pipe cannot encode now prints as `?`.

### Documentation

- The catalog's publishing status is no longer written into the README,
  where it went stale; SECURITY names the per-server state files; and
  the eval's claim is the narrower true one - these exact questions
  cannot have been memorised, the templates are public. The README says
  plainly that running model code without a container is not a sandbox,
  and that `find`'s ranking is a heuristic for a shortlist.

## [0.6.6] - 2026-09-18

### Fixed

- `mdl ui` chat: the view no longer snaps to the bottom on every token,
  so a reply can be read back while it streams. It follows while you are
  at the bottom, lets go when you scroll up, and takes hold again when
  you scroll back down.
- `mdl ui` chat: replies are no longer cut off at 1024 tokens, which a
  model that thinks could spend before it began answering. Esc stops a
  reply that runs on; `-n` in a model's `args` still caps it server-side.

## [0.6.5] - 2026-09-17

### Changed

- `mdl eval` suite v4. Tools: eighteen of thirty items are multi-step
  worlds (was twelve), two in three of them hard, with two new ones - a
  customer to pick out of four namesakes, and counters behind optimistic
  locking where another writer gets in between a read and a write.
  Scores from v3 are not comparable, and `--compare` says so.
- The nightly catalog covers the 3,000 most-downloaded GGUF repos plus
  the 500 newest (was 800 and 200). The first full crawl used 289
  requests of its 40-minute budget and left three in four GGUF models
  with no benchmark results to rank on.
- CI and release workflows use the Node 24 majors of the GitHub actions
  (checkout and setup-python v7, upload-artifact v7, download-artifact
  v8); Node 20 is deprecated on the runners.

### Fixed

- `mdl eval`: "12% of 3840" wanted 460 and marked 460.8 wrong. A reply
  that asks for missing details as a list rather than a question now
  counts as asking. In format items, rules that only forbid something
  ("no commas", "at most 60 words") count half toward part marks, since
  a reply that is not writing keeps them.
- `mdl find` no longer ranks files that are not the model: an 11 GB
  DSpark drafter was shown as DeepSeek-V4-Flash, all on GPU, and
  `imatrix.gguf` as Qwen3.5-122B. Files whose header holds a parameter
  count far from their model's are turned away.
- `tests/test_live.py` read `PromptScreen.log_text`, which no longer
  exists; it now waits for the reply in the transcript.

## [0.6.4] - 2026-09-17

### Fixed

- `mdl ui`: copying the log (`y`) or the command (`c`) works on Linux.
  Textual copies only by OSC 52, which GNOME Terminal and other VTE
  terminals ignore, tmux drops unless `set-clipboard` is on, and many
  terminals cap below the size of a server log - while the app said
  "log copied" regardless. The text now also goes to `wl-copy`, `xclip`
  or `xsel` (`pbcopy` on macOS), and with none of them installed the
  notice says it went via the terminal instead of claiming the copy.
- The copied command is quoted for sh on Linux and macOS, not for cmd.

## [0.6.3] - 2026-09-17

### Added

- `mdl config` opens models.toml in `$VISUAL`, then `$EDITOR`, falling
  back to Notepad on Windows or vi elsewhere. `mdl config --path` prints
  its location, including when `XDG_CONFIG_HOME` overrides the default.

### Changed

- The published catalog is built from the 800 most-downloaded GGUF repos
  plus the 200 newest, with only their files and direct source models
  fetched. A crawl has a 40-minute budget; when it runs out, the snapshot
  is published as a partial with its work queue inside, and the next run
  resumes from it. `mdl catalog stats`, `pull` and `find` say when a
  snapshot is partial.

### Fixed

- Vision projectors, LoRA adapters, speculative drafts and MTP heads are
  no longer listed as quants. A 1.4 GB MTP head read as the smallest
  "Q4_0" of a 27B model, which `find` could pick when nothing else fit.
  Snapshots already published are filtered when read.
- `mdl catalog tree` names files that share a quant by what sets them
  apart (`noMTP-Q4_K_M`, `IQ1_S-multilingual`) instead of repeating it.
  NVFP4 is recognised as a quant.

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
