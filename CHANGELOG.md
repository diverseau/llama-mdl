# Changelog

Notable changes. Dates are ISO; versions follow [semver](https://semver.org/).

## [0.2.0] - 2026-09-01

### Changed

- `mdl check` no longer counts the `[example]` table `mdl init` writes
  as a problem. It is a to-do, so a freshly initialised config passes.
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
