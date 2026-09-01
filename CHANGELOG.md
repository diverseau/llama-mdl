# Changelog

Notable changes. Dates are ISO; versions follow [semver](https://semver.org/).

## [Unreleased]

## [0.1.0]

First release.

### Added

- `mdl run`, `stop`, `ps`, `list` - one llama.cpp server at a time, driven
  from `~/.config/mdl/models.toml`.
- `mdl init` writes a commented starter config and finds `llama-server` on
  your PATH if it is there.
- `mdl logs [-f] [name]` prints or follows a server's log.
- `mdl add <model.gguf> [name] [port]` appends a config entry with sane
  defaults, naming it after the file and reporting its layer count.
- `mdl check` validates every model - binary, model files, unknown keys,
  `ngl` below the layer count - without launching anything.
- `mdl ps --json` for scripts and status bars.
- `ready_timeout` in the config, for a model that takes longer than 300s.
- Logs rotate: `<name>.log` shuffles to `.1` and `.2` on each run.
- `mdl ui` (and bare `mdl`) - a terminal dashboard over the same config and
  state file: model list, live `llama-server` command preview, VRAM and
  KV-cache meters, a tokens/sec sparkline, a colour-coded log tail, inline
  parameter editing that saves back to the config, and a streaming prompt
  pane. Needs `pip install "mdl[ui]"`; the four commands need nothing.
- `--version`, and `$XDG_CONFIG_HOME` / `$XDG_STATE_HOME` support.
- Pre-flight checks: a missing binary, a missing model file or a busy port
  fails in milliseconds with one line, rather than after a model load.
- Test suites in `tests/`, including POSIX process semantics and an opt-in
  suite that drives a real model.

### Notes

- Readiness is an HTTP probe of `/health`, not a regex over the log.
  llama.cpp has reworded that line between builds; the endpoint is stable.
- `stop` signals the whole process tree. If `llama_server` is a wrapper
  script, signalling only the recorded pid orphans the real server and
  leaves the port held.
- The state file records the OS process creation time, so a recycled pid
  is not mistaken for a running server.
- Python 3.11+ (for `tomllib`). Older versions exit with one line.
