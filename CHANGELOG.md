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
- Python 3.11+ (for `tomllib`). Older versions exit with one line.
