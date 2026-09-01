# Contributing

Thanks for looking. This is a small tool with a deliberately small scope, so
the most useful thing you can do before writing code is open an issue and
check the change is wanted.

## The scope

mdl runs **one** local llama.cpp server from a config file. Things it will not
grow: a daemon, model downloading, hot-swapping, auto-unload, running several
servers at once, a web UI. Those are all reasonable things to want, and all of
them are somebody else's tool. The narrow scope is the point.

Things that fit: better llama.cpp flag coverage, better failure messages,
shell completions, more output formats, anything that makes the two files
smaller.

## Getting set up

No build step, no virtualenv required.

```sh
git clone https://github.com/diverseau/llama-mdl
cd llama-mdl
python mdl.py --help
python -m pip install textual      # only if you want the dashboard
```

## Running the tests

```sh
python tests/run.py           # fast, no real model needed
python tests/run.py --live    # also drives a real model through the UI
python tests/integration_posix.py   # Linux only; process semantics
ruff check .
```

The fast suites need no model, no server and no network: they run against a
fake `llama-server` in `tests/`. They must stay that way.

On any machine with Docker, the POSIX suite runs anywhere:

```sh
docker run --rm -v "$PWD:/repo:ro" python:3.12-slim \
    sh -c 'cp -r /repo /w && cd /w && python3 tests/integration_posix.py'
```

## House style

- **Two files.** `mdl.py` is the CLI and imports only the standard library.
  `mdl_ui.py` is the dashboard and is the only place Textual may appear. The
  CLI must keep working with Textual uninstalled.
- **Errors are one plain line on stderr and a non-zero exit.** No tracebacks
  at the user, no emoji, no colour in the CLI.
- **Comments explain why, not what.** If a line looks odd, the comment should
  say what would go wrong without it.
- Keep it short. If a change needs a new layer of abstraction to fit, it
  probably does not fit.

## Pull requests

- One change per PR, with a test that fails without it.
- Say what you actually ran. "Tests pass on Windows, POSIX suite in Docker"
  is worth more than "should be fine".
- CI runs the suites on Linux, macOS and Windows across Python 3.11-3.13, the
  pinned Textual floor and the current release, Ruff, and a packaging check.

## Reporting a bug

Include `mdl --version`, your OS, the `llama-server` build, and the relevant
part of `~/.local/state/mdl/<name>.log`. `mdl check` output helps too.

Security issues go through [SECURITY.md](SECURITY.md) instead, not the public
tracker.
