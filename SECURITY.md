# Security

## Reporting a vulnerability

Please report privately through GitHub, not in a public issue:

**[Report a vulnerability](https://github.com/contactsturnover-prog/llama-mdl/security/advisories/new)**

That opens a private advisory only you and the maintainers can see. Expect a
first reply within a week. If a fix is needed, it ships as a patch release and
you get credit in the advisory unless you would rather not.

## What is in scope

mdl launches a local `llama-server` process from a config file you wrote, and
talks to it over localhost. The parts worth attacking are:

- **The config.** `models.toml` becomes command-line arguments. A path that
  escapes the argument list, or an `args` entry that makes `mdl` run something
  other than the configured binary, is a bug worth reporting.
- **The state file.** `~/.local/state/mdl/state.json` records a pid that `mdl`
  will later signal. Anything that lets another user's file, or a crafted one,
  make `mdl` signal a process it did not start is in scope.
- **Log and config writes.** Anywhere `mdl` writes through a symlink it did not
  expect, or leaves a file readable that should not be.

## What is not

- **The server's port is unauthenticated by design.** `llama-server` binds
  localhost with no auth; `mdl` neither adds nor removes any. Exposing that
  port to a network is your decision and its consequences are yours.
- **`llama_server` runs whatever you point it at**, including a wrapper script.
  That is the feature. `$MDL_LLAMA_SERVER` overriding the config is likewise
  intended.
- **A config you did not write is already game over.** If someone can edit
  `models.toml` they can already run commands as you.
- **Model behaviour.** What the model says is not this project's concern.

## Supported versions

The latest release. This is a small tool with a single maintainer; there are
no backports.
