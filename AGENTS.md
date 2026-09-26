# AGENTS.md

Working notes for coding agents in the `mdl` repo. This is the canonical
file; `CLAUDE.md` points here and adds only Claude Code specifics.

Written against **0.12.0**. Where a fact is likely to drift, this says how to
re-derive it instead of quoting it.

## What this is

`mdl` runs local [llama.cpp](https://github.com/ggml-org/llama.cpp) servers
from a config file, and works out what to run and how. Two halves:

- **Running servers** — `run`, `stop`, `ps`, `list`, `add`, `check`, `init`,
  `config`, `logs`, `ui`. All of this is `mdl.py` alone.
- **Deciding what to run** — `fit` (what a GGUF will do on this machine and
  with which flags), `eval` (score it on a private graded suite), `catalog`
  and `find` (what is worth running at all), `lab` (measure variants of a
  config against each other). All of this is `mdl_fit/`.

## Layout

```
mdl.py            the whole CLI for running servers. Standard library only.
mdl_ui.py         the Textual dashboard (`mdl tui`). The only dependency.
mdl_update.py     `mdl update`, the daily check, the dashboard's restart
mdl_web/          the web UI (`mdl ui`, `mdl snapshot`): stdlib server,
                  static/ page with no build step; view.js is pure.
                  usage.py is the recorder `mdl run` starts; agents.py
                  opens coding agents; tailnet.py wraps `tailscale serve`
mdl_fit/          everything behind fit / eval / catalog / find / pull / lab
tests/            run.py drives the suites; support.py + fake_llama_server.py
docs/fit-plan.md  the design notes for mdl fit
docs/mdl-lab.md   the design of mdl lab, and where the build departs from it
```

Every `mdl_fit` module opens with a docstring saying what it is for. Read
those before guessing — they are accurate and they are the cheapest map:

```sh
for f in mdl_fit/*.py; do python -c "
import ast,sys; print(sys.argv[1], '-', (ast.get_docstring(
  ast.parse(open(sys.argv[1],encoding='utf-8').read())) or '').split(chr(10))[0])
" "$f"; done
```

Roughly: `gguf` parses headers, `hw`/`usage` measure the machine, `model`
places tensors, `perf` predicts speed, `search` enumerates configs,
`calib` holds measured corrections, `explain`/`emit`/`cli` are presentation,
`remote` reads headers of models you have not downloaded, `catalog` holds
and queries the hub snapshot and `catalog_crawl` builds it,
`quality`/`find` rank models, `evalsuite`/`evalrun` are the eval, `lab`
runs and records measured variants.

## Commands you will need

```sh
python tests/run.py            # the gate. Fast, no real model needed.
python tests/run.py --live     # also drives a real model through the UI
ruff check .                   # line-length 88; E, F, W, B
python -m build && twine check dist/*   # packaging, if you touched metadata
```

To run one suite: `python tests/test_eval.py`. Each file is standalone and
prints `<name>: N failure(s)`.

`--live` needs a real GPU and real models from your `models.toml`, starts
llama-server, and is **not** in CI. It can leave a server running if it
fails — check `mdl ps` afterwards. Treat a `--live` failure as information,
not as a gate, and confirm against `python tests/run.py` before believing
you broke something.

`tests/integration_posix.py` needs Linux:

```sh
docker run --rm -v "$PWD:/repo:ro" python:3.12-slim \
    sh -c 'cp -r /repo /w && cd /w && python3 tests/integration_posix.py'
```

## Hard rules

These are load-bearing. Breaking one passes review and fails in CI, or
worse, silently.

1. **`mdl.py` imports the standard library and nothing else, at module
   level.** It reaches `mdl_fit` lazily, inside the handlers that need it
   (`fit`, `eval`, `catalog`, `find`, `manifest`, `lab`, and the passive
   calibration after a launch), so the everyday commands load nothing
   extra. `pyproject` declares one dependency, Textual, for the dashboard;
   the `ui` extra is kept, empty, so pre-0.13 install commands still work.
2. **`mdl_ui.py` is the only file that imports a dependency** (Textual,
   pinned `>=3,<9` — a tested floor, not a guess). It installs with mdl
   since 0.13, but the CLI must never import it except inside the `tui`
   handler: `test_cli.py` checks that `--version`, `ps`, `list`, `check`
   and `--help` leave it unloaded. `mdl_web` is standard library, and
   its page loads nothing from the network: no CDN, no build step.
3. **`mdl_fit` and `mdl_update` are standard library too.** Keep them
   that way. `mdl_update` is reached lazily, like `mdl_fit`: by
   `mdl update`, `mdl doctor` and the dashboard, never by `run` or `ps`.
4. **Errors are one line on stderr and exit 1. Never a traceback.** Raise
   `mdl.MdlError` (or call `die()`); `main()` catches it and prints
   `mdl: <message>`. There is a CI job whose entire purpose is checking that
   Python 3.10 gets one polite line rather than a stack trace.
5. **Ruff: `E, F, W, B`, line length 88.** Deliberately *not* isort (the test
   files set `sys.path` before importing `mdl`, and reordering breaks them)
   and *not* pyupgrade (it would rewrite the `%` formatting for no behaviour
   change). Do not add them.
6. **Run `mdl` as an installed command or via `python mdl.py`** — the latter
   works only because of the `sys.modules.setdefault("mdl", ...)` line at the
   bottom of `mdl.py`. `mdl_fit` does `import mdl`; without that alias you
   get a *second* module object whose `MdlError` the handler cannot catch, and
   every error becomes a traceback. If you see a traceback where you expected
   one line, suspect a double import.
7. **No test may reach the real PyPI.** `tests/support.py` sets
   `MDL_NO_UPDATE_CHECK=1` for every suite; `test_update.py` turns it back
   on against a fake index (`MDL_PYPI_URL`). A new suite that does not
   import `support` has to do the same. Nor GitHub (`MDL_CHANGELOG_URL`
   points at nothing), nor the developer's own models: `support` makes
   `scan.places()` return nothing, so `mdl setup` and an empty config's
   web page never list someone's LM Studio folder; `test_setup.py` gives
   it folders of its own.

## Style

The prose in this codebase is part of it. Comments explain *why*, in full
sentences, and are worth matching rather than stripping. Docstrings say what
a thing is for and what it refuses to do. `%`-formatting throughout. If a
comment states a number, that number was measured — do not adjust it to fit
a change without re-measuring.

## Releasing

Pushing a `v*` tag publishes to PyPI. There is no token: PyPI trusts the
workflow via OIDC.

```sh
# bump VERSION in mdl.py and add a CHANGELOG.md entry, commit, push main,
# wait for CI to pass, then
git tag -a v0.12.0 -m "0.12.0" && git push origin v0.12.0
```

### Versions

[Semantic Versioning 2.0.0](https://semver.org/): `MAJOR.MINOR.PATCH`,
counted from the last release, never from the last development build.
Bump `VERSION` once per release, in the release commit - not per change
(0.6.7 to 0.6.20 were numbered a patch at a time by mistake and never
shipped).

- **PATCH** (0.8.0 -> 0.8.1): backward-compatible bug fixes, and nothing
  else. Every change makes mdl do what it already claimed to do.
- **MINOR** (0.8.1 -> 0.9.0, PATCH resets to 0): new backward-compatible
  functionality - a command, a flag, a config key, a field in `--json`
  output - or existing behaviour a user can see doing something new or
  different, or something marked deprecated.
- **MAJOR** (0.9.0 -> 1.0.0, MINOR and PATCH reset to 0): an incompatible
  change to the public API.

Classify by what a user can now do or see, not by why it was done. A
"fix" that lets the dashboard set a value it could not, or makes
`mdl fit` hold back memory it did not and print a line saying so, is
new behaviour: MINOR. The release takes the highest bump any of its
changes needs - one feature among twenty fixes is MINOR.

The public API is what users and their scripts rely on: the commands and
their flags, `models.toml`'s keys and meaning, `--json` output, the exit
codes, the files mdl reads that people write (`models.toml`, the custom
eval tasks), and `mdl` on PATH as the command. Internal functions, test
helpers, messages meant for people, the layout of state and cache files,
and CI are not.

**While at 0.y.z** (initial development, SemVer item 4) the public API
may change at any time: a breaking change to it bumps MINOR, not MAJOR,
and its changelog entry starts with **Breaking:** (0.3.0 did this for
`mdl ps --json`). 1.0.0 is not a version bump; it declares the public API
stable, and only the maintainer makes that call. From 1.0.0 on, a breaking
change bumps MAJOR.

The changelog's sections name the bump: **Added**, **Changed**,
**Deprecated** or **Removed** mean at least MINOR; a release with only
**Fixed** and **Security** is a PATCH.

### Publishing

- The tag must equal `mdl.VERSION` or the build job fails on purpose. The
  version lives only in `mdl.py`; `pyproject` reads it from there.
- **This is irreversible.** PyPI will not accept a re-upload of a version, so
  a mistake ships as the next patch. Do not push a `v*` tag without the
  maintainer saying so explicitly.
- The package is `llama-mdl`; the command is `mdl`. Plain `mdl` on PyPI is
  someone else's project.
- `pyproject`'s `description` is what PyPI shows, so it goes stale invisibly.

CI on every push: the fast suites on Linux/macOS/Windows across Python
3.11–3.13, the pinned Textual floor and latest, ruff, the POSIX process
suite, the 3.10 rejection, and a wheel+sdist install check.

## Runtime paths

```
~/.config/mdl/models.toml     config
~/.config/mdl/eval-seed       the eval seed (see below)
~/.config/mdl/evals.jsonl     eval results
~/.config/mdl/hashes.json     model sha256s, kept against size and mtime
~/.local/state/mdl/           pids, ports, logs
~/.cache/mdl/catalog.sqlite   the pulled catalog snapshot
~/.cache/mdl/update.json      the daily update check, and a skipped version
```

`$XDG_CONFIG_HOME` / `$XDG_STATE_HOME` / `$XDG_CACHE_HOME` are honoured, and
`$MDL_FIT_HOME` moves everything `mdl_fit` keeps (cache included) under one
directory - the tests and the catalog workflow both use it; on Windows the same
layout sits under `%USERPROFILE%`. The fast tests run against a temp config
and a fake `llama-server`, so they never touch a real one — keep it that way
when adding tests.

## The UI, if you touch it

- **Copying goes through `_clip()`, never `copy_to_clipboard()` alone.**
  Textual's copy is an OSC 52 escape that nothing acknowledges; VTE
  terminals and tmux drop it silently. `_clip()` also runs `wl-copy`,
  `xclip`/`xsel` or `pbcopy`, and only claims "copied" when one of them
  took it.
- **Tests that patch `mdl_ui` globals must put them back.** `mdl_ui.shutil`
  and `mdl_ui.subprocess` are the real modules, so replacing `which` or
  `run` leaks into every later check in the run. One such leak hid on
  Windows and only failed on Linux.

## The eval, if you touch it

`mdl eval` generates its items from a seed secret to each install
(`~/.config/mdl/eval-seed`), so no model can have memorised these exact
questions (the templates and problem families are public - do not claim
more) and no two machines share a suite. Consequences worth holding on to:

- **Scores are not comparable across machines.** They are comparable across
  *models on one machine*, which is the point: the item set depends only on
  the seed and `SUITE_VERSION`, never on the model, so two entries in
  `models.toml` always answer identical questions. That is what makes
  `mdl eval --compare A B` a valid paired test — and the headline use, since
  it answers "is this model worse at 3 bits than at 4, here".
- **Every result carries an `items_hash`** fingerprinting the questions
  actually asked. Change a template and the hash changes and `--compare`
  refuses old runs rather than subtracting incomparable numbers. That is
  correct behaviour; do not weaken it.
- **Graders must never pay for a non-answer.** `test_eval.py` walks every
  item with replies containing no knowledge — nothing, a refusal, a guessed
  number, the instructions read back — and asserts none of them score. Four
  real holes were found this way (a "no tool fits" item paying full marks for
  "I don't know"; format rules paying for silence; long-context substring
  matching paying for a quoted slab of the document; the hidden-test count
  being forgeable by printing `passed 9 of 9`). If you add a grader, add it
  to that sweep.
- **Half the hard code and reason items are generated, not named.** The point
  is that recognising an item must not help — a model has read a thousand
  solutions to "longest increasing subsequence". If you add a hard item,
  prefer one whose constants, rules and names come from the seed.
- **Model-written code executes.** In a throwaway podman/docker container
  with no network whenever one answers; otherwise, or with `--no-sandbox`, a
  subprocess in a temp dir with a timeout - not a sandbox, and the run says
  so. The expected answers stay in the grading process. Do not loosen this.

## Catalog

The nightly hub crawl (`.github/workflows/catalog.yml`) is gated behind the
repo variable `CATALOG_ENABLED == "true"` and does nothing on forks. It is
**on** (since 2026-09-17), running daily at 03:17 UTC and publishing to the
`CATALOG_REPO` dataset (`diversemate/mdl-catalog`). Do not turn it off,
change its scope or dispatch it without the maintainer's approval; manual
dispatch bypasses the enable flag but still requires the owner check.

The default builder (`mdl_fit/catalog_crawl.py`) takes 3,000 GGUF repos by
downloads plus 500 by creation date, deduplicates them, and fetches only
file lists and direct source-model metadata. Its 40-minute budget saves a
partial with the work queue in SQLite; the workflow publishes that file.
Data and cursors commit per HTTP page. Do not prune a partial refresh or
publish an incomplete shard inventory. Explicit --org/--base builds still
use the old, unbudgeted lineage traversal in catalog.py.

Known sharp edge: the Hugging Face API **ignores `other=`** on the
quantized-children query — both conditions have to go in `filter=`. And
card fields arrive as lists or dicts, which SQLite will not bind; `_text()`
coerces them. A crawl died 50 nodes in over exactly that.

GGUF repos also hold files that are not quants: vision projectors, LoRA
adapters, speculative drafts (`-draft-`, `DFlash`) and MTP heads (`mtp-…`,
`MTP/`). `remote.auxiliary()` drops them by name, because size cannot - a
real Q1_0 is as small per weight as a draft. `-mtp` as a *suffix* is a whole
model with its MTP layers, and stays.

## Updating, if you touch it

`mdl update` upgrades through whatever installed mdl (pipx, uv tool, pip),
pinned to the version it offered where the tool allows, and believes a
fresh interpreter rather than the installer's exit status. Measured on
Windows before it was built: pip and pipx replace a running `mdl.exe`;
`uv tool upgrade` fails with os error 32, so the running launcher is
renamed aside first (Windows allows that) and put back if the install
fails. `uv tool upgrade` also honours a version pinned at install time,
which is why a no-op upgrade says the install may be pinned.

The dashboard restarts through `mdl._after_ui` once Textual has handed
the terminal back: `os.execve` on POSIX, a waited-on child on Windows
(where exec is not a real replace), always `python -P -m mdl` so a stray
`mdl.py` in the working directory is never the one that starts.

## Non-goals

Stated in the README and enforced in review: no daemon and no hot-swap.
(`mdl pull` and `mdl ui` are features: the non-goals once ruled out
downloading weights and a web UI, and no longer do.) `mdl eval` is the one
carve-out on hot-swap, and a narrow one — it starts a server for the run and stops it
after, but only one it started itself; an already-running server is used as
it stands and left up. Read the README's non-goals before implementing
anything that sounds like model management.

## Before you say you are done

```sh
python tests/run.py && ruff check .
```

Report what actually happened. If a suite fails, say so with the output; if
you skipped something, say that. `--live` failing is not the same as the
gate failing, and a stale result from another worktree is not a result.
