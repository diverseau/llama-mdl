# AGENTS.md

Working notes for coding agents in the `mdl` repo. This is the canonical
file; `CLAUDE.md` points here and adds only Claude Code specifics.

Written against **0.6.2**. Where a fact is likely to drift, this says how to
re-derive it instead of quoting it.

## What this is

`mdl` runs local [llama.cpp](https://github.com/ggml-org/llama.cpp) servers
from a config file, and works out what to run and how. Two halves:

- **Running servers** — `run`, `stop`, `ps`, `list`, `add`, `check`, `init`,
  `logs`, `ui`. All of this is `mdl.py` alone.
- **Deciding what to run** — `fit` (what a GGUF will do on this machine and
  with which flags), `eval` (score it on a private graded suite), `catalog`
  and `find` (what is worth running at all). All of this is `mdl_fit/`.

## Layout

```
mdl.py            the whole CLI for running servers. Standard library only.
mdl_ui.py         optional Textual dashboard (`mdl ui`). The only dependency.
mdl_fit/          everything behind fit / eval / catalog / find
tests/            run.py drives the suites; support.py + fake_llama_server.py
docs/fit-plan.md  the design notes for mdl fit
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
`remote` reads headers of models you have not downloaded, `catalog` crawls
the hub, `quality`/`find` rank models, `evalsuite`/`evalrun` are the eval.

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
   level.** It reaches `mdl_fit` lazily, inside the four handlers that need
   it, so the everyday commands load nothing extra. `pyproject` declares
   `dependencies = []`; only the `ui` extra has one.
2. **`mdl_ui.py` is the only file allowed a dependency** (Textual, pinned
   `>=3,<9` — a tested floor, not a guess). The CLI must never import it
   except inside the `ui` handler.
3. **`mdl_fit` is standard library too.** Keep it that way.
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
git tag -a v0.6.2 -m "0.6.2" && git push origin v0.6.2
```

- The tag must equal `mdl.VERSION` or the build job fails on purpose.
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
~/.local/state/mdl/           pids, ports, logs
```

`$XDG_CONFIG_HOME` / `$XDG_STATE_HOME` are honoured; on Windows the same
layout sits under `%USERPROFILE%`. The fast tests run against a temp config
and a fake `llama-server`, so they never touch a real one — keep it that way
when adding tests.

## The eval, if you touch it

`mdl eval` generates its items from a seed secret to each install
(`~/.config/mdl/eval-seed`), so no model can have trained on them and no two
machines share a suite. Consequences worth holding on to:

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
- **Model-written code executes.** A subprocess in a temp dir with a timeout
  by default; `--sandbox` uses a throwaway podman/docker container with no
  network. Do not loosen this.

## Catalog

The nightly hub crawl (`.github/workflows/catalog.yml`) is gated behind the
repo variable `CATALOG_ENABLED == "true"` and does nothing on forks. Leave
it off unless the maintainer asks. Manual dispatch bypasses the schedule's
enable flag, but still requires the owner check; do not dispatch without
the maintainer's approval.

The default builder (`mdl_fit/catalog_crawl.py`) takes 800 GGUF repos by
downloads plus 200 by creation date, deduplicates them, and fetches only
file lists and direct source-model metadata. Its 40-minute budget saves a
partial with the work queue in SQLite; the workflow publishes that file.
Data and cursors commit per HTTP page. Do not prune a partial refresh or
publish an incomplete shard inventory. Explicit --org/--base builds still
use the old, unbudgeted lineage traversal in catalog.py.

Known sharp edge: the Hugging Face API **ignores `other=`** on the
quantized-children query — both conditions have to go in `filter=`. And
card fields arrive as lists or dicts, which SQLite will not bind; `_text()`
coerces them. A crawl died 50 nodes in over exactly that.

## Non-goals

Stated in the README and enforced in review: no daemon, no downloading model
weights, no hot-swap, no web UI. `mdl eval` is the one carve-out on
hot-swap, and a narrow one — it starts a server for the run and stops it
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
