# CLAUDE.md

**Read [AGENTS.md](AGENTS.md) first — it is the canonical guide.** Layout,
the hard rules, how to run the tests, the release process, and the traps in
the eval and the catalog all live there.

Two copies of the same guidance drift apart, and a contradictory doc is
worse than none: this file carries only what is specific to Claude Code.

## The short version

```sh
python tests/run.py && ruff check .      # the gate, before you claim done
```

`mdl.py` is standard library only and imports `mdl_fit` lazily. `mdl_ui.py`
is the only file allowed a dependency. Errors are one line on stderr, never
a traceback. Ruff is `E, F, W, B` at line length 88 — no isort, no pyupgrade.

## Things worth knowing here

**This repo is used through git worktrees.** Check `git worktree list` and
`git status` before you start, and again before you commit. More than one
agent has been in this tree at once: if files you did not touch are modified,
they are someone else's in-flight work. Stage your own paths explicitly
(`git add <file>`), never `git add -A`, and say so rather than sweeping them
into your commit.

**Never use bare `git stash` / `git stash pop`.** The stash stack is shared
across every worktree, so you can pop someone else's work. Prefer a WIP
commit. If you must stash, use `git stash push -u -m "<unique-tag>"`, capture
the SHA from `git stash list --format='%H %gs'`, restore with
`git stash apply <sha>`, and drop it by re-finding it by tag.

**Do not push a `v*` tag.** It publishes to PyPI, and PyPI will not take a
re-upload of a version — a mistake ships as the next patch. Wait for the
maintainer to say so in as many words.

**Heredocs eat backslashes.** Writing Python patch scripts through
`cat <<'PY'` in the Bash tool silently collapses `\\n` to `\n`, which turns a
string literal into a real newline and breaks the file. If a patch script's
`assert old in s` fails for no visible reason, that is why. Use the Write
tool for anything containing escapes or regexes.

**`python tests/run.py` is the gate, not the individual files.** `test_live`
needs a real GPU and real models, is not in CI, and can leave a server
running — check `mdl ps` if it fails. A `--live` failure is information, not
a broken build.

**Claims in this codebase are measured, not estimated.** If you are about to
write a number into a comment, a docstring or the README, go and read it out
of the source. "A few kilobytes" became "the first few megabytes" that way,
because `remote.py` has `FIRST = 4 << 20`.

## When the eval is involved

Scores are per-machine by design — the seed is secret to the install. They
compare *models on one machine*, which is the whole point. Anything that
changes a suite's items changes its `items_hash`, and `--compare` then
refuses older runs instead of subtracting incomparable numbers; that is
correct, so do not weaken it. If you add a grader, add it to the adversarial
sweep in `test_eval.py` that asserts no reply containing zero knowledge ever
scores. See AGENTS.md for the rest.
