"""Bars and spinners: what they say, on a terminal and off one."""
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                                 # noqa: E402

from mdl_fit import progress                                   # noqa: E402

t = support.Tally("test_progress")
check = t.check
MB, GB = 1 << 20, 1 << 30


class Term(io.StringIO):
    """A stream that says it is a terminal."""

    def __init__(self, encoding="utf-8"):
        super().__init__()
        self._enc = encoding

    @property
    def encoding(self):
        return self._enc

    def isatty(self):
        return True


check("sizes: MB under a gigabyte, GB with a decimal over",
      [progress.size_words(n) for n in (0, 300 << 10, 397 * MB, int(1.25 * GB))],
      ["0 MB", "300 KB", "397 MB", "1.2 GB"])
check("done and total in the total's unit",
      [progress.pair_words(176 * MB, 397 * MB),
       progress.pair_words(GB // 2, 5 * GB)], ["176/397 MB", "0.5/5.0 GB"])
check("times", [progress.time_words(s) for s in (7, 125, 3720)],
      ["7s", "2m05s", "1h02m"])

# on a terminal: one line, redrawn, with speed and time left
term = Term()
bar = progress.Bar("Qwen3-0.6B-Q4_K_M.gguf", 400 * MB, stream=term)
now = time.monotonic()
bar.samples.clear()
bar.samples.extend([(now - 4, 0), (now, 160 * MB)])
bar.done = 160 * MB
line = bar.render()
check("the bar says how far, how fast and how long",
      ("Qwen3-0.6B-Q4_K_M.gguf" in line, " 40%" in line,
       "160/400 MB · 40 MB/s · 6s left" in line), (True, True, True))
check("and fits the terminal", len(line) <= bar.line.width(), True)
bar.update(400 * MB)
check("drawn in place, not line by line", ("\r" in term.getvalue(),
                                           "\n" in term.getvalue()),
      (True, False))
bar.finish("400 MB in 10s")
check("finish leaves the summary on a line of its own",
      term.getvalue().rstrip().endswith("400 MB in 10s"), True)

# a cp1252 stream gets characters it can print
old = Term("cp1252")
bar = progress.Bar("x", 10, stream=old)
bar.update(5)
check("no block characters where they cannot be encoded",
      ("#" in old.getvalue(), "█" in old.getvalue()), (True, False))

# off a terminal: a plain line every tenth, never a carriage return
pipe = io.StringIO()
bar = progress.Bar("model.gguf", 100 * MB, stream=pipe)
for i in range(0, 101):
    bar.update(i * MB)
lines = pipe.getvalue().splitlines()
check("a pipe gets eleven plain lines, 0% to 100%, and no redraws",
      (len(lines), "\r" in pipe.getvalue(), lines[-1].split()[1]),
      (11, False, "100%"))

# a spinner off a terminal says its label once and nothing else
pipe = io.StringIO()
with progress.Spinner("fitting to this machine", stream=pipe):
    time.sleep(0.25)
check("a spinner in a pipe: its label, once",
      pipe.getvalue(), "fitting to this machine...\n")
term = Term()
with progress.Spinner("asking the Hub", stream=term) as spin:
    time.sleep(0.35)
    spin.print("picked Q6_K")
    time.sleep(0.25)
out = term.getvalue()
check("on a terminal it turns, a line printed over it stays, and it "
      "clears itself at the end",
      ("asking the Hub" in out, "picked Q6_K\n" in out, out.endswith("\r")),
      (True, True, True))

sys.exit(t.done())
