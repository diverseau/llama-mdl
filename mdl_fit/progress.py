"""Progress on a terminal: a bar for work with a size, a spinner for work
without one. Standard library only, and it writes to stderr, so stdout
stays what a script reads.

On a terminal a line is redrawn in place, at most ten times a second.
Anywhere else - a pipe, a CI log, a file - redrawing makes a mess, so a
bar prints a plain line every tenth of the way and a spinner prints its
label once. A stream that cannot encode the block characters (cp1252,
piped on Windows) gets # and - instead.

The numbers are for people: sizes in MB below a gigabyte and in GB with
one decimal above, a speed averaged over the last five seconds rather
than since the start (a download that stalled and recovered should say
how fast it is going now), and an ETA from that speed.
"""

import shutil
import sys
import threading
import time
from collections import deque

REDRAW_S = 0.1           # a terminal line at most this often
WINDOW_S = 5.0           # the speed is averaged over this much of the past
SPIN = "|/-\\"


def size_words(n):
    """397 MB, 1.2 GB, 512 KB."""
    n = max(0, int(n))
    if n >= 1 << 30:
        return "%.1f GB" % (n / (1 << 30))
    if n >= 1 << 20:
        return "%d MB" % (n >> 20)
    return "%d KB" % max(1, n >> 10) if n else "0 MB"


def pair_words(done, total):
    """176/397 MB, 0.4/1.2 GB: the two in one unit, the total's."""
    if total >= 1 << 30:
        return "%.1f/%.1f GB" % (done / (1 << 30), total / (1 << 30))
    if total >= 1 << 20:
        return "%d/%d MB" % (done >> 20, total >> 20)
    return "%d/%d KB" % (done >> 10, max(1, total >> 10))


def time_words(seconds):
    """7s, 2m05s, 1h02m."""
    s = int(round(max(0, seconds)))
    if s >= 3600:
        return "%dh%02dm" % (s // 3600, s % 3600 // 60)
    if s >= 60:
        return "%dm%02ds" % (s // 60, s % 60)
    return "%ds" % s


def _tty(stream):
    try:
        return stream.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def _blocks(stream):
    enc = getattr(stream, "encoding", None) or "ascii"
    try:
        "█░".encode(enc)
        return "█", "░"
    except (UnicodeError, LookupError):
        return "#", "-"


class Line:
    """One line of stderr that can be redrawn, then cleared or kept."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self.tty = _tty(self.stream)
        self.shown = 0            # characters on the line now
        self.lock = threading.RLock()

    def width(self):
        return max(40, shutil.get_terminal_size((80, 24)).columns - 1)

    def draw(self, text):
        """Redraw in place (a terminal only)."""
        with self.lock:
            text = text[:self.width()]
            pad = max(0, self.shown - len(text))
            self.stream.write("\r" + text + " " * pad)
            self.stream.flush()
            self.shown = len(text)

    def clear(self):
        with self.lock:
            if self.tty and self.shown:
                self.stream.write("\r" + " " * self.shown + "\r")
                self.stream.flush()
            self.shown = 0

    def print(self, text):
        """A line of its own, above whatever is being redrawn."""
        with self.lock:
            self.clear()
            self.stream.write(text + "\n")
            self.stream.flush()


class Bar:
    """A bar for `total` of something, bytes unless said otherwise.

        bar = Bar("Qwen3-8B-Q4_K_M.gguf", total)
        bar.update(done)            # as often as you like
        bar.finish("5.0 GB in 2m10s")
    """

    def __init__(self, label, total, stream=None, unit="bytes", line=None):
        self.label, self.total, self.unit = label, max(1, int(total)), unit
        self.line = line or Line(stream)
        self.full, self.empty = _blocks(self.line.stream)
        self.done, self.started = 0, time.monotonic()
        self.samples = deque([(self.started, 0)])
        self.drawn_at, self.step = 0.0, -1

    # what it says, also for others to show (the web UI's card)
    def rate(self):
        """Units a second over the last WINDOW_S; None until it can say."""
        t0, d0 = self.samples[0]
        t1, d1 = self.samples[-1]
        if t1 - t0 < 0.5:
            return None
        return (d1 - d0) / (t1 - t0)

    def eta(self):
        r = self.rate()
        if not r or r <= 0:
            return None
        return (self.total - self.done) / r

    def percent(self):
        return min(100, self.done * 100 // self.total)

    def words(self):
        """'176/397 MB · 31 MB/s · 7s left'."""
        if self.unit != "bytes":
            return "%d/%d %s" % (self.done, self.total, self.unit)
        parts = [pair_words(self.done, self.total)]
        r = self.rate()
        if r:
            parts.append(size_words(r) + "/s")
        e = self.eta()
        if e is not None and self.done < self.total:
            parts.append(time_words(e) + " left")
        return " · ".join(parts)

    def update(self, done, label=None):
        now = time.monotonic()
        self.done = min(int(done), self.total)
        if label:
            self.label = label
        self.samples.append((now, self.done))
        while len(self.samples) > 2 and now - self.samples[0][0] > WINDOW_S:
            self.samples.popleft()
        if self.line.tty:
            if now - self.drawn_at >= REDRAW_S or self.done >= self.total:
                self.drawn_at = now
                self.line.draw(self.render())
        else:
            step = self.percent() // 10
            if step > self.step:
                self.step = step
                self.line.stream.write("%s  %3d%%  %s\n" % (
                    self.label, self.percent(), self.words()))
                self.line.stream.flush()

    def render(self):
        tail = "  %3d%%  %s" % (self.percent(), self.words())
        room = self.line.width() - len(tail) - 2
        width = max(10, min(30, room - min(len(self.label), 36)))
        label = self.label
        if len(label) > room - width:
            label = label[:max(8, room - width - 1)] + "…"
        filled = width * self.done // self.total
        return "%s  %s%s%s" % (label, self.full * filled,
                               self.empty * (width - filled), tail)

    def finish(self, summary=None):
        """Take the bar off the line; leave `summary` in its place."""
        if summary:
            self.line.print(summary)
        else:
            self.line.clear()

    def elapsed(self):
        return time.monotonic() - self.started


class Spinner:
    """For a wait with no size: a label and the seconds so far, turning.

        with Spinner("fitting to this machine") as spin:
            spin.label = "fitting to this machine: 3 of 12"   # any time
            ...
    """

    def __init__(self, label, stream=None, line=None):
        self.label = label
        self.line = line or Line(stream)
        self.started = time.monotonic()
        self.stop_event = threading.Event()
        self.thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def start(self):
        if self.line.tty:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        else:
            self.line.stream.write(self.label + "...\n")
            self.line.stream.flush()
        return self

    def _run(self):
        i = 0
        while not self.stop_event.wait(REDRAW_S):
            secs = time.monotonic() - self.started
            self.line.draw("%s %s%s" % (
                SPIN[i % len(SPIN)], self.label,
                "  %s" % time_words(secs) if secs >= 2 else ""))
            i += 1

    def print(self, text):
        """A line of its own, the spinner carrying on under it."""
        self.line.print(text)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.line.clear()
