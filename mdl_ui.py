"""mdl ui - a terminal dashboard for the mdl llama.cpp server runner.

Kept deliberately separate from mdl.py: the CLI stays dependency-free and
quiet, this file is allowed to be pretty. Both drive the same functions and
the same state file, so they can never disagree about what is running.
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

import mdl
import mdl_update

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (Button, DataTable, Footer, Input, Label, RichLog,
                             Static)

POLL_SECONDS = 1.0
SPARK_POINTS = 40

# Wordmark animation. "always" drifts slowly forever, "sweep" runs one pass
# at launch then parks, "off" paints it flat. Set ui_fx in models.toml,
# $MDL_UI_FX, or pass --no-fx.
FX_DEFAULT = "always"
FX_TICK = 0.06          # ~16 fps
FX_PERIOD = 3.0         # seconds per colour cycle; ui_fx_period overrides
FX_SWEEP_SECONDS = 1.2  # how long the one-shot "sweep" pass takes
FX_MIN_PERIOD = 0.4     # below this it strobes rather than drifts

# ANSI Shadow. Drawn once at launch, then parked in the header.
WORDMARK = r"""
 ███╗   ███╗ ██████╗  ██╗
 ████╗ ████║ ██╔══██╗ ██║
 ██╔████╔██║ ██║  ██║ ██║
 ██║╚██╔╝██║ ██║  ██║ ██║
 ██║ ╚═╝ ██║ ██████╔╝ ███████╗
 ╚═╝     ╚═╝ ╚═════╝  ╚══════╝
""".strip("\n")

GRADIENT = ["#7dcfff", "#7aa2f7", "#8a7af7", "#9d7cf7", "#bb7af7"]
WEIGHTS_COLOUR, KV_COLOUR = "#7aa2f7", "#bb7af7"
SPARK_CHARS = " ▁▂▃▄▅▆▇█"
NAME_WIDTH = 15          # keeps the size and ctx columns on screen
GROUP_KEY = chr(0) + "g:"   # row keys a model name can never take
NEWLINE = chr(10)


def fx_period(override=None):
    """Seconds per colour cycle. --fx-period beats $MDL_UI_FX_PERIOD
    beats ui_fx_period in the config."""
    raw = override or os.environ.get("MDL_UI_FX_PERIOD")
    if raw is None:
        try:
            mdl.load_config()
        except mdl.MdlError:
            pass
        raw = (mdl.CONFIG_DATA or {}).get("ui_fx_period")
    try:
        return max(FX_MIN_PERIOD, float(raw))
    except (TypeError, ValueError):
        return FX_PERIOD


def fx_mode(override=None):
    """--no-fx beats $MDL_UI_FX beats ui_fx in the config."""
    if override:
        return str(override).strip().lower()
    env = os.environ.get("MDL_UI_FX")
    if env:
        return env.strip().lower()
    try:
        mdl.load_config()
    except mdl.MdlError:
        pass
    value = (mdl.CONFIG_DATA or {}).get("ui_fx")
    return str(value).strip().lower() if value else FX_DEFAULT


def _rgb(hex_colour):
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def colour_at(pos, colors=GRADIENT):
    """Colour at 0..1 through `colors`, wrapping so it can drift seamlessly."""
    pos = pos % 1.0
    span = pos * len(colors)
    i = int(span)
    a = _rgb(colors[i % len(colors)])
    b = _rgb(colors[(i + 1) % len(colors)])
    t = span - i
    return "#%02x%02x%02x" % tuple(
        round(a[k] + (b[k] - a[k]) * t) for k in range(3))


def gradient_text(lines, colors=GRADIENT, phase=0.0):
    """Colour a block of text left-to-right, offset by `phase` (0..1)."""
    width = max((len(line) for line in lines), default=1)
    out = Text()
    for row, line in enumerate(lines):
        for col, ch in enumerate(line):
            if ch.strip():
                out.append(ch, style=colour_at(
                    col / max(width - 1, 1) + phase, colors))
            else:
                out.append(ch)
        if row != len(lines) - 1:
            out.append("\n")
    return out


def sparkline(values, width=SPARK_POINTS):
    """Unicode sparkline over the last `width` values."""
    pts = list(values)[-width:]
    if not pts:
        return " " * width
    hi = max(pts) or 1.0
    scaled = [SPARK_CHARS[min(int(v / hi * (len(SPARK_CHARS) - 1)),
              len(SPARK_CHARS) - 1)]
              for v in pts]
    return "".join(scaled).rjust(width)


def bar(fraction, width=20, fill="█", empty="░"):
    fraction = 0.0 if fraction is None else max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return fill * filled + empty * (width - filled)


def stacked_bar(parts, capacity, width=18, empty="░"):
    """One bar, several coloured runs, scaled to capacity.

    Past capacity it scales to the total instead, so the parts stay
    visible: once a bar is full, how full stopped being the question
    and what is in it started being one. Rounds each run up, because a
    part that is present should occupy at least a cell.
    """
    t = Text()
    capacity = max(capacity, sum(size for size, _ in parts))
    used = 0
    for size, colour in parts:
        cells = min(width - used, max(1, round(size / max(capacity, 1) * width))
                    if size > 0 else 0)
        if cells > 0:
            t.append("█" * cells, style=colour)
            used += cells
        if used >= width:
            break
    t.append(empty * (width - used), style="#1f2430")
    return t


def bar_colour(fraction):
    if fraction is None:
        return "#565f89"
    if fraction >= 0.92:
        return "#f7768e"
    if fraction >= 0.75:
        return "#e0af68"
    return "#9ece6a"


# --------------------------------------------------------------------------
# telemetry. Every one of these degrades to None rather than raising: a panel
# going grey is always better than the dashboard falling over.
# --------------------------------------------------------------------------

def http_get(port, path, timeout=1.5):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                    timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def http_json(port, path, timeout=1.5):
    raw = http_get(port, path, timeout)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


METRIC_LINE = re.compile(r"^([a-zA-Z_:][\w:]*)\s+([0-9.eE+-]+)$")


def parse_metrics(raw):
    """Prometheus text exposition -> {name: float}. Comments and labels ignored."""
    out = {}
    if not raw:
        return out
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = METRIC_LINE.match(line)
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


GPU_TTL = 3.0
_GPU_CACHE = {"at": -1e9, "value": None}


def gpu_memory():
    """(used_mib, total_mib) across every card, or None. Cached for GPU_TTL.

    Summed, because a box with two cards has two lines here and reading
    only the first reported the wrong card's memory as the whole system's.
    """
    now = time.monotonic()
    if now - _GPU_CACHE["at"] < GPU_TTL:
        return _GPU_CACHE["value"]
    _GPU_CACHE["at"] = now
    exe = shutil.which("nvidia-smi")
    if not exe:
        _GPU_CACHE["value"] = None
        return None
    try:
        raw = subprocess.run(
            [exe, "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        used = total = 0
        for line in raw.strip().splitlines():
            u, t = line.split(",")
            used, total = used + int(u), total + int(t)
        _GPU_CACHE["value"] = (used, total) if total else None
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        _GPU_CACHE["value"] = None
    return _GPU_CACHE["value"]


LAYER_RE = re.compile(r"offloaded (\d+)/(\d+) layers")
LOADING_RE = re.compile(r"load_tensors|llama_model_loader|loading model")


def log_level(line):
    """Colour key for a llama.cpp log line, across both log formats."""
    if re.search(r"\b(error|failed|cannot|unsupported|missing tensor)\b", line, re.I):
        return "error"
    if re.search(r"\bW\b|warn|NOTICE|deprecat", line):
        return "warn"
    if mdl.READY.search(line):
        return "ready"
    if re.search(r"slot (launch|release|update)|prompt (eval|done)", line):
        return "slot"
    return "info"


# --------------------------------------------------------------------------
# widgets
# --------------------------------------------------------------------------

class Wordmark(Static):
    """The big MDL. Sweeps its gradient once on mount, then sits still."""

    def on_mount(self):
        self.lines = WORDMARK.split("\n")
        self._phase = 0.0
        self._step = FX_TICK / FX_PERIOD
        self.update(gradient_text(self.lines))
        mode = getattr(self.app, "fx", FX_DEFAULT)
        if mode == "off":
            return
        if mode == "sweep":
            self._step = FX_TICK / FX_SWEEP_SECONDS
            self.set_interval(FX_TICK, self._drift,
                              repeat=round(FX_SWEEP_SECONDS / FX_TICK))
        else:
            period = getattr(self.app, "fx_period", FX_PERIOD)
            self._step = FX_TICK / period
            self.set_interval(FX_TICK, self._drift)

    def _drift(self):
        self._phase = (self._phase + self._step) % 1.0
        self.update(gradient_text(self.lines, phase=self._phase))


class Meter(Static):
    """One labelled bar: `label  ██████░░░░  value`."""

    def set_value(self, label, fraction, caption, width=22):
        colour = bar_colour(fraction)
        t = Text()
        t.append(f"{label:<7}", style="#565f89")
        t.append(bar(fraction, width), style=colour)
        t.append(f"  {caption}", style="#c0caf5")
        self.update(t)


class Spark(Static):
    """Sparkline plus current/peak readout."""

    def set_series(self, label, values, unit="", peak=None):
        t = Text()
        t.append(f"{label:<7}", style="#565f89")
        t.append(sparkline(values), style="#7dcfff")
        if values:
            # peak is the server's lifetime high-water mark. max(values)
            # would be the best of the last 40 samples, which decays to
            # zero a matter of seconds after a reply finishes.
            if peak is None:
                peak = max(values)
            t.append(f"  {values[-1]:.1f}{unit} now", style="#c0caf5")
            t.append(f" · {peak:.1f}{unit} peak", style="#565f89")
        else:
            t.append("  no data yet", style="#565f89")
        self.update(t)


def system_clipboard(text):
    """Put text on the desktop clipboard with the platform's own tool.

    Returns the tool's name, or None when there was none to use. Textual
    copies by OSC 52 alone - an escape sequence the terminal may or may
    not act on, with nothing coming back either way. GNOME Terminal and
    the other VTE terminals ignore it, tmux drops it unless set-clipboard
    is on, and many cap its length well short of a server log, so on
    Linux the app said "log copied" and nothing was.

    Windows is left to OSC 52, which Windows Terminal honours: clip.exe
    reads the console code page and mangles anything outside it.
    """
    if os.name == "nt":
        return None
    if sys.platform == "darwin":
        tools = [["pbcopy"]]
    else:
        tools = []
        if os.environ.get("WAYLAND_DISPLAY"):
            tools.append(["wl-copy"])
        if os.environ.get("DISPLAY"):
            tools += [["xclip", "-selection", "clipboard"],
                      ["xsel", "--clipboard", "--input"]]
    for argv in tools:
        exe = shutil.which(argv[0])
        if not exe:
            continue
        # xclip and wl-copy stay behind to serve the selection; with
        # their output on DEVNULL, only the parent is waited for
        try:
            done = subprocess.run([exe, *argv[1:]], input=text.encode("utf-8"),
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if done.returncode == 0:
            return argv[0]
    return None


class CopyableLog(RichLog):
    """A RichLog you can get the text out of.

    Widget.get_selection only knows how to read a widget that renders
    to a single Text, so every scrolling widget in textual - this one,
    Log, DataTable - contributes nothing to a selection and copies as
    nothing. The lines are right here; hand them over.

    A drag marks the whole widget rather than a span, because textual
    cannot map a coordinate into rendered strips, so this copies the
    whole buffer. Whole log or no log is the honest choice available.
    """

    def get_selection(self, selection):
        return selection.extract(self.as_text()), NEWLINE

    def as_text(self):
        return NEWLINE.join(strip.text.rstrip() for strip in self.lines)


class ArgvPreview(VerticalScroll):
    """The exact llama-server command mdl would run.

    Scrolls, because a long command must never be silently truncated -
    a half-shown command reads as a differently-configured one.
    """

    WRAP = 58

    def compose(self) -> ComposeResult:
        yield Static(id="argv-body")

    @property
    def text(self):
        rendered = self.query_one("#argv-body", Static).render()
        return rendered.plain if hasattr(rendered, "plain") else str(rendered)

    @staticmethod
    def _pairs(argv):
        """[-ngl, 99, -fa, on] -> [(-ngl, 99), (-fa, on)]."""
        out, i = [], 1
        while i < len(argv):
            flag = argv[i]
            if (flag.startswith("-") and i + 1 < len(argv)
                    and not argv[i + 1].startswith("-")):
                out.append((flag, argv[i + 1]))
                i += 2
            else:
                out.append((flag, None))
                i += 1
        return out

    def set_argv(self, argv):
        body = self.query_one("#argv-body", Static)
        if not argv:
            body.update(Text("(select a model)", style="#565f89"))
            return
        t = Text()
        t.append("$ ", style="#565f89")
        t.append(Path(argv[0]).name, style="bold #9ece6a")
        used = self.WRAP + 1          # force a break before the first flag
        for flag, value in self._pairs(argv):
            shown = value
            if value and len(value) > 44:
                shown = ".../" + Path(value).name
            width = len(flag) + (len(shown) + 1 if shown else 0) + 2
            if used + width > self.WRAP or flag == "-m":
                t.append(NEWLINE + "    ")
                used = 4
            t.append(flag, style="#7aa2f7")
            used += len(flag)
            if shown:
                t.append(" " + shown, style="#e0af68")
                used += len(shown) + 1
            t.append("  ")
            used += 2
        body.update(t)


def vram_estimate(argv, size_bytes, mmproj_bytes=0):
    """Rough (weights, kv, partial) in MB. Enough to warn, not to trust.

    Read off the command rather than the config, because the same flag
    can arrive either way: --cache-type-k in args has to count for as
    much as kv_type does, and both land here.

    `partial` says the weights figure is an upper bound: some flag keeps
    part of the model off the GPU, and how much cannot be known without
    reading the tensor table and replaying llama.cpp's placement rules.
    """
    ctx, quantised, partial = 4096, False, False   # 4096 is llama.cpp's default
    for i, flag in enumerate(argv):
        value = argv[i + 1] if i + 1 < len(argv) else ""
        if flag in ("-c", "--ctx-size"):
            try:
                ctx = int(value)
            except ValueError:
                pass
        elif flag in ("-ctk", "--cache-type-k"):
            quantised = value not in ("f16", "f32", "")
        elif flag in ("--n-cpu-moe", "-ncmoe", "--cpu-moe", "-cmoe",
                      "-ot", "--override-tensor"):
            partial = True
        elif flag in ("-ngl", "--n-gpu-layers", "--gpu-layers"):
            try:
                partial = partial or int(value) < 99
            except ValueError:
                pass
    # The projector loads onto the card with everything else, and on a
    # small one a gigabyte of it is not a rounding error.
    return (((size_bytes or 0) + (mmproj_bytes or 0)) / (1 << 20),
            ctx / 1024 * (32 if quantised else 64), partial)


COMPUTE_COLOUR = "#e0af68"
_SHAPES = {}          # (path, size, mtime) -> mdl_fit Shape; parsing is 0.3 s
_FIT_ENV = {}


def fit_placement(argv):
    """(Memory, Speed, calibrated?, Shape) from mdl fit for this command.

    None when the model is not a GGUF the engine can read, and the pane
    falls back to the old estimate. Shapes are cached, because this runs
    every time the selection moves.
    """
    try:
        from mdl_fit import calib, gguf, hw, model, perf
    except ImportError:
        return None
    flags, _, path, mmproj = model.parse_argv(argv)
    if not path:
        return None
    try:
        st = Path(path).stat()
        key = (path, st.st_size, st.st_mtime)
        if key not in _SHAPES:
            _SHAPES[key] = model.Shape(gguf.load(path))
        shape = _SHAPES[key]
        if mmproj and Path(mmproj).is_file():
            flags.mmproj = Path(mmproj).stat().st_size
    except (OSError, ValueError, KeyError, IndexError, gguf.Truncated):
        return None
    if "res" not in _FIT_ENV:
        _FIT_ENV["res"] = calib.Residuals()
    # per build: a model can name its own llama_server, and the first
    # backend booked for any build - CUDA if none was - is not this one's
    binary = argv[0] if argv else "llama-server"
    if ("mach", binary) not in _FIT_ENV:
        saved = hw.load_saved()
        _FIT_ENV["mach", binary] = hw.Machine(
            bench=saved.get("bench", {}),
            backend=hw.backend_for(binary, saved))
    mach = _FIT_ENV["mach", binary]
    residual = _FIT_ENV["res"].lookup(calib.signature(shape.inv),
                                      shape.arch, flags)
    mem = model.memory(shape, flags, None, mach.max_alloc, residual)
    speed = perf.speed(shape, flags, mach, depth=8192)
    return mem, speed, mach.calibrated, shape


def placement_text(placed, total_mib):
    """The real placement, replacing the est-VRAM guess: what sits on the
    card, what stays on the CPU, and how fast it should decode."""
    mem, speed, calibrated, shape = placed
    gib, mib = 1 << 30, 1 << 20
    v = Text()
    v.append("VRAM      ", style="#565f89")
    v.append(stacked_bar([(mem.gpu_weights / mib, WEIGHTS_COLOUR),
                          ((mem.gpu_kv + mem.gpu_rs) / mib, KV_COLOUR),
                          ((mem.gpu_compute + mem.gpu_mmproj) / mib,
                           COMPUTE_COLOUR)], total_mib))
    over = mem.gpu / mib > total_mib
    v.append("  %.1f / %.1f G" % (mem.gpu / gib, total_mib / 1024),
             style="#f7768e" if over else "#c0caf5")
    v.append(NEWLINE + " " * 10)
    v.append("weights", style=WEIGHTS_COLOUR)
    v.append(" %.1f" % (mem.gpu_weights / gib), style="#565f89")
    v.append("  kv", style=KV_COLOUR)
    v.append(" %.1f" % ((mem.gpu_kv + mem.gpu_rs) / gib), style="#565f89")
    v.append("  compute", style=COMPUTE_COLOUR)
    v.append(" %.1f" % ((mem.gpu_compute + mem.gpu_mmproj) / gib),
             style="#565f89")
    # every weight byte that is not on the card, named: leaving the
    # embeddings out made the two figures miss the size of the file
    if mem.host_weights > 64 * mib:
        v.append(NEWLINE + "CPU       ", style="#565f89")
        parts = []
        if mem.host_exps and mem.host_exps_layers:
            parts.append("experts L0–%d %.1f G" % (
                mem.host_exps_layers - 1, mem.host_exps / gib))
        rest = mem.host_weights - mem.host_exps - mem.host_embd
        if rest > 64 * mib:
            parts.append("layers %.1f G" % (rest / gib))
        if mem.host_embd > 64 * mib:
            parts.append("embd %.1f G" % (mem.host_embd / gib))
        v.append("  ".join(parts) or "%.1f G" % (mem.host_weights / gib),
                 style="#c0caf5")
    v.append(NEWLINE + "decode    ", style="#565f89")
    v.append("~%.0f t/s" % speed.decode0, style="#c0caf5")
    v.append("  %s" % ("measured box" if calibrated
                       else "uncalibrated · mdl fit hw"), style="#565f89")
    return v


class ParamPane(VerticalScroll):
    """Model metadata, tunable params, and the argv preview."""

    def compose(self) -> ComposeResult:
        yield Static(id="p-title")
        yield Static(id="p-meta")
        yield Static(id="p-params")
        yield Static(id="p-vram")

    def show(self, name, cfg, argv, size_bytes, status, mmproj_bytes=0):
        title = Text()
        title.append(name, style="bold #bb7af7")
        badge = {"ok": (" verified", "#9ece6a"), "fail": (" failed to load", "#f7768e"),
                 "new": (" never run", "#565f89")}[status]
        title.append("   " + badge[0], style=badge[1])
        self.query_one("#p-title", Static).update(title)

        meta = Text()
        meta.append(Path(cfg["model"]).name + "\n", style="#565f89")
        if size_bytes:
            meta.append(mdl.human_size(size_bytes), style="#c0caf5")
            meta.append("  on disk", style="#565f89")
        else:
            meta.append("not on disk", style="#f7768e")
        self.query_one("#p-meta", Static).update(meta)

        rows = Text()
        for key in ("mmproj", "ngl", "n_cpu_moe", "ctx", "flash_attn",
                    "kv_type", "parallel", "port"):
            if key not in cfg:
                continue
            rows.append(f"  {key:<12}", style="#565f89")
            if key == "mmproj":     # a path; the filename is the useful part
                here = Path(str(cfg[key])).is_file()
                rows.append(Path(str(cfg[key])).name
                            + ("" if here else "  missing") + "\n",
                            style="#c0caf5" if here else "#f7768e")
            else:
                rows.append(f"{cfg[key]}\n", style="#c0caf5")
        self.query_one("#p-params", Static).update(rows)

        gpu = gpu_memory()
        placed = fit_placement(argv) if size_bytes else None
        if gpu and placed:
            self.query_one("#p-vram", Static).update(
                placement_text(placed, gpu[1]))
            return
        weights, kv, partial = vram_estimate(argv, size_bytes, mmproj_bytes)
        v = Text()
        if gpu:
            total, room = weights + kv, gpu[1]
            v.append("est VRAM  ", style="#565f89")
            v.append(stacked_bar([(weights, WEIGHTS_COLOUR), (kv, KV_COLOUR)],
                                 room))
            # An upper bound that exceeds the card is not news: the flag
            # that makes it an upper bound is there to keep it off the
            # card. Red is for a figure we actually believe.
            over = total > room and not partial
            v.append("  %s%.1f / %.1f G" % ("≤ " if partial else "",
                                            total / 1024, room / 1024),
                     style="#f7768e" if over else "#c0caf5")
            v.append(NEWLINE + " " * 10)
            v.append("weights", style=WEIGHTS_COLOUR)
            v.append(" %.1fG" % (weights / 1024), style="#565f89")
            v.append("  kv", style=KV_COLOUR)
            v.append(" %.1fG" % (kv / 1024), style="#565f89")
            if partial:
                v.append(NEWLINE + " " * 10)
                v.append("some weights stay on the cpu; the real figure "
                         "is lower", style="#565f89")
        self.query_one("#p-vram", Static).update(v)


class Dashboard(Static):
    """Live panel shown while a server is up."""

    def compose(self) -> ComposeResult:
        yield Static(id="d-head")
        yield Meter(id="d-vram")
        yield Meter(id="d-ctx")
        yield Spark(id="d-toks")
        yield Static(id="d-slots")

    def update_all(self, state, metrics, slots, gpu, tok_history, metrics_ok,
                   health_ok=True, peak=None):
        head = Text()
        head.append(state["name"], style="bold #bb7af7")
        head.append("  ● RUNNING", style="bold #9ece6a")
        up = mdl.uptime(time.time() - state.get("started", 0))
        head.append(f"   pid {state['pid']} · :{state['port']} · up {up}",
                    style="#565f89")
        self.query_one("#d-head", Static).update(head)

        if gpu:
            used, total = gpu
            self.query_one("#d-vram", Meter).set_value(
                "VRAM", used / max(total, 1),
                                   f"{used / 1024:.1f} / {total / 1024:.1f} G")
        else:
            self.query_one("#d-vram", Meter).set_value("VRAM", None, "no nvidia-smi")

        kv = metrics.get("llamacpp:kv_cache_usage_ratio")
        if kv is None and slots:
            # Current builds do not export that gauge at all. The slots do
            # carry what is in the cache, and keep carrying it once the
            # slot is released - the conversation is still resident.
            held = sum(x.get("n_prompt_tokens") or 0 for x in slots)
            room = sum(x.get("n_ctx") or 0 for x in slots)
            if room:
                kv = held / room
        if kv is not None:
            caption = "{:.0f}% of kv cache".format(kv * 100)
        elif not health_ok:
            caption = "loading"
        elif metrics_ok:
            caption = "idle"
        else:
            caption = "metrics off"
        self.query_one("#d-ctx", Meter).set_value("ctx", kv, caption)

        self.query_one("#d-toks", Spark).set_series("tok/s", tok_history, "",
                                                    peak)

        s = Text()
        if slots is not None:
            busy = sum(1 for x in slots if x.get("is_processing"))
            s.append("slots  ", style="#565f89")
            s.append("▣ " * busy, style="#9ece6a")
            s.append("▢ " * (len(slots) - busy), style="#565f89")
            s.append(f" {busy}/{len(slots)} busy", style="#c0caf5")
        else:
            s.append("slots  ", style="#565f89")
            s.append("loading" if not health_ok else "unavailable", style="#565f89")
        reqs = metrics.get("llamacpp:n_decode_total")
        if reqs:
            s.append(f"   {int(reqs)} decodes", style="#565f89")
        if not health_ok:
            s.append("   server still loading", style="#565f89")
        elif not metrics_ok:
            s.append("   metrics off · add --metrics to args", style="#e0af68")
        self.query_one("#d-slots", Static).update(s)


class HelpScreen(ModalScreen):
    """Keybinding cheatsheet."""

    BINDINGS = [Binding("escape,q,question_mark", "dismiss", "close")]

    ROWS = [
        ("↑ ↓ / j k", "select a model"),
        ("enter / r", "run the selected model, or fold a group"),
        ("s", "stop the selected server"),
        ("R", "restart (stop, then run again)"),
        ("e", "edit params for the selected model"),
        ("c", "copy the llama-server command"),
        ("y", "copy the whole log"),
        ("p", "prompt the running model"),
        ("b", "measure the selected config (mdl lab); b again stops it"),
        ("l", "focus the log pane"),
        ("/", "filter the log"),
        ("g", "reload models.toml and refresh telemetry"),
        ("u", "update mdl, when a newer release is out"),
        ("?", "this help"),
        ("q", "quit the UI (the server keeps running)"),
    ]

    def compose(self) -> ComposeResult:
        body = Text()
        body.append("  keys\n\n", style="bold #bb7af7")
        for key, what in self.ROWS:
            body.append(f"  {key:<12}", style="#7aa2f7")
            body.append(f"{what}\n", style="#c0caf5")
        body.append("\n  quitting never stops a server. use s for that.\n",
                    style="#565f89")
        yield Static(body, id="help-box")


class EditScreen(ModalScreen):
    """Edit one model's params and save them to the config."""

    BINDINGS = [Binding("escape", "dismiss", "cancel")]
    FIELDS = ["ngl", "n_cpu_moe", "ctx", "kv_type", "parallel", "port"]

    def __init__(self, name, cfg):
        super().__init__()
        # NB: `name` is a reserved DOMNode property, so it cannot be self.name.
        self.model_name, self.cfg = name, dict(cfg)
        self.shown = mdl.form_values(cfg)

    def compose(self) -> ComposeResult:
        with Vertical(id="edit-box"):
            yield Label(Text(" edit " + self.model_name + " ", style="bold #bb7af7"))
            for f in self.FIELDS:
                with Horizontal(classes="edit-row"):
                    yield Label(f"{f:<11}", classes="edit-label")
                    yield Input(value=self.shown[f], id=f"f-{f}",
                                placeholder="unset", classes="edit-input")
            # A group is not an object to create and delete: it is a
            # name two models share. Type the same one twice and they
            # fold together; clear the last one and it is gone.
            with Horizontal(classes="edit-row"):
                yield Label(f"{'group':<11}", classes="edit-label")
                yield Input(value=self.shown["group"],
                            id="f-group", placeholder="unset",
                            classes="edit-input")
            # The vision half of a multimodal model. Its own row rather
            # than a line in args, because it is a path to a file that
            # can go missing, and check has to be able to say so.
            with Horizontal(classes="edit-row"):
                yield Label(f"{'mmproj':<11}", classes="edit-label")
                yield Input(value=self.shown["mmproj"],
                            id="f-mmproj", placeholder="path to mmproj-*.gguf",
                            classes="edit-input")
            # Three states, as in the config: false is -fa off, and unset
            # is whatever the build defaults to. Shown as "off" when unset,
            # a save used to turn one into the other either way.
            with Horizontal(classes="edit-row"):
                yield Label(f"{'flash_attn':<11}", classes="edit-label")
                yield Input(value=self.shown["flash_attn"],
                            id="f-flash_attn", placeholder="unset",
                            classes="edit-input")
            # Everything llama-server takes that mdl has no key for. Better
            # one row here than a form that chases llama.cpp's flag list.
            with Horizontal(classes="edit-row"):
                yield Label(f"{'args':<11}", classes="edit-label")
                yield Input(value=self.shown["args"],
                            id="f-args", placeholder="--metrics --no-mmap",
                            classes="edit-input")
            yield Label(Text(" enter saves to models.toml · esc cancels",
                        style="#565f89"))
            yield Button("apply", variant="primary", id="apply")

    def _collect(self):
        # the page's reader too, so the two dashboards take the same text
        form = {f: self.query_one(f"#f-{f}", Input).value for f in mdl.FORM}
        try:
            changes, cleared = mdl.form_read(form)
        except mdl.MdlError as e:
            self.notify(str(e), severity="error")
            return None
        cfg = {k: v for k, v in self.cfg.items() if k not in cleared}
        cfg.update(changes)
        # the rules `mdl run` holds the config to: a save the form took
        # but run refused left a models.toml nothing would start from
        try:
            mdl.check_cfg(self.model_name, cfg)
        except mdl.MdlError as e:
            self.notify(str(e), severity="error")
            return None
        return cfg

    def on_button_pressed(self, _):
        self._apply()

    def on_input_submitted(self, _):
        self._apply()

    def _apply(self):
        cfg = self._collect()
        if cfg is not None:
            self.dismiss(cfg)


SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
BUSY = ("waiting", "thinking", "typing")
PHASES = {"idle":     ("#565f89", "ready"),
          "waiting":  ("#e0af68", "reading the prompt"),
          "thinking": ("#bb7af7", "reasoning"),
          "typing":   ("#9ece6a", "generating"),
          "done":     ("#565f89", "done"),
          "stopped":  ("#e0af68", "interrupted"),
          "error":    ("#f7768e", "failed")}


def partial_tag(s, tag):
    """How many trailing characters of s could be the start of tag."""
    for n in range(min(len(tag) - 1, len(s)), 0, -1):
        if tag.startswith(s[-n:]):
            return n
    return 0


class PromptScreen(ModalScreen):
    """A conversation with the running server.

    The server is a whole process away, so the only honest thing to show
    while it works is what it is actually doing - reading the prompt,
    reasoning, or emitting - and how fast. Twenty silent seconds in a box
    reads as a hang, which is how the first version of this felt.
    """

    BINDINGS = [Binding("escape", "close", "close"),
                Binding("ctrl+l", "clear", "clear")]

    def __init__(self, port, model="model"):
        super().__init__()
        self.port, self.model = port, model
        self.history = []              # the whole exchange, so it has context
        self.transcript = Text()
        self.phase = "idle"
        self.frame = 0
        self.cancel = False
        self.stream = None
        # Whether new text pulls the view down with it. Scrolling up to
        # read lets go; scrolling back to the bottom takes hold again.
        self.follow = True
        self._reset()

    def _reset(self):
        self.buf, self.thinking = "", False
        self.tokens = self.reasoned = 0
        self.began = self.first = self.think_at = 0.0
        self.think_secs = 0.0
        self.timings = {}
        self.reply = []

    # -- layout ------------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Vertical(id="prompt-box"):
            yield Static(id="chat-head")
            with VerticalScroll(id="prompt-scroll"):
                yield Static(id="prompt-out")
            yield Static(id="chat-stats")
            yield Input(placeholder="ask something, enter to send",
                        id="prompt-input")

    def on_mount(self):
        self._paint()
        self.set_interval(0.08, self._tick)
        self.watch(self.query_one("#prompt-scroll"), "scroll_y", self._scrolled,
                   init=False)
        self.query_one("#prompt-input", Input).focus()

    def _scrolled(self, old, new):
        """Every token used to scroll to the end, so reading back while a
        reply streamed was impossible. Going by direction rather than by
        "is it at the bottom" matters: tokens land faster than the layout
        grows, so a view that is following is often a line short of a
        bottom that has already moved, and would let go on its own."""
        pane = self.query_one("#prompt-scroll")
        # at the bottom first: clearing the pane drops scroll_y too, and a
        # view clamped to a shorter transcript has not been scrolled away
        if new >= pane.max_scroll_y - 1:
            self.follow = True
        elif new < old - 0.5:
            self.follow = False

    def _pull(self):
        if self.follow:
            self.query_one("#prompt-scroll").scroll_end(animate=False)

    def _tick(self):
        self.frame += 1
        if self.phase in BUSY:
            self._paint()

    def _paint(self):
        colour, label = PHASES[self.phase]
        self.query_one("#chat-head", Static).update(Text.assemble(
            ("● ", colour), (self.model, "bold #c0caf5"), ("  ·  ", "#1f2430"),
            ("127.0.0.1:%d" % self.port, "#565f89"), ("  ·  ", "#1f2430"),
            (label, colour)))
        body = self.transcript.copy()
        if self.phase in ("thinking", "typing") and self.frame % 8 < 5:
            body.append("▌", "#bb7af7")
        self.query_one("#prompt-out", Static).update(body)
        self.query_one("#chat-stats", Static).update(self._stats())

    def _rate(self):
        return self.tokens / max(time.time() - self.first, 1e-6) if self.first else 0

    def _summary(self):
        bits = ["%d tok" % self.tokens]
        rate = self.timings.get("predicted_per_second") or self._rate()
        if rate:
            bits.append("%.1f tok/s" % rate)
        if self.first:
            bits.append("ttft %.2fs" % (self.first - self.began))
        if self.think_secs:
            bits.append("%.1fs reasoning" % self.think_secs)
        return "  ·  ".join(bits)

    def _stats(self):
        if self.phase not in BUSY:      # the summary is in the transcript
            return Text("enter to send  ·  ctrl+l to clear  ·  esc to close",
                        "#3b4261")
        spin = SPINNER[self.frame % len(SPINNER)]
        label = "interrupting" if self.cancel else PHASES[self.phase][1]
        bits = [(spin + " ", "#7aa2f7"), (label, "#7aa2f7")]
        if self.phase == "waiting":
            bits.append(("  %.1fs" % (time.time() - self.began), "#565f89"))
        else:
            bits.append(("  %d tok  ·  %.1f tok/s" % (self.tokens, self._rate()),
                         "#565f89"))
        bits.append(("   esc to interrupt", "#3b4261"))
        return Text.assemble(*bits)

    # -- streaming ---------------------------------------------------------
    def _split(self, piece):
        """(kind, text) pairs, honouring <think> tags split across chunks.

        Some builds hand reasoning back in its own delta field; others just
        emit the tags inline, and a tag can straddle two chunks.
        """
        self.buf += piece
        out = []
        while True:
            tag = THINK_CLOSE if self.thinking else THINK_OPEN
            kind = "reason" if self.thinking else "text"
            i = self.buf.find(tag)
            if i < 0:
                keep = len(self.buf) - partial_tag(self.buf, tag)
                text, self.buf = self.buf[:keep], self.buf[keep:]
                if text:
                    out.append((kind, text))
                return out
            if i:
                out.append((kind, self.buf[:i]))
            self.buf = self.buf[i + len(tag):]
            self.thinking = not self.thinking

    def _feed(self, kind, text):
        if not self.first:
            self.first = time.time()
        if kind == "reason":
            if not self.reasoned:
                self.transcript.append("⟩ reasoning" + NEWLINE,
                                       "italic #565f89")
                self.think_at = time.time()
            self.phase = "thinking"
            self.reasoned += 1
            # Flush left, not inset: the pane wraps, so an indent would
            # land on the first line only and read as a stray one.
            self.transcript.append(text, "italic #4b5478")
        else:
            if self.phase == "thinking":
                self.think_secs = time.time() - self.think_at
                self.transcript.append(NEWLINE)
            self.phase = "typing"
            self.reply.append(text)
            self.transcript.append(text, "#c0caf5")
        self.tokens += 1
        self._paint()
        self._pull()

    def _finish(self, phase, note=""):
        self.phase = phase
        style = {"error": "#f7768e", "stopped": "#e0af68"}.get(phase, "#3b4261")
        self.transcript.append(NEWLINE * 2 + "  " + (note or self._summary())
                               + NEWLINE, style)
        reply = "".join(self.reply).strip()
        if reply:
            self.history.append({"role": "assistant", "content": reply})
        box = self.query_one("#prompt-input", Input)
        box.disabled = False
        box.placeholder = "ask something, enter to send"
        box.focus()
        self._paint()
        self._pull()

    def on_input_submitted(self, event):
        text = event.value.strip()
        if not text or self.phase in BUSY:
            return
        box = self.query_one("#prompt-input", Input)
        box.value = ""
        box.disabled = True
        box.placeholder = "streaming, esc to interrupt"
        if self.transcript.plain:
            self.transcript.append(NEWLINE)
        self.transcript.append("▌ you" + NEWLINE, "bold #7aa2f7")
        self.transcript.append(text + NEWLINE * 2, "#7aa2f7")
        self.transcript.append("▌ " + self.model + NEWLINE, "bold #bb7af7")
        self.history.append({"role": "user", "content": text})
        self._reset()
        self.began = time.time()
        self.cancel = False
        self.phase = "waiting"
        self.follow = True                  # a new question is worth seeing
        self._paint()
        self._pull()
        self._send()

    def _body(self):
        """No max_tokens. It was 1024, and a model that thinks can spend
        that before it starts answering; the reply then stops mid-sentence
        for no reason on screen. Esc stops a reply that runs on, and the
        server's own -n, set through args, still applies."""
        return json.dumps({"messages": self.history, "temperature": 0.7,
                           "stream": True,
                           "timings_per_token": True}).encode()

    @work(thread=True)
    def _send(self):
        """Stream the reply. Runs off the UI thread so the spinner keeps up."""
        body = self._body()
        req = urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % self.port, data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        call = self.app.call_from_thread
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                self.stream = response
                for raw in response:
                    if self.cancel:
                        break
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0]["delta"]
                    except (ValueError, KeyError, IndexError):
                        continue
                    self.timings = chunk.get("timings") or self.timings
                    think = delta.get("reasoning_content") or ""
                    if think:
                        call(self._feed, "reason", think)
                    for kind, text in self._split(delta.get("content") or ""):
                        call(self._feed, kind, text)
        except Exception as e:                   # noqa: BLE001 - shown, not raised
            if not self.cancel:                  # a cancel closes the socket
                call(self._finish, "error", "failed: %s" % e)
                return
        finally:
            self.stream = None
        if self.cancel:
            call(self._finish, "stopped", "interrupted  ·  " + self._summary())
        else:
            call(self._finish, "done")

    # -- keys --------------------------------------------------------------
    def action_close(self):
        if self.phase not in BUSY:
            self.dismiss()
            return
        self.cancel = True
        if self.stream is not None:              # cut it off now, not next token
            try:
                self.stream.close()
            except Exception:                    # noqa: BLE001
                pass

    def action_clear(self):
        if self.phase in BUSY:
            return
        self.history.clear()
        self.transcript = Text()
        self.phase = "idle"
        self.follow = True
        self._reset()
        self._paint()


# --------------------------------------------------------------------------
# the app
# --------------------------------------------------------------------------

class MdlApp(App):
    CSS = """
    Screen { background: #0d1017; color: #c0caf5;
             scrollbar-size-vertical: 1;
             scrollbar-background: #0d1017; scrollbar-color: #1f2430;
             scrollbar-background-hover: #0d1017;
             scrollbar-color-hover: #2a3050;
             scrollbar-background-active: #0d1017;
             scrollbar-color-active: #2a3050;
             scrollbar-corner-color: #0d1017; }
    /* Textual's own footer theme is a bright blue band; everything
       else here is hand-picked, and the two did not sit together. */
    Footer { background: #10141c; }
    FooterKey { background: #10141c; }
    FooterKey .footer-key--key { color: #7aa2f7; background: #10141c;
                                 text-style: bold; }
    FooterKey .footer-key--description { color: #565f89;
                                        background: #10141c; }
    FooterKey:hover { background: #1f2430; }
    #top { height: 8; background: #10141c; border-bottom: solid #1f2430; }
    #wordmark { width: 34; padding: 1 0 0 1; }
    #sysinfo { padding: 2 0 0 2; color: #565f89; }
    #body { height: 1fr; }
    #left { width: 38; border-right: solid #1f2430; }
    #right { width: 1fr; }
    #models { height: 1fr; background: #0d1017; border: round #1f2430; }
    /* auto so the command pane gives its spare rows to the log on a
       short terminal, where 1fr used to leave the log with none. */
    #p-argv { height: auto; max-height: 9; min-height: 3;
              padding: 0 2; border: round #1f2430; }
    #models > .datatable--cursor { background: #2a3050; color: #c0caf5; }
    #models > .datatable--header { background: #10141c; color: #565f89; }
    ParamPane { padding: 0 2; height: 1fr; border: round #1f2430; }
    Dashboard { height: 8; padding: 0 2; border: round #2a3050; }
    #p-title, #d-head { padding-bottom: 1; }
    #log { height: 1fr; min-height: 5; padding: 0 1; background: #0b0e14;
           border: round #1f2430; }
    #status { height: 1; background: #10141c; color: #565f89; padding: 0 1; }
    #help-box { width: 62; height: auto; padding: 1 2; background: #151a23;
                border: round #7aa2f7; }
    #edit-box { width: 66; height: auto; padding: 1 2;
                background: #151a23; border: round #7aa2f7; }
    #update-box { width: 72; height: auto; padding: 1 2;
                  background: #151a23; border: round #e0af68; }
    #update-buttons { height: auto; margin-top: 1; }
    #update-buttons Button { margin-right: 1; }
    #update-log { height: 10; margin-top: 1; background: #0b0e14;
                  border: round #1f2430; display: none; }
    #prompt-box { width: 88; height: auto; padding: 1 2;
                background: #151a23; border: round #bb7af7; }
    #chat-head { height: 1; padding: 0 1; }
    #prompt-scroll { height: 18; background: #0b0e14; margin-top: 1;
                border: round #1f2430; }
    #prompt-out { padding: 0 1; }
    #chat-stats { height: 1; padding: 0 1; margin-top: 1; }
    #prompt-input { margin-top: 1; }
    .edit-row { height: 3; }
    .edit-label { padding: 1 0 0 0; color: #565f89; width: 12; }
    .edit-input { width: 1fr; }
    HelpScreen, EditScreen, PromptScreen, UpdateScreen {
        align: center middle; }
    """

    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("enter,r", "run", "run"),
        Binding("s", "stop", "stop"),
        Binding("R", "restart", "restart"),
        Binding("e", "edit", "edit"),
        Binding("c", "copy", "copy cmd"),
        Binding("y", "copy_log", "copy log"),
        Binding("p", "prompt", "prompt"),
        Binding("b", "bench", "measure"),
        Binding("l", "focus_log", "logs"),
        Binding("slash", "filter", "filter"),
        Binding("g", "refresh", "refresh"),
        Binding("u", "update", "update"),
        Binding("question_mark", "help", "help"),
        Binding("j", "cursor_down", "", show=False),
        Binding("k", "cursor_up", "", show=False),
    ]

    status_line = reactive("")

    def __init__(self, fx=None, fx_period_override=None):
        super().__init__()
        self.fx = fx_mode(fx)
        self.fx_period = fx_period(fx_period_override)
        self.models, self.binary = {}, ""
        self.marks = {}                # name -> "ok" | "fail" | "new"
        self.collapsed = set()         # group names folded shut
        self.sizes = {}
        self.mmproj_sizes = {}
        self.tok_history = deque(maxlen=SPARK_POINTS)
        self.tok_peak = 0.0       # for this server, until it is restarted
        self._last_decode = self._last_decode_at = None
        self._log_pos = 0
        self._log_path = None
        self._filter = ""
        self._metrics_ok = False
        self._health_ok = False
        self._tele = {"metrics": {}, "slots": None, "gpu": None}
        self.states = {}          # name -> state, refreshed every tick
        self._following = None    # whose tok/s the sparkline is showing
        self._update_ver = None   # a newer mdl on offer, once one is known

    # ---- layout ----
    def compose(self) -> ComposeResult:
        with Horizontal(id="top"):
            yield Wordmark(id="wordmark")
            yield Static(id="sysinfo")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield DataTable(id="models", cursor_type="row", zebra_stripes=False)
            with Vertical(id="right"):
                yield Dashboard(id="dash")
                yield ParamPane(id="params")
                yield ArgvPreview(id="p-argv")
                yield CopyableLog(id="log", wrap=False, markup=False,
                                  max_lines=2000)
        yield Static(id="status")
        yield Footer()

    def on_mount(self):
        self.title = "mdl"
        self.query_one("#models").border_title = "models"
        self.query_one("#params").border_title = "config"
        self.query_one("#p-argv").border_title = "command"
        self.query_one("#log").border_title = "log"
        self.query_one("#dash").border_title = "live"
        self._load_config()
        self._build_table()
        self._sysinfo()
        # popped, so a server or an editor started from here does not
        # inherit it and a later restart does not repeat it
        came_from = os.environ.pop("MDL_UPDATED_FROM", None)
        if came_from:
            self.status_line = "updated mdl %s -> %s" % (came_from, mdl.VERSION)
            self._whats_new(came_from)
        self.set_interval(POLL_SECONDS, self._tick)
        self._tick()
        self._check_update()

    @work(thread=True, group="news")
    def _whats_new(self, came_from):
        """After an update, the headlines of what it brought, from the
        changelog at the new tag. A courtesy, like the check: nothing
        that goes wrong here may reach the app."""
        try:
            entries = mdl_update.changes(came_from, mdl.VERSION)
        except Exception:
            return
        heads = [h for _, hs in entries for h in hs]
        if heads:
            more = len(heads) - 3
            self.call_from_thread(
                self.notify, "\n".join("· " + h for h in heads[:3])
                + ("\nand %d more in the changelog" % more if more > 0 else ""),
                title="new in mdl %s" % mdl.VERSION, timeout=20)

    # ---- config / table ----
    def _load_config(self):
        try:
            self.models, self.binary = mdl.load_config(missing_ok=True)
        except mdl.MdlError as e:
            self.models, self.binary = {}, ""
            self.status_line = str(e)
            return
        if not self.models:
            self.status_line = mdl.NO_MODELS
            # a first model may be on disk already, from LM Studio or a
            # pull by another tool: say so, and what adds them
            try:
                from mdl_fit import scan
                n = len(scan.found())
            except OSError:
                n = 0
            if n:
                self.status_line = (
                    "no models yet, but %d GGUF%s on this machine: quit (q) "
                    "and run `mdl setup` to add %s" % (
                        n, "" if n == 1 else "s", "it" if n == 1 else "them"))
        marks = self._marks_path()
        try:
            self.marks = json.loads(marks.read_text())
        except (OSError, ValueError):
            self.marks = {}
        try:
            self.collapsed = set(json.loads(
                self._groups_path().read_text()))
        except (OSError, ValueError, TypeError):
            self.collapsed = set()
        for name, cfg in self.models.items():
            self.marks.setdefault(name, "new")
            try:
                self.sizes[name] = Path(cfg["model"]).stat().st_size
            except (OSError, KeyError):
                self.sizes[name] = None      # not 0: the file is not there
            try:
                self.mmproj_sizes[name] = Path(cfg["mmproj"]).stat().st_size
            except (OSError, KeyError):
                self.mmproj_sizes[name] = 0

    def _dot(self, name):
        if name in self.states:      # up right now, whatever it did before
            return ("▶", "#9ece6a")
        return {"ok": ("●", "#9ece6a"), "fail": ("✗", "#f7768e"),
                "new": ("○", "#565f89")}[self.marks.get(name, "new")]

    def _group_bytes(self, names):
        """What the group costs on disk, counting each file once.

        Presets of one model are the usual reason to group: six
        context lengths of the same gguf are six rows but one file,
        and adding the sizes up claimed 67.7G of a 12.3G download.
        """
        seen, total = set(), 0
        for name in names:
            raw = self._cfg(name).get("model")
            size = self.sizes.get(name)
            if not raw or not size:
                continue
            key = os.path.normcase(str(Path(raw)))
            if key not in seen:
                seen.add(key)
                total += size
        return total

    def _add_group_row(self, table, group):
        """A folded group still has to show that something inside it is
        up, or collapsing hides the one thing this pane is for."""
        members = [n for g, n in self._ordered() if g == group]
        shut = group in self.collapsed
        running = [n for n in members if n in self.states]
        head = Text((("▸ " if shut else "▾ ") + group)[:NAME_WIDTH],
                    style="bold #bb7af7")
        total = self._group_bytes(members)
        table.add_row(head,
                      Text(mdl.human_size(total) if total else "-",
                           style="#3b4261"),
                      Text("%d" % len(members), style="#3b4261"),
                      Text("▶" if running else "", style="#9ece6a"),
                      key=GROUP_KEY + group)

    def _marks_path(self):
        return mdl.STATE_DIR / "ui-marks.json"

    def _groups_path(self):
        return mdl.STATE_DIR / "ui-groups.json"

    def _save_collapsed(self):
        try:
            mdl.STATE_DIR.mkdir(parents=True, exist_ok=True)
            mdl.write_atomic(self._groups_path(),
                             json.dumps(sorted(self.collapsed)))
        except OSError:
            pass

    def _save_marks(self):
        try:
            mdl.STATE_DIR.mkdir(parents=True, exist_ok=True)
            mdl.write_atomic(self._marks_path(), json.dumps(self.marks))
        except OSError:
            pass

    def _build_table(self):
        try:
            table = self.query_one("#models", DataTable)
        except NoMatches:
            return          # a worker finishing after the screen has gone
        keep = self._selected_key()
        table.clear(columns=True)
        table.add_columns("model", "size", "ctx", "")
        shown = None
        for group, name in self._ordered():
            if group and group != shown:
                self._add_group_row(table, group)
                shown = group
            if group in self.collapsed:
                continue
            cfg = self._cfg(name)
            # The pane is a fixed width, so a long name used to push the
            # numbers off the right edge - 65536 rendered as 65, silently.
            room = NAME_WIDTH - (2 if group else 0)
            label = name if len(name) <= room else name[:room - 1] + "…"
            size = self.sizes.get(name)
            cell = (Text("missing", style="#f7768e") if size is None
                    else Text(mdl.human_size(size), style="#565f89"))
            dot = self._dot(name)
            table.add_row(Text(("  " if group else "") + label,
                               style="#c0caf5"), cell,
                          Text(str(cfg.get("ctx", "-")), style="#565f89"),
                          Text(dot[0], style=dot[1]), key=name)
        if keep:
            for row in range(table.row_count):
                if table.ordered_rows[row].key.value == keep:
                    table.move_cursor(row=row)
                    break
        if self.models:
            table.focus()

    def _cfg(self, name):
        return self.models.get(name, {})

    def _group(self, name):
        return str(self._cfg(name).get("group") or "")

    def _ordered(self):
        """(group, name) pairs, grouped models after the loose ones.

        Ungrouped first so a config that uses no groups looks exactly
        as it did before the feature existed.
        """
        return sorted(((self._group(n), n) for n in self.models),
                      key=lambda gn: (gn[0] != "", gn[0].lower(),
                                      gn[1].lower()))

    def _selected_key(self):
        """The row under the cursor, group rows included."""
        try:
            table = self.query_one("#models", DataTable)
        except NoMatches:
            return None
        if not table.row_count:
            return None
        try:
            # The row key, not the cell: a long name is shown truncated.
            return table.ordered_rows[table.cursor_row].key.value
        except (IndexError, AttributeError):
            return None

    def _selected(self):
        """The model under the cursor, or None on a group row.

        None rather than the group name, because every caller already
        handles None by doing nothing - which is what run, stop, edit
        and prompt should do when the cursor is on a folder.
        """
        key = self._selected_key()
        return None if key is None or key.startswith(GROUP_KEY) else key

    def _selected_group(self):
        key = self._selected_key()
        return key[len(GROUP_KEY):] if key and key.startswith(GROUP_KEY) else None

    def action_toggle_group(self):
        group = self._selected_group()
        if group is None:
            return False
        self.collapsed ^= {group}
        self._save_collapsed()
        self._build_table()
        return True

    def _sysinfo(self):
        gpu = gpu_memory()
        t = Text()
        t.append(Path(self.binary).name if self.binary else "no binary",
                 style="#9ece6a")
        t.append("\n")
        if gpu:
            t.append(f"{gpu[1] / 1024:.0f} GB VRAM", style="#565f89")
        t.append(f"\n{len(self.models)} models", style="#565f89")
        self.query_one("#sysinfo", Static).update(t)

    # ---- polling ----
    def _tick(self):
        """Poll, and draw the result - unless the screen has gone.

        The guard is here rather than on each widget because the tick
        is what they have in common: quitting mid-poll took out #models,
        then #dash, then #status, one release at a time. Every selector
        below is exercised by the tests on every run, so a typo shows up
        as a pane that never updates rather than being swallowed here.
        """
        try:
            self._tick_once()
        except NoMatches:
            return

    def _tick_once(self):
        # Whatever is running, the panes follow the selection: with
        # several servers up, 'the' running one is not a thing.
        self.states = mdl.read_states()
        dash = self.query_one("#dash", Dashboard)
        params = self.query_one("#params", ParamPane)
        state = self.states.get(self._selected())
        if state and state["name"] != self._following:
            self._following = state["name"]   # a different server's rate
            self.tok_history.clear()
            self.tok_peak = 0.0
            self._last_decode = self._last_decode_at = None
            self._tele = {"metrics": {}, "slots": None, "gpu": None}
        if state:
            dash.display = True
            params.display = False      # config pane is for choosing, not watching
            self._poll(state)
            dash.update_all(state, self._tele["metrics"], self._tele["slots"],
                            self._tele["gpu"], list(self.tok_history),
                            self._metrics_ok, self._health_ok, self.tok_peak)
            # a measurement in progress keeps the pane on its own lines
            if self._log_path != Path(state["log"]) and not getattr(
                    self, "_benching", None):
                self._follow(state["log"], "log")
        else:
            dash.display = False
            params.display = True
            self._following = None
            self.tok_history.clear()
            self.tok_peak = 0.0
            self._last_decode = self._last_decode_at = None
        self._drain_log()
        name = self._selected()
        if name and name in self.models:
            cfg = self._cfg(name)
            try:
                argv = mdl.build_argv(name, cfg, self.binary)
            except mdl.MdlError as e:
                argv, self.status_line = [], str(e)
            params.show(name, cfg, argv, self.sizes.get(name, 0),
                        self.marks.get(name, "new"),
                        self.mmproj_sizes.get(name, 0))
            self.query_one("#p-argv", ArgvPreview).set_argv(argv)
        self._render_status(state)

    @work(thread=True, exclusive=True, group="poll")
    def _poll(self, state):
        port = state["port"]
        self._health_ok = http_get(port, "/health") is not None
        raw = http_get(port, "/metrics")
        self._metrics_ok = raw is not None
        metrics = parse_metrics(raw)
        slots = http_json(port, "/slots")
        if not isinstance(slots, list):
            slots = None
        gpu = gpu_memory()

        # llamacpp:n_decode_total is the only one of these that moves while
        # a reply is streaming. predicted_tokens_seconds sits at 0 for the
        # whole generation and then publishes one average when the request
        # finishes, and tokens_predicted_total does not move until then
        # either - which is why this used to draw zeros and a single spike.
        now = time.monotonic()
        rate = None
        total = metrics.get("llamacpp:n_decode_total")
        if total is not None:
            if self._last_decode is not None:
                # Measured elapsed, not POLL_SECONDS: the poll drifts.
                gap = max(now - self._last_decode_at, 1e-3)
                rate = max(0.0, (total - self._last_decode) / gap)
            self._last_decode, self._last_decode_at = total, now
        if rate is None:      # no counter; the end-of-request average
            rate = metrics.get("llamacpp:predicted_tokens_seconds")
        if rate is not None:
            self.tok_history.append(rate)
            self.tok_peak = max(self.tok_peak, rate)

        self._tele = {"metrics": metrics, "slots": slots, "gpu": gpu}

    def _drain_log(self):
        if not self._log_path or not self._log_path.exists():
            return
        try:
            with open(self._log_path, "r", errors="replace") as fh:
                fh.seek(self._log_pos)
                chunk = fh.read()
                self._log_pos = fh.tell()
        except OSError:
            return
        if not chunk:
            return
        out = self.query_one("#log", CopyableLog)
        colours = {"error": "#f7768e", "warn": "#e0af68", "ready": "bold #9ece6a",
                   "slot": "#7dcfff", "info": "#565f89"}
        for line in chunk.splitlines():
            if self._filter and self._filter.lower() not in line.lower():
                continue
            out.write(Text(line, style=colours[log_level(line)]))
            m = LAYER_RE.search(line)
            if m:
                self.status_line = "loading - {}/{} layers on GPU".format(
                    m.group(1), m.group(2))

    def _render_status(self, state):
        t = Text()
        if state:
            t.append(" ▶ ", style="#9ece6a")
            t.append("{} on :{}".format(state["name"], state["port"]),
                     style="#c0caf5")
        elif self.states:
            t.append(" ● ", style="#9ece6a")
            t.append(", ".join(sorted(self.states)), style="#c0caf5")
        else:
            t.append(" ○ nothing running", style="#565f89")
        if len(self.states) > 1:
            t.append("   %d running" % len(self.states), style="#565f89")
        if self._filter:
            t.append("   filter: " + self._filter, style="#e0af68")
        if self.status_line:
            t.append("   " + self.status_line, style="#7aa2f7")
        if self._update_ver:
            t.append("   mdl %s is out: u" % self._update_ver, style="#e0af68")
        self.query_one("#status", Static).update(t)

    def on_data_table_row_highlighted(self, _):
        self._tick()

    def on_data_table_row_selected(self, _):
        """Enter inside the table never reaches the app binding; forward it."""
        self.action_run()

    # ---- actions ----
    def action_cursor_down(self):
        self.query_one("#models", DataTable).action_cursor_down()

    def action_cursor_up(self):
        self.query_one("#models", DataTable).action_cursor_up()

    def action_help(self):
        self.push_screen(HelpScreen())

    # ---- update ----
    def check_action(self, action, parameters):
        """`u` is in the footer only while there is an update to take."""
        if action == "update":
            return self._update_ver is not None
        return True

    @work(thread=True, group="update-check")
    def _check_update(self):
        """Ask, at most once a day, whether a newer mdl is out. A courtesy:
        whatever goes wrong in here, the dashboard carries on without it,
        so nothing may escape the worker and take the app down."""
        try:
            if mdl_update.disabled():
                return
            mdl_update.sweep()
            version = mdl_update.offer()
        except Exception:
            return
        if version:
            self.call_from_thread(self._offer_update, version)

    def _offer_update(self, version):
        self._update_ver = version
        self.refresh_bindings()
        self._tick()
        # never on top of something the user is in the middle of; the
        # status line and `u` still say it
        if len(self.screen_stack) == 1:
            self.push_screen(UpdateScreen(version))

    def action_update(self):
        if self._update_ver and not isinstance(self.screen, UpdateScreen):
            self.push_screen(UpdateScreen(self._update_ver))

    def begin_update(self, screen):
        """Update and restart, unless something this process is doing
        would be cut off by the restart."""
        doing = {"start": "a server is starting", "stop": "a server is stopping",
                 "lab": "a measurement is running"}
        busy = sorted({doing[w.group] for w in self.workers
                       if w.is_running and w.group in doing})
        if busy:
            self.notify("%s; update once it is done" % "; ".join(busy),
                        severity="warning")
            return
        screen.installing_now()
        self._do_update(screen, screen.version,
                        screen.query_one("#update-log", RichLog))

    @work(thread=True, group="update")
    def _do_update(self, screen, version, log):
        def out(line):
            self.call_from_thread(log.write, Text(line, style="#565f89"))
        try:
            now = mdl_update.install(version, out=out)
        except mdl.MdlError as e:
            self.call_from_thread(screen.failed, str(e))
            return
        self.call_from_thread(screen.done, now)
        time.sleep(0.8)                 # long enough to read that it worked
        self.call_from_thread(self.exit, mdl_update.RESTART)

    def action_bench(self):
        """A measured run of the selected config through mdl lab: one
        warmup and one repetition of its default workload, from a config
        of its own, recorded with the rest for mdl lab report. A server of
        yours on its port is left alone, and lab says so. Pressed while
        one runs, it stops that one: its server down, its temp files gone,
        what it measured kept."""
        running = getattr(self, "_benching", None)
        if running:
            if self._bench_stop.is_set():
                self.notify("already stopping " + running, severity="warning")
                return
            self._bench_stop.set()
            self.status_line = "stopping the measurement of %s..." % running
            self.notify("stopping the measurement of " + running)
            return
        name = self._selected()
        if not name:
            return
        from mdl_fit import lab
        self._benching, self._bench_stop = name, lab.Stop()
        self.status_line = "measuring %s with mdl lab (b stops it)..." % name
        self._do_bench(name, self._bench_stop)

    async def action_quit(self):
        # a measurement is stopped rather than left to finish behind a
        # closed dashboard: the process waits for its thread either way,
        # and this way that is seconds, with its server taken down
        stop = getattr(self, "_bench_stop", None)
        if getattr(self, "_benching", None) and stop is not None:
            stop.set()
        self.exit()

    def _later(self, fn, *args, **kw):
        """call_from_thread, when there is still an app to call: a worker
        finishing after quit has nothing to tell."""
        try:
            self.call_from_thread(fn, *args, **kw)
        except RuntimeError:
            pass

    def _follow(self, path, title):
        """Point the log pane at a file of our own, from its start."""
        self._log_path, self._log_pos = Path(path), 0
        log = self.query_one("#log", CopyableLog)
        log.clear()
        log.border_title = title

    @work(thread=True, group="lab")
    def _do_bench(self, name, stop=None):
        import io

        from mdl_fit import lab

        # the lab's own lines, in the log pane as they come: which depth,
        # which repetition, how many are left
        logs = lab.lab_dir() / "logs"
        try:
            logs.mkdir(parents=True, exist_ok=True)
            for old in sorted(logs.glob("b-*.log"))[:-9]:
                old.unlink(missing_ok=True)     # the last ten are enough
        except OSError:
            pass
        path = logs / ("b-%s.log" % time.strftime("%Y%m%d-%H%M%S"))
        try:
            with open(path, "w", encoding="utf-8", buffering=1) as fh:
                self._later(self._follow, path, "log · mdl lab " + name)
                run_id = lab.main(["run", name, "--reps", "1", "--warmup",
                                   "1", "--cooldown", "0"], fh, stop=stop)
            got = [r for r in lab.load() if r.get("run") == run_id
                   and not r.get("warmup") and not r.get("skipped")]
            rows = lab.report({"names": [run_id], "format": "json"},
                              io.StringIO()) if got else []
            if stop is not None and stop.is_set():
                self._later(self.notify, "stopped measuring %s; %d "
                            "repetition%s kept%s" % (
                                name, len(got), "" if len(got) == 1 else "s",
                                ". mdl lab report " + run_id if got else ""),
                            severity="warning", timeout=15)
            elif rows:
                # a row per depth: empty, and the context full
                msg = "%s: %s; VRAM %s, RAM %s. mdl lab report %s" % (
                    name, "; ".join(
                        "%s t/s at depth %s, first token %ss" % (
                            lab.cell("decode", r).split(" ")[0],
                            lab.cell("depth", r), lab.cell("ttft", r))
                        for r in rows),
                    lab.cell("vram_peak", rows[-1]),
                    lab.cell("rss_peak", rows[-1]), run_id)
                self._later(self.notify, msg, timeout=30)
            else:
                skipped = [x.get("skipped") for x in lab.load()
                           if x.get("run") == run_id and x.get("skipped")]
                self._later(
                    self.notify, "%s was not measured: %s" % (
                        name, skipped[0] if skipped else "nothing ran"),
                    severity="warning", timeout=15)
        except (mdl.MdlError, OSError) as e:
            self._later(self.notify, str(e), severity="error", timeout=15)
        finally:
            self._benching = None
            self._later(setattr, self, "status_line", "")

    def action_refresh(self):
        self._load_config()
        self._build_table()
        self._sysinfo()
        self._tick()
        self.notify("reloaded config")

    def action_focus_log(self):
        self.query_one("#log", CopyableLog).focus()

    def action_filter(self):
        def apply(value):
            self._filter = (value or "").strip()
            self._log_pos = 0
            self.query_one("#log", CopyableLog).clear()
            self._drain_log()
        self.push_screen(FilterScreen(self._filter), apply)

    def action_copy(self):
        name = self._selected()
        if not name:
            return
        try:
            argv = mdl.build_argv(name, self._cfg(name), self.binary)
        except mdl.MdlError as e:
            self.notify(str(e), severity="error")
            return
        # pasted into the shell it came from: cmd quoting is not sh quoting
        line = (subprocess.list2cmdline(argv) if os.name == "nt"
                else shlex.join(str(a) for a in argv))
        self._clip(line, "command copied")

    def action_copy_log(self):
        text = self.query_one("#log", CopyableLog).as_text().strip()
        if not text:
            self.notify("the log is empty")
            return
        self._clip(text, "log copied (%d lines)" % len(text.split(NEWLINE)))

    def _clip(self, text, said):
        """The system clipboard where there is a tool for it, and OSC 52
        as well - over ssh the escape is the only way to the desk you
        are sitting at. Without a tool, say that it went to the terminal,
        not that it was copied: nothing reports whether it arrived."""
        try:
            via = system_clipboard(text)
            self.copy_to_clipboard(text)
        except Exception:                            # noqa: BLE001
            self.notify("could not reach the clipboard", severity="warning")
            return
        if via or os.name == "nt":
            self.notify(said)
        else:
            self.notify("%s via the terminal - if nothing pastes, install "
                        "wl-clipboard or xclip" % said.split(" (")[0],
                        timeout=8)

    def action_edit(self):
        name = self._selected()
        if not name:
            return

        def apply(cfg):
            if cfg is None:
                return
            try:
                mdl.write_params(name, cfg)
            # MdlError is how the writer refuses: the table renamed or
            # removed on disk while the dashboard had it open. Uncaught,
            # it took the whole dashboard down with a traceback.
            except (OSError, ValueError, mdl.MdlError) as e:
                self.notify("could not save: %s" % e, severity="error")
                return
            self._load_config()
            self._build_table()
            self._tick()
            self.notify(name + " saved to models.toml")
        self.push_screen(EditScreen(name, self._cfg(name)), apply)

    def action_prompt(self):
        state = self.states.get(self._selected())
        if not state:
            self.notify("that model is not running", severity="warning")
            return
        self.push_screen(PromptScreen(state["port"], state["name"]))

    def action_run(self):
        # enter does the obvious thing for whatever is under the cursor,
        # rather than needing a key of its own for folders.
        if self.action_toggle_group():
            return
        name = self._selected()
        if not name:
            return
        if name in self.states:
            self.notify(
                name + " is already running - press s to stop it",
                severity="warning")
            return
        try:
            proc, log, port = mdl.spawn(name, self.models, self.binary)
        except mdl.MdlError as e:
            self.notify(str(e), severity="error")
            return
        self._log_path, self._log_pos = Path(log), 0
        self.query_one("#log", CopyableLog).clear()
        self.status_line = "starting " + name
        self._watch_start(name, proc, port)

    @work(thread=True, group="start")
    def _watch_start(self, name, proc, port):
        """Mark the model verified once it reports ready, or failed if it dies.

        Readiness is /health answering, not a line in the log. The log
        wording has already changed once between llama.cpp builds; the
        endpoint is the contract, and mdl run has always used it.
        """
        limit = mdl.ready_timeout()
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            if mdl.server_ready(port):
                self.marks[name] = "ok"
                self._save_marks()
                self.call_from_thread(self._after_start, name, True, None)
                return
            if proc.poll() is not None:
                self.marks[name] = "fail"
                self._save_marks()
                mdl.state_path(name).unlink(missing_ok=True)
                self.call_from_thread(self._after_start, name, False, proc.returncode)
                return
            time.sleep(0.3)
        self.call_from_thread(self._after_start, name, False, None, limit)

    def _after_start(self, name, ok, code, limit=None):
        self._build_table()
        self.status_line = ""
        if ok:
            self.notify(name + " is ready")
        elif code is not None:
            self.notify(
                "{} exited with status {} during startup".format(name, code),
                severity="error", timeout=10)
        else:
            self.notify(
                "{} did not report ready in {:g}s".format(
                    name, limit if limit is not None else mdl.READY_TIMEOUT),
                severity="warning", timeout=10)
        self._tick()

    def action_stop(self):
        # The selected one if it is up, otherwise the only one that is.
        name = self._selected() if self._selected() in self.states else None
        if name is None and len(self.states) == 1:
            name = next(iter(self.states))
        if name is None:
            self.notify("that model is not running")
            return
        self.status_line = "stopping " + name
        self._do_stop(name)

    @work(thread=True, group="stop")
    def _do_stop(self, name, then=None):
        try:
            mdl.cmd_stop([name])
            self.call_from_thread(self.notify, "stopped " + name)
        except mdl.MdlError as e:
            self.call_from_thread(self.notify, str(e), severity="error")
            then = None
        self.call_from_thread(setattr, self, "status_line", "")
        if then:
            self.call_from_thread(then, name)
        self.call_from_thread(self._tick)

    def action_restart(self):
        name = self._selected()
        if not name:
            return
        if name not in self.states:
            self._restart_run(name)      # nothing to stop; just start it
            return
        self.status_line = "restarting " + name
        # Chained off the stop, not a timer: a stop that takes longer than
        # the guess used to leave the run colliding with its own old port.
        self._do_stop(name, then=self._restart_run)

    def _restart_run(self, name):
        # read_states, because the stop that led here happened in a worker
        # and self.states does not catch up until the next poll.
        self.states = mdl.read_states()
        try:
            table = self.query_one("#models", DataTable)
        except NoMatches:
            return
        for row in range(table.row_count):
            # The key, not the cell: a long name is displayed truncated,
            # and a miss here would leave the cursor on another model and
            # start that one instead.
            if table.ordered_rows[row].key.value == name:
                table.move_cursor(row=row)
                break
        self.action_run()


class UpdateScreen(ModalScreen):
    """A newer mdl is out: update and restart, later, or skip that one."""

    # q too: the app's q would quit halfway through an install
    BINDINGS = [Binding("escape,q", "later", "later")]
    AUTO_FOCUS = "#update-go"

    def __init__(self, version):
        super().__init__()
        self.version, self.installing = version, False

    def compose(self) -> ComposeResult:
        with Vertical(id="update-box"):
            yield Label(Text(" update ", style="bold #e0af68"))
            body = Text()
            body.append("\nmdl %s is out" % self.version, style="bold #c0caf5")
            body.append(" - you have %s\n" % mdl.VERSION, style="#565f89")
            body.append("what changed: %s\n" % mdl_update.CHANGES,
                        style="#565f89")
            body.append("servers keep running; the dashboard restarts on "
                        "the new version", style="#565f89")
            yield Static(body, id="update-msg")
            with Horizontal(id="update-buttons"):
                yield Button("update & restart", variant="primary",
                             id="update-go")
                yield Button("later", id="update-later")
                yield Button("skip %s" % self.version, id="update-skip")
            yield RichLog(id="update-log", wrap=True, markup=False,
                          max_lines=500)

    def on_button_pressed(self, event):
        if event.button.id == "update-go":
            self.app.begin_update(self)
        elif event.button.id == "update-skip":
            mdl_update.skip(self.version)
            self.app._update_ver = None
            self.app.refresh_bindings()
            self.dismiss()
        else:
            self.action_later()

    def action_later(self):
        if self.installing:
            # killing pip halfway leaves a worse install than either version
            self.notify("updating; this cannot be interrupted",
                        severity="warning")
            return
        self.dismiss()

    def _buttons(self, enabled):
        for button in self.query(Button):
            button.disabled = not enabled

    def installing_now(self):
        self.installing = True
        self._buttons(False)
        self.query_one("#update-log", RichLog).display = True

    def failed(self, message):
        self.installing = False
        self._buttons(True)
        self.query_one("#update-log", RichLog).write(
            Text(message, style="#f7768e"))
        self.notify("update failed; still on %s" % mdl.VERSION,
                    severity="error", timeout=10)

    def done(self, version):
        self.query_one("#update-log", RichLog).write(
            Text("updated to %s - restarting" % version, style="bold #9ece6a"))


class FilterScreen(ModalScreen):
    """One-line log filter prompt."""

    BINDINGS = [Binding("escape", "dismiss", "cancel")]

    def __init__(self, current):
        super().__init__()
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="edit-box"):
            yield Label(Text(" filter log (empty clears) ", style="bold #bb7af7"))
            yield Input(value=self.current, id="filter-input")

    def on_mount(self):
        self.query_one("#filter-input", Input).focus()

    def on_input_submitted(self, event):
        self.dismiss(event.value)


def run_ui(fx=None, fx_period_override=None):
    """Returns mdl_update.RESTART when the user updated mdl from here."""
    app = MdlApp(fx, fx_period_override)
    result = app.run()
    if getattr(app, "_benching", None):
        # quit mid-measurement: its thread takes its server down before
        # the process can exit, and a silent pause would look like a hang
        print("mdl: stopping the mdl lab measurement of %s..."
              % app._benching, file=sys.stderr, flush=True)
    return result
