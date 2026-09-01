"""mdl ui - a terminal dashboard for the mdl llama.cpp server runner.

Kept deliberately separate from mdl.py: the CLI stays dependency-free and
quiet, this file is allowed to be pretty. Both drive the same functions and
the same state file, so they can never disagree about what is running.
"""

import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

import mdl

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
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
FX_TICK = 0.08          # ~12 fps; cheap enough to leave running
FX_STEP = 0.010         # a full colour cycle every ~8 seconds
FX_SWEEP_STEP = 0.067   # "sweep" does that one cycle in ~1.2s, then parks

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
SPARK_CHARS = " ▁▂▃▄▅▆▇█"


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
    width = max((len(l) for l in lines), default=1)
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
    scaled = [SPARK_CHARS[min(int(v / hi * (len(SPARK_CHARS) - 1)), len(SPARK_CHARS) - 1)]
              for v in pts]
    return "".join(scaled).rjust(width)


def bar(fraction, width=20, fill="█", empty="░"):
    fraction = 0.0 if fraction is None else max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return fill * filled + empty * (width - filled)


def bar_colour(fraction):
    if fraction is None:
        return "#565f89"
    if fraction >= 0.92:
        return "#f7768e"
    if fraction >= 0.75:
        return "#e0af68"
    return "#9ece6a"


def human_size(nbytes):
    for unit, div in (("T", 1 << 40), ("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if nbytes >= div:
            return f"{nbytes / div:.1f}{unit}"
    return f"{nbytes}B"


# --------------------------------------------------------------------------
# telemetry. Every one of these degrades to None rather than raising: a panel
# going grey is always better than the dashboard falling over.
# --------------------------------------------------------------------------

def http_get(port, path, timeout=1.5):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
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


def gpu_memory():
    """(used_mib, total_mib) from nvidia-smi, or None if unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        raw = subprocess.run(
            [exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        used, total = raw.strip().splitlines()[0].split(",")
        return int(used), int(total)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


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
        self._step = FX_STEP
        self.update(gradient_text(self.lines))
        mode = getattr(self.app, "fx", FX_DEFAULT)
        if mode == "off":
            return
        if mode == "sweep":
            self._step = FX_SWEEP_STEP
            self.set_interval(FX_TICK, self._drift,
                              repeat=round(1.0 / FX_SWEEP_STEP))
        else:
            self._step = FX_STEP
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

    def set_series(self, label, values, unit=""):
        t = Text()
        t.append(f"{label:<7}", style="#565f89")
        t.append(sparkline(values), style="#7dcfff")
        if values:
            t.append(f"  {values[-1]:.1f}{unit} now", style="#c0caf5")
            t.append(f" · {max(values):.1f}{unit} peak", style="#565f89")
        else:
            t.append("  no data yet", style="#565f89")
        self.update(t)


class ArgvPreview(Static):
    """The exact llama-server command mdl would run. Updates as params change."""

    def set_argv(self, argv):
        if not argv:
            self.update(Text("(select a model)", style="#565f89"))
            return
        t = Text()
        t.append("$ ", style="#565f89")
        t.append(Path(argv[0]).name, style="bold #9ece6a")
        i = 1
        while i < len(argv):
            tok = argv[i]
            if tok.startswith("-"):
                t.append("\n    ")
                t.append(tok, style="#7aa2f7")
                if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                    val = argv[i + 1]
                    shown = ".../" + Path(val).name if len(val) > 46 else val
                    t.append(" " + shown, style="#e0af68")
                    i += 1
            else:
                t.append(" " + tok, style="#c0caf5")
            i += 1
        self.update(t)


class ParamPane(VerticalScroll):
    """Model metadata, tunable params, and the argv preview."""

    def compose(self) -> ComposeResult:
        yield Static(id="p-title")
        yield Static(id="p-meta")
        yield Static(id="p-params")
        yield Static(id="p-vram")

    def show(self, name, cfg, argv, size_bytes, status):
        title = Text()
        title.append(name, style="bold #bb7af7")
        badge = {"ok": (" verified", "#9ece6a"), "fail": (" failed to load", "#f7768e"),
                 "new": (" never run", "#565f89")}[status]
        title.append("   " + badge[0], style=badge[1])
        self.query_one("#p-title", Static).update(title)

        meta = Text()
        meta.append(Path(cfg["model"]).name + "\n", style="#565f89")
        meta.append(human_size(size_bytes) if size_bytes else "?", style="#c0caf5")
        meta.append("  on disk", style="#565f89")
        self.query_one("#p-meta", Static).update(meta)

        rows = Text()
        for key in ("ngl", "n_cpu_moe", "ctx", "flash_attn", "kv_type", "parallel", "port"):
            if key in cfg:
                rows.append(f"  {key:<12}", style="#565f89")
                rows.append(f"{cfg[key]}\n", style="#c0caf5")
        self.query_one("#p-params", Static).update(rows)

        # Rough: weights plus a KV allowance. Enough to warn, not to trust.
        est = (size_bytes or 0) / (1 << 20)
        kv = cfg.get("ctx", 4096) / 1024 * (32 if cfg.get("kv_type") else 64)
        gpu = gpu_memory()
        v = Text()
        if gpu:
            frac = min((est + kv) / max(gpu[1], 1), 1.0)
            v.append("est VRAM  ", style="#565f89")
            v.append(bar(frac, 18), style=bar_colour(frac))
            v.append(f"  {(est + kv) / 1024:.1f} / {gpu[1] / 1024:.1f} G", style="#c0caf5")
        self.query_one("#p-vram", Static).update(v)


class Dashboard(Static):
    """Live panel shown while a server is up."""

    def compose(self) -> ComposeResult:
        yield Static(id="d-head")
        yield Meter(id="d-vram")
        yield Meter(id="d-ctx")
        yield Spark(id="d-toks")
        yield Static(id="d-slots")

    def update_all(self, state, metrics, slots, gpu, tok_history, metrics_ok):
        head = Text()
        head.append(state["name"], style="bold #bb7af7")
        head.append("  ● RUNNING", style="bold #9ece6a")
        up = mdl.uptime(time.time() - state.get("started", 0))
        head.append(f"   pid {state['pid']} · :{state['port']} · up {up}", style="#565f89")
        self.query_one("#d-head", Static).update(head)

        if gpu:
            used, total = gpu
            self.query_one("#d-vram", Meter).set_value(
                "VRAM", used / max(total, 1), f"{used / 1024:.1f} / {total / 1024:.1f} G")
        else:
            self.query_one("#d-vram", Meter).set_value("VRAM", None, "no nvidia-smi")

        kv = metrics.get("llamacpp:kv_cache_usage_ratio")
        self.query_one("#d-ctx", Meter).set_value(
            "ctx", kv, f"{kv * 100:.0f}% of kv cache" if kv is not None
            else ("idle" if metrics_ok else "metrics off"))

        self.query_one("#d-toks", Spark).set_series("tok/s", tok_history, "")

        s = Text()
        if slots is not None:
            busy = sum(1 for x in slots if x.get("is_processing"))
            s.append("slots  ", style="#565f89")
            s.append("▣ " * busy, style="#9ece6a")
            s.append("▢ " * (len(slots) - busy), style="#565f89")
            s.append(f" {busy}/{len(slots)} busy", style="#c0caf5")
        else:
            s.append("slots  ", style="#565f89")
            s.append("unavailable", style="#565f89")
        reqs = metrics.get("llamacpp:n_decode_total")
        if reqs:
            s.append(f"   {int(reqs)} decodes", style="#565f89")
        if not metrics_ok:
            s.append("   metrics off · add --metrics to args", style="#e0af68")
        self.query_one("#d-slots", Static).update(s)


class HelpScreen(ModalScreen):
    """Keybinding cheatsheet."""

    BINDINGS = [Binding("escape,q,question_mark", "dismiss", "close")]

    ROWS = [
        ("↑ ↓ / j k", "select a model"),
        ("enter / r", "run the selected model"),
        ("s", "stop the running server"),
        ("R", "restart (stop, then run again)"),
        ("e", "edit params for the selected model"),
        ("c", "copy the llama-server command"),
        ("p", "prompt the running model"),
        ("l", "focus the log pane"),
        ("/", "filter the log"),
        ("g", "refresh GPU / telemetry now"),
        ("?", "this help"),
        ("q", "quit the UI (the server keeps running)"),
    ]

    def compose(self) -> ComposeResult:
        body = Text()
        body.append("  keys\n\n", style="bold #bb7af7")
        for key, what in self.ROWS:
            body.append(f"  {key:<12}", style="#7aa2f7")
            body.append(f"{what}\n", style="#c0caf5")
        body.append("\n  quitting never stops a server. use s for that.\n", style="#565f89")
        yield Static(body, id="help-box")


class EditScreen(ModalScreen):
    """Edit one model's params. Writes nothing to disk; applies for this run."""

    BINDINGS = [Binding("escape", "dismiss", "cancel")]
    FIELDS = ["ngl", "n_cpu_moe", "ctx", "kv_type", "parallel", "port"]

    def __init__(self, name, cfg):
        super().__init__()
        # NB: `name` is a reserved DOMNode property, so it cannot be self.name.
        self.model_name, self.cfg = name, dict(cfg)

    def compose(self) -> ComposeResult:
        with Vertical(id="edit-box"):
            yield Label(Text(" edit " + self.model_name + " ", style="bold #bb7af7"))
            for f in self.FIELDS:
                with Horizontal(classes="edit-row"):
                    yield Label(f"{f:<11}", classes="edit-label")
                    yield Input(value=str(self.cfg.get(f, "")), id=f"f-{f}",
                                placeholder="unset", classes="edit-input")
            with Horizontal(classes="edit-row"):
                yield Label(f"{'flash_attn':<11}", classes="edit-label")
                yield Input(value="on" if self.cfg.get("flash_attn") else "off",
                            id="f-flash_attn", classes="edit-input")
            yield Label(Text(" enter applies · esc cancels", style="#565f89"))
            yield Button("apply", variant="primary", id="apply")

    def _collect(self):
        cfg = dict(self.cfg)
        for f in self.FIELDS:
            raw = self.query_one(f"#f-{f}", Input).value.strip()
            if not raw:
                cfg.pop(f, None)
                continue
            if f == "kv_type":
                cfg[f] = raw
            else:
                try:
                    cfg[f] = int(raw)
                except ValueError:
                    self.notify(f"{f} must be a whole number", severity="error")
                    return None
        fa = self.query_one("#f-flash_attn", Input).value.strip().lower()
        if fa in ("on", "true", "yes", "1"):
            cfg["flash_attn"] = True
        else:
            cfg.pop("flash_attn", None)
        return cfg

    def on_button_pressed(self, _):
        self._apply()

    def on_input_submitted(self, _):
        self._apply()

    def _apply(self):
        cfg = self._collect()
        if cfg is not None:
            self.dismiss(cfg)


class PromptScreen(ModalScreen):
    """Send one completion to the running server without leaving the UI."""

    BINDINGS = [Binding("escape", "dismiss", "close")]

    def __init__(self, port):
        super().__init__()
        self.port = port

    def compose(self) -> ComposeResult:
        with Vertical(id="prompt-box"):
            yield Label(Text(" prompt the model ", style="bold #bb7af7"))
            yield Input(placeholder="ask something, enter to send", id="prompt-input")
            yield RichLog(id="prompt-out", wrap=True, markup=False)

    def on_mount(self):
        self.query_one("#prompt-input", Input).focus()

    def on_input_submitted(self, event):
        text = event.value.strip()
        if not text:
            return
        out = self.query_one("#prompt-out", RichLog)
        out.write(Text(f"> {text}", style="#7aa2f7"))
        self.query_one("#prompt-input", Input).value = ""
        self._send(text, out)

    @work(thread=True)
    def _send(self, text, out):
        body = json.dumps({"messages": [{"role": "user", "content": text}],
                           "max_tokens": 256, "temperature": 0.7}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            reply = data["choices"][0]["message"]["content"].strip()
            used = data.get("usage", {}).get("completion_tokens", 0)
            rate = used / max(time.time() - started, 1e-6)
            self.app.call_from_thread(out.write, Text(reply, style="#c0caf5"))
            self.app.call_from_thread(
                out.write, Text(f"  {used} tokens · {rate:.1f} tok/s\n", style="#565f89"))
        except Exception as e:                      # noqa: BLE001 - surfaced, not raised
            self.app.call_from_thread(out.write, Text(f"  request failed: {e}\n",
                                                      style="#f7768e"))


# --------------------------------------------------------------------------
# the app
# --------------------------------------------------------------------------

class MdlApp(App):
    CSS = """
    Screen { background: #0d1017; color: #c0caf5; }
    #top { height: 8; background: #10141c; border-bottom: solid #1f2430; }
    #wordmark { width: 34; padding: 1 0 0 1; }
    #sysinfo { padding: 2 0 0 2; color: #565f89; }
    #body { height: 1fr; }
    #left { width: 38; border-right: solid #1f2430; }
    #right { width: 1fr; }
    #models { height: 1fr; background: #0d1017; border: round #1f2430; }
    #p-argv { height: 9; padding: 0 2; border: round #1f2430; }
    #models > .datatable--cursor { background: #2a3050; color: #c0caf5; }
    #models > .datatable--header { background: #10141c; color: #565f89; }
    ParamPane { padding: 0 2; height: 1fr; border: round #1f2430; }
    Dashboard { height: 9; padding: 0 2; border: round #2a3050; }
    #p-title, #d-head { padding-bottom: 1; }
    #log { height: 1fr; padding: 0 1; background: #0b0e14; border: round #1f2430; }
    #status { height: 1; background: #10141c; color: #565f89; padding: 0 1; }
    #help-box { width: 62; height: auto; padding: 1 2; background: #151a23;
                border: round #7aa2f7; }
    #edit-box, #prompt-box { width: 66; height: auto; padding: 1 2;
                background: #151a23; border: round #7aa2f7; }
    #prompt-out { height: 14; background: #0b0e14; margin-top: 1; }
    .edit-row { height: 3; }
    .edit-label { padding: 1 0 0 0; color: #565f89; width: 12; }
    .edit-input { width: 1fr; }
    HelpScreen, EditScreen, PromptScreen { align: center middle; }
    """

    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("enter,r", "run", "run"),
        Binding("s", "stop", "stop"),
        Binding("R", "restart", "restart"),
        Binding("e", "edit", "edit"),
        Binding("c", "copy", "copy cmd"),
        Binding("p", "prompt", "prompt"),
        Binding("l", "focus_log", "logs"),
        Binding("slash", "filter", "filter"),
        Binding("g", "refresh", "refresh"),
        Binding("question_mark", "help", "help"),
        Binding("j", "cursor_down", "", show=False),
        Binding("k", "cursor_up", "", show=False),
    ]

    status_line = reactive("")

    def __init__(self, fx=None):
        super().__init__()
        self.fx = fx_mode(fx)
        self.models, self.binary = {}, ""
        self.overrides = {}            # name -> cfg edited this session
        self.marks = {}                # name -> "ok" | "fail" | "new"
        self.sizes = {}
        self.tok_history = deque(maxlen=SPARK_POINTS)
        self._last_decode = None
        self._log_pos = 0
        self._log_path = None
        self._filter = ""
        self._metrics_ok = False

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
                yield RichLog(id="log", wrap=False, markup=False, max_lines=2000)
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
        self.set_interval(POLL_SECONDS, self._tick)
        self._tick()

    # ---- config / table ----
    def _load_config(self):
        try:
            self.models, self.binary = mdl.load_config()
        except mdl.MdlError as e:
            self.models, self.binary = {}, ""
            self.status_line = str(e)
            return
        marks = self._marks_path()
        try:
            self.marks = json.loads(marks.read_text())
        except (OSError, ValueError):
            self.marks = {}
        for name, cfg in self.models.items():
            self.marks.setdefault(name, "new")
            try:
                self.sizes[name] = Path(cfg["model"]).stat().st_size
            except (OSError, KeyError):
                self.sizes[name] = 0

    def _marks_path(self):
        return mdl.STATE_DIR / "ui-marks.json"

    def _save_marks(self):
        try:
            mdl.STATE_DIR.mkdir(parents=True, exist_ok=True)
            self._marks_path().write_text(json.dumps(self.marks))
        except OSError:
            pass

    def _build_table(self):
        table = self.query_one("#models", DataTable)
        keep = self._selected()
        table.clear(columns=True)
        table.add_columns("model", "size", "ctx", "")
        for name in sorted(self.models):
            cfg = self._cfg(name)
            dot = {"ok": ("●", "#9ece6a"), "fail": ("✗", "#f7768e"),
                   "new": ("○", "#565f89")}[self.marks.get(name, "new")]
            table.add_row(Text(name, style="#c0caf5"),
                          Text(human_size(self.sizes.get(name, 0)), style="#565f89"),
                          Text(str(cfg.get("ctx", "-")), style="#565f89"),
                          Text(dot[0], style=dot[1]), key=name)
        if keep:
            for row in range(table.row_count):
                if table.get_row_at(row)[0].plain == keep:
                    table.move_cursor(row=row)
                    break
        if self.models:
            table.focus()

    def _cfg(self, name):
        return self.overrides.get(name, self.models.get(name, {}))

    def _selected(self):
        table = self.query_one("#models", DataTable)
        if not table.row_count:
            return None
        try:
            return table.get_row_at(table.cursor_row)[0].plain
        except (IndexError, AttributeError):
            return None

    def _sysinfo(self):
        gpu = gpu_memory()
        t = Text()
        t.append(Path(self.binary).name if self.binary else "no binary", style="#9ece6a")
        t.append("\n")
        if gpu:
            t.append(f"{gpu[1] / 1024:.0f} GB VRAM", style="#565f89")
        t.append(f"\n{len(self.models)} models", style="#565f89")
        self.query_one("#sysinfo", Static).update(t)

    # ---- polling ----
    def _tick(self):
        state = mdl.read_state()
        dash = self.query_one("#dash", Dashboard)
        params = self.query_one("#params", ParamPane)
        if state:
            dash.display = True
            params.display = False      # config pane is for choosing, not watching
            self._poll(state)
            if self._log_path != Path(state["log"]):
                self._log_path, self._log_pos = Path(state["log"]), 0
                self.query_one("#log", RichLog).clear()
        else:
            dash.display = False
            params.display = True
            self.tok_history.clear()
            self._last_decode = None
        self._drain_log()
        name = self._selected()
        if name and name in self.models:
            cfg = self._cfg(name)
            try:
                argv = mdl.build_argv(name, cfg, self.binary)
            except mdl.MdlError as e:
                argv, self.status_line = [], str(e)
            params.show(name, cfg, argv, self.sizes.get(name, 0),
                        self.marks.get(name, "new"))
            self.query_one("#p-argv", ArgvPreview).set_argv(argv)
        self._render_status(state)

    @work(thread=True, exclusive=True, group="poll")
    def _poll(self, state):
        port = state["port"]
        raw = http_get(port, "/metrics")
        self._metrics_ok = raw is not None
        metrics = parse_metrics(raw)
        slots = http_json(port, "/slots")
        if not isinstance(slots, list):
            slots = None
        gpu = gpu_memory()

        rate = metrics.get("llamacpp:predicted_tokens_seconds")
        if rate is None:
            total = metrics.get("llamacpp:tokens_predicted_total")
            if total is not None and self._last_decode is not None:
                rate = max(0.0, (total - self._last_decode) / POLL_SECONDS)
            if total is not None:
                self._last_decode = total
        if rate is not None:
            self.tok_history.append(rate)

        self.call_from_thread(
            self.query_one("#dash", Dashboard).update_all,
            state, metrics, slots, gpu, list(self.tok_history), self._metrics_ok)

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
        out = self.query_one("#log", RichLog)
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
            t.append(" ● ", style="#9ece6a")
            t.append("{} on :{}".format(state["name"], state["port"]), style="#c0caf5")
        else:
            t.append(" ○ nothing running", style="#565f89")
        if self._filter:
            t.append("   filter: " + self._filter, style="#e0af68")
        if self.status_line:
            t.append("   " + self.status_line, style="#7aa2f7")
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

    def action_refresh(self):
        self._load_config()
        self._build_table()
        self._sysinfo()
        self._tick()
        self.notify("reloaded config")

    def action_focus_log(self):
        self.query_one("#log", RichLog).focus()

    def action_filter(self):
        def apply(value):
            self._filter = (value or "").strip()
            self._log_pos = 0
            self.query_one("#log", RichLog).clear()
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
        try:
            self.copy_to_clipboard(subprocess.list2cmdline(argv))
            self.notify("command copied")
        except Exception:                            # noqa: BLE001
            self.notify("could not reach the clipboard", severity="warning")

    def action_edit(self):
        name = self._selected()
        if not name:
            return

        def apply(cfg):
            if cfg is not None:
                self.overrides[name] = cfg
                self._build_table()
                self._tick()
                self.notify(name + " params updated for this session")
        self.push_screen(EditScreen(name, self._cfg(name)), apply)

    def action_prompt(self):
        state = mdl.read_state()
        if not state:
            self.notify("nothing is running", severity="warning")
            return
        self.push_screen(PromptScreen(state["port"]))

    def action_run(self):
        name = self._selected()
        if not name:
            return
        state = mdl.read_state()
        if state:
            self.notify(
                state["name"] + " is already running - press s to stop it",
                severity="warning")
            return
        models = dict(self.models)
        models[name] = self._cfg(name)
        try:
            proc, log, port = mdl.spawn(name, models, self.binary)
        except mdl.MdlError as e:
            self.notify(str(e), severity="error")
            return
        self._log_path, self._log_pos = Path(log), 0
        self.query_one("#log", RichLog).clear()
        self.status_line = "starting " + name
        self._watch_start(name, proc)

    @work(thread=True, group="start")
    def _watch_start(self, name, proc):
        """Mark the model verified once it reports ready, or failed if it dies."""
        deadline = time.monotonic() + mdl.READY_TIMEOUT
        log = mdl.STATE_DIR / (name + ".log")
        while time.monotonic() < deadline:
            try:
                text = log.read_text(errors="replace")
            except OSError:
                text = ""
            if mdl.READY.search(text):
                self.marks[name] = "ok"
                self._save_marks()
                self.call_from_thread(self._after_start, name, True, None)
                return
            if proc.poll() is not None:
                self.marks[name] = "fail"
                self._save_marks()
                mdl.STATE.unlink(missing_ok=True)
                self.call_from_thread(self._after_start, name, False, proc.returncode)
                return
            time.sleep(0.3)
        self.call_from_thread(self._after_start, name, False, None)

    def _after_start(self, name, ok, code):
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
                "{} did not report ready in {}s".format(name, mdl.READY_TIMEOUT),
                severity="warning", timeout=10)
        self._tick()

    def action_stop(self):
        state = mdl.read_state()
        if not state:
            self.notify("nothing is running")
            return
        self.status_line = "stopping " + state["name"]
        self._do_stop(state["name"])

    @work(thread=True, group="stop")
    def _do_stop(self, name):
        try:
            mdl.cmd_stop([])
            self.call_from_thread(self.notify, "stopped " + name)
        except mdl.MdlError as e:
            self.call_from_thread(self.notify, str(e), severity="error")
        self.call_from_thread(setattr, self, "status_line", "")
        self.call_from_thread(self._tick)

    def action_restart(self):
        state = mdl.read_state()
        name = state["name"] if state else self._selected()
        if not name:
            return
        self.action_stop()
        self.set_timer(1.5, lambda: self._restart_run(name))

    def _restart_run(self, name):
        table = self.query_one("#models", DataTable)
        for row in range(table.row_count):
            if table.get_row_at(row)[0].plain == name:
                table.move_cursor(row=row)
                break
        self.action_run()


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


def run_ui(fx=None):
    MdlApp(fx).run()
