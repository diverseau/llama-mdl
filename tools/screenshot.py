#!/usr/bin/env python3
"""Render docs/screenshot.svg and docs/chat.svg from the real TUI.

Uses invented models and a scripted reply, so the images do not depend on
whatever happens to be on the author's disk, and do not need a server.
Run it after any visual change:

    python tools/screenshot.py
"""
import asyncio
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import mdl        # noqa: E402
import mdl_ui     # noqa: E402
from mdl_ui import MdlApp                     # noqa: E402
from textual.widgets import DataTable, Input  # noqa: E402

DEMO = '''llama_server = "/opt/llama.cpp/build/bin/llama-server"

[ornith]
model = "/srv/models/Ornith-1.5-35B-A3B-Q4_K_M.gguf"
ngl = 99
n_cpu_moe = 24
ctx = 65536
flash_attn = true
kv_type = "q8_0"
parallel = 1
port = 8080
args = ["--metrics"]

[qwen-small]
model = "/srv/models/Qwen3-8B-Q5_K_M.gguf"
ngl = 99
ctx = 16384
flash_attn = true
port = 8080

[gemma-12b]
model = "/srv/models/Gemma-3-12B-it-Q4_K_M.gguf"
ngl = 99
ctx = 8192
flash_attn = true
kv_type = "q8_0"
port = 8080
'''

SIZES = {"ornith": 21_400_000_000, "qwen-small": 5_800_000_000,
         "gemma-12b": 7_300_000_000}
MARKS = {"ornith": "ok", "qwen-small": "ok", "gemma-12b": "new"}

QUESTION = "in one sentence, what is a GGUF file?"
REASONING = [
    "The user asked for one sentence, so no preamble. Lead with what the ",
    "format actually is - a container for the weights - then the part they ",
    "care about: it is what lets a quantised model load quickly and run on ",
    "hardware they already own. Expanding the acronym would spend the ",
    "sentence on trivia they did not ask for.",
]
ANSWER = [
    "A GGUF file is a single-file container holding an LLM's weights and ",
    "the metadata needed to load them, usually quantised so the model fits ",
    "in memory and runs on ordinary CPUs and consumer GPUs.",
]


async def main():
    root = Path(tempfile.mkdtemp(prefix="mdl-shot-"))
    (root / "config").mkdir()
    (root / "state").mkdir()
    mdl.CONFIG = root / "config" / "models.toml"
    mdl.STATE_DIR = root / "state"
    mdl.STATE = root / "state" / "state.json"
    mdl.CONFIG.write_text(DEMO, encoding="utf-8")

    app = MdlApp(fx="off")
    async with app.run_test(size=(104, 38)) as pilot:
        await pilot.pause()
        app.sizes.update(SIZES)               # plausible sizes, no real files
        app.marks.update(MARKS)
        app._build_table()
        table = app.query_one("#models", DataTable)
        table.move_cursor(row=1)              # land on ornith
        await pilot.pause()
        app._tick()
        await pilot.pause()
        save(app, "screenshot.svg")

        await chat(app, pilot)
        save(app, "chat.svg")


def save(app, name):
    out = ROOT / "docs" / name
    out.parent.mkdir(exist_ok=True)
    out.write_text(app.export_screenshot(title="mdl"), encoding="utf-8")
    print("wrote %s (%d KB)" % (out, out.stat().st_size // 1024))


async def chat(app, pilot):
    """Replay a turn straight into the screen, so no server is needed."""
    screen = mdl_ui.PromptScreen(8080, "ornith")
    await app.push_screen(screen)
    await pilot.pause()
    screen.transcript.append("▌ you" + chr(10), "bold #7aa2f7")
    screen.transcript.append(QUESTION + chr(10) * 2, "#7aa2f7")
    screen.transcript.append("▌ ornith" + chr(10), "bold #bb7af7")
    screen.began = time.time()
    for chunk in REASONING:
        screen._feed("reason", chunk)
    for chunk in ANSWER:
        screen._feed("text", chunk)
    # Plausible numbers for a still image; the arithmetic is the real one.
    screen.tokens, screen.think_secs = 148, 9.4
    screen.first = time.time() - 3.2
    screen.phase, screen.frame = "typing", 3
    box = screen.query_one("#prompt-input", Input)
    box.disabled = True                   # as it is while a reply streams
    box.placeholder = "streaming, esc to interrupt"
    screen._paint()
    await pilot.pause()


asyncio.run(main())
