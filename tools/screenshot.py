#!/usr/bin/env python3
"""Render docs/screenshot.svg from the real TUI, against a demo config.

Uses invented models so the image does not depend on whatever happens to be
on the author's disk. Run it after any visual change:

    python tools/screenshot.py
"""
import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import mdl        # noqa: E402
import mdl_ui     # noqa: E402
from mdl_ui import MdlApp                     # noqa: E402
from textual.widgets import DataTable         # noqa: E402

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
        out = ROOT / "docs" / "screenshot.svg"
        out.parent.mkdir(exist_ok=True)
        out.write_text(app.export_screenshot(title="mdl"), encoding="utf-8")
        print("wrote %s (%d KB)" % (out, out.stat().st_size // 1024))


asyncio.run(main())
