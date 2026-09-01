"""Opt-in: drives the TUI against YOUR real config and a real model.

Slow (it loads a model) and machine-specific, so run.py only includes it
with --live. It never writes to the config.

    python tests/test_live.py [model-name]
"""
import asyncio
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                   # noqa: E402
from support import mdl                          # noqa: E402

import mdl_ui                                    # noqa: E402
from mdl_ui import MdlApp                        # noqa: E402
from textual.widgets import DataTable, Input, Static  # noqa: E402

t = support.Tally("test_live")
check = t.check


def pick_model():
    if len(sys.argv) > 1:
        return sys.argv[1]
    models, _ = mdl.load_config()
    if not models:
        print("no models in %s" % mdl.CONFIG)
        sys.exit(0)
    return sorted(models, key=lambda n: Path(models[n]["model"]).stat().st_size
                  if Path(models[n]["model"]).is_file() else 1 << 60)[0]


async def until(pilot, predicate, seconds=180):
    for _ in range(int(seconds * 2)):
        await asyncio.sleep(0.5)
        await pilot.pause()
        if predicate():
            return True
    return False


async def main():
    if mdl.read_state():
        print("something is already running; stop it first")
        return 1
    name = pick_model()
    print("      using model: %s" % name)
    app = MdlApp(fx="off")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        table = app.query_one("#models", DataTable)
        for row in range(table.row_count):
            if table.get_row_at(row)[0].plain == name:
                table.move_cursor(row=row)
                break
        await pilot.pause()
        check("selected it", app._selected(), name)

        app.marks[name] = "new"          # marks persist; a stale ok would fake this
        await pilot.press("r")
        await pilot.pause()

        loading = []
        ready = await until(pilot, lambda: (
            loading.append(app.query_one("#d-slots", Static).render()) or
            app.marks.get(name) == "ok"))
        check("launched from the UI and became ready", ready, True)
        if not ready:
            return t.done()

        seen = [getattr(x, "plain", str(x)) for x in loading if str(x).strip()]
        check("no bogus --metrics hint while loading",
              any("add --metrics" in x for x in seen[:6]), False)

        state = mdl.read_state()
        check("state written", state["name"], name)
        code = None
        for _ in range(20):
            try:
                code = urllib.request.urlopen(
                    "http://127.0.0.1:%d/health" % state["port"], timeout=5).status
                break
            except Exception as e:                        # noqa: BLE001
                code = str(e)
                await asyncio.sleep(0.5)
        check("server serving", code, 200)
        check("dashboard visible", await until(pilot, lambda: app.query_one("#dash").display,
                                               10), True)
        check("metrics reachable", await until(pilot, lambda: app._metrics_ok, 15), True)

        # streaming prompt
        await pilot.press("p")
        await pilot.pause()
        check("prompt opened", isinstance(app.screen, mdl_ui.PromptScreen), True)
        app.screen.query_one("#prompt-input", Input).value = "Say hello in five words."
        await pilot.press("enter")
        grew = await until(pilot, lambda: len(app.screen.log_text.plain) > 40, 180)
        check("reply streamed into the pane", grew, True)
        print("      reply: %r" % app.screen.log_text.plain[-90:].replace("\n", " "))
        await pilot.press("escape")
        await pilot.pause()
        check("tok/s series populated",
              await until(pilot, lambda: bool(app.tok_history), 15), True)

        await pilot.press("s")
        check("stopped from the UI", await until(pilot, lambda: mdl.read_state() is None,
                                                 30), True)
    return t.done()


sys.exit(asyncio.run(main()))
