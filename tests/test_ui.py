"""Headless drive of the TUI. No real server, no real config."""
import asyncio
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                        # noqa: E402
from support import mdl, sandbox, teardown            # noqa: E402

import mdl_ui                                         # noqa: E402
from mdl_ui import MdlApp                             # noqa: E402
from textual.widgets import DataTable, Input, Static  # noqa: E402

t = support.Tally("test_ui")
check = t.check
EXTRA = ('\n[second]\nmodel = "%s"\nngl = 10\nctx = 512\nport = 9999\n'
         % str(support.FAKE).replace("\\", "/"))


def plain(widget):
    r = widget.render()
    return r.plain if hasattr(r, "plain") else str(r)


async def main():
    root, port = sandbox(extra=EXTRA)
    app = MdlApp(fx="off")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        table = app.query_one("#models", DataTable)
        names = [table.get_row_at(r)[0].plain for r in range(table.row_count)]
        check("both models listed, sorted", names, ["demo", "second"])
        check("a model is selected on boot", app._selected(), "demo")

        argv = app.query_one("#p-argv", mdl_ui.ArgvPreview)
        check("command preview shows the binary", "llama-server" in argv.text, True)
        check("command preview shows mapped flags",
              "-ngl 99" in argv.text and "--port %d" % port in argv.text, True)
        check("long commands are not truncated",
              "--cache-type-v q8_0" in argv.text, True)

        await pilot.press("down")
        await pilot.pause()
        check("cursor moves", app._selected(), "second")
        check("detail pane follows", plain(app.query_one("#p-title", Static)
                                           ).startswith("second"), True)
        check("command preview follows", "-ngl 10" in argv.text, True)
        await pilot.press("k")
        await pilot.pause()
        check("k goes back up", app._selected(), "demo")

        check("dashboard hidden when idle", app.query_one("#dash").display, False)
        check("status says nothing running",
              "nothing running" in plain(app.query_one("#status", Static)), True)

        for key, screen, label in (("question_mark", mdl_ui.HelpScreen, "help"),
                                   ("e", mdl_ui.EditScreen, "edit"),
                                   ("slash", mdl_ui.FilterScreen, "filter")):
            await pilot.press(key)
            await pilot.pause()
            check("%s opens" % label, isinstance(app.screen, screen), True)
            await pilot.press("escape")
            await pilot.pause()
            check("%s closes" % label, isinstance(app.screen, screen), False)

        # editing persists to the config file
        await pilot.press("e")
        await pilot.pause()
        app.screen.query_one("#f-ctx", Input).value = "2048"
        app.screen.query_one("#f-ngl", Input).value = ""      # clearing removes it
        await pilot.pause()
        app.screen._apply()
        await pilot.pause()
        saved = tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8"))
        check("edit saved ctx to the config", saved["demo"]["ctx"], 2048)
        check("clearing a field removes the key", "ngl" in saved["demo"], False)
        check("other model untouched", saved["second"]["ctx"], 512)
        check("top-level keys survive", saved["llama_server"] is not None, True)
        check("command preview reflects the save",
              "-c 2048" in argv.text and "-ngl" not in argv.text, True)

        _, _, code = support.run(mdl.load_config)
        check("config still parses after the write", code, 0)

        await pilot.press("p")
        await pilot.pause()
        check("prompt refused while idle",
              isinstance(app.screen, mdl_ui.PromptScreen), False)
        await pilot.press("s")
        await pilot.pause()
        check("stop while idle does not crash", app.is_running, True)

    teardown(root)
    return t.done()


sys.exit(asyncio.run(main()))
