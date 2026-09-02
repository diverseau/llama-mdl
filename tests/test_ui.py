"""Headless drive of the TUI. No real server, no real config."""
import asyncio
import sys
import shlex
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                        # noqa: E402
from support import mdl, sandbox, teardown            # noqa: E402

import mdl_ui                                         # noqa: E402
from mdl_ui import MdlApp                             # noqa: E402
from textual.geometry import Region                  # noqa: E402
from textual.widgets import DataTable, Input, Static  # noqa: E402

t = support.Tally("test_ui")
check = t.check
LONG = ('\n[a-model-name-far-too-long-for-the-pane]\n'
        'model = "%s"\nctx = 65536\nport = 9998\n'
        % str(support.FAKE).replace("\\", "/"))
EXTRA = ('\n[second]\nmodel = "%s"\nngl = 10\nctx = 512\nport = 9999\n'
         % str(support.FAKE).replace("\\", "/"))


def pane_text(widget):
    """What the widget actually puts on screen, clipping included."""
    region = Region(0, 0, widget.size.width, widget.size.height)
    return "\n".join(strip.text
                     for strip in widget.render_lines(region))


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

        # a quoted value must survive the round trip through models.toml.
        # --chat-template-kwargs takes JSON, which is nothing but quotes;
        # unescaped they close the string and leave the args array open,
        # corrupting the config on save.
        json_arg = '{"reasoning_effort":"low"}'
        await pilot.press("e")
        await pilot.pause()
        app.screen.query_one("#f-args", Input).value = shlex.join(
            ["--chat-template-kwargs", json_arg])
        app.screen._apply()
        await pilot.pause()
        saved = tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8"))
        check("a quoted arg survives the save",
              saved["demo"]["args"], ["--chat-template-kwargs", json_arg])

        await pilot.press("e")
        await pilot.pause()
        app.screen._apply()          # saving again must not double-escape
        await pilot.pause()
        saved = tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8"))
        check("and again on the next save",
              saved["demo"]["args"], ["--chat-template-kwargs", json_arg])
        await pilot.press("e")
        await pilot.pause()
        app.screen.query_one("#f-args", Input).value = ""
        app.screen._apply()
        await pilot.pause()

        # the VRAM estimate must not care which field a flag came from
        est, kv = mdl_ui.vram_estimate(
            ["llama-server", "-c", "65536", "--cache-type-k", "q8_0"], 1 << 30)
        check("a quantised cache halves the kv allowance", (est, kv),
              (1024.0, 2048.0))
        _, plain_kv = mdl_ui.vram_estimate(["llama-server", "-c", "65536"], 0)
        check("an unquantised one does not", plain_kv, 4096.0)
        _, from_key = mdl_ui.vram_estimate(
            mdl.build_argv("demo", {"model": "m.gguf", "ctx": 65536,
                                    "kv_type": "q8_0"}, "llama-server"), 0)
        _, from_args = mdl_ui.vram_estimate(
            mdl.build_argv("demo", {"model": "m.gguf", "ctx": 65536,
                                    "args": ["--cache-type-k", "q8_0"]},
                           "llama-server"), 0)
        check("kv_type and args agree", from_key, from_args)
        check("and a missing ctx falls back to llama.cpp's default",
              mdl_ui.vram_estimate(["llama-server"], 0)[1], 256.0)

        # args is the escape hatch for every flag mdl has no key for
        await pilot.press("e")
        await pilot.pause()
        app.screen.query_one("#f-args", Input).value = '--metrics --alias "a b"'
        app.screen._apply()
        await pilot.pause()
        saved = tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8"))
        check("args are saved as a list, quotes respected",
              saved["demo"]["args"], ["--metrics", "--alias", "a b"])
        check("and reach the command line",
              "--metrics" in argv.text and "--alias" in argv.text, True)

        await pilot.press("e")
        await pilot.pause()
        check("args round-trip back into the field",
              app.screen.query_one("#f-args", Input).value,
              "--metrics --alias 'a b'")
        app.screen.query_one("#f-args", Input).value = ""
        app.screen._apply()
        await pilot.pause()
        saved = tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8"))
        check("clearing args removes the key", "args" in saved["demo"], False)

        await pilot.press("p")
        await pilot.pause()
        check("prompt refused while idle",
              isinstance(app.screen, mdl_ui.PromptScreen), False)
        await pilot.press("s")
        await pilot.pause()
        check("stop while idle does not crash", app.is_running, True)

    teardown(root)

    # --- the live panel, driven by hand: no server to talk to --------------
    # Both of these used to read zero in the case that matters most: idle.
    root, port = sandbox()
    app = MdlApp(fx="off")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        dash = app.query_one("#dash", mdl_ui.Dashboard)
        state = {"name": "demo", "pid": 1, "port": port, "started": time.time()}
        idle = [{"n_prompt_tokens": 2048, "n_ctx": 8192, "is_processing": False}]

        # No kv_cache_usage_ratio: current llama.cpp builds do not export it.
        dash.update_all(state, {}, idle, None, [0.0, 0.0], True, True, 12.5)
        ctx = plain(app.query_one("#d-ctx", Static))
        check("ctx falls back to the slots when the gauge is missing",
              "25% of kv cache" in ctx, True)
        check("and does not report an idle slot as empty", "idle" in ctx, False)

        toks = plain(app.query_one("#d-toks", Static))
        check("peak is the high-water mark, not the best of the window",
              "12.5 peak" in toks, True)
        check("now still comes from the series", "0.0 now" in toks, True)

        # The gauge wins wherever a build still exports it.
        dash.update_all(state, {"llamacpp:kv_cache_usage_ratio": 0.5}, idle,
                        None, [1.0], True, True, 1.0)
        check("the gauge is preferred when present",
              "50% of kv cache" in plain(app.query_one("#d-ctx", Static)), True)
    teardown(root)

    # --- a name too long for the pane -------------------------------------
    root, port = sandbox(extra=LONG)
    app = MdlApp(fx="off")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        table = app.query_one("#models", DataTable)
        shown = [table.get_row_at(r)[0].plain for r in range(table.row_count)]
        check("the long name is truncated", any(x.endswith("…")
                                                for x in shown), True)
        check("but no ctx value is clipped off the pane",
              "65536" in pane_text(table), True)
        for row in range(table.row_count):
            table.move_cursor(row=row)
            await pilot.pause()
            check("selection resolves to the real name, not the label",
                  app._selected() in app.models, True)
    teardown(root)
    return t.done()


sys.exit(asyncio.run(main()))
