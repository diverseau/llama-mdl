"""Headless drive of the TUI. No real server, no real config."""
import asyncio
import sys
import inspect
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
from textual.selection import Selection               # noqa: E402
from textual.widgets import (DataTable, Input, RichLog,  # noqa: E402
                             Static)

BACKSLASH = chr(92)
t = support.Tally("test_ui")
check = t.check
GROUPED = ('\n[one]\n'
           'model = "%s"\n'
           'group = "qwen"\nctx = 4096\nport = 9997\n'
           '\n[two]\n'
           'model = "%s"\n'
           'group = "qwen"\nctx = 4096\nport = 9996\n'
           % ((str(support.FAKE).replace("\\", "/"),) * 2))
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

        # a vision model: its projector is a second file, and it is
        # weights on the same card as everything else
        await pilot.press("e")
        await pilot.pause()
        app.screen.query_one("#f-mmproj", Input).value = str(support.FAKE)
        app.screen._apply()
        await pilot.pause()
        saved = tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8"))
        check("mmproj is saved with forward slashes",
              saved["demo"]["mmproj"], str(support.FAKE).replace(BACKSLASH, "/"))
        check("and reaches the command line",
              "--mmproj" in argv.text, True)
        check("the params pane names the file, not the path",
              support.FAKE.name in plain(app.query_one("#p-params", Static)),
              True)

        w_with, _, _ = mdl_ui.vram_estimate(["llama-server"], 1 << 30, 1 << 29)
        w_without, _, _ = mdl_ui.vram_estimate(["llama-server"], 1 << 30)
        check("the projector counts towards vram",
              (w_with, w_without), (1536.0, 1024.0))

        await pilot.press("e")
        await pilot.pause()
        app.screen.query_one("#f-mmproj", Input).value = ""
        app.screen._apply()
        await pilot.pause()
        saved = tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8"))
        check("clearing mmproj removes it", "mmproj" in saved["demo"], False)

        # the VRAM estimate must not care which field a flag came from
        est, kv, partial = mdl_ui.vram_estimate(
            ["llama-server", "-c", "65536", "--cache-type-k", "q8_0"], 1 << 30)
        check("a quantised cache halves the kv allowance", (est, kv),
              (1024.0, 2048.0))
        check("nothing here keeps weights off the gpu", partial, False)
        _, plain_kv, _ = mdl_ui.vram_estimate(["llama-server", "-c", "65536"], 0)
        check("an unquantised one does not", plain_kv, 4096.0)
        _, from_key, _ = mdl_ui.vram_estimate(
            mdl.build_argv("demo", {"model": "m.gguf", "ctx": 65536,
                                    "kv_type": "q8_0"}, "llama-server"), 0)
        _, from_args, _ = mdl_ui.vram_estimate(
            mdl.build_argv("demo", {"model": "m.gguf", "ctx": 65536,
                                    "args": ["--cache-type-k", "q8_0"]},
                           "llama-server"), 0)
        check("kv_type and args agree", from_key, from_args)
        check("and a missing ctx falls back to llama.cpp's default",
              mdl_ui.vram_estimate(["llama-server"], 0)[1], 256.0)

        # weights that are not all on the card make the figure a ceiling
        for flags in (["--n-cpu-moe", "14"], ["-ngl", "20"],
                      ["-ot", "exps=CPU"], ["--cpu-moe"]):
            check("%s means the estimate is an upper bound" % flags[0],
                  mdl_ui.vram_estimate(["llama-server"] + flags, 0)[2], True)
        check("a full offload does not",
              mdl_ui.vram_estimate(["llama-server", "-ngl", "99"], 0)[2], False)

        # the bar shows what is in it, not just how full it is
        parts = [(6144.0, "#7aa2f7"), (2048.0, "#bb7af7")]
        drawn = mdl_ui.stacked_bar(parts, 12288, width=12).plain
        check("segments are sized in proportion",
              (drawn.count("█"), len(drawn)), (8, 12))
        over = mdl_ui.stacked_bar([(12288.0, "a"), (4096.0, "b")], 12288,
                                  width=12)
        check("an overflowing bar still shows both parts", over.plain,
              "█" * 12)
        check("in the proportion of the total, not the card",
              [sp.end - sp.start for sp in over.spans], [9, 3])

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

    # --- the small print ---------------------------------------------------
    root, port = sandbox(extra=LONG)
    app = MdlApp(fx="off")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        # two cards are one pool, not whichever one nvidia-smi lists first
        two = "1000, 8192" + chr(10) + "2000, 8192" + chr(10)
        mdl_ui._GPU_CACHE["at"] = 0
        real_run = mdl_ui.subprocess.run
        mdl_ui.subprocess.run = lambda *a, **k: type(
            "R", (), {"stdout": two})()
        mdl_ui.shutil.which = lambda x: "nvidia-smi"
        try:
            check("both cards are counted", mdl_ui.gpu_memory(), (3000, 16384))
        finally:
            mdl_ui.subprocess.run = real_run
            mdl_ui._GPU_CACHE["at"] = 0

        # readiness is /health, and the timeout is the configured one
        src = inspect.getsource(app._watch_start)
        check("startup waits on /health, not on a log line",
              "server_ready" in src and "READY.search" not in src, True)
        check("and honours ready_timeout from the config",
              "ready_timeout()" in src, True)

        # restarting must not pick a row by its displayed label
        table = app.query_one("#models", DataTable)
        long_name = [n for n in app.models if len(n) > mdl_ui.NAME_WIDTH][0]
        elsewhere = [r for r in range(table.row_count)
                     if table.ordered_rows[r].key.value != long_name][0]
        table.move_cursor(row=elsewhere)     # so a miss leaves it on the wrong one
        await pilot.pause()
        ran = []
        app.action_run = lambda: ran.append(app._selected())
        try:
            app._restart_run(long_name)
            await pilot.pause()
        finally:
            del app.action_run
        check("restart lands on the model it was given", ran, [long_name])

        # a model whose file is gone says so rather than reading 0B
        app.sizes[long_name] = None
        app._build_table()
        await pilot.pause()
        shown = [table.get_row_at(r)[1].plain for r in range(table.row_count)]
        check("a missing file is named, not sized", "missing" in shown, True)
        check("0B is no longer how absence looks", "0B" in shown, False)

    teardown(root)

    # --- a worker finishing after the widgets have gone -------------------
    # Removing them is not teardown, so this asks the question that
    # matters rather than whether the app is still up: does it raise?
    root, port = sandbox()
    app = MdlApp(fx="off")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for widget in app.query("#dash, #params, #models, #status, #log, #p-argv"):
            widget.remove()
        await pilot.pause()
        for label, call in (("the poll", app._tick),
                            ("the table builder", app._build_table),
                            ("a finished start", lambda: app._after_start(
                                "demo", True, None))):
            try:
                call()
                raised = None
            except Exception as e:               # noqa: BLE001
                raised = type(e).__name__
            check("%s outliving its widgets is not a crash" % label,
                  raised, None)
    teardown(root)

    # --- folders ----------------------------------------------------------
    root, port = sandbox(extra=GROUPED)
    app = MdlApp(fx="off")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        table = app.query_one("#models", DataTable)

        def rows():
            return [table.ordered_rows[r].key.value
                    for r in range(table.row_count)]

        check("ungrouped models come first, so a config with no groups"
              " looks untouched",
              [n for g, n in app._ordered()],
              ["demo", "one", "two"])
        check("a group gets a row of its own",
              rows(), ["demo", mdl_ui.GROUP_KEY + "qwen", "one", "two"])
        check("whose key can never collide with a model name",
              mdl_ui.GROUP_KEY.isalnum(), False)

        # the cursor on a folder is not a model
        table.move_cursor(row=rows().index(mdl_ui.GROUP_KEY + "qwen"))
        await pilot.pause()
        check("a group row selects as no model", app._selected(), None)
        check("but it knows which group it is", app._selected_group(),
              "qwen")

        # enter folds it rather than running anything
        ran = []
        real_spawn, mdl.spawn = mdl.spawn, lambda *a, **k: ran.append(a)
        try:
            await pilot.press("enter")
            await pilot.pause()
        finally:
            mdl.spawn = real_spawn
        check("enter on a group starts nothing", ran, [])
        check("it folds it instead", app.collapsed, {"qwen"})
        check("and its members go away", rows(),
              ["demo", mdl_ui.GROUP_KEY + "qwen"])

        # a server inside a folded group still has to show
        app.states = {"one": {"name": "one", "pid": 1, "port": port,
                              "started": time.time()}}
        app._build_table()
        await pilot.pause()
        header = table.get_row_at(rows().index(mdl_ui.GROUP_KEY + "qwen"))
        check("a folded group says something inside it is up",
              header[3].plain, "\u25b6")
        check("and counts what it is hiding", header[2].plain, "2")

        # one and two are presets of the same file, as grouped models
        # usually are, so the group costs one download and not two
        app.sizes["one"] = app.sizes["two"] = 4 << 30
        check("a shared gguf is counted once",
              app._group_bytes(["one", "two"]), 4 << 30)
        app.models["two"] = dict(app.models["two"],
                                 model=str(support.FAKE) + "-other")
        check("and a genuine second file is added",
              app._group_bytes(["one", "two"]), 8 << 30)
        check("a file that is not there adds nothing",
              app._group_bytes(["one", "missing-model"]), 4 << 30)

        # and the row itself must show the deduplicated figure, which
        # is where the sum was being claimed
        app.models["two"] = dict(app.models["two"],
                                 model=str(support.FAKE))
        app._build_table()
        await pilot.pause()
        header = table.get_row_at(rows().index(mdl_ui.GROUP_KEY + "qwen"))
        check("the group row shows one file, not six of it",
              header[1].plain, "4.0G")
        app.states = {}

        await pilot.press("enter")
        await pilot.pause()
        check("enter again unfolds it", app.collapsed, set())
        check("the fold is remembered on disk",
              app._groups_path().exists(), True)

        # and the editor sets the key, because a group is just a field
        table.move_cursor(row=rows().index("demo"))
        await pilot.pause()
        await pilot.press("e")
        await pilot.pause()
        app.screen.query_one("#f-group", Input).value = "qwen"
        app.screen._apply()
        await pilot.pause()
        saved = tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8"))
        check("the editor writes group", saved["demo"]["group"], "qwen")
        await pilot.press("e")
        await pilot.pause()
        app.screen.query_one("#f-group", Input).value = ""
        app.screen._apply()
        await pilot.pause()
        saved = tomllib.loads(mdl.CONFIG.read_text(encoding="utf-8"))
        check("and clearing it removes the key",
              "group" in saved["demo"], False)
    teardown(root)

    # --- the log can be got out of ----------------------------------------
    root, port = sandbox()
    app = MdlApp(fx="off")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        log = app.query_one("#log", mdl_ui.CopyableLog)
        copied = []
        app.copy_to_clipboard = copied.append

        await pilot.press("y")
        await pilot.pause()
        check("an empty log copies nothing", copied, [])

        for i in range(4):
            log.write("slot print_timing: id 0 | task %d" % i)
        await pilot.pause()
        check("a plain RichLog would have handed back nothing",
              RichLog.get_selection(log, Selection(None, None)), None)
        got = log.get_selection(Selection(None, None))
        check("ours hands back its lines",
              got[0].splitlines()[0], "slot print_timing: id 0 | task 0")

        await pilot.press("y")
        await pilot.pause()
        check("y copies the whole buffer", len(copied), 1)
        check("all of it", len(copied[0].splitlines()), 4)
        check("and it is the text, not the styling",
              "task 3" in copied[0], True)
    teardown(root)
    return t.done()


sys.exit(asyncio.run(main()))
