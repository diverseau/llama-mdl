"""The prompt screen: reasoning, live stats, interrupt, conversation memory.

Driven against the fake server's streaming chat endpoint, in both shapes
llama.cpp hands reasoning back: its own delta field, and inline <think>
tags that can straddle a chunk boundary.
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                        # noqa: E402
from support import mdl, run, sandbox, teardown       # noqa: E402

import mdl_ui                                         # noqa: E402
from mdl_ui import MdlApp, PromptScreen               # noqa: E402
from textual.widgets import Input, Static             # noqa: E402

t = support.Tally("test_chat")
check = t.check


def plain(widget):
    r = widget.render()
    return r.plain if hasattr(r, "plain") else str(r)


async def settle(pilot, screen, phases=("done", "stopped", "error"), limit=15.0):
    """Wait for the stream to land rather than guessing at a sleep."""
    for _ in range(int(limit / 0.05)):
        if screen.phase in phases:
            return True
        await pilot.pause()
        await asyncio.sleep(0.05)
    return False


# ------------------------------------------------------- tag splitting ----
# No server needed: this is the part that goes wrong silently.
check("partial_tag spots a straddled opener",
      mdl_ui.partial_tag("hello <th", "<think>"), 3)
check("partial_tag ignores a false lead",
      mdl_ui.partial_tag("hello there", "<think>"), 0)
check("partial_tag will not over-claim",
      mdl_ui.partial_tag("<", "<think>"), 1)

screen = PromptScreen(1234, "demo")
check("plain text is plain", screen._split("hello"), [("text", "hello")])
check("only the partial tag is held back, not the text before it",
      screen._split(" <th"), [("text", " ")])
check("and resolves when the rest arrives",
      screen._split("ink>why"), [("reason", "why")])
check("closing returns to normal text",
      screen._split(" so</think>done"), [("reason", " so"), ("text", "done")])
check("a lone < is not mistaken for a tag",
      PromptScreen(1, "x")._split("a < b"), [("text", "a < b")])


async def main():
    for mode, label in (("", "reasoning_content"), ("tags", "<think> tags")):
        root, port = sandbox()
        os.environ["MDL_FAKE_MODE"] = mode
        run(mdl.cmd_run, ["demo"])

        app = MdlApp(fx="off")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("p")
            await pilot.pause()
            screen = app.screen
            check("%s: prompt screen opens" % label,
                  isinstance(screen, PromptScreen), True)
            check("%s: it knows the model" % label, screen.model, "demo")
            check("%s: header names model and port" % label,
                  "demo" in plain(app.screen.query_one("#chat-head", Static))
                  and str(port) in plain(app.screen.query_one("#chat-head", Static)), True)

            box = app.screen.query_one("#prompt-input", Input)
            box.value = "hi"
            await pilot.press("enter")
            await pilot.pause()
            check("%s: input locks while streaming" % label,
                  box.disabled or screen.phase in ("done", "error"), True)

            check("%s: the stream completes" % label,
                  await settle(pilot, screen), True)
            check("%s: it finished cleanly" % label, screen.phase, "done")

            body = plain(app.screen.query_one("#prompt-out", Static))
            check("%s: the question is echoed" % label, "hi" in body, True)
            check("%s: reasoning is shown" % label, "reasoning" in body, True)
            check("%s: reasoning text is shown" % label,
                  "thinking about it" in body, True)
            check("%s: the answer is shown" % label, "hello there!" in body, True)
            check("%s: think tags never reach the transcript" % label,
                  "<think>" in body or "</think>" in body, False)
            check("%s: the rate is reported" % label, "tok/s" in body, True)
            check("%s: server timings win over our own guess" % label,
                  "42.5 tok/s" in body, True)
            check("%s: time to first token is reported" % label,
                  "ttft" in body, True)
            check("%s: reasoning is timed" % label, screen.think_secs > 0, True)

            check("%s: the reply is remembered, reasoning is not" % label,
                  screen.history,
                  [{"role": "user", "content": "hi"},
                   {"role": "assistant", "content": "hello there!"}])

            box.value = "again"
            await pilot.press("enter")
            check("%s: the stream completes twice" % label,
                  await settle(pilot, screen), True)
            check("%s: context carries across turns" % label,
                  len(screen.history), 4)
            check("%s: input is usable again" % label, box.disabled, False)

            await pilot.press("ctrl+l")
            await pilot.pause()
            check("%s: ctrl+l clears both views" % label,
                  (screen.history, plain(app.screen.query_one("#prompt-out", Static))),
                  ([], ""))

            await pilot.press("escape")
            await pilot.pause()
            check("%s: escape closes when idle" % label,
                  isinstance(app.screen, PromptScreen), False)

        run(mdl.cmd_stop, [])
        teardown(root)

    # --- interrupting a stream -------------------------------------------
    root, port = sandbox()
    os.environ["MDL_FAKE_MODE"] = "slowchat"   # long enough to interrupt
    run(mdl.cmd_run, ["demo"])
    app = MdlApp(fx="off")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("p")
        await pilot.pause()
        screen = app.screen
        app.screen.query_one("#prompt-input", Input).value = "hi"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("escape")
        check("escape stops the stream instead of closing",
              isinstance(app.screen, PromptScreen), True)
        check("the interrupt lands", await settle(pilot, screen), True)
        check("it is reported as interrupted, not failed", screen.phase, "stopped")
        check("interrupted is said out loud",
              "interrupted" in plain(app.screen.query_one("#prompt-out", Static)), True)
        check("the input is handed back",
              app.screen.query_one("#prompt-input", Input).disabled, False)
        await pilot.press("escape")
        await pilot.pause()
        check("escape closes once it has stopped",
              isinstance(app.screen, PromptScreen), False)
    run(mdl.cmd_stop, [])
    teardown(root)

    # --- the server is gone ----------------------------------------------
    root, port = sandbox()
    app = MdlApp(fx="off")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = PromptScreen(support.free_port(), "ghost")
        await app.push_screen(screen)
        await pilot.pause()
        app.screen.query_one("#prompt-input", Input).value = "hi"
        await pilot.press("enter")
        check("a dead server settles", await settle(pilot, screen), True)
        check("it reports failure rather than hanging", screen.phase, "error")
        check("the reason is shown, not raised",
              "failed" in plain(app.screen.query_one("#prompt-out", Static)), True)
    teardown(root)


asyncio.run(main())
sys.exit(t.done())
