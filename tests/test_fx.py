"""Wordmark animation: modes, and that the configured speed is what you get.

Phase wraps modulo 1.0, so nothing here compares raw phase values across a
sample gap - it counts wraps, or measures when motion stops. Timing comes
from real clocks, never index*interval: pilot.pause() makes iterations
longer than the nominal step and silently skews anything derived from it.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                # noqa: E402
from support import sandbox, teardown         # noqa: E402

import mdl_ui                                 # noqa: E402
from mdl_ui import MdlApp                     # noqa: E402

t = support.Tally("test_fx")
check = t.check
STEP = 0.05


async def sample(mode, seconds, period=None):
    """(samples, cycles_travelled, elapsed); samples is [(t, phase), ...]."""
    app = MdlApp(fx=mode, fx_period_override=period)
    samples, wraps = [], 0
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        wm = app.query_one("#wordmark", mdl_ui.Wordmark)
        start = prev = wm._phase
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            await asyncio.sleep(STEP)
            await pilot.pause()
            if wm._phase < prev:
                wraps += 1
            prev = wm._phase
            samples.append((time.monotonic() - t0, round(wm._phase, 4)))
        elapsed = time.monotonic() - t0
    return samples, wraps + (prev - start), elapsed


def last_motion(samples):
    last = 0.0
    for i in range(1, len(samples)):
        if samples[i][1] != samples[i - 1][1]:
            last = samples[i][0]
    return last


async def main():
    root, _ = sandbox()

    samples, travelled, elapsed = await sample("always", 4.0, period=1.0)
    moved = last_motion(samples)
    print("      always: %.1f cycles in %.1fs, last motion %.1fs"
          % (travelled, elapsed, moved))
    check("always: moves", travelled > 0, True)
    check("always: never stops", moved > elapsed - 0.5, True)
    check("always: wraps past a full cycle", travelled > 1.0, True)

    samples, travelled, _ = await sample("sweep", 4.0)
    stopped = last_motion(samples)
    print("      sweep: stopped at ~%.1fs" % stopped)
    check("sweep: moves", travelled > 0, True)
    check("sweep: parks", stopped < 3.0, True)
    check("sweep: lasts about the 1.2s intended", 0.6 < stopped < 2.5, True)
    check("sweep: frozen at the end", samples[-1][1], samples[-2][1])

    samples, travelled, _ = await sample("off", 1.2)
    check("off: never moves", travelled, 0.0)
    check("off: phase stays at zero", {p for _, p in samples}, {0.0})

    for want in (2.0, 1.0):
        _, travelled, elapsed = await sample("always", 2.5, period=want)
        got = elapsed / travelled if travelled else 1e9
        print("      period %.1fs -> measured %.2fs" % (want, got))
        check("cycle takes ~%.1fs as configured" % want,
              abs(got - want) < want * 0.3, True)

    check("colour interpolation wraps seamlessly",
          mdl_ui.colour_at(0.0), mdl_ui.colour_at(1.0))
    check("phase changes the rendered colour",
          mdl_ui.gradient_text(["MDL"], phase=0.0).spans[0].style
          != mdl_ui.gradient_text(["MDL"], phase=0.4).spans[0].style, True)
    check("period floor is enforced", mdl_ui.fx_period(0.01), mdl_ui.FX_MIN_PERIOD)
    check("garbage period falls back", mdl_ui.fx_period("nonsense"), mdl_ui.FX_PERIOD)

    teardown(root)
    return t.done()


sys.exit(asyncio.run(main()))
