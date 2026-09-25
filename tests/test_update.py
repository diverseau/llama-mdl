"""mdl update, the daily check, and the dashboard's offer. Never the real
PyPI and never a real installer: a fake index on localhost, and installers
that are a line of Python."""
import asyncio
import http.server
import json
import os
import sys
import threading
import time
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                        # noqa: E402
from support import mdl, run, sandbox, teardown       # noqa: E402

import mdl_update                                     # noqa: E402

t = support.Tally("test_update")
check = t.check
NEWER = "99.0.0"
CUR = mdl.VERSION


def raises(fn, *args, **kw):
    """The MdlError message, or None if fn returned."""
    try:
        fn(*args, **kw)
    except mdl.MdlError as e:
        return str(e)
    return None


def pypi(*versions, requires=">=3.11", yanked=(), files=True):
    """PyPI's JSON shape, as much of it as mdl reads."""
    return {"info": {"version": versions[-1] if versions else ""},
            "releases": {v: ([{"filename": "llama_mdl-%s.whl" % v,
                               "requires_python": requires,
                               "yanked": v in yanked}] if files else [])
                         for v in versions}}


class Index(http.server.BaseHTTPRequestHandler):
    """Answers every GET with whatever the test last put in `reply`."""
    reply = (200, b"{}")
    delay = 0.0
    hits = 0

    def do_GET(self):
        type(self).hits += 1
        time.sleep(type(self).delay)
        code, body = type(self).reply
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def serve(data=None, code=200, raw=None, delay=0.0):
    Index.reply = (code, raw if raw is not None else json.dumps(data).encode())
    Index.delay = delay


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Index)
server.handle_error = lambda *a: None     # a client that timed out and left
threading.Thread(target=server.serve_forever, daemon=True).start()
os.environ["MDL_PYPI_URL"] = "http://127.0.0.1:%d/pypi/llama-mdl/json" % (
    server.server_address[1])
os.environ.pop("MDL_NO_UPDATE_CHECK", None)
root, port = sandbox()
os.environ["XDG_CACHE_HOME"] = str(root / "cache")
os.environ["MDL_FIT_HOME"] = str(root / "fit")


def fresh_cache():
    mdl_update.cache_path().unlink(missing_ok=True)


class Dist:
    """Enough of importlib.metadata.Distribution for detect()."""

    def __init__(self, where, direct=None, record=True):
        self.where, self.direct, self.record = Path(where), direct, record

    def read_text(self, name):
        if name == "RECORD":
            # an installed .dist-info lists its files; an egg-info has none
            return "mdl.py,,\n" if self.record else None
        return json.dumps(self.direct) if self.direct else None

    def locate_file(self, name):
        return self.where / name


here = Path(mdl.__file__).resolve()

try:
    # ---------------------------------------------------------- versions
    check("plain versions parse", mdl_update.parse("0.10.0"), (0, 10, 0))
    for bad in ("0.11.0rc1", "1.0.dev2", "0.10.0.post1+local", "", None, "v1"):
        check("never offered: %r" % (bad,), mdl_update.parse(bad), None)
    check("0.10 is newer than 0.9.9", mdl_update.is_newer("0.10", "0.9.9"), True)
    check("1.0 equals 1.0.0", mdl_update.is_newer("1.0", "1.0.0"), False)
    check("an unparseable current is never behind",
          mdl_update.is_newer("1.0.0", "1.0.0rc1"), False)
    for spec, have, ok in [(">=3.11", (3, 11, 0), True),
                           (">=3.12", (3, 11, 9), False),
                           (">=3.11,<3.14", (3, 14, 0), False),
                           ("==3.11.*", (3, 11, 4), True),
                           ("==3.11.*", (3, 12, 0), False),
                           ("!=3.12.*", (3, 12, 1), False),
                           ("~=3.11", (3, 13, 0), True),
                           ("~=3.11", (4, 0, 0), False),
                           ("", (3, 11, 0), True),
                           (None, (3, 11, 0), True),
                           ("something odd", (3, 11, 0), True)]:
        check("requires_python %r on %s" % (spec, have),
              mdl_update.python_ok(spec, have), ok)

    check("newest is by number, not by listing order",
          mdl_update.newest(pypi("0.10.0", "0.9.0", "0.2.0")), "0.10.0")
    check("pre-releases skipped",
          mdl_update.newest(pypi("0.10.0", "0.11.0rc1")), "0.10.0")
    check("yanked skipped",
          mdl_update.newest(pypi("0.10.0", "0.11.0", yanked={"0.11.0"})),
          "0.10.0")
    check("a release without files skipped",
          mdl_update.newest(pypi("0.11.0", files=False)), None)
    check("a release this Python cannot run skipped",
          mdl_update.newest(pypi("0.11.0", requires=">=3.99"), (3, 11, 0)),
          None)

    # ------------------------------------------------------------- fetch
    serve(pypi(CUR, NEWER))
    check("fetch reads the index", mdl_update.fetch(), NEWER)
    for label, kw in [("HTTP 500", {"code": 500, "raw": b"oops"}),
                      ("HTTP 404", {"code": 404, "raw": b"{}"}),
                      ("not JSON", {"raw": b"<html>"}),
                      ("JSON, wrong shape", {"raw": b"[1, 2]"}),
                      ("nothing installable", {"data": pypi("1.0rc1")})]:
        serve(**kw)
        msg = raises(mdl_update.fetch)
        check("fetch fails in one line: " + label,
              (msg is not None, "\n" not in (msg or "")), (True, True))
    serve(pypi(NEWER), delay=1.0)
    check("a slow index times out", raises(mdl_update.fetch, 0.2) is not None,
          True)
    serve(pypi(NEWER))

    # ------------------------------------------------------------- cache
    fresh_cache()
    Index.hits = 0
    check("first check asks", (mdl_update.check(), Index.hits), (NEWER, 1))
    check("second check is the cache", (mdl_update.check(), Index.hits),
          (NEWER, 1))
    check("a stale cache asks again",
          (mdl_update.check(max_age=0), Index.hits), (NEWER, 2))
    serve(code=500, raw=b"")
    check("a failed check keeps what it knew",
          mdl_update.check(max_age=0), NEWER)
    Index.hits = 0
    check("and does not ask again for a day",
          (mdl_update.check(), Index.hits), (NEWER, 0))
    serve(pypi(NEWER))
    fresh_cache()
    check("offered", mdl_update.offer(), NEWER)
    mdl_update.skip(NEWER)
    check("skipped is not offered", mdl_update.offer(), None)
    serve(pypi(NEWER, "99.0.1"))
    check("a newer one than the skipped is",
          mdl_update.check(max_age=0) and mdl_update.offer(), "99.0.1")
    serve(pypi(CUR))
    fresh_cache()
    check("the running version is not offered", mdl_update.offer(), None)
    mdl_update.cache_path().write_text("not json")
    serve(pypi(NEWER))
    check("a corrupt cache is asked past", mdl_update.offer(), NEWER)

    # ------------------------------------------------ nothing ever pays
    # every way the index can be wrong: no offer, and nothing raised
    for label, kw in [("500", {"code": 500, "raw": b""}),
                      ("html", {"raw": b"<html></html>"}),
                      ("list", {"raw": b"[]"}),
                      ("no releases", {"raw": b'{"info": {}}'}),
                      ("releases a list", {"raw": b'{"releases": []}'}),
                      ("files not a list",
                       {"raw": b'{"releases": {"99.0.0": {}}}'}),
                      ("files not dicts",
                       {"raw": b'{"releases": {"99.0.0": [1, "x"]}}'}),
                      ("all yanked", {"data": pypi(NEWER, yanked={NEWER})}),
                      ("only pre-releases", {"data": pypi("99.0.0b1")}),
                      ("python too new",
                       {"data": pypi(NEWER, requires=">=3.99")}),
                      ("older than this", {"data": pypi("0.0.1")})]:
        fresh_cache()
        serve(**kw)
        try:
            got = mdl_update.offer()
        except Exception as e:           # the point is that nothing escapes
            got = "raised %r" % e
        check("no offer from a bad index: " + label, got, None)
    serve(pypi(NEWER))

    # ---------------------------------------------------------- disabled
    with patch.object(mdl_update, "detect",
                      return_value=mdl_update.Install("pip", root)):
        check("on by default", mdl_update.disabled(), None)
        with patch.dict(os.environ, {"MDL_NO_UPDATE_CHECK": "1"}):
            check("off by environment", "MDL_NO_UPDATE_CHECK"
                  in (mdl_update.disabled() or ""), True)
        with patch.dict(os.environ, {"MDL_NO_UPDATE_CHECK": "0"}):
            check("=0 leaves it on", mdl_update.disabled(), None)
        original = mdl.CONFIG.read_text()
        mdl.CONFIG.write_text("update_check = false\n" + original)
        check("off by config", "update_check" in (mdl_update.disabled() or ""),
              True)
        mdl.CONFIG.write_text(original)
    check("off in a source checkout",
          "source" in (mdl_update.disabled() or ""), True)

    # ------------------------------------------------------------ detect
    check("the tests run from a checkout", mdl_update.detect().kind, "source")
    prefix = root / "prefix"
    prefix.mkdir()
    def dists(*found):
        return patch("importlib.metadata.distributions",
                     return_value=list(found))

    with dists(Dist(here.parent)), patch.object(sys, "prefix", str(prefix)):
        check("pip, when the running file is the installed one",
              mdl_update.detect().kind, "pip")
        (prefix / "uv-receipt.toml").write_text("")
        check("uv tool", mdl_update.detect().kind, "uv")
        (prefix / "pipx_metadata.json").write_text("{}")
        check("pipx", mdl_update.detect().kind, "pipx")
    with dists(Dist(root / "elsewhere")):
        check("an installed copy that is not the running one",
              mdl_update.detect().kind, "source")
    with dists(Dist(here.parent, {"url": "file:///x",
                                  "dir_info": {"editable": True}})):
        check("editable", mdl_update.detect().kind, "source")
    # what CI had: `pip install .` leaves an egg-info in the checkout,
    # pointing at the checkout's own mdl.py, found before the install
    with dists(Dist(here.parent, record=False), Dist(root / "site")):
        check("a build's leftover egg-info is not an install",
              mdl_update.detect().kind, "source")
    with dists(Dist(root / "site", record=False), Dist(here.parent)), \
            patch.object(sys, "prefix", str(root / "plain")):
        check("the real install is found behind a leftover",
              mdl_update.detect().kind, "pip")
    # pip's mdl.exe is a zip at the head of sys.path; reading metadata out
    # of it held it open, and the update could not move it aside
    launcher_zip = root / "mdl.exe"
    launcher_zip.write_bytes(b"PK\x05\x06" + bytes(18))
    asked = []
    with patch.object(sys, "path", [str(launcher_zip), str(root)] + sys.path), \
            patch("importlib.metadata.distributions",
                  lambda **kw: asked.append(kw["path"]) or []):
        mdl_update.detect()
    check("metadata is read from directories only, never a zip",
          (str(launcher_zip) in asked[0], str(root) in asked[0]),
          (False, True))

    # ----------------------------------------------------------- command
    Inst = mdl_update.Install
    check("pipx upgrades by name", mdl_update.command(Inst("pipx", root), NEWER),
          ["pipx", "upgrade", "llama-mdl"])
    check("uv upgrades by name", mdl_update.command(Inst("uv", root), NEWER),
          ["uv", "tool", "upgrade", "llama-mdl"])
    with patch("importlib.util.find_spec", return_value=object()):
        argv = mdl_update.command(Inst("pip", root), NEWER)
        check("pip pins, keeps the extra, and is this interpreter's",
              (argv[:3], argv[-1]), ([sys.executable, "-m", "pip"],
                                     "llama-mdl[ui]==" + NEWER))
        check("--user kept", mdl_update.command(Inst("pip", root, True),
                                                NEWER)[-1], "--user")
    with patch("importlib.util.find_spec", return_value=None):
        check("no extra without textual",
              mdl_update.command(Inst("pip", root), NEWER)[-1],
              "llama-mdl==" + NEWER)
    check("a checkout is never installed over",
          "git pull" in (raises(mdl_update.command, Inst("source", root), NEWER)
                         or ""), True)

    # ----------------------------------------------------------- install
    def fake(*lines, code=0):
        body = "".join("print(%r);" % x for x in lines)
        return patch.object(mdl_update, "command", return_value=[
            sys.executable, "-c", body + "raise SystemExit(%d)" % code])

    pip_here = patch.object(mdl_update, "detect",
                            return_value=Inst("pip", root))
    lines = []
    with pip_here, fake("Collecting llama-mdl", "Successfully installed"), \
            patch.object(mdl_update, "installed_version", return_value=NEWER):
        check("install reports the version now there",
              mdl_update.install(NEWER, out=lines.append), NEWER)
    check("installer output streamed", lines[1:],
          ["Collecting llama-mdl", "Successfully installed"])
    with pip_here, fake("ERROR: No matching distribution", code=1):
        msg = raises(mdl_update.install, NEWER, out=lambda _: None)
    check("a failed installer: its error, and still on this version",
          ("No matching distribution" in (msg or ""),
           ("still on " + CUR) in (msg or "")), (True, True))
    with pip_here, fake("ok"), \
            patch.object(mdl_update, "installed_version", return_value=CUR):
        msg = raises(mdl_update.install, NEWER, out=lambda _: None)
    check("an installer that changed nothing is a failure",
          "still " + CUR in (msg or ""), True)
    with pip_here, fake("ok"), \
            patch.object(mdl_update, "installed_version", return_value=None):
        msg = raises(mdl_update.install, NEWER, out=lambda _: None)
    check("an install that no longer imports says how to repair it",
          "no longer imports" in (msg or ""), True)
    with patch.object(mdl_update, "detect", return_value=Inst("pipx", root)), \
            fake("ok"), \
            patch.object(mdl_update, "installed_version", return_value=CUR):
        msg = raises(mdl_update.install, NEWER, out=lambda _: None)
    check("pipx left behind hints at a pinned install",
          "pinned" in (msg or ""), True)

    # pip's reason is its first ERROR line; the notes after it are not
    pep668 = ["error: externally-managed-environment", "",
              "x This environment is externally managed",
              "  To install Python packages system-wide, try apt install",
              "note: If you believe this is a mistake, please contact your "
              "Python installation or OS distribution provider.",
              "hint: See PEP 668 for the detailed specification."]
    with pip_here, fake(*pep668, code=1):
        msg = raises(mdl_update.install, NEWER, out=lambda _: None)
    check("an OS-managed Python (PEP 668) says so, and what to do",
          ("PEP 668" in (msg or ""), "pipx install" in (msg or ""),
           "hint:" in (msg or "")), (True, True, False))
    with pip_here, fake("Collecting llama-mdl",
                        "ERROR: Could not install packages due to an OSError: "
                        "[Errno 13] Permission denied: '/usr/lib/x'",
                        "", "[notice] A new release of pip is available",
                        code=1):
        msg = raises(mdl_update.install, NEWER, out=lambda _: None)
    check("a system-wide install: pip's error, and who can update it",
          ("Permission denied" in (msg or ""),
           "where you cannot write" in (msg or ""),
           "[notice]" in (msg or "")), (True, True, False))
    with patch.object(mdl_update, "detect", return_value=Inst("pipx", root)), \
            patch.object(mdl_update, "command",
                         return_value=["no-such-pipx-here", "upgrade"]):
        msg = raises(mdl_update.install, NEWER, out=lambda _: None)
    check("pipx missing from PATH says so",
          "on your PATH" in (msg or ""), True)

    # another mdl mid-eval: refused, unless forced
    runs = root / "fit" / "eval-runs"
    runs.mkdir(parents=True)
    lock = runs / "abc.lock"
    lock.write_text(str(os.getppid()))
    with pip_here, fake("ok"), \
            patch.object(mdl_update, "installed_version", return_value=NEWER):
        msg = raises(mdl_update.install, NEWER, out=lambda _: None)
        check("refused while an eval holds its lock",
              "mdl eval (pid %d)" % os.getppid() in (msg or ""), True)
        check("--force goes ahead",
              mdl_update.install(NEWER, out=lambda _: None, force=True), NEWER)
    lock.write_text("999999999")
    check("a dead holder is no obstacle", mdl_update.busy(), [])

    # the launcher Windows will not let uv overwrite
    if os.name == "nt":
        bindir = root / "bin"
        bindir.mkdir()
        exe = bindir / "mdl.exe"
        exe.write_bytes(b"old launcher")
        writes = ("open(%r, 'wb').write(b'new launcher')" % str(exe))
        with patch.object(sys, "argv", [str(bindir / "mdl"), "update"]), \
                pip_here:
            check("the running launcher is found", mdl_update._launcher(), exe)
            with patch.object(mdl_update, "command", return_value=[
                    sys.executable, "-c", writes]), \
                    patch.object(mdl_update, "installed_version",
                                 return_value=NEWER):
                mdl_update.install(NEWER, out=lambda _: None)
            check("a new launcher lands at the free name",
                  exe.read_bytes(), b"new launcher")
            left = list(bindir.glob("mdl.exe.old-*"))
            check("the old one is set aside", len(left), 1)
            mdl_update.sweep()
            check("and swept up next time", list(bindir.glob("mdl.exe.old-*")),
                  [])
            with fake("boom", code=2):
                raises(mdl_update.install, NEWER, out=lambda _: None)
            check("a failed install puts the launcher back",
                  (exe.read_bytes(), list(bindir.glob("mdl.exe.old-*"))),
                  (b"new launcher", []))
            # a launcher that cannot be moved aside: pip would fail on the
            # same file halfway through, so the installer never starts
            ran = root / "installer-ran"
            with patch.object(mdl_update, "command", return_value=[
                    sys.executable, "-c",
                    "open(%r, 'w').write('x')" % str(ran)]), \
                    patch.object(mdl_update.os, "replace",
                                 side_effect=PermissionError(13, "in use")):
                msg = raises(mdl_update.install, NEWER, out=lambda _: None)
            check("a launcher held open stops the update before it starts",
                  ("nothing was changed" in (msg or ""), ran.exists()),
                  (True, False))
        with patch.object(sys, "argv", [str(root / "mdl.py")]):
            check("python mdl.py has no launcher", mdl_update._launcher(), None)

    # ------------------------------------------------------------ the CLI
    serve(pypi(CUR, NEWER))
    out, err, code = run(mdl.cmd_update, ["--check"])
    check("--check says what is out", (code, ("%s is out" % NEWER) in out),
          (0, True))
    out, err, code = run(mdl.cmd_update, [])
    check("from a checkout: refused, pointing at git",
          (code, "git pull" in err), (1, True))
    check("one line", err.count("\n"), 1)
    serve(pypi(CUR))
    out, err, code = run(mdl.cmd_update, [])
    check("up to date", (code, out.strip()),
          (0, "mdl %s is the newest release" % CUR))
    serve(pypi("0.0.1"))
    out, err, code = run(mdl.cmd_update, ["--check"])
    check("a build ahead of PyPI says so", "newer than the newest" in out,
          True)
    out, err, code = run(mdl.cmd_update, ["--now"])
    check("an unknown flag is usage", (code, "usage: mdl update" in err),
          (1, True))
    serve(code=503, raw=b"")
    out, err, code = run(mdl.cmd_update, ["--check"])
    check("an index that is down: one line", (code, err.count("\n")), (1, 1))
    serve(pypi(CUR, NEWER))
    lines = []
    with pip_here, fake("ok"), \
            patch.object(mdl_update, "installed_version", return_value=NEWER), \
            patch.object(mdl_update, "path_note", return_value=None):
        out, err, code = run(mdl.cmd_update, [])
    check("an update, start to finish",
          (code, "updated mdl %s -> %s" % (CUR, NEWER) in out), (0, True))
    check("the dashboard's cache learns from it",
          json.loads(mdl_update.cache_path().read_text())["latest"], NEWER)
    check("usage lists update", "update [--check]" in mdl.USAGE, True)

    # the mdl first on PATH being another install
    other = support._launcher(root)      # prints "version: 1234 (fake)"
    with patch("shutil.which", return_value=str(other)):
        note = mdl_update.path_note(NEWER)
    check("another mdl on PATH is pointed out",
          note is not None and str(other) in note, True)

    # ------------------------------------------------------------ restart
    calls = []
    with patch.object(os, "execve", lambda *a: calls.append(a)), \
            patch("subprocess.call", lambda argv, env: calls.append(
                (argv, env)) or 0), \
            patch.object(sys, "exit", lambda code=0: calls.append(code)):
        mdl_update.restart(["ui", "--no-fx"])
    argv, env = (calls[0][1], calls[0][2]) if os.name != "nt" else calls[0]
    check("restarts the new code, never a stray mdl.py",
          argv, [sys.executable, "-P", "-m", "mdl", "ui", "--no-fx"])
    check("and tells it what it came from", env.get("MDL_UPDATED_FROM"), CUR)
    seen = []
    with patch.object(mdl_update, "restart", seen.append):
        mdl._after_ui(None, ["ui"])
        mdl._after_ui(mdl_update.RESTART, ["ui", "--no-fx"])
    check("only a dashboard that updated is restarted", seen,
          [["ui", "--no-fx"]])
    if os.name != "nt":
        def refuse(*a):
            raise PermissionError(13, "Permission denied")

        with patch.object(os, "execve", refuse):
            msg = raises(mdl_update.restart, ["ui"])
        check("a restart that cannot exec is one line, not a traceback",
              "could not start again" in (msg or ""), True)

    # macOS python.org builds: no CA certificates until their installer
    # script runs; the hint is for that platform only
    def no_certs(*a, **kw):
        raise urllib.error.URLError(
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")

    with patch("urllib.request.urlopen", no_certs):
        with patch.object(sys, "platform", "darwin"):
            mac = raises(mdl_update.fetch)
        with patch.object(sys, "platform", "linux"):
            linux = raises(mdl_update.fetch)
    check("missing certificates on macOS point at the fix",
          ("Install Certificates.command" in (mac or ""),
           "Install Certificates" in (linux or "")), (True, False))

    # ------------------------------------------------------------- doctor
    def doctor():
        out, _, _ = run(mdl.cmd_doctor, ["--json"])
        return [n for n in json.loads(out)["global"] if n["check"] == "update"]

    check("doctor quiet in a checkout", doctor(), [])
    with patch.object(mdl_update, "disabled", return_value=None):
        with patch.object(mdl_update, "check", return_value=NEWER):
            check("doctor warns of a newer release",
                  [n["level"] for n in doctor()], ["warn"])
        with patch.object(mdl_update, "check", return_value=CUR):
            check("doctor: newest", [n["level"] for n in doctor()], ["ok"])
        with patch.object(mdl_update, "check", return_value=None):
            check("doctor quiet when PyPI cannot be reached", doctor(), [])

    # --------------------------------------------------------- the dashboard
    from mdl_ui import MdlApp, UpdateScreen
    from textual.widgets import Button, RichLog, Static

    async def until(pilot, cond, limit=5.0):
        end = time.monotonic() + limit
        while not cond() and time.monotonic() < end:
            await pilot.pause(0.05)
        return cond()

    def status(app):
        r = app.query_one("#status", Static).render()
        return r.plain if hasattr(r, "plain") else str(r)

    def offered(app):
        """The popup is up and composed: pushed alone is not enough to
        press its buttons."""
        return (isinstance(app.screen, UpdateScreen)
                and bool(app.screen.query("#update-go")))

    async def ui():
        offering = patch.multiple(mdl_update, disabled=lambda: None,
                                  offer=lambda: NEWER, sweep=lambda: None)
        with offering:
            app = MdlApp(fx="off")
            async with app.run_test(size=(120, 40)) as pilot:
                check("the offer pops up", await until(
                    pilot, lambda: offered(app)), True)
                check("u is live", app.check_action("update", ()), True)
                check("the status line says so",
                      ("%s is out" % NEWER) in status(app), True)
                await pilot.press("escape")
                await pilot.pause()
                check("later closes it", isinstance(app.screen, UpdateScreen),
                      False)
                check("and the hint stays", ("%s is out" % NEWER) in status(app),
                      True)
                await pilot.press("u")
                await pilot.pause()
                check("u brings it back", isinstance(app.screen, UpdateScreen),
                      True)

                # a measurement running: the restart would cut it off
                gate = threading.Event()
                app.run_worker(gate.wait, thread=True, group="lab")
                await pilot.pause()
                app.screen.query_one("#update-go", Button).press()
                await pilot.pause()
                check("refused while a measurement runs",
                      app.screen.installing, False)
                gate.set()
                await pilot.pause(0.2)

                # an install that fails: the popup stays, with why
                with patch.object(mdl_update, "install",
                                  side_effect=mdl.MdlError("pip exploded")):
                    app.screen.query_one("#update-go", Button).press()
                    await until(pilot, lambda: not app.screen.installing
                                and app.screen.query_one(
                                    "#update-log", RichLog).lines)
                screen = app.screen
                log = "\n".join(s.text for s in screen.query_one(
                    "#update-log", RichLog).lines)
                check("a failure stays on screen", (isinstance(
                    screen, UpdateScreen), "pip exploded" in log), (True, True))
                check("and can be retried",
                      screen.query_one("#update-go", Button).disabled, False)

                # mid-install, nothing closes it
                gate = threading.Event()

                def slow(version, out=print, force=False):
                    out("Collecting llama-mdl")
                    gate.wait(5)
                    raise mdl.MdlError("stopped by the test")

                with patch.object(mdl_update, "install", slow):
                    screen.query_one("#update-go", Button).press()
                    await pilot.pause(0.2)
                    await pilot.press("escape")
                    await pilot.press("q")
                    await pilot.pause()
                    check("escape and q wait for the install",
                          (app.screen is screen, app.is_running), (True, True))
                    gate.set()
                    await until(pilot, lambda: not screen.installing)

                # skip: gone, and remembered
                fresh_cache()
                screen.query_one("#update-skip", Button).press()
                await pilot.pause()
                check("skip closes it", isinstance(app.screen, UpdateScreen),
                      False)
                check("skip is remembered", json.loads(
                    mdl_update.cache_path().read_text()).get("skip"), NEWER)
                check("and u goes quiet", app.check_action("update", ()),
                      False)

        # an install that works exits asking for the restart
        with offering, patch.object(mdl_update, "install",
                                    return_value=NEWER):
            app = MdlApp(fx="off")
            async with app.run_test(size=(120, 40)) as pilot:
                await until(pilot, lambda: offered(app))
                app.screen.query_one("#update-go", Button).press()
                await until(pilot, lambda: not app.is_running)
            check("a finished update exits for the restart",
                  app.return_value, mdl_update.RESTART)

        # not over another popup: the hint only
        with patch.multiple(mdl_update, disabled=lambda: None,
                            offer=lambda: None, sweep=lambda: None):
            app = MdlApp(fx="off")
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                check("u is hidden with nothing to offer",
                      app.check_action("update", ()), False)
                await pilot.press("question_mark")
                await pilot.pause()
                app._offer_update(NEWER)
                await pilot.pause()
                check("no popup over the help",
                      type(app.screen).__name__, "HelpScreen")
                check("but the hint is there",
                      ("%s is out" % NEWER) in status(app), True)

        # off, or broken: never a popup, never a crash
        asked = []
        os.environ["MDL_UPDATED_FROM"] = "0.9.0"
        with patch.multiple(mdl_update, disabled=lambda: "off",
                            offer=lambda: asked.append(1)):
            app = MdlApp(fx="off")
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.3)
                check("off means not asked", asked, [])
                check("a restart says it updated",
                      "updated mdl 0.9.0 -> %s" % CUR in status(app), True)
                check("and does not pass that on",
                      "MDL_UPDATED_FROM" in os.environ, False)

        def boom():
            raise RuntimeError("anything at all")

        with patch.multiple(mdl_update, disabled=lambda: None, offer=boom,
                            sweep=lambda: None):
            app = MdlApp(fx="off")
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.3)
                check("a check that blows up is ignored",
                      (app.is_running, isinstance(app.screen, UpdateScreen)),
                      (True, False))

    asyncio.run(ui())
finally:
    server.shutdown()
    teardown(root)

sys.exit(t.done())
