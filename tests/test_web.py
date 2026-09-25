"""mdl ui (the web page) and mdl snapshot: the gate on every request, the
snapshot, the event stream, run and stop, and the window it opens in."""
import http.client
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                                 # noqa: E402
from support import free_port, mdl, run, sandbox, teardown     # noqa: E402

from mdl_web import server, snapshot                           # noqa: E402

t = support.Tally("test_web")
check = t.check
HERE = Path(__file__).resolve().parent

# -------------------------------------------------------- pure helpers --
check("api_key from argv", [snapshot.api_key(a) for a in
                            (["-m", "x", "--api-key", "k1"], ["--api-key=k2"],
                             ["--api-key"], None)],
      ["k1", "k2", None, None])
got = snapshot.parse_metrics("# HELP x\nllamacpp:n_decode_total 12\n"
                             'llamacpp:thing{a="b"} 3.5\nbroken line\n')
check("parse_metrics reads values and skips comments",
      got.get("llamacpp:n_decode_total"), 12.0)

tr = snapshot.Tracker(points=3)
first = tr.observe(7, 100.0, {"llamacpp:n_decode_total": 10,
                              "llamacpp:prompt_tokens_total": 5,
                              "llamacpp:tokens_predicted_total": 10})
rate = tr.observe(7, 102.0, {"llamacpp:n_decode_total": 50,
                             "llamacpp:prompt_tokens_total": 5,
                             "llamacpp:tokens_predicted_total": 50})
check("tracker: no rate from one reading, then the delta over time",
      (first, rate), (None, 20.0))
for i in range(5):
    tr.observe(7, 103.0 + i, {"llamacpp:n_decode_total": 50,
                              "llamacpp:prompt_tokens_total": 5,
                              "llamacpp:tokens_predicted_total": 50})
check("tracker: the series is capped", len(tr.series[7]), 3)
tr.forget({8})
check("tracker: a gone server is forgotten", 7 in tr.series, False)

# ------------------------------------------------------------ a server --
keyed = free_port()
root, port = sandbox(extra=(
    '\n[keyed]\nmodel = "%s"\nctx = 4096\nport = %d\ngroup = "g"\n'
    'args = ["--metrics", "--api-key", "sekrit"]\n'
    % (str(support.FAKE).replace("\\", "/"), keyed)))

httpd, hub, url = server.serve(0)
parts = urllib.parse.urlsplit(url)
UI = parts.port
TOKEN = urllib.parse.parse_qs(parts.query)["t"][0]
HOST = "127.0.0.1:%d" % UI
ORIGIN = "http://" + HOST
COOKIE = "%s=%s" % (server.COOKIE, TOKEN)


def req(method, path, headers=None, body=None, host=HOST, cookie=True):
    h = {"Host": host}
    if cookie:
        h["Cookie"] = COOKIE
    h.update(headers or {})
    conn = http.client.HTTPConnection("127.0.0.1", UI, timeout=10)
    conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    if body is not None:
        body = json.dumps(body).encode() if not isinstance(body, bytes) else body
        h["Content-Length"] = str(len(body))
        h.setdefault("Content-Type", "application/json")
    for k, v in h.items():
        conn.putheader(k, v)
    conn.endheaders(body)
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, dict(r.getheaders()), data


def action(verb, name, **kw):
    kw.setdefault("headers", {"Origin": ORIGIN})
    code, _, body = req("POST", "/api/action", body={"verb": verb,
                                                      "name": name}, **kw)
    try:
        return code, json.loads(body)
    except ValueError:
        return code, body.decode(errors="replace").strip()


def snap():
    return json.loads(req("GET", "/api/snapshot")[2])


def model(s, name):
    return next((m for m in s["models"] if m["name"] == name), None)


def until(fn, secs=20):
    end = time.monotonic() + secs
    while time.monotonic() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.25)
    return fn()


try:
    # -- the token: once in a URL, then a cookie ---------------------------
    code, _, body = req("GET", "/", cookie=False)
    check("no token, no page", (code, body.strip()), (403, b"no token"))
    code, _, _ = req("GET", "/?t=wrong", cookie=False)
    check("a wrong token in the URL is refused", code, 403)
    code, h, _ = req("GET", "/?t=" + TOKEN, cookie=False)
    cookie = h.get("Set-Cookie", "")
    check("the token URL swaps for a cookie and a clean URL",
          (code, h.get("Location"), cookie.startswith(COOKIE),
           "HttpOnly" in cookie, "SameSite=Strict" in cookie),
          (303, "/", True, True, True))
    code, h, body = req("GET", "/")
    check("with the cookie, the page",
          (code, b"/static/app.js" in body, "script-src 'self'" in
           h.get("Content-Security-Policy", "")), (200, True, True))
    code, _, _ = req("GET", "/api/snapshot", cookie=False,
                     headers={"X-Mdl-Token": TOKEN})
    check("or with the token as a header", code, 200)
    code, _, _ = req("GET", "/api/snapshot", cookie=False,
                     headers={"Cookie": server.COOKIE + "=nope"})
    check("a wrong cookie is refused", code, 403)

    # -- Host and Origin -----------------------------------------------------
    code, _, _ = req("GET", "/", host="localhost:%d" % UI)
    check("localhost names it too", code, 200)
    for bad in ("evil.example:%d" % UI, "127.0.0.1:%d" % (UI + 1),
                "127.0.0.1", ""):
        code, _, body = req("GET", "/api/snapshot", host=bad)
        check("Host %r is refused (DNS rebinding)" % bad,
              (code, body.strip()), (403, b"wrong host"))
    code, _, _ = req("GET", "/?t=" + TOKEN, cookie=False,
                     host="evil.example:%d" % UI)
    check("even with the token in the URL", code, 403)
    code, _, body = req("GET", "/api/snapshot",
                        headers={"Origin": "http://evil.example"})
    check("a GET from another origin is refused",
          (code, body.strip()), (403, b"wrong origin"))

    # -- the files ---------------------------------------------------------
    code, h, _ = req("GET", "/static/app.js")
    check("the page's files are served",
          (code, h.get("Content-Type", "").startswith("text/javascript")),
          (200, True))
    for bad in ("/static/../server.py", "/static/%2e%2e/server.py",
                "/static/snapshot.py", "/static/", "/static/app.js/",
                "/nope"):
        check("nothing else: %s" % bad, req("GET", bad)[0], 404)

    # -- the snapshot --------------------------------------------------------
    s = snap()
    check("snapshot: schema, version and both models",
          (s["schema"], s["version"], sorted(m["name"] for m in s["models"])),
          (snapshot.SCHEMA, mdl.VERSION, ["demo", "keyed"]))
    d = model(s, "demo")
    check("snapshot: a stopped model's config",
          (d["state"], d["ctx"], d["ngl"], d["kv_type"], d["port"],
           "run" in d), ("stopped", 4096, 99, "q8_0", port, False))
    check("snapshot: its group", model(s, "keyed").get("group"), "g")

    # -- actions: only the right origin, a known verb, a known name ----------
    check("POST without an Origin is refused",
          action("run", "demo", headers={})[0], 403)
    check("POST from another origin is refused",
          action("run", "demo", headers={"Origin": "http://evil.example"})[0],
          403)
    check("POST without the token is refused",
          action("run", "demo", cookie=False)[0], 403)
    check("an unknown verb", action("delete", "demo"),
          (400, {"ok": False, "error": "unknown verb 'delete'"}))
    check("an unknown model", action("run", "nope")[0], 404)
    code, _, _ = req("POST", "/api/action", headers={"Origin": ORIGIN},
                     body=b"not json")
    check("a body that is not JSON", code, 400)
    check("stop what is not running", action("stop", "demo")[0], 409)

    # -- the event stream ----------------------------------------------------
    conn = http.client.HTTPConnection("127.0.0.1", UI, timeout=10)
    conn.request("GET", "/api/events", headers={"Host": HOST,
                                                "Cookie": COOKIE})
    r = conn.getresponse()
    lines = []
    while True:
        line = r.fp.readline().decode()
        if line in ("\n", ""):
            break
        lines.append(line.rstrip("\n"))
    check("events: a stream, starting with the snapshot",
          (r.status, r.getheader("Content-Type"), lines[0]),
          (200, "text/event-stream", "event: snapshot"))
    first = json.loads(lines[1][len("data: "):])
    check("events: the snapshot in it", sorted(m["name"] for m in
                                                first["models"]),
          ["demo", "keyed"])
    check("events: the hub counts the page", hub.clients, 1)

    # -- run: from stopped to ready, pushed down the stream ------------------
    check("run demo", action("run", "demo"), (200, {"ok": True}))
    seen = []
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        line = r.fp.readline().decode()
        if line.startswith("data: "):
            state = model(json.loads(line[6:]), "demo")["state"]
            if not seen or seen[-1] != state:
                seen.append(state)
            if state == "ready":
                break
    check("events: the start arrives without asking", seen[-1:], ["ready"])
    r.close()                   # the response holds the socket, not conn
    conn.close()
    # noticed on the second write after it went: a few seconds, not more
    check("events: a closed page leaves", until(lambda: hub.clients == 0, 15),
          True)
    check("run it again is refused", action("run", "demo")[0], 409)
    d = model(snap(), "demo")
    check("a server without --metrics: running, nothing to count",
          (d["state"], d["run"]["metrics"], d["run"]["n_ctx"],
           d["run"]["api_key"]), ("ready", None, 4096, False))
    code, _, body = req("GET", "/api/log?name=demo")
    check("its log", (code, b"server is listening" in body), (200, True))
    check("no log for a name without one", req("GET", "/api/log?name=zz")[0],
          404)

    # -- a server with an API key and --metrics ------------------------------
    check("run keyed", action("run", "keyed")[0], 200)
    k = until(lambda: model(snap(), "keyed")["state"] == "ready"
              and model(snap(), "keyed"))
    check("the key reaches /props and /metrics: counters come through",
          (k["run"]["api_key"], k["run"]["n_ctx"],
           k["run"]["metrics"] is not None), (True, 4096, True))
    chat = http.client.HTTPConnection("127.0.0.1", keyed, timeout=10)
    chat.request("POST", "/v1/chat/completions", json.dumps(
        {"messages": [{"role": "user", "content": "hi"}]}))
    chat.getresponse().read()
    chat.close()
    m = model(snap(), "keyed")["run"]["metrics"]
    check("the tokens it served are counted",
          (m["generated"], m["tokens"]), (6.0, 16.0))
    check("and it has a line to draw",
          len(model(snap(), "keyed")["run"].get("series", [])) >= 2, True)

    # -- stop ----------------------------------------------------------------
    for name in ("demo", "keyed"):
        check("stop %s" % name, action("stop", name), (200, {"ok": True}))
    check("both stopped", sorted(m["state"] for m in snap()["models"]),
          ["stopped", "stopped"])

    # -- a start that fails shows why ----------------------------------------
    os.environ["MDL_FAKE_MODE"] = "fail"
    check("run a model that will fail", action("run", "demo")[0], 200)
    d = until(lambda: model(snap(), "demo")["state"] == "failed"
              and model(snap(), "demo"))
    check("it is marked failed, with the log's words",
          (d and d["state"], d and "missing tensor" in d.get("error", "")),
          ("failed", True))
    os.environ.pop("MDL_FAKE_MODE")
finally:
    for name, st in mdl.read_states().items():
        mdl.stop_one(name, st)
    hub.halt.set()
    httpd.shutdown()
    httpd.server_close()

# ------------------------------------------------------- mdl snapshot --
out, _, code = run(mdl.cmd_snapshot, [])
check("mdl snapshot prints the snapshot",
      (code, sorted(m["name"] for m in json.loads(out)["models"])),
      (0, ["demo", "keyed"]))
out2, _, _ = run(mdl.cmd_snapshot, ["--json"])
check("--json is the same", json.loads(out2)["schema"], snapshot.SCHEMA)
_, err, code = run(mdl.cmd_snapshot, ["--yaml"])
check("anything else is a usage line", (code, "usage: mdl snapshot" in err),
      (1, True))
mdl.CONFIG.write_text("[broken\n", encoding="utf-8")
s = snapshot.build()
check("a config mdl cannot read is an error, not a raise",
      (s["models"], bool(s["error"])), ([], True))
teardown(root)

# ------------------------------------------------- the names, the window --
calls = []
real_launch = mdl._launch_ui
mdl._launch_ui = lambda fx=None: calls.append(fx)
try:
    _, err, code = run(mdl.cmd_ui, ["--tui", "--no-fx"])
    check("mdl ui --tui still opens the terminal UI, and says it moved",
          (code, calls, "is now `mdl tui`" in err), (0, ["off"], True))
    _, _, code = run(mdl.cmd_tui, [])
    check("mdl tui opens it", (code, calls[-1]), (0, None))
    _, err, code = run(mdl.cmd_tui, ["--bogus"])
    check("mdl tui refuses what it does not know",
          (code, "usage: mdl tui" in err), (1, True))
finally:
    mdl._launch_ui = real_launch
_, err, code = run(server.main, ["--bogus"])
check("mdl ui refuses what it does not know", (code, "usage: mdl ui" in err),
      (1, True))

opened = []
real_chromium, real_open = server.chromium, server.webbrowser.open
server.webbrowser.open = lambda u: opened.append(u) or True
profile = Path(support.tempfile.mkdtemp(prefix="mdl-web-")) / "profile"
try:
    server.chromium = lambda: None
    check("no Chromium: the default browser",
          (server.open_window("http://x/", profile), opened),
          ("browser", ["http://x/"]))
    server.chromium = lambda: str(profile.parent / "no-such-browser.exe")
    check("a Chromium that will not start: the default browser too",
          server.open_window("http://y/", profile), "browser")
    probe = profile.parent / "argv.json"
    fake = profile.parent / "fake_browser.py"
    fake.write_text("import json, sys\nopen(%r, 'w').write(json.dumps("
                    "sys.argv[1:]))\n" % str(probe), encoding="utf-8")
    if os.name == "nt":
        exe = profile.parent / "fake_browser.cmd"
        exe.write_text('@"%s" "%s" %%*\r\n' % (sys.executable, fake),
                       encoding="ascii")
    else:
        exe = profile.parent / "fake_browser"
        exe.write_text('#!/bin/sh\nexec "%s" "%s" "$@"\n'
                       % (sys.executable, fake), encoding="ascii")
        exe.chmod(0o755)
    server.chromium = lambda: str(exe)
    how = server.open_window("http://127.0.0.1:1/?t=x", profile)
    argv = until(lambda: probe.is_file() and probe.read_text()
                 and json.loads(probe.read_text()), 10)
    check("a Chromium: an app window on its own profile",
          (how, argv and argv[0], argv and argv[1], profile.is_dir()),
          ("window", "--app=http://127.0.0.1:1/?t=x",
           "--user-data-dir=" + str(profile), True))
finally:
    server.chromium, server.webbrowser.open = real_chromium, real_open
    shutil.rmtree(profile.parent, ignore_errors=True)

# ------------------------------------------------------- the page's JS --
node = shutil.which("node")
if node:
    for f in ("app.js", "view.js"):
        p = subprocess.run([node, "--check", str(server.STATIC / f)],
                           capture_output=True, text=True)
        check("%s parses" % f, (p.returncode, p.stderr.strip()), (0, ""))
    p = subprocess.run([node, str(HERE / "web_view.js")], capture_output=True,
                       text=True)
    sys.stdout.write(p.stdout)
    check("view.js: its checks pass", p.returncode, 0)
else:
    print("SKIP the page's JS: no node on PATH")

sys.exit(t.done())
