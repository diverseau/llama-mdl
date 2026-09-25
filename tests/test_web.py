"""mdl ui (the web page), mdl snapshot and the usage recorder: the gate on
every request, the snapshot in the panel's shape, the event stream, the
verbs, what is booked from each server's counters, and the agents."""
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                                 # noqa: E402
from support import free_port, mdl, run, sandbox, teardown     # noqa: E402

from mdl_web import agents, server, snapshot, usage            # noqa: E402

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
check("a model's logo comes from its name or file",
      [snapshot.family("qwen9u", ""), snapshot.family("x", "/m/LFM2-8B.gguf"),
       snapshot.family("gemma", "/m/gemma.gguf")], ["qwen", "lfm", ""])
check("weights from the hub's cache path",
      snapshot.weights("/c/hub/models--unsloth--Qwen3-8B-GGUF/snapshots/"
                       "0123abcd/Qwen3-8B-Q4_K_M.gguf"),
      [{"repository": "unsloth/Qwen3-8B-GGUF", "revision": "0123abcd"}])
check("and none from any other path", snapshot.weights("/models/x.gguf"), [])

# ------------------------------------------------------------- usage ----
check("a counter's move is the difference, and all of it after a reset",
      [usage.delta(None, 5), usage.delta(5, 9), usage.delta(9, 3)], [5, 4, 3])
data = {"hours": {}, "first": None, "last": None}
check("nothing moved books nothing", usage.book(data, 7200, [0, 0, 0, 0, 0]),
      False)
usage.book(data, 7300, [100, 50, 0.5, 1.0, 2])
usage.book(data, 7400, [100, 50, 0.5, 1.0, 1])
usage.book(data, 11000, [0, 10, 0, 0.2, 1])
check("readings go to the hour they fell in",
      (sorted(data["hours"]), data["hours"]["7200"], data["first"],
       data["last"]),
      (["10800", "7200"], [200, 100, 1.0, 2.0, 3], 7200, 11000))
s = usage.summarize(data, now=11000)
check("totals and averages: decode, prefill, the wait for a first token",
      (s["tokens"], s["requests"], s["decode"], s["prefill"], s["ttft"]),
      (310, 4, 50, 200, 250))
check("the line rises from nothing to the total",
      (s["line"][0], s["line"][-1], s["line"] == sorted(s["line"])),
      (0, 310, True))
check("no usage, no figures", usage.summarize({}, now=1)["decode"], None)

# a Thursday afternoon: the grid ends on the Thursday of its last column
start, today, cells = usage.days({"m": data}, now=time.mktime(
    (2026, 9, 24, 14, 0, 0, 0, 0, -1)))
check("the grid: 20 weeks, a column a week, Monday first",
      (len(cells), time.localtime(start).tm_wday, today), (140, 0, 136))


class FakeGet:
    """/metrics and /slots answers in turn, as a server gives them."""

    def __init__(self, answers):
        self.answers = list(answers)

    def __call__(self, port, path, timeout, key):
        return self.answers.pop(0)


def metrics(p, g, ps, gs):
    return (200, "llamacpp:prompt_tokens_total %d\n"
            "llamacpp:tokens_predicted_total %d\n"
            "llamacpp:prompt_seconds_total %s\n"
            "llamacpp:tokens_predicted_seconds_total %s\n" % (p, g, ps, gs))


def slots(task=None):
    s = {"id": 0}
    if task is not None:
        s["id_task"] = task
    return 200, json.dumps([s])


cur, st = usage.Cursor(), {"port": 1, "argv": []}
first = usage.read(st, cur, FakeGet([metrics(10, 5, 0.1, 0.2), slots()]))
again = usage.read(st, cur, FakeGet([metrics(40, 25, 0.2, 0.6), slots(7)]))
same = usage.read(st, cur, FakeGet([metrics(40, 25, 0.2, 0.6), slots(7)]))
reset = usage.read(st, cur, FakeGet([metrics(3, 1, 0.01, 0.02), (404, None)]))
check("read: the first reading whole, then what moved",
      (first[:2], again[:2], [round(v, 2) for v in again[2:4]]),
      ([10, 5], [30, 20], [0.1, 0.4]))
check("read: a slot whose task changed is a request; one that did not is not",
      (first[4], again[4], same[4]), (0, 1, 0))
check("read: a server that restarted counts from zero", reset[:2], [3, 1])
check("read: no metrics, nothing read",
      usage.read(st, usage.Cursor(), FakeGet([(501, None)])), None)

# -- speed by context depth, from the server's log ---------------------------
LOG = """\
12.00.000.000 I slot launch_slot_: id  0 | task 7 | processing task, is_child = 0
12.00.100.000 W srv          stop: cancel task, id_task = 9
12.02.000.000 I slot print_timing: id  0 | task 7 | prompt processing, n_tokens =   2048, progress = 0.50, t =   4.00 s / 512.00 tokens per second
12.04.000.000 I slot print_timing: id  0 | task 7 | prompt processing, n_tokens =   4096, progress = 1.00, t =   8.00 s / 512.00 tokens per second
12.07.000.000 I slot print_timing: id  0 | task 7 | n_gen =     90, tg =  30.00 t/s, tg_3s =  30.00 t/s
12.10.000.000 I slot print_timing: id  0 | task 7 | n_gen =    180, tg =  30.00 t/s, tg_3s =  30.00 t/s
12.10.500.000 I slot print_timing: id  0 | task 7 | prompt eval time =    8100.00 ms /  4100 tokens (    1.98 ms per token,   506.17 tokens per second)
12.10.500.001 I slot print_timing: id  0 | task 7 |        eval time =    6400.00 ms /   192 tokens (   33.33 ms per token,    30.00 tokens per second)
12.10.500.002 I slot      release: id  0 | task 7 | stop processing: n_tokens = 4291, truncated = 0
12.20.000.000 I slot launch_slot_: id  0 | task 12 | processing task, is_child = 0
12.21.000.000 I slot print_timing: id  0 | task 12 | prompt eval time =    1000.00 ms /   500 tokens (    2.00 ms per token,   500.00 tokens per second)
12.21.000.001 I slot print_timing: id  0 | task 12 |        eval time =    2000.00 ms /    50 tokens (   40.00 ms per token,    25.00 tokens per second)
12.21.000.002 I slot      release: id  0 | task 12 | stop processing: n_tokens = 4841, truncated = 0
"""  # noqa: E501 - llama-server's own lines, verbatim
logf = Path(tempfile.mkdtemp(prefix="mdl-log-")) / "m.log"
cut = LOG.index("n_gen =    180")
logf.write_bytes(LOG[:cut].encode())
reader = usage.LogReader()
check("the log: nothing until a request is released", reader.read(logf), [])
with open(logf, "ab") as fh:
    fh.write(LOG[cut:].encode())
done = reader.read(logf)
check("the log: a request per release, a line split across reads kept whole",
      len(done), 2)
(pre, dec), (pre2, dec2) = done
check("prefill batch by batch, at the depth each reached",
      [(int(d), n, round(t, 2)) for d, n, t in pre],
      [(1024, 2048, 4.0), (3072, 2048, 4.0)])
check("decode three seconds at a time, past the prompt",
      [(int(d), round(n / t, 1)) for d, n, t in dec], [(4190, 30.0), (4280, 30.0)])
check("a request on a cached prompt starts as deep as its cache",
      ([(int(d), n) for d, n, _ in pre2], [(int(d), n, t) for d, n, t in dec2]),
      ([(4541, 500)], [(4816, 50, 2.0)]))
logf.write_bytes(LOG[:40].encode())
check("a log that starts again is read from its top",
      (reader.read(logf), reader.offset), ([], 40))

check("a config's id leaves out the port and key, not the context",
      [usage.config_key(["s", "-c", "4096", "--port", "1", "--api-key", "a"])
       == usage.config_key(["s", "-c", "4096", "--port", "2"]),
       usage.config_key(["s", "-c", "4096"]) == usage.config_key(["s", "-c", "8192"])],
      [True, False])
check("the context a server runs with", [usage.ctx_of(["s", "-c", "8192"]),
                                         usage.ctx_of(["s"])], [8192, None])
data = {}
usage.book_speed(data, "cfg", 8192, done)
sp = usage.speed(data, "cfg", points=8)
check("speed by depth: steps across the context, a step once it has time",
      sp, {"n_ctx": 8192, "decode": [[4380, 28.8]],
           "prefill": [[1024, 512.0], [3072, 512.0], [4541, 500.0]]})
check("no speed for a config never seen", usage.speed(data, "other"), None)
cur = usage.Cursor()
cur.resume({"id": "1:2", "counters": [1, 2, 3, 4], "log": 99}, "1:2")
check("a recorder carries on where the last left this server",
      (cur.counters, cur.log.offset, cur.baseline), ([1, 2, 3, 4], 99, False))
cur = usage.Cursor()
cur.resume({"id": "1:1", "log": 99}, "1:2", booked_since=True)
check("and with none for it, counts from what it reads now",
      (cur.counters, cur.log.offset, cur.baseline), (None, 0, True))
shutil.rmtree(logf.parent, ignore_errors=True)

# ------------------------------------------------------------ agents ----
home = Path(tempfile.mkdtemp(prefix="mdl-agent-"))
argv, env, files = agents.argv("claude", "claude", "http://127.0.0.1:9", "m",
                               8192, False, home, "k")
check("claude: the Anthropic API at the server, the key in its env",
      (argv, env["ANTHROPIC_BASE_URL"], env["ANTHROPIC_AUTH_TOKEN"],
       env["ANTHROPIC_DEFAULT_HAIKU_MODEL"], files),
      (["claude", "--model", "m"], "http://127.0.0.1:9", "k", "m", {}))
argv, env, _ = agents.argv("codex", "codex", "http://h:9", "m", 8192, False,
                           home, "k")
check("codex: a provider on the responses API, the key by name",
      ("model_providers.mdl.base_url=http://h:9/v1" in argv,
       "model_providers.mdl.wire_api=responses" in argv,
       "model_context_window=8192" in argv, env),
      (True, True, True, {"MDL_API_KEY": "k"}))
argv, env, _ = agents.argv("opencode", "opencode", "http://h:9", "m", 4096,
                           True, home, "k")
cfg = json.loads(env["OPENCODE_CONFIG_CONTENT"])
check("opencode: its config in the env, the key beside it, images when the "
      "model sees",
      (argv[-1], cfg["provider"]["mdl"]["options"],
       cfg["provider"]["mdl"]["models"]["m"]["modalities"]["input"],
       env["MDL_API_KEY"]),
      ("mdl/m", {"baseURL": "http://h:9/v1", "apiKey": "{env:MDL_API_KEY}"},
       ["text", "image"], "k"))
argv, env, files = agents.argv("pi", "pi", "http://h:9", "m", 4096, False,
                               home, "k")
check("pi: a models file of its own, in its own directory",
      (argv[1:], env["PI_CODING_AGENT_DIR"],
       list(files) == [home / "models.json"],
       json.loads(files[home / "models.json"])["providers"]["mdl"]["apiKey"]),
      (["--provider", "mdl", "--model", "m"], str(home), True, "k"))
check("anything else: the OpenAI variables",
      agents.argv("aider", "aider", "http://h:9", "m", 1, False, home,
                  "k")[1]["OPENAI_BASE_URL"], "http://h:9/v1")
shutil.rmtree(home, ignore_errors=True)
if os.name == "nt":
    how = agents.terminal(str(Path.home()), ["x", "y"])
    check("a terminal on Windows: Windows Terminal, else a console",
          (Path(how[0]).stem.lower() in ("wt", "cmd"), how[-2:]),
          (True, ["x", "y"]))

# ------------------------------------------------------------ a server --
keyed = free_port()
root, port = sandbox(extra=(
    '\n[keyed]\nmodel = "%s"\nctx = 4096\nport = %d\ngroup = "g"\n'
    'args = ["--metrics", "--api-key", "sekrit"]\n'
    % (str(support.FAKE).replace("\\", "/"), keyed)))
folder = Path(tempfile.mkdtemp(prefix="mdl-folder-")).resolve()

# -- an agent and folder of a run's own, while that run lasts ---------------
try:
    agents.open_agent("x", {"port": 1}, {}, {"agent": None, "folder": "."})
    check("no agent installed: says so", "raised", "an error")
except mdl.MdlError as e:
    check("no agent installed: says so",
          str(e).startswith("no coding agent installed"), True)
agents.remember("demo", {"pid": 11}, folder=str(folder))
check("a run keeps the folder chosen for it",
      Path(agents.run_config("demo", {"pid": 11})["folder"]), folder)
check("and the next run of it starts from the default",
      Path(agents.run_config("demo", {"pid": 12})["folder"]) == folder, False)

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


def action(verb, name="", extra=None, **kw):
    kw.setdefault("headers", {"Origin": ORIGIN})
    body = dict({"verb": verb, "name": name}, **(extra or {}))
    code, _, raw = req("POST", "/api/action", body=body, **kw)
    try:
        return code, json.loads(raw)
    except ValueError:
        return code, raw.decode(errors="replace").strip()


def snap():
    return json.loads(req("GET", "/api/snapshot")[2])


def dep(s, name):
    return next((d for d in s["deployments"] if d["id"] == name), None)


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
    for f, ctype in (("app.js", "text/javascript"),
                     ("mono.woff2", "font/woff2"),
                     ("qwen.svg", "image/svg+xml")):
        code, h, _ = req("GET", "/static/" + f)
        check("the page's %s is served as %s" % (f, ctype),
              (code, h.get("Content-Type", "").startswith(ctype)), (200, True))
    for bad in ("/static/../server.py", "/static/%2e%2e/server.py",
                "/static/snapshot.py", "/static/", "/static/app.js/",
                "/nope"):
        check("nothing else: %s" % bad, req("GET", bad)[0], 404)
    check("every file the page is made of is served, and only those",
          sorted(server.FILES), sorted(p.name for p in server.STATIC.iterdir()))

    # -- the snapshot, in the panel's shape ----------------------------------
    s = snap()
    check("snapshot: schema, version, and what the page reads",
          (s["schema"], s["version"], sorted(k for k in s if k in (
              "gpus", "kinds", "deployments", "life", "total", "week",
              "agents", "defaults", "folders", "tailnet"))),
          (snapshot.SCHEMA, mdl.VERSION,
           ["agents", "defaults", "deployments", "folders", "gpus", "kinds",
            "life", "tailnet", "total", "week"]))
    check("snapshot: a card to run on, even with no GPU mdl can read",
          bool(s["gpus"]) and all("key" in g and "name" in g
                                  for g in s["gpus"]), True)
    kind = s["kinds"][0]
    check("snapshot: every model in the config is the card's to run",
          sorted(m["id"] for m in kind["models"]), ["demo", "keyed"])
    demo = next(m for m in kind["models"] if m["id"] == "demo")
    check("snapshot: a model's facts",
          (demo["ctx"], demo["format"], demo["port"],
           next(m for m in kind["models"] if m["id"] == "keyed")["group"]),
          (4096, "GGUF", port, "g"))
    check("snapshot: nothing running, no card held",
          (s["deployments"], sorted(kind["free"] + kind["taken"])),
          ([], sorted(kind["keys"])))
    check("snapshot: no history yet",
          (s["total"], s["week"], s["life"]["requests"], len(s["life"]["days"])),
          (0, 0, 0, 140))

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
    for bad in (b"not json", b"[1, 2]"):
        code, _, _ = req("POST", "/api/action", headers={"Origin": ORIGIN},
                         body=bad)
        check("a body that is not an object: %r" % bad, code, 400)
    check("stop what is not running", action("stop", "demo")[0], 409)
    check("open an agent on what is not running", action("open", "demo")[0],
          409)
    opened = []
    real_open = server.webbrowser.open
    server.webbrowser.open = opened.append
    try:
        check("url: a model's weights page opens",
              action("url", extra={"url": "https://huggingface.co/a/b"}),
              (200, {"ok": True}))
        check("url: nothing else does",
              action("url", extra={"url": "https://evil.example/"})[0], 400)
    finally:
        server.webbrowser.open = real_open
    check("url: and only the one asked for", opened,
          ["https://huggingface.co/a/b"])

    # -- the default agent and folder ------------------------------------------
    check("set: an agent mdl does not know",
          action("set", extra={"key": "agent", "value": "rm"})[0], 409)
    check("set: a folder",
          action("set", extra={"key": "folder", "value": str(folder)}),
          (200, {"ok": True}))
    s = snap()
    check("set: it is the default, and offered again",
          (Path(s["defaults"]["folder"]), [Path(f) for f in s["folders"]]),
          (folder, [folder]))
    check("set: a folder that is not there",
          action("set", extra={"key": "folder",
                               "value": str(folder) + "-gone"})[0], 409)
    check("set: nothing else",
          action("set", extra={"key": "shell", "value": "x"})[0], 409)

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
    check("events: the snapshot in it",
          json.loads(lines[1][len("data: "):])["schema"], snapshot.SCHEMA)
    check("events: the hub counts the page", hub.clients, 1)

    # -- run: from stopped to ready, pushed down the stream ------------------
    check("run demo", action("run", "demo"), (200, {"ok": True}))
    seen = []
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        line = r.fp.readline().decode()
        if line.startswith("data: "):
            d = dep(json.loads(line[6:]), "demo")
            state = d and d["state"]
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
    s = snap()
    d = dep(s, "demo")
    check("a server without --metrics: running, nothing to count",
          (d["state"], d.get("metrics"), d["api_key"], d["port"]),
          ("ready", False, False, port))
    check("it runs in the folder chosen last",
          Path(d["folder"]), folder)
    check("the card it runs on is held, neither free nor taken",
          s["kinds"][0]["free"] + s["kinds"][0]["taken"], [])
    check("share: not without an --api-key", action("share", "demo")[0], 409)
    code, _, body = req("GET", "/api/log?name=demo")
    check("its log", (code, b"server is listening" in body), (200, True))
    check("no log for a name without one", req("GET", "/api/log?name=zz")[0],
          404)

    # -- a server with an API key and --metrics ------------------------------
    check("run keyed", action("run", "keyed")[0], 200)
    k = until(lambda: (dep(snap(), "keyed") or {}).get("state") == "ready"
              and dep(snap(), "keyed"))
    check("the key reaches /metrics", (k["api_key"], "metrics" in k),
          (True, False))
    chat = http.client.HTTPConnection("127.0.0.1", keyed, timeout=10)
    chat.request("POST", "/v1/chat/completions", json.dumps(
        {"messages": [{"role": "user", "content": "hi"}]}))
    chat.getresponse().read()
    chat.close()
    check("its session counts what it served",
          dep(snap(), "keyed")["session"]["tokens"], 16)

    # -- the recorder, one pass by hand ----------------------------------------
    cursors = {}
    check("the recorder sees both servers", usage.tick(cursors), 2)
    booked = usage.load("keyed")
    row = next(iter(booked["hours"].values()), None)
    check("it books what the keyed server served, with its key",
          row and (row[0], row[1], row[4]), (10, 6, 1))
    check("and nothing for a server without metrics",
          usage.load("demo")["hours"], {})
    s = snap()
    mine = dep(s, "keyed")["session"]["all"]
    check("the page's history: lifetime, week, requests, the model's own",
          (s["total"], s["week"], s["life"]["requests"],
           s["life"]["days"][s["life"]["today"]], mine["tokens"],
           mine["line"][-1], bool(mine["since"])),
          (16, 16, 1, 16, 16, 16, True))
    usage.tick(cursors)
    check("a second pass books nothing new", usage.load("keyed")["hours"],
          booked["hours"])
    usage.tick({})
    check("a recorder that starts again counts nothing twice",
          usage.load("keyed")["hours"], booked["hours"])
    check("the page has its speed and lab entries",
          sorted(k for k in dep(snap(), "keyed")["session"]
                 if k in ("speed", "lab")), ["lab", "speed"])
    rec = {"argv": mdl.read_states()["keyed"]["argv"], "warmup": False,
           "workload": {"prompt_tokens": 2000},
           "metrics": {"tokens": 100, "decode": {"p50": 41.5},
                       "prefill": {"tps": 900.0}}}
    lab_dir = mdl.STATE_DIR / "lab"
    lab_dir.mkdir(parents=True, exist_ok=True)
    (lab_dir / "records.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps(
        dict(rec, warmup=True, metrics={"decode": {"p50": 1}})) + "\n")
    check("what mdl lab measured for this config, warmups left out",
          dep(snap(), "keyed")["session"]["lab"],
          {"decode": [[2050, 41.5]], "prefill": [[1000, 900.0]]})

    # -- stop, as the page does it: in the background ------------------------
    for name in ("demo", "keyed"):
        check("stop %s" % name, action("stop", name), (200, {"ok": True}))
    check("both stop", until(lambda: not snap()["deployments"]), True)

    # -- a start that fails shows why, and can be dismissed ------------------
    os.environ["MDL_FAKE_MODE"] = "fail"
    check("run a model that will fail", action("run", "demo")[0], 200)
    d = until(lambda: (dep(snap(), "demo") or {}).get("state") == "error"
              and dep(snap(), "demo"))
    check("it is shown crashed, with the log's words",
          (d and d["state"], d and "missing tensor" in d.get("error", "")),
          ("error", True))
    os.environ.pop("MDL_FAKE_MODE")
    check("dismiss it", action("stop", "demo"), (200, {"ok": True}))
    check("and it is gone", dep(snap(), "demo"), None)
finally:
    os.environ.pop("MDL_FAKE_MODE", None)
    for name, st in mdl.read_states().items():
        mdl.stop_one(name, st)
    hub.halt.set()
    httpd.shutdown()
    httpd.server_close()
    shutil.rmtree(folder, ignore_errors=True)

# ------------------------------------------------------- mdl snapshot --
out, _, code = run(mdl.cmd_snapshot, [])
check("mdl snapshot prints the snapshot",
      (code, sorted(m["id"] for m in json.loads(out)["kinds"][0]["models"])),
      (0, ["demo", "keyed"]))
out2, _, _ = run(mdl.cmd_snapshot, ["--json"])
check("--json is the same", json.loads(out2)["schema"], snapshot.SCHEMA)
_, err, code = run(mdl.cmd_snapshot, ["--yaml"])
check("anything else is a usage line", (code, "usage: mdl snapshot" in err),
      (1, True))
mdl.CONFIG.write_text("[broken\n", encoding="utf-8")
s = snapshot.build()
check("a config mdl cannot read is an error, not a raise",
      (s["kinds"], bool(s["error"])), ([], True))
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
profile = Path(tempfile.mkdtemp(prefix="mdl-web-")) / "profile"
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
    sys.stdout.write(p.stdout + p.stderr)
    check("view.js: its checks pass", p.returncode, 0)
else:
    print("SKIP the page's JS: no node on PATH")

sys.exit(t.done())
