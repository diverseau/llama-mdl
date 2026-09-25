"""`mdl ui`: the page, its snapshot, and its buttons, on 127.0.0.1.

Only this machine's browser, and only the window mdl opened, may use it.
The server binds 127.0.0.1 on a port of its own and makes a token each
start. The token comes in the URL mdl opens once, is swapped for a cookie
on the first request and never appears in a URL again, and every request
after - the page, its files, the snapshot, the event stream, an action -
must carry it. Past the token:

  Host    must name this server as 127.0.0.1 or localhost with its port,
          so a page on some other site whose name was pointed at
          127.0.0.1 (DNS rebinding) reaches nothing;
  Origin  must be this server on anything that changes state, and on
          anything else a browser sent one with.

The page gets the snapshot pushed over server-sent events as it changes,
not by polling, and an action is one of a fixed few verbs on a model the
config names. The page itself is omarchy-local-ai's panel (see NOTICE).
"""

import hmac
import http.server
import json
import os
import secrets
import shutil
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

from . import snapshot

STATIC = Path(__file__).resolve().parent / "static"
FILES = {"index.html": "text/html; charset=utf-8",
         "app.css": "text/css; charset=utf-8",
         "app.js": "text/javascript; charset=utf-8",
         "view.js": "text/javascript; charset=utf-8",
         "mono.woff2": "font/woff2",
         "qwen.svg": "image/svg+xml", "lfm.svg": "image/svg+xml",
         "hf.svg": "image/svg+xml"}
# where a page may send you: the weights' pages, and mdl's own
URLS = ("https://huggingface.co/", "https://github.com/diverseau/llama-mdl")
COOKIE = "mdl_ui"
BUSY_S, IDLE_S = 1.0, 2.0       # a snapshot this often while a load runs, else
PING_S = 15.0                   # an SSE comment this often keeps it open
IDLE_EXIT_S = 20.0              # no page for this long after one was: done
FIRST_PAGE_S = 300.0            # and none ever arrived within this: done
LOG_LINES = 400


class Hub:
    """The latest snapshot, rebuilt on a thread while anyone watches, and
    the event streams waiting on it."""

    def __init__(self):
        self.cond = threading.Condition()
        self.snap, self.text, self.seq = None, "null", 0
        self.clients = 0
        self.seen_client = False
        self.last_client = time.monotonic()
        self.failed = {}        # name -> why its start, from here, failed
        self.stopping = set()   # names being stopped from here
        self.halt = threading.Event()

    def rebuild(self):
        snap = snapshot.build(self.failed, set(self.stopping))
        text = json.dumps(snap, separators=(",", ":"))
        with self.cond:
            if text != self.text:
                self.snap, self.text = snap, text
                self.seq += 1
                self.cond.notify_all()
        return snap

    def loop(self):
        while not self.halt.is_set():
            try:
                snap = self.rebuild()
                busy = any(d["state"] in ("starting", "stopping", "download")
                           for d in snap["deployments"])
            except Exception as e:      # noqa: BLE001 - the page says why
                with self.cond:
                    self.text = json.dumps({"schema": snapshot.SCHEMA,
                                            "error": "snapshot failed: %s"
                                            % e, "gpus": [],
                                            "deployments": []})
                    self.seq += 1
                    self.cond.notify_all()
                busy = False
            self.halt.wait(BUSY_S if busy else IDLE_S)

    def wait(self, seq, timeout):
        """The snapshot after `seq`, or None when nothing changed."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq or self.halt.is_set(),
                               timeout)
            return (self.seq, self.text) if self.seq != seq else None

    def join(self, delta):
        with self.cond:
            self.clients += delta
            self.seen_client = True
            self.last_client = time.monotonic()


# -------------------------------------------------------------- actions --

def _watch(hub, name, proc, port, cfg, binary, log):
    """A start from the page, watched to its end: ready, or failed with the
    log's last words, which the page shows on the model. Also reaps it."""
    import mdl

    from . import usage
    started = time.monotonic()
    deadline = started + mdl.ready_timeout()
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            hub.failed[name] = _last_words(log) or (
                "exited with status %s while loading" % proc.returncode)
            return
        if mdl.server_ready(port):
            hub.failed.pop(name, None)
            try:
                usage.note_load(name, time.monotonic() - started)
            except OSError:
                pass
            mdl.learn_from_log(name, cfg, binary, log)
            proc.wait()
            return
        time.sleep(0.5)
    hub.failed[name] = "not ready after %d s" % mdl.ready_timeout()


def _last_words(log):
    try:
        lines = Path(log).read_text(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        low = line.lower()
        if "error" in low or "failed" in low or "unable" in low:
            return line.strip()[:300]
    return lines[-1].strip()[:300] if lines else None


def _stop(hub, name, state):
    """Stop a server in the background, the page showing it stopping."""
    import mdl

    from . import agents, tailnet
    try:
        if agents.run_config(name, state).get("shared"):
            tailnet.unshare(state["port"])
        if not mdl.stop_one(name, state):
            hub.failed[name] = "%s would not stop" % name
    finally:
        hub.stopping.discard(name)
        hub.rebuild()


def act(hub, req):
    """Run one of a fixed few verbs: (status, {"ok": ...}). A model must be
    one the config or the running servers have."""
    import mdl

    from . import agents, tailnet, usage
    verb, name = str(req.get("verb")), str(req.get("name") or "")
    if verb == "url":
        url = str(req.get("url") or "")
        if not url.startswith(URLS):
            return 400, {"ok": False, "error": "not a page mdl opens"}
        webbrowser.open(url)
        return 200, {"ok": True}
    if verb == "set":
        states = mdl.read_states()
        try:
            agents.choose(str(req.get("key")), str(req.get("value") or ""),
                          name if name in states else None, states.get(name))
        except mdl.MdlError as e:
            return 409, {"ok": False, "error": str(e)}
        return 200, {"ok": True}
    try:
        models, binary = mdl.load_config()
    except mdl.MdlError as e:
        return 409, {"ok": False, "error": str(e)}
    states = mdl.read_states()
    if verb in ("run", "again") and name.startswith("hf:"):
        return _pull(name, str(req.get("keys") or ""), states)
    if verb in ("run", "again") and name not in models:
        from mdl_fit import pull
        spec = (pull.read_all().get(name) or {}).get("spec")
        if spec:                         # a download that stopped: again
            return _pull(spec, str(req.get("keys") or ""), states)
    if verb in ("run", "again"):
        if name not in models:
            return 404, {"ok": False, "error": "no model named %r" % name}
        if verb == "again" and name in states:
            if not mdl.stop_one(name, states[name]):
                return 500, {"ok": False, "error": "%s would not stop" % name}
        try:
            proc, log, port = mdl.spawn(name, models, binary)
        except mdl.MdlError as e:
            return 409, {"ok": False, "error": str(e)}
        hub.failed.pop(name, None)
        usage.ensure()
        threading.Thread(target=_watch, args=(hub, name, proc, port,
                                              models[name], binary, log),
                         daemon=True).start()
        return 200, {"ok": True}
    if verb == "stop":
        from mdl_fit import pull
        if name not in states and pull.stop(name):
            return 200, {"ok": True}     # a download stopped, or dismissed
        if name in hub.failed and name not in states:
            del hub.failed[name]         # dismiss a start that failed
            return 200, {"ok": True}
        if name not in states:
            return 409, {"ok": False, "error": "%s is not running" % name}
        hub.stopping.add(name)
        threading.Thread(target=_stop, args=(hub, name, states[name]),
                         daemon=True).start()
        return 200, {"ok": True}
    if verb in ("open", "share"):
        if name not in states:
            return 409, {"ok": False, "error": "%s is not running" % name}
        state = states[name]
        try:
            if verb == "open":
                agents.open_agent(name, state, models.get(name, {}),
                                  agents.run_config(name, state))
            elif not snapshot.api_key(state.get("argv")):
                return 409, {"ok": False, "error": "a share needs the "
                             "server to have an --api-key"}
            else:
                agents.remember(name, state,
                                shared=tailnet.share(state["port"]))
        except (mdl.MdlError, OSError) as e:
            return 409, {"ok": False, "error": str(e)}
        return 200, {"ok": True}
    return 400, {"ok": False, "error": "unknown verb %r" % verb}


def _pull(spec, keys, states):
    """Run on a model find picked: download it, add it, start it - a
    process of its own, so it outlives the page."""
    from mdl_fit import pull, remote

    from . import proc
    try:
        repo, _ = remote.parse_spec(spec)
    except remote.RemoteError as e:
        return 404, {"ok": False, "error": str(e)}
    name = pull.default_name(repo)
    if name in states:
        return 409, {"ok": False, "error": "%s is running" % name}
    going = pull.read_all().get(name)
    if going and going.get("state") != "error":
        return 409, {"ok": False, "error": "%s is downloading" % name}
    pull.stop(name)                      # forget a pull that failed before
    if not all(k.isalnum() for k in keys.split(",") if k):
        keys = ""
    try:
        proc.detach(["pull", spec, "--run", "--quiet", "--keys", keys])
    except OSError as e:
        return 409, {"ok": False, "error": str(e)}
    return 200, {"ok": True}


def picks_loop(hub):
    """mdl find for the Config pages, again when it goes stale."""
    while not hub.halt.is_set():
        try:
            snapshot.refresh_picks()
        except Exception:               # noqa: BLE001 - no picks, no harm
            pass
        hub.halt.wait(600)


def log_tail(name):
    """The last lines of a model's log: its running server's, else the
    last one it wrote."""
    import mdl
    state = mdl.read_states().get(name)
    path = Path(state["log"]) if state and state.get("log") else (
        mdl.STATE_DIR / (name + ".log"))
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 256 * 1024))
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    return "\n".join(text.splitlines()[-LOG_LINES:])


# -------------------------------------------------------------- handler --

def make_handler(hub, token, port):
    hosts = {"127.0.0.1:%d" % port, "localhost:%d" % port}
    origins = {"http://" + h for h in hosts}

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "mdl"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):       # quiet: a UI, not a server
            pass

        # -- the checks every request passes ----------------------------
        def _refuse(self, code, why):
            body = (why + "\n").encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            return False

        def _cookie(self):
            for part in (self.headers.get("Cookie") or "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == COOKIE:
                    return v
            return None

        def _allowed(self, writes):
            if self.headers.get("Host") not in hosts:
                return self._refuse(403, "wrong host")
            origin = self.headers.get("Origin")
            if (writes or origin is not None) and origin not in origins:
                return self._refuse(403, "wrong origin")
            got = self._cookie() or self.headers.get("X-Mdl-Token") or ""
            if not hmac.compare_digest(got.encode(), token.encode()):
                return self._refuse(403, "no token")
            return True

        def _send(self, code, body, ctype, extra=()):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj), "application/json")

        # -- routes -----------------------------------------------------
        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            if url.path == "/" and url.query:
                # the one URL with the token in it: swap it for a cookie
                given = urllib.parse.parse_qs(url.query).get("t", [""])[0]
                if self.headers.get("Host") not in hosts:
                    return self._refuse(403, "wrong host")
                if not hmac.compare_digest(given.encode(), token.encode()):
                    return self._refuse(403, "no token")
                self.send_response(303)
                self.send_header("Location", "/")
                self.send_header("Set-Cookie", "%s=%s; HttpOnly; "
                                 "SameSite=Strict; Path=/" % (COOKIE, token))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if not self._allowed(writes=False):
                return
            if url.path == "/":
                return self._file("index.html")
            if url.path.startswith("/static/"):
                return self._file(url.path[len("/static/"):])
            if url.path == "/api/snapshot":
                hub.rebuild()
                return self._send(200, hub.text, "application/json")
            if url.path == "/api/events":
                return self._events()
            if url.path == "/api/log":
                name = urllib.parse.parse_qs(url.query).get("name", [""])[0]
                text = log_tail(name)
                if text is None:
                    return self._json(404, {"ok": False,
                                            "error": "no log for %r" % name})
                return self._send(200, text, "text/plain; charset=utf-8")
            return self._refuse(404, "not found")

        def do_POST(self):
            if not self._allowed(writes=True):
                return
            if urllib.parse.urlsplit(self.path).path != "/api/action":
                return self._refuse(404, "not found")
            try:
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(min(n, 65536)) or b"{}")
                if not isinstance(req, dict):
                    raise ValueError(req)
            except (ValueError, AttributeError):
                return self._json(400, {"ok": False, "error": "bad request"})
            code, out = act(hub, req)
            hub.rebuild()
            return self._json(code, out)

        def _file(self, name):
            # only the files the page is made of: no path reaches past them
            if name not in FILES:
                return self._refuse(404, "not found")
            try:
                body = (STATIC / name).read_bytes()
            except OSError:
                return self._refuse(404, "not found")
            csp = ("default-src 'self'; style-src 'self'; script-src 'self'; "
                   "font-src 'self'; "
                   "img-src 'self' data:; connect-src 'self'; "
                   "frame-ancestors 'none'; base-uri 'none'; "
                   "form-action 'none'")
            return self._send(200, body, FILES[name],
                              (("Content-Security-Policy", csp),))

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            hub.join(+1)
            seq = -1
            try:
                while not hub.halt.is_set():
                    got = hub.wait(seq, PING_S)
                    if got is None:
                        self.wfile.write(b": ping\n\n")
                    else:
                        seq, text = got
                        self.wfile.write(("event: snapshot\ndata: %s\n\n"
                                          % text).encode("utf-8"))
                    self.wfile.flush()
            except (OSError, ValueError):
                pass                # the page went away
            finally:
                hub.join(-1)

    return Handler


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def serve(port=0, token=None):
    """(server, hub, url): listening, with the snapshot thread running."""
    token = token or secrets.token_urlsafe(24)
    hub = Hub()
    httpd = Server(("127.0.0.1", port), lambda *a: None)
    real_port = httpd.server_address[1]
    httpd.RequestHandlerClass = make_handler(hub, token, real_port)
    threading.Thread(target=hub.loop, daemon=True).start()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, hub, "http://127.0.0.1:%d/?t=%s" % (real_port, token)


# ------------------------------------------------------------- the window --

def chromium():
    """A Chromium browser to open the page as an app window, or None."""
    names = ("msedge", "chrome", "google-chrome", "google-chrome-stable",
             "chromium", "chromium-browser", "microsoft-edge",
             "microsoft-edge-stable", "brave", "brave-browser")
    for n in names:
        found = shutil.which(n)
        if found:
            return found
    if sys.platform == "win32":
        roots = [os.environ.get(k) for k in ("ProgramFiles(x86)",
                                             "ProgramFiles", "LOCALAPPDATA")]
        tails = (r"Microsoft\Edge\Application\msedge.exe",
                 r"Google\Chrome\Application\chrome.exe",
                 r"BraveSoftware\Brave-Browser\Application\brave.exe",
                 r"Chromium\Application\chrome.exe")
        for root in filter(None, roots):
            for tail in tails:
                p = Path(root) / tail
                if p.is_file():
                    return str(p)
    if sys.platform == "darwin":
        for app in ("Google Chrome", "Microsoft Edge", "Brave Browser",
                    "Chromium"):
            p = Path("/Applications/%s.app/Contents/MacOS/%s" % (app, app))
            if p.is_file():
                return str(p)
    return None


def open_window(url, profile):
    """The page in an app window of its own - its own profile, so it is a
    process of its own and none of your browser's extensions or sessions
    see it - else in your default browser. What was used, as a word."""
    import subprocess
    exe = chromium()
    if exe:
        try:
            Path(profile).mkdir(parents=True, exist_ok=True)
            subprocess.Popen(
                [exe, "--app=" + url, "--user-data-dir=" + str(profile),
                 "--no-first-run", "--no-default-browser-check",
                 "--window-size=520,1000"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            return "window"
        except OSError:
            pass
    return "browser" if webbrowser.open(url) else None


def main(args):
    import mdl
    usage = "usage: mdl ui [--no-open] [--port N]"
    port, opening = 0, True
    rest = list(args)
    while rest:
        a = rest.pop(0)
        if a == "--no-open":
            opening = False
        elif a == "--port" and rest:
            port = mdl.check_port(rest.pop(0))
        else:
            mdl.die(usage)
    from . import usage
    usage.ensure()              # servers started before mdl ui are booked too
    try:
        httpd, hub, url = serve(port)
    except OSError as e:
        mdl.die("cannot listen on 127.0.0.1:%s: %s" % (port or "any", e))
    threading.Thread(target=picks_loop, args=(hub,), daemon=True).start()
    how = open_window(url, mdl.STATE_DIR / "ui-browser") if opening else None
    print({"window": "mdl ui: opened in an app window",
           "browser": "mdl ui: opened in your browser"}.get(
               how, "mdl ui: open this in a browser") + " - " + url,
          flush=True)
    print("mdl ui: Ctrl-C stops it; it also stops a little after its last "
          "page closes", flush=True)
    started = time.monotonic()
    try:
        while True:
            time.sleep(1)
            with hub.cond:
                clients, seen = hub.clients, hub.seen_client
                quiet = time.monotonic() - hub.last_client
            if clients == 0 and seen and quiet > IDLE_EXIT_S:
                break
            if not seen and time.monotonic() - started > FIRST_PAGE_S:
                print("mdl ui: no page opened in %d s; stopping"
                      % FIRST_PAGE_S, flush=True)
                break
    except KeyboardInterrupt:
        pass
    finally:
        hub.halt.set()
        with hub.cond:
            hub.cond.notify_all()
        httpd.shutdown()
        httpd.server_close()
