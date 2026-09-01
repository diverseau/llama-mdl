"""A stand-in for llama-server: logs like it, serves /health like it.

Behaviour is driven by $MDL_FAKE_MODE so it never collides with the flags
mdl passes:

    (unset)    log a little, then serve /health
    slow       take ~3s before serving, so "still loading" is observable
    fail       print an error and exit 1 without ever listening
    silent     listen, but never answer /health (readiness must time out)
    stubborn   ignore SIGTERM, so stop() has to escalate to SIGKILL
"""
import http.server
import os
import signal
import sys
import threading
import time

MODE = os.environ.get("MDL_FAKE_MODE", "")


def port_from_argv():
    for i, a in enumerate(sys.argv):
        if a == "--port" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return 8080


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health" and MODE != "silent":
            body = b'{"status":"ok"}'
        elif self.path == "/metrics":
            body = b"llamacpp:kv_cache_usage_ratio 0.25\n"
        else:
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass                                  # keep our stdout clean


def main():
    print("args: " + " ".join(sys.argv[1:]), flush=True)
    if MODE == "fail":
        print("error loading model: missing tensor 'blk.0.attn_q.weight'", flush=True)
        sys.exit(1)
    if MODE == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print("load_tensors: offloaded 33/33 layers to GPU", flush=True)
    if MODE == "slow":
        time.sleep(3.0)
    server = http.server.HTTPServer(("127.0.0.1", port_from_argv()), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("main: server is listening on http://127.0.0.1:%d" % port_from_argv(),
          flush=True)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
