"""A stand-in for llama-server: logs like it, serves /health like it.

Behaviour is driven by $MDL_FAKE_MODE so it never collides with the flags
mdl passes:

    (unset)    log a little, then serve /health
    slow       take ~3s before serving, so "still loading" is observable
    fail       print an error and exit 1 without ever listening
    silent     listen, but never answer /health (readiness must time out)
    stubborn   ignore SIGTERM, so stop() has to escalate to SIGKILL
    tags       stream reasoning as inline <think> tags, not its own field
    slowchat   stream a long reply slowly, so an interrupt has something
               to interrupt
"""
import http.server
import json
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

    def do_POST(self):
        """Stream a reply the way llama-server does, chunk per token."""
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        pause = 0.02
        if MODE == "slowchat":
            pause = 0.1
            deltas = [{"content": "word "} for _ in range(200)]
            self._stream(deltas, pause)
            return
        reason = ["thinking ", "about ", "it"]
        words = ["hello", " there", "!"]
        if MODE == "tags":                     # reasoning inline in content
            deltas = [{"content": c} for c in
                      ["<th", "ink>"] + reason + ["</thi", "nk>"] + words]
        else:
            deltas = ([{"reasoning_content": c} for c in reason]
                      + [{"content": c} for c in words])
        self._stream(deltas, pause)

    def _stream(self, deltas, pause):
        for i, delta in enumerate(deltas):
            chunk = {"choices": [{"delta": delta}]}
            if i == len(deltas) - 1:
                chunk["timings"] = {"predicted_per_second": 42.5}
            try:
                self.wfile.write(b"data: " + json.dumps(chunk).encode()
                                 + b"\n\n")
                self.wfile.flush()
            except OSError:
                return                     # the client hung up mid-stream
            time.sleep(pause)
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            pass

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
