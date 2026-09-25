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
    othermodel /props names a model file other than the -m it was given

/props says what it loaded and its context, as llama-server does, and
/tokenize counts four characters to a token. A chat request with
ignore_eos streams max_tokens chunks, as mdl lab asks for. /metrics has
llama-server's counters, counting what this fake streamed; with --api-key,
/metrics and /props want it as a Bearer token and /health does not, as
llama-server has it.
"""
import http.server
import json
import os
import signal
import sys
import threading
import time

MODE = os.environ.get("MDL_FAKE_MODE", "")
# tasks are numbered as llama-server numbers them: its own reads take
# one too, and a slot keeps the number of the last reply it ran
COUNT = {"prompt": 0, "predicted": 0, "decode": 0, "task": 0, "slot": None}
COUNT_LOCK = threading.Lock()


def flag(name, default=None):
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def port_from_argv():
    return int(flag("--port", 8080))


def metrics():
    with COUNT_LOCK:
        c = dict(COUNT)
    lines = ["# HELP llamacpp:prompt_tokens_total Number of prompt tokens",
             "# TYPE llamacpp:prompt_tokens_total counter",
             "llamacpp:prompt_tokens_total %d" % c["prompt"],
             "llamacpp:prompt_seconds_total %.3f" % (c["prompt"] / 900.0),
             "llamacpp:tokens_predicted_total %d" % c["predicted"],
             "llamacpp:tokens_predicted_seconds_total %.3f"
             % (c["predicted"] / 50.0),
             "llamacpp:n_decode_total %d" % c["decode"],
             "llamacpp:predicted_tokens_seconds 50",
             "llamacpp:requests_processing 0",
             "llamacpp:requests_deferred 0",
             "llamacpp:kv_cache_usage_ratio 0.25"]
    return ("\n".join(lines) + "\n").encode()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        key = flag("--api-key")
        if (key and self.path in ("/metrics", "/props", "/slots")
                and self.headers.get("Authorization") != "Bearer " + key):
            self.send_error(401)
            return
        if self.path == "/health" and MODE != "silent":
            body = b'{"status":"ok"}'
        elif self.path == "/metrics" and "--metrics" in sys.argv:
            with COUNT_LOCK:
                COUNT["task"] += 1
            body = metrics()
        elif self.path == "/slots":
            with COUNT_LOCK:
                COUNT["task"] += 1
                slot = {"id": 0, "n_ctx": int(flag("-c", 4096)),
                        "is_processing": False}
                if COUNT["slot"] is not None:
                    slot["id_task"] = COUNT["slot"]
            body = json.dumps([slot]).encode()
        elif self.path == "/props":
            loaded = flag("-m", "")
            if MODE == "othermodel":
                loaded = os.path.join(os.path.dirname(loaded), "other.gguf")
            body = json.dumps({"model_path": loaded,
                               "default_generation_settings": {
                                   "n_ctx": int(flag("-c", 4096))}}).encode()
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
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/tokenize":
            text = json.loads(raw or b"{}").get("content", "")
            body = json.dumps({"tokens": list(range(len(text) // 4))}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        req = json.loads(raw or b"{}")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if req.get("ignore_eos"):              # run to max_tokens, as lab asks
            n = int(req.get("max_tokens") or 16)
            # a rate no stream here comes near, however slow the runner:
            # 42.5 is what a macOS CI runner's stretched sleeps can land
            # on, and then lab has no disagreement to flag
            self._stream([{"content": "w "} for _ in range(n)], 0.004,
                         {"prompt_n": len(json.dumps(req["messages"])) // 4,
                          "prompt_per_second": 900.0,
                          "predicted_per_second": 10000.0})
            return
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

    def _stream(self, deltas, pause, timings=None):
        with COUNT_LOCK:
            COUNT["prompt"] += 10
            COUNT["task"] += 1
            COUNT["slot"] = COUNT["task"]
        for i, delta in enumerate(deltas):
            with COUNT_LOCK:
                COUNT["predicted"] += 1
                COUNT["decode"] += 1
            chunk = {"choices": [{"delta": delta}]}
            if i == len(deltas) - 1:
                chunk["timings"] = dict({"predicted_per_second": 42.5},
                                        **(timings or {}))
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
    if "--version" in sys.argv:
        print("version: 1234 (fake)")
        return
    if "--list-devices" in sys.argv:
        print("Available devices:")        # none: a CPU machine
        return
    if "--help" in sys.argv:
        print("-m --mmproj -ngl --n-cpu-moe -c -np --port -fa "
              "--cache-type-k --cache-type-v --metrics --api-key")
        return
    print("args: " + " ".join(sys.argv[1:]), flush=True)
    if MODE == "fail":
        print("error loading model: missing tensor 'blk.0.attn_q.weight'", flush=True)
        sys.exit(1)
    if MODE == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print("load_tensors: offloaded 33/33 layers to GPU", flush=True)
    if MODE == "slow":
        # $MDL_FAKE_LOAD_S makes it a long one: a load a stop must not wait
        time.sleep(float(os.environ.get("MDL_FAKE_LOAD_S", "3.0")))
    server = http.server.HTTPServer(("127.0.0.1", port_from_argv()), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("main: server is listening on http://127.0.0.1:%d" % port_from_argv(),
          flush=True)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
