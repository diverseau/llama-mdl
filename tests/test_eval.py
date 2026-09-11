"""mdl eval: the suites, their graders, the tool loop, the client, and
the results store. Nothing here needs a model: reference solutions and a
scripted agent stand in for one, and a stand-in server for llama-server.
"""
import http.server
import io
import json
import os
import random
import re
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                   # noqa: E402

from mdl_fit import evalrun, evalsuite           # noqa: E402

t = support.Tally("test_eval")
check = t.check
TMP = Path(tempfile.mkdtemp(prefix="mdl-eval-test-"))
os.environ["MDL_FIT_HOME"] = str(TMP / "home")   # never the real ~/.config
R, Call = evalrun.Reply, evalrun.Call
env = evalrun.Env()

# ============================================================== items ===

seed = "test-seed"
items = evalsuite.build(evalsuite.SUITES, seed)
by = {}
for it in items:
    by.setdefault(it.suite, []).append(it)
check("every suite at its size", {s: len(v) for s, v in by.items()},
      evalsuite.SIZE)
check("ids are unique", len({i.id for i in items}), len(items))
again = evalsuite.build(evalsuite.SUITES, seed)
check("the same seed gives the same items",
      [i.text()[:300] for i in items if i.suite != "longctx"],
      [i.text()[:300] for i in again if i.suite != "longctx"])
check("another seed gives other items",
      [i.text() for i in evalsuite.build(["reason"], "other")]
      != [i.text() for i in by["reason"]], True)
check("the seed is made once and kept", evalsuite.secret(),
      evalsuite.secret())
check("results name the item set, not the seed",
      len(evalsuite.seed_id(seed)), 8)
check("limit takes the first N of each suite",
      len(evalsuite.build(["code", "reason"], seed, limit=3)), 6)

# =============================================================== code ===

ok = [it.grade(R("```python\n%s```" % it.meta["reference"]), env)[0]
      for it in by["code"]]
check("every code item's reference solution passes its hidden tests",
      sum(ok), len(ok))
first = by["code"][0]
fname = first.id.split("-", 2)[2]
check("a wrong solution fails",
      first.grade(R("```python\ndef %s(*a):\n    return None\n```" % fname),
                  env)[0], 0.0)
check("no function, no score", first.grade(R("I cannot do that."), env),
      (0.0, "no function %s in the reply" % fname))
slow = evalrun.Env(timeout=2)
score, why = first.grade(R("```python\ndef %s(*a):\n    while True:\n"
                           "        pass\n```" % fname), slow)
check("a solution that never returns times out", (score, "timed out" in why),
      (0.0, True))
check("a __main__ block in the reply does not run",
      first.grade(R("```python\n%s\nif __name__ == '__main__':\n"
                    "    input()\n```" % first.meta["reference"]), env)[0],
      1.0)
check("the code block that defines the function wins",
      evalsuite.extract_code("```python\nprint(1)\n```\n```python\ndef f(x):"
                             "\n    return x\n```\n```\nf(2)\n```", "f"),
      "def f(x):\n    return x\n")

# ============================================================= reason ===

ok = [it.grade(R("Let me work it out.\nAnswer: %s" % it.meta["answer"]),
               env)[0] for it in by["reason"]]
check("exact answers grade right", sum(ok), len(ok))
check("and a wrong one does not",
      by["reason"][0].grade(R("Answer: 1234567"), env)[0], 0.0)
check("answers are read the way people write them",
      [evalsuite.same_answer(evalsuite.final_answer(x), w) for x, w in (
          ("so it is\n**Answer:** $1,250.", "1250"),
          ("Answer: Friday", "Friday"), ("answer: It's Friday.", "Friday"),
          ("Answer: 1011", "1011"), ("Answer: 12", "13"))],
      [True, True, True, True, False])

# ============================================================== tools ===

grade = evalsuite._grade_call("get_weather", {
    "city": evalsuite._has("Oslo"), "unit": evalsuite._ci("celsius")})
check("the right call with the right arguments",
      grade(R(calls=[Call("1", "get_weather",
                          {"city": "Oslo, Norway", "unit": "Celsius"},
                          "")]), env)[0], 1.0)
check("wrong unit, wrong tool, no call", [
    grade(R(calls=[Call("1", "get_weather", {"city": "Oslo",
                                              "unit": "fahrenheit"}, "")]),
          env)[0],
    grade(R(calls=[Call("1", "set_timer", {}, "")]), env)[0],
    grade(R("It is cold."), env)[0]], [0.0, 0.0, 0.0])
check("times are read in 24-hour form", [evalsuite._hhmm("09:30")(x) for x in
                                         ("9:30", "09:30", "21:30", 930)],
      [True, True, False, False])
none = [i for i in by["tools"] if i.id.endswith("-none")][0]
check("when no tool fits, answering is right and calling is wrong",
      [none.grade(R("It is 1,234."), env)[0],
       none.grade(R(calls=[Call("1", "get_weather", {}, "")]), env)[0]],
      [1.0, 0.0])
single = [i for i in by["tools"] if not i.world]
check("single-call items offer three tools, the right one among them",
      all(len(i.tools) in (2, 3) for i in single), True)


class Agent:
    """Plays a careful agent against the mock worlds: reads the task,
    calls the tools in order, reads what comes back."""

    def __init__(self):
        self.turns = 0

    def chat(self, messages, tools=None, max_tokens=0, seed=None):
        self.turns += 1
        task = messages[-1]["content"] if messages[-1]["role"] == "user" \
            else next(m["content"] for m in messages if m["role"] == "user")
        seen = [json.loads(m["content"]) for m in messages
                if m["role"] == "tool"]
        names = {t["function"]["name"] for t in tools}
        return getattr(self, "_" + sorted(names)[0])(task, seen)

    @staticmethod
    def call(name, **args):
        return R(calls=[Call("c%d" % random.randint(0, 99999), name, args,
                             json.dumps(args))])

    def _book(self, task, seen):            # calendar
        need = int(re.search(r"(\d+)-minute", task).group(1))
        day = re.search(r"on (\d{4}-\d{2}-\d{2})", task).group(1)
        if not seen:
            return self.call("get_busy", date=day)
        if len(seen) == 1:
            busy = [(evalsuite._mins(a), evalsuite._mins(b))
                    for a, b in seen[0]["busy"]]
            for s in [540] + [b for _, b in busy]:
                if s + need <= 1020 and all(s + need <= a or s >= b
                                            for a, b in busy):
                    return self.call("book", date=day, start=evalsuite._hm(s),
                                     minutes=need, title="sync")
        return R("Booked.")

    def _cancel_order(self, task, seen):    # orders
        email = re.search(r"Customer (\S+@\S+)", task).group(1)
        item = re.search(r"cancel their (.+) order", task).group(1)
        if not seen:
            return self.call("find_customer", email=email)
        if len(seen) == 1:
            return self.call("list_orders",
                             customer_id=seen[0]["customer_id"])
        if len(seen) == 2:
            oid = next(o["order_id"] for o in seen[1]["orders"]
                       if o["item"] == item)
            return self.call("cancel_order", order_id=oid)
        return R("Done, it is cancelled.")

    def _get_price(self, task, seen):       # prices
        wants = re.findall(r"(\d+) (\w+?)s\b", task.split("?")[0])
        if not seen:
            return R(calls=[Call("p%d" % i, "get_price", {"item": n},
                                 json.dumps({"item": n}))
                            for i, (_, n) in enumerate(wants)])
        prices = {s["item"]: s["unit_price"] for s in seen}
        total = sum(int(q) * prices[n] for q, n in wants)
        return R("That comes to...\nAnswer: %.2f" % total)

    def _list_dir(self, task, seen):        # files
        svc = re.search(r"is the (\w+) service", task).group(1)
        if not seen:
            return self.call("list_dir", path="/srv")
        if len(seen) == 1:
            return self.call("list_dir", path="/srv/" + svc)
        if len(seen) == 2:
            return self.call("read_file", path="/srv/%s/config.ini" % svc)
        port = re.search(r"port = (\d+)", seen[2]["content"]).group(1)
        return R("It listens on %s.\nAnswer: %s" % (port, port))


multi = [i for i in by["tools"] if i.world]
got = [evalrun.run_item(Agent(), i, env) for i in multi]
check("a careful agent solves every multi-step item",
      [(r["id"], r["score"]) for r in got],
      [(r["id"], 1.0) for r in got])
check("all four worlds are in the suite",
      sorted({i.id.split("-", 2)[2] for i in multi}),
      ["calendar", "files", "orders", "prices"])


class Lazy(Agent):
    def chat(self, messages, tools=None, max_tokens=0, seed=None):
        return R("I would call the tools, but I will not.")


check("an agent that never calls a tool solves none of them",
      sum(evalrun.run_item(Lazy(), i, env)["score"] for i in multi), 0.0)


class Loop(Agent):
    def chat(self, messages, tools=None, max_tokens=0, seed=None):
        self.turns += 1
        return self.call(tools[0]["function"]["name"], nonsense=1)


loop = Loop()
evalrun.run_item(loop, multi[0], env)
check("a model stuck calling tools is cut off", loop.turns, evalrun.MAX_TURNS)
w = multi[0].world()
check("bad calls get an error back, never an exception",
      [("error" in w.call("nope", {})), ("error" in w.call(
          w.tools[0], "not a dict")), ("error" in w.call(w.tools[0],
                                                          {"bogus": 1}))],
      [True, True, True])

# ============================================================ longctx ===

lc = by["longctx"][0]
doc = lc.text(4.0)
check("a document is about the size asked for",
      0.95 * 30000 * 4 < len(doc) < 1.05 * 30000 * 4 + 3000, True)
check("and every answer is in its document",
      all(i.meta["answer"] in i.text(4.0) for i in by["longctx"]), True)
check("five questions per length, three lengths",
      sorted({i.meta["doc"] for i in by["longctx"]}), ["128k", "32k", "64k"])
check("a right answer, a wrong one",
      [lc.grade(R("It is %s." % lc.meta["answer"]), env)[0],
       lc.grade(R("I could not find it."), env)[0]], [1.0, 0.0])


class Echo:
    def chat(self, messages, tools=None, max_tokens=0, seed=None):
        return R("nothing")


done = evalrun.run_items(Echo(), by["longctx"], env, n_ctx=65536)
check("documents longer than the context are skipped, not failed",
      [("skipped" in r) for r in done],
      [i.meta["tokens"] + evalsuite.ROOM > 65536 for i in by["longctx"]])

# =========================================================== instruct ===

rules = {}
rng = random.Random(3)
while len(rules) < 11:
    for kind, text, fn in (evalsuite._structure(rng), evalsuite._lexical(rng)):
        if kind and kind not in rules:
            rules[kind] = (text, fn)


def rule_ok(kind, text):
    return rules[kind][1](text)


n_bul = int(re.search(r"exactly (\d+)", rules["bullets"][0]).group(1))
n_par = int(re.search(r"exactly (\d+)", rules["paragraphs"][0]).group(1))
keys = re.findall(r'"(\w+)"', rules["json"][0])
kw = re.search(r"'(\w+)'", rules["keyword"][0]).group(1)
n_kw = int(re.search(r"at least (\d+)", rules["keyword"][0]).group(1))
end = rules["ends"][0].split("sentence: ", 1)[1]
opener = re.search(r"'(\w+)'", rules["starts"][0]).group(1)
check("format rules pass what they ask for", [
    rule_ok("bullets", "Intro\n" + "\n".join("- point %d" % i
                                             for i in range(n_bul))),
    rule_ok("paragraphs", "\n***\n".join("para %d" % i for i in range(n_par))),
    rule_ok("json", "```json\n%s\n```" % json.dumps({k: 1 for k in keys})),
    rule_ok("quotes", '"all of it"'), rule_ok("title", "<<Tides>>\ntext"),
    rule_ok("lower", "all lower here"), rule_ok("nocomma", "no commas here"),
    rule_ok("keyword", " ".join([kw] * n_kw)),
    rule_ok("maxwords", "short"), rule_ok("ends", "Words. " + end),
    rule_ok("starts", "%s, yes." % opener)], [True] * 11)
check("and fail what they forbid", [
    rule_ok("bullets", "- one"), rule_ok("json", '{"x": 1}'),
    rule_ok("lower", "Capital"), rule_ok("nocomma", "a, b"),
    rule_ok("keyword", kw), rule_ok("maxwords", "word " * 100),
    rule_ok("ends", end + " More."), rule_ok("starts", "No.")],
    [False] * 8)
check("no item pairs rules that cannot both be kept",
      all(tuple(i.id.split("-")[2:4]) not in evalsuite.CLASH
          for i in by["instruct"] if i.id.count("-") == 3), True)

# ============================================================= custom ===

cdir = TMP / "custom"
cdir.mkdir()
(cdir / "mine.toml").write_text('''
[[task]]
id = "cap"
prompt = "Capital of France?"
expect = "paris"

[[task]]
id = "re"
prompt = "A date?"
check = "regex"
expect = '\\d{4}-\\d{2}-\\d{2}'
domain = "reasoning"

[[task]]
id = "py"
prompt = "Say hi."
check = "python"
test = "assert reply.strip().lower().startswith('hi')"
''', encoding="utf-8")
mine = evalsuite.build(["custom"], seed, custom_dir=cdir)
check("your own tasks load, with their domains",
      [(i.id, i.domain) for i in mine],
      [("custom-mine-cap", "general"), ("custom-mine-re", "reasoning"),
       ("custom-mine-py", "general")])
check("and grade", [mine[0].grade(R("It is Paris."), env)[0],
                    mine[1].grade(R("On 2026-09-11."), env)[0],
                    mine[2].grade(R("Hi there"), env)[0],
                    mine[2].grade(R("Hello"), env)[0]], [1.0, 1.0, 1.0, 0.0])
(cdir / "bad.toml").write_text('[[task]]\nid = "x"\n', encoding="utf-8")
try:
    evalsuite.build(["custom"], seed, custom_dir=cdir)
    raised = None
except ValueError as e:
    raised = str(e)
check("a task with no prompt is refused, with the file named",
      raised is not None and "bad.toml" in raised, True)

# ============================================================= client ===

seen_bodies = []


class Stand(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"default_generation_settings": {"n_ctx": 4096}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen_bodies.append(body)
        if self.path == "/tokenize":
            self._send(200, {"tokens": list(range(len(body["content"]) // 4))})
        elif body.get("tools") and body["tools"][0]["function"]["name"] == "x":
            self._send(500, {"error": "tools param requires --jinja flag"})
        else:
            self._send(200, {"choices": [{"message": {
                "content": "<think>let me see</think>Hi there",
                "tool_calls": [{"id": "abc", "type": "function", "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Oslo"}'}}]},
                "finish_reason": "length"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7}})


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stand)
threading.Thread(target=srv.serve_forever, daemon=True).start()
client = evalrun.Client(srv.server_port)
r = client.chat([{"role": "user", "content": "hi"}], max_tokens=64, seed=5)
check("replies: answer, inline thinking, tool calls, usage, finish",
      (r.content, r.reasoning.startswith("<think>"), r.calls[0].name,
       r.calls[0].args, r.prompt_tokens, r.completion_tokens, r.finish),
      ("Hi there", True, "get_weather", {"city": "Oslo"}, 11, 7, "length"))
check("the request asks for the prompt cache and a fixed seed",
      (seen_bodies[-1]["cache_prompt"], seen_bodies[-1]["seed"],
       seen_bodies[-1]["stream"]), (True, 5, False))
bad = client.chat([{"role": "user", "content": "hi"}],
                  tools=[{"type": "function", "function": {"name": "x"}}])
check("a server error is an error, not a crash",
      (bad.error or "").startswith("HTTP 500") and "jinja" in bad.error, True)
check("tokens and props", (client.tokens("abcd" * 10),
                           client.props()["default_generation_settings"]),
      (10, {"n_ctx": 4096}))
check("characters per token come from the server's tokenizer",
      round(evalrun.chars_per_token(client), 1), 4.0)
dead = evalrun.Client(1, timeout=2).chat([{"role": "user", "content": "x"}])
check("no server: an error", bool(dead.error), True)

# ======================================================== score, store ==

check("an interval around a perfect score is a point",
      evalrun.bootstrap([1.0] * 20), (1.0, 1.0, 1.0))
m, lo, hi = evalrun.bootstrap([1.0] * 12 + [0.0] * 8, seed=1)
check("a mixed one brackets the mean", (lo < m < hi, round(m, 2)),
      (True, 0.6))
res = [{"id": "a", "suite": "code", "domain": "coding", "score": 1.0,
        "capped": False, "error": False},
       {"id": "b", "suite": "code", "domain": "coding", "score": 0.0,
        "capped": True, "error": False},
       {"id": "c", "suite": "longctx", "domain": "long-context",
        "skipped": "too long"}]
s = evalrun.summarize(res, "suite")
check("skipped items do not count", (sorted(s), s["code"]["n"],
                                     s["code"]["capped"]), (["code"], 2, 1))
rec = {"at": "2026-09-11 21:00", "name": "tiny", "quant": "Q8_0",
       "kv": "q8_0/q8_0", "build": 1, "suite_version": 1,
       "suites": s, "domains": evalrun.summarize(res, "domain"),
       "items": res, "minutes": 1.5}
evalrun.save(rec)
out = io.StringIO()
evalrun.main(["--results"], out)
check("results are kept and listed", "tiny" in out.getvalue()
      and "0.50" in out.getvalue(), True)
out = io.StringIO()
evalrun.report(rec, out.write)
check("the report shows scores, skips and misses",
      [s in out.getvalue() for s in ("0.50", "skipped", "miss")],
      [True, True, True])
check("the file hash reads the ends, not the whole file",
      len(evalrun.file_hash(Path(__file__))), 16)
check("the command line says whether it thinks",
      [evalrun.thinking_flag(a) for a in (["--reasoning", "on"],
                                          ["--reasoning", "off"],
                                          ["--reasoning-budget", "0"], [])],
      [True, False, False, None])

sys.exit(t.done())
