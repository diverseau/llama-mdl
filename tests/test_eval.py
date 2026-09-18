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
# B10: the fingerprint names the task, not just its id and prompt
base = evalsuite.Item("x", "x-1", "Q?", None, system="Be brief.",
                      tools=[{"type": "function", "function": {
                          "name": "f", "parameters": {"type": "object",
                                                      "properties": {}}}}])


def variant(**kw):
    d = dict(suite="x", iid="x-1", prompt="Q?", grade=None,
             system="Be brief.", tools=base.tools)
    d.update(kw)
    return evalsuite.fingerprint([evalsuite.Item(**d)])


check("the same task, the same fingerprint",
      variant(), evalsuite.fingerprint([base]))
check("another system prompt is another task",
      variant(system="Be thorough.") != variant(), True)
check("so is another parameter schema under the same tool name",
      variant(tools=[{"type": "function", "function": {
          "name": "f", "parameters": {"type": "object", "properties": {
              "x": {"type": "string"}}}}}]) != variant(), True)
check("and another document behind the same name and size",
      variant(prompt=lambda cpt: "doc A") != variant(prompt=lambda cpt: "doc B"),
      True)
check("a server moved with --port is the same server (B15)",
      evalrun.sans_port(["ls", "-m", "a", "--port", "9", "-c", "4"]),
      ["ls", "-m", "a", "-c", "4"])
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
# B02: the score used to come from the harness's own stdout and exit code,
# both of which the code under test shares. Each of these scored 1.0.
cheats = {
    "exits clean at import": "import os\nos._exit(0)\ndef %s(*a):\n"
                             "    return \"no such answer\"\n",
    "raises SystemExit(0)": "raise SystemExit(0)\ndef %s(*a):\n"
                            "    return \"no such answer\"\n",
    "prints a pass line": "import re, sys\nsrc = open('main.py').read()\n"
                          "print('passed 99 of 99')\nsys.stdout.flush()\n"
                          "import os\nos._exit(0)\ndef %s(*a):\n"
                          "    return \"no such answer\"\n",
    "reads the cases": "import json\nT = json.load(open('cases.json'))\n"
                       "def %s(*a):\n    return T\n",
    "writes its own results": "import json, atexit, os\n"
                              "def _w():\n"
                              "    json.dump([['ok', 1]] * 99, "
                              "open('results.json', 'w'))\n"
                              "    os._exit(0)\n"
                              "atexit.register(_w)\n"
                              "def %s(*a):\n    return \"no such answer\"\n",
    "floods its output": "import sys\nsys.stdout.write('x' * 5000000)\n"
                         "def %s(*a):\n    return \"no such answer\"\n",
}
for label, src in cheats.items():
    for it in by["code"][:3]:
        name = it.id.split("-", 2)[2]
        check("a submission that %s scores nothing (%s)" % (label, name),
              it.grade(R("```python\n%s```" % (src % name)), env)[0], 0.0)
ok, out = env.run_python({"main.py": "print('y' * 3000000)\n"})
check("output is kept to a bounded tail", (ok, len(out) <= 4000), (True, True))
check("and a reference still scores in full after all that",
      first.grade(R("```python\n%s```" % first.meta["reference"]), env)[0],
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
check("when no tool fits, the answer still has to be right",
      [none.grade(R("Answer: %s" % none.meta["answer"]), env)[0],
       none.grade(R("Answer: 1234567"), env)[0],
       none.grade(R("I don't know."), env)[0],
       none.grade(R(calls=[Call("1", "get_weather", {}, "")]), env)[0]],
      [1.0, 0.0, 0.0, 0.0])
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
    def call(tool, **args):                 # `tool`, not `name`: one of
        return R(calls=[Call("c%d" % random.randint(0, 99999),  # the tools
                             tool, args, json.dumps(args))])    # takes a
                                                                # `name` arg

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

    def _get_freeze(self, task, seen):      # a freeze that starts late
        target = re.search(r"behind version (\S+) forward", task).group(1)
        if not seen:
            return self.call("list_services")
        behind = [s["name"] for s in seen[0]["services"]
                  if s["version"] != target]
        done = [s["rolled_out"] for s in seen if "rolled_out" in s]
        left = [n for n in behind if n not in done]
        if not left:
            return R("All of them are on %s now." % target)
        if "frozen" not in seen[-1]:        # look again before each one
            return self.call("get_freeze")
        if seen[-1]["frozen"]:
            return R("The board froze deploys, so I stopped.")
        return self.call("rollout", service=left[0], version=target)

    def _export_statement(self, task, seen):   # ledger: paged, long
        holder = task.split("Go through ")[1].split("'s account")[0]
        limit = int(re.search(r"more than (\d+) US dollars", task).group(1))
        if not seen:
            return self.call("find_account", name=holder)
        acct = next(s["account_id"] for s in seen if "account_id" in s)
        pages = [s for s in seen if "transactions" in s]
        if not pages:
            return self.call("list_transactions", account_id=acct, page=1)
        if pages[-1]["next_page"]:
            return self.call("list_transactions", account_id=acct,
                             page=pages[-1]["next_page"])
        txns = [x for p in pages for x in p["transactions"]]
        rates = {s["currency"]: s["rate_to_usd"]
                 for s in seen if "rate_to_usd" in s}
        need = {x["currency"] for x in txns} - set(rates)
        if need:
            return self.call("get_rate", currency=sorted(need)[0])
        done = {s["flagged"] for s in seen if "flagged" in s}
        for x in txns:
            if (x["amount"] * rates[x["currency"]] > limit
                    and x["txn_id"] not in done):
                return self.call("flag_transaction", txn_id=x["txn_id"],
                                 reason="over limit")
        return R("Flagged every one that was over.")

    def _find_order(self, task, seen):      # refunds, with a rule to obey
        oid = re.search(r"(ORD-\d+)", task).group(1)
        if not seen:
            return self.call("find_order", order_id=oid)
        if len(seen) == 1:
            return self.call("get_policy", reason="faulty")
        if len(seen) == 2:
            order, policy = seen
            if order["delivered_days_ago"] <= policy["window_days"]:
                return self.call("refund", order_id=order["order_id"],
                                 amount=order["total"])
            return R("Delivered %d days ago, past the %d-day window, so no "
                     "refund." % (order["delivered_days_ago"],
                                  policy["window_days"]))
        return R("Refunded in full.")

    def _create_order(self, task, seen):    # restock, adding up as it goes
        target = int(re.search(r"hold (\d+) units", task).group(1))
        sku = re.search(r"units of (SKU-\d+)", task).group(1)
        if not seen:
            return self.call("list_warehouses")
        if len(seen) == 1:
            return R(calls=[Call("s%d" % i, "get_stock",
                                 {"warehouse": w, "sku": sku},
                                 json.dumps({"warehouse": w, "sku": sku}))
                            for i, w in enumerate(seen[0]["warehouses"])])
        if not any("ordered" in x for x in seen):
            short = sum(target - x["units"] for x in seen if "units" in x)
            return self.call("create_order", sku=sku, quantity=short)
        return R("Ordered.")

    def _lookup_code(self, task, seen):     # triage, through a flaky read
        svc = re.search(r"The (\w+) service", task).group(1)
        last = seen[-1] if seen else None
        if last is None or "error" in last:
            return self.call("read_log", service=svc)
        if "lines" in last:
            code = next(w for line in last["lines"] for w in line.split()
                        if w.startswith("E") and w[1:].isdigit())
            return self.call("lookup_code", code=code)
        if "recommended_action" in last:
            return self.call("%s_service" % last["recommended_action"],
                             service=svc)
        return R("Done.")

    def _book_meeting(self, task, seen):    # a slot that suits everyone
        need = int(re.search(r"(\d+)-minute", task).group(1))
        day = re.search(r"on (\d{4}-\d{2}-\d{2})", task).group(1)
        if not seen:
            return self.call("list_team")
        team = seen[0]["team"]
        if len(seen) == 1:
            return R(calls=[Call("b%d" % i, "get_busy",
                                 {"person": who, "date": day},
                                 json.dumps({"person": who, "date": day}))
                            for i, who in enumerate(team)])
        if not any("ok" in x for x in seen):
            busy = [(evalsuite._mins(a), evalsuite._mins(b))
                    for x in seen if "busy" in x for a, b in x["busy"]]
            for start in range(540, 1021 - need, 15):
                if all(start + need <= a or start >= b for a, b in busy):
                    return self.call("book_meeting", date=day,
                                     start=evalsuite._hm(start),
                                     minutes=need, attendees=team)
        return R("Booked.")

    def _cancel_subscription(self, task, seen):  # namesakes: read them all
        name = re.search(r"subscription for (.+?) - the one", task).group(1)
        city = re.search(r"lives in (.+?) and", task).group(1)
        domain = re.search(r"is at (\S+)\. There", task).group(1)
        if not seen:
            return self.call("search_customers", name=name)
        ids = [c["customer_id"] for c in seen[0]["customers"]]
        read = {s["customer_id"]: s for s in seen[1:] if "city" in s}
        if len(read) < len(ids):
            return self.call("get_customer", customer_id=ids[len(read)])
        if not any("plan" in s and s.get("plan") == "cancelled"
                   for s in seen):
            match = next(i for i, s in read.items() if s["city"] == city
                         and s["email"].endswith("@" + domain))
            return self.call("cancel_subscription", customer_id=match)
        return R("Cancelled.")

    def _get_record(self, task, seen):      # counters: re-read on conflict
        adds = dict((k, int(n)) for k, n in
                    re.findall(r"- (\S+): add (\d+)", task))
        done = {s["key"] for s in seen if s.get("ok")}
        left = [k for k in adds if k not in done]
        if not left:
            return R("All received.")
        k = left[0]
        last = seen[-1] if seen else {}
        if last.get("key") == k and "value" in last:
            return self.call("update_record", key=k,
                             value=last["value"] + adds[k],
                             version=last["version"])
        return self.call("get_record", key=k)

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
check("every world is in the suite, the hard ones too",
      sorted({i.id.split("-", 2)[2] for i in multi}),
      ["calendar", "conflict", "files", "freeze", "incident", "ledger",
       "namesake", "orders", "prices", "refund", "restock", "team"])
check("most of the tools suite is worlds, and most of those are hard",
      (len(multi), sum(i.meta["tier"] == "hard" for i in multi)), (18, 12))


class Hasty(Agent):
    """Takes the first namesake, and retries a conflicted write with the
    value it already worked out - the two mistakes those worlds are for."""

    def _cancel_subscription(self, task, seen):
        name = re.search(r"subscription for (.+?) - the one", task).group(1)
        if not seen:
            return self.call("search_customers", name=name)
        if len(seen) == 1:
            return self.call("cancel_subscription",
                             customer_id=seen[0]["customers"][0]["customer_id"])
        return R("Cancelled.")

    def _get_record(self, task, seen):
        adds = dict((k, int(n)) for k, n in
                    re.findall(r"- (\S+): add (\d+)", task))
        reads = {s["key"]: s for s in seen if "value" in s}
        done = {s["key"] for s in seen if s.get("ok")}
        left = [k for k in adds if k not in done]
        if not left:
            return R("All received.")
        k = left[0]
        if k not in reads:
            return self.call("get_record", key=k)
        if seen[-1].get("error"):               # retry, same value, new
            version = reads[k]["version"] + 1   # version guessed
            return self.call("update_record", key=k,
                             value=reads[k]["value"] + adds[k],
                             version=version)
        return self.call("update_record", key=k,
                         value=reads[k]["value"] + adds[k],
                         version=reads[k]["version"])


hasty = {i.id.split("-", 2)[2]: evalrun.run_item(Hasty(), i, env)
         for i in multi if i.id.endswith(("namesake", "conflict"))}
# the first search hit is the right customer one time in four, so try the
# draws where it is not
unlucky = []
for s in range(40):
    prompt, world, schemas, grade, turns = evalsuite._world_namesake(
        random.Random(s))
    w = world()
    name = re.search(r"subscription for (.+?) - the one", prompt).group(1)
    first = w.t_search_customers(name)["customers"][0]["customer_id"]
    it = evalsuite.Item("tools", "tools-99-namesake", prompt, grade,
                        evalsuite.AGENT, schemas, world=world,
                        meta={"tier": "hard", "turns": turns})
    careful = evalrun.run_item(Agent(), it, env)
    if careful["score"] == 1.0 and not w.cancelled:
        probe = world()
        probe.t_cancel_subscription(first)
        if grade(R("done"), env, probe)[0] == 0.0:
            unlucky.append(it)
    if len(unlucky) == 3:
        break
check("taking the first namesake cancels a stranger, and is paid nothing",
      [evalrun.run_item(Hasty(), it, env)["score"] for it in unlucky],
      [0.0, 0.0, 0.0])
check("and a retry that erases another writer's change loses that counter",
      (hasty["conflict"]["score"] < 1.0,
       "overwrote the other writer" in hasty["conflict"]["why"]),
      (True, True))


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

# The format items have a length floor, so that an empty reply cannot
# bank "no commas" and "do not use the letter t" for free. It must never
# cost a model that did exactly what it was told, and three of the rules
# allow a correct answer to be very short indeed.

need = {"sentences": None, "acrostic": None, "typed_json": None}
guard = 0
while not all(need.values()) and guard < 500:
    guard += 1
    k, text, chk = evalsuite._hard_rule(random.Random(guard))
    if k in need and need[k] is None:
        need[k] = (text, chk)

n = int(re.search(r"exactly (\d+) sentences", need["sentences"][0]).group(1))
terse = {"sentences": " ".join(["Bees hum."] * n)}
spell = re.search(r"spell (\w+)", need["acrostic"][0]).group(1)
terse["acrostic"] = "\n".join("%sees hums" % c for c in spell)
keys = re.findall(r'"(\w+)"', need["typed_json"][0])
terse["typed_json"] = json.dumps(
    {k: ["a", "b", "c"] if k == "tags" else "x" for k in keys},
    separators=(",", ":"))

check("the shortest correct answer to a rule still keeps it",
      [need[k][1](v) for k, v in sorted(terse.items())], [True, True, True])
check("and is never short enough to trip the floor",
      [evalsuite._wordish(v) >= 6 for v in terse.values()],
      [True, True, True])
check("while a reply that is not writing is under it",
      [evalsuite._wordish(t) for t in
       ("", "I don't know.", "Answer: 42", "none")], [0, 3, 2, 1])


# ========================================================== adversary ===
#
# A grader that can be satisfied without doing the work is worse than no
# grader, because it still reads as evidence. Every item is fed replies
# that contain no knowledge at all, and none of them may be paid for.
# (The code suite needs a subprocess per reply, so its cheats - a clean
# exit, a forged pass line, a results file of its own - are under "code".)

SINK = (
    "Answer: 0\nAnswer: 1\nAnswer: 42\nAnswer: yes\nAnswer: no\n"
    "true false none null ok done 10 12 20 42 100 1000\n"
    "passed 99 of 99\n"
    + ", ".join(evalsuite.STAFF) + "\n"
    + " ".join(evalsuite.WEEKDAYS) + "\n")

worst = {}
for it in [i for i in items if i.suite != "code"]:
    asked = it.text(4.0) if it.suite == "longctx" else it.text()
    junk = {"nothing": "", "a guess": "Answer: 42",
            "the question back": asked[-500:], "everything at once": SINK}
    if not it.id.endswith("-7"):
        # the one question the document does not answer is *meant* to be
        # passed by a refusal, so it is the one item that is not probed
        # with one; every other item must score nothing for it
        junk["refusal"] = "I don't know."
    world = it.world() if it.world else None
    for label, text in junk.items():
        got = (it.grade(R(text), env, world) if it.world
               else it.grade(R(text), env))[0]
        if got > worst.get(label, (-1, ""))[0]:
            worst[label] = (got, it.id)

check("no reply that knows nothing is ever paid in full",
      sorted(k for k, (v, _) in worst.items() if v >= 0.5), [])
check("and saying nothing is worth nothing anywhere",
      worst["nothing"][0], 0.0)
check("a guess and a refusal earn nothing either",
      [worst["a guess"][0], worst["refusal"][0]], [0.0, 0.0])
# a kitchen-sink reply can still keep one mechanical rule by accident -
# a lipogram forbids a letter, and a list of digits does not use it - so
# a third of a three-rule item is the most it may ever be worth
check("and the most an accident can be worth is one rule of three",
      [worst["the question back"][0] <= 0.34,
       worst["everything at once"][0] <= 0.34], [True, True])


# ============================================================ longctx ===

lc = by["longctx"][0]
doc = lc.text(4.0)
check("a document is about the size asked for",
      0.95 * 30000 * 4 < len(doc) < 1.05 * 30000 * 4 + 3000, True)
planted = [i for i in by["longctx"]
           if not i.id.endswith(("-5", "-6", "-7"))]
check("and every planted answer is in its document",
      all(i.meta["answer"] in i.text(4.0) for i in planted), True)
check("eight questions per length, three lengths",
      (sorted({i.meta["doc"] for i in by["longctx"]}), len(by["longctx"])),
      (["128k", "32k", "64k"], 24))
count = [i for i in by["longctx"] if i.id.endswith("-5")][0]
check("the counting question counts what the document really says",
      (count.meta["tier"],
       count.text(4.0).count(" works from room "),
       count.grade(R(count.meta["answer"]), env)[0],
       count.grade(R(str(int(count.meta["answer"]) + 1)), env)[0]),
      ("hard", int(count.meta["answer"]), 1.0, 0.0))
check("a right answer, a wrong one",
      [lc.grade(R("It is %s." % lc.meta["answer"]), env)[0],
       lc.grade(R("I could not find it."), env)[0]], [1.0, 0.0])

many = [i for i in by["longctx"] if i.id.endswith("-6")][0]
crowd = many.meta["answer"].split(", ")
check("three people share the floor, and the document says so",
      (len(crowd),
       [many.text(4.0).count("%s works from " % n) for n in crowd]),
      (3, [1, 1, 1]))
wrong = next(n for n in evalsuite.STAFF if n not in many.text(4.0))
check("finding two of three beats finding one, and a wrong name costs",
      [many.grade(R(", ".join(crowd)), env)[0],
       many.grade(R(", ".join(crowd[:2])), env)[0],
       many.grade(R(", ".join(crowd) + ", " + wrong), env)[0],
       many.grade(R("nobody works there"), env)[0]],
      [1.0, round(2 / 3, 4), 0.75, 0.0])

gone = [i for i in by["longctx"] if i.id.endswith("-7")][0]
check("the absent question asks about a project that is really there",
      (gone.meta["tier"], gone.text(4.0).count(" is led by ") > 0),
      ("hard", True))
check("saying it is not recorded passes, inventing a code does not",
      [gone.grade(R("The records do not give an access code for it."),
                  env)[0],
       gone.grade(R("N/A - no code is listed."), env)[0],
       gone.grade(R("The access code is KP-4417."), env)[0],
       gone.grade(R("It is not stated; the closest is KP-4417."), env)[0],
       gone.grade(R("hunter2"), env)[0],
       gone.grade(R(""), env)[0]],
      [1.0, 1.0, 0.0, 0.0, 0.0, 0.0])


class Echo:
    def chat(self, messages, tools=None, max_tokens=0, seed=None):
        return R("nothing")


done = evalrun.run_items(Echo(), by["longctx"], env, n_ctx=65536)
check("documents longer than the context are skipped, not failed",
      [("skipped" in r) for r in done],
      [i.meta["tokens"] + evalsuite.ROOM > 65536 for i in by["longctx"]])

# ------------------------------------------------ resume and retries ---
evalrun.RETRIES = (0, 0)            # no real waiting in tests
reason = by["reason"][:4]


class Flaky:
    """Fails the first `bad` calls with a transport error, then answers."""

    def __init__(self, bad):
        self.bad, self.calls = bad, 0

    def chat(self, messages, tools=None, max_tokens=0, seed=None):
        self.calls += 1
        if self.calls <= self.bad:
            return R(error="connection reset")
        return R("Answer: 0")


flaky = Flaky(2)
got = evalrun.run_items(flaky, reason[:1], env)
check("a server error is retried, then graded as an answer",
      ("score" in got[0], "failed" in got[0], flaky.calls), (True, False, 3))
dead = evalrun.run_items(Flaky(99), reason[:1], env)
check("one that keeps failing is recorded as failed, with no score",
      ("score" in dead[0], dead[0].get("failed"), dead[0]["error"]),
      (False, "error: connection reset", True))
check("and summaries leave it out rather than count it wrong",
      evalrun.summarize(dead, "suite"), {})

ck = TMP / "runs" / "x.jsonl"
keep = evalrun.checkpoint(ck)
first = evalrun.run_items(Echo(), reason[:2], env, keep=keep)
keep.close()
with open(ck, "a", encoding="utf-8") as fh:
    fh.write('{"id": "cut off mid-wri')          # the interrupt's last line
    fh.write("\n" + json.dumps(dead[0]) + "\n")
prior = evalrun.load_checkpoint(ck)
check("a checkpoint keeps finished items, drops a torn line and failures",
      sorted(prior), sorted(r["id"] for r in first))
counted = Flaky(0)
rest = evalrun.run_items(counted, reason, env, done_ids=set(prior))
check("a resumed run asks only what is left",
      ([r["id"] for r in rest], counted.calls),
      ([i.id for i in reason[2:]], 2))
check("the run id changes with the items, the runtime or the seed",
      len({evalrun.run_id("m", *k) for k in (
          ("items", "rt", "s"), ("items2", "rt", "s"),
          ("items", "rt2", "s"), ("items", "rt", "s2"))}), 4)

os.environ["MDL_FIT_HOME"] = str(TMP / "home-save")
evalrun.save({"name": "m", "run": "m-1", "partial": True, "n": 1})
evalrun.save({"name": "other", "partial": True})
evalrun.save({"name": "m", "run": "m-1", "partial": True, "n": 2})
evalrun.save({"name": "m", "run": "m-1", "partial": False, "n": 3})
check("a finished run replaces its own partial records, and nothing else",
      [(r["name"], r.get("n")) for r in evalrun.load()],
      [("other", None), ("m", 3)])
os.environ["MDL_FIT_HOME"] = str(TMP / "home")
held = TMP / "held.lock"
held.write_text(str(os.getpid()))           # a live eval holds it
import mdl  # noqa: E402

_, err, code = support.run(mdl.file_lock(held, "another mdl eval is "
                                         "running these items").__enter__)
check("a second eval of the same items on the same server is refused "
      "(peer review)", ("another mdl eval" in err, code), (True, 1))
held.unlink()
check("a container runtime that is not there is not usable",
      evalrun.usable_runtime(str(TMP / "no-such-docker")), False)

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
hot = io.StringIO()
evalrun.report(dict(rec, sampling={"temp": "0.7"}), hot.write)
cold = io.StringIO()
evalrun.report(dict(rec, sampling={"temp": "0"}), cold.write)
check("a run that was sampled says the interval is not over runs",
      ["not over runs" in hot.getvalue(),
       "not over runs" in cold.getvalue(),
       "overall" in hot.getvalue()],
      [True, False, True])
check("the report shows scores, skips and misses",
      [s in out.getvalue() for s in ("0.50", "skipped", "miss")],
      [True, True, True])
check("the file hash reads the ends, not the whole file",
      len(evalrun.file_hash(Path(__file__))), 16)
spend = [{"at": "2026-01-01 00:00", "hash": "h1", "items": [
            {"suite": "code", "completion_tokens": 100},
            {"suite": "code", "completion_tokens": 300}]},
         {"at": "2026-02-02 00:00", "hash": "h1", "items": [
            {"suite": "code", "completion_tokens": 50},
            {"suite": "reason", "completion_tokens": 80}]},
         {"at": "2026-03-03 00:00", "hash": "other", "items": [
            {"suite": "code", "completion_tokens": 9999}]},
         {"at": "2026-04-04 00:00", "hash": "h1", "partial": True,
          "items": [{"suite": "code", "completion_tokens": 7777}]}]
def run_of(name, scores):
    return {"at": "2026-05-05 00:00", "name": name, "items_hash": "same",
            "suite_version": 2, "suites": {}, "minutes": 1.0,
            "items": [{"id": "code-%02d" % i, "domain": "coding",
                       "score": v} for i, v in enumerate(scores)]}


good = run_of("good", [1.0] * 16 + [0.0] * 4)
poor = run_of("poor", [1.0] * 6 + [0.0] * 14)
same = run_of("same", [1.0] * 15 + [0.0] * 5)
out = io.StringIO()
evalrun.compare(good, poor, out.write)
check("a real gap is called, item by item",
      ("good ahead" in out.getvalue(), "0.80" in out.getvalue()),
      (True, True))
out = io.StringIO()
evalrun.compare(good, same, out.write)
check("and a gap inside the noise is not",
      "too close to call" in out.getvalue(), True)
out = io.StringIO()
evalrun.compare(good, dict(poor, items_hash="other"), out.write)
check("runs on different questions are refused, not compared",
      "not comparable" in out.getvalue(), True)
check("what a model spent last time beats a constant, per suite",
      (evalrun.spent("h1", spend), evalrun.spent("nope", spend)),
      ({"code": 50.0, "reason": 80.0}, {}))
check("the command line says whether it thinks",
      [evalrun.thinking_flag(a) for a in (["--reasoning", "on"],
                                          ["--reasoning", "off"],
                                          ["--reasoning-budget", "0"], [])],
      [True, False, False, None])

sys.exit(t.done())
