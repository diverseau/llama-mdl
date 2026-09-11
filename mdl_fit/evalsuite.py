"""mdl eval - the suites.

Every item is generated from a template, with parameters drawn from a
seed that is secret to this install (~/.config/mdl/eval-seed), so the
questions exist on this machine only and cannot have been trained on.
The same seed and suite version give the same items: two models, or two
runs of one, answer the same questions. Bump SUITE_VERSION when a
template changes, so old results stop being compared with new ones.

Everything is graded by code - unit tests executed, tool calls checked
against what was asked, exact answers, format rules. No model judges
another.

A grader is `grade(reply, env, world=None) -> (score 0..1, why)`; reply
is evalrun.Reply, env runs code (evalrun.Env), world is the mock a
multi-step tool item acts on.
"""

import copy
import datetime
import hashlib
import json
import math
import random
import re
import secrets
import tomllib
from pathlib import Path

from . import hw

SUITE_VERSION = 1
SUITES = ("code", "tools", "longctx", "instruct", "reason")
DOMAIN = {"code": "coding", "tools": "agentic", "longctx": "long-context",
          "instruct": "general", "reason": "reasoning", "custom": "general"}
SIZE = {"code": 40, "tools": 30, "longctx": 15, "instruct": 20, "reason": 20}
# max_tokens per request. A thinking model spends most of it thinking; a
# reply that hits the cap is graded as it stands and counted as capped.
CAP = {"code": 6144, "tools": 3072, "longctx": 2048, "instruct": 3072,
       "reason": 6144, "custom": 4096}
# Reply tokens a model that does not think typically spends, for the
# time estimate. A thinking model is costed at THINK_FACTOR times this.
EXPECT = {"code": 450, "tools": 120, "longctx": 40, "instruct": 300,
          "reason": 450, "custom": 400}
THINK_FACTOR = 4
LONG = (("32k", 30000), ("64k", 60000), ("128k", 120000))
ROOM = 2048            # tokens a long document leaves for question + answer


class Item:
    def __init__(self, suite, iid, prompt, grade, system=None, tools=None,
                 world=None, meta=None):
        self.suite, self.id = suite, iid
        self.prompt = prompt       # a string, or cpt -> string (long docs)
        self.grade, self.system = grade, system
        self.tools = tools         # OpenAI tool schemas, or None
        self.world = world         # () -> fresh mock world, multi-step only
        self.meta = meta or {}
        self.domain = DOMAIN.get(suite, "general")
        self.max_tokens = CAP.get(suite, 4096)

    def text(self, cpt=4.0):
        return self.prompt(cpt) if callable(self.prompt) else self.prompt

    def messages(self, cpt=4.0):
        msgs = [{"role": "system", "content": self.system}] if self.system \
            else []
        return msgs + [{"role": "user", "content": self.text(cpt)}]

    def __repr__(self):
        return "Item(%s)" % self.id


# ------------------------------------------------------------- seeding --

def secret():
    """This install's eval seed, made on first use."""
    path = hw.config_dir() / "eval-seed"
    try:
        s = path.read_text(encoding="utf-8").strip()
        if s:
            return s
    except OSError:
        pass
    s = secrets.token_hex(16)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s + "\n", encoding="utf-8")
    return s


def seed_id(seed):
    """Names the item set in results without giving the seed away."""
    return hashlib.sha256(seed.encode()).hexdigest()[:8]


def rng_for(seed, suite):
    return random.Random("%s:%s:v%d" % (seed, suite, SUITE_VERSION))


def build(suites, seed, limit=None, custom_dir=None):
    items = []
    for suite in suites:
        if suite == "custom":
            got = load_custom(custom_dir) if custom_dir else []
        else:
            got = GENERATORS[suite](rng_for(seed, suite))
        items += got[:limit] if limit else got
    return items


# -------------------------------------------------------------- common --

WORDS = ("apple river stone cloud maple ember quartz lantern violet harbor "
         "meadow copper falcon orbit pebble thistle willow cobalt saffron "
         "tundra glacier bramble cinder hollow juniper marble nectar "
         "prairie sparrow timber").split()
NAMES = ("Alice Bruno Chen Dana Elif Farid Greta Hugo Ines Jonas Kiri "
         "Lena Mateo Nadia Oskar Priya Quinn Rosa Sami Tara").split()
WEEKDAYS = ("Monday Tuesday Wednesday Thursday Friday Saturday "
            "Sunday").split()
MONTHS = ("January February March April May June July August September "
          "October November December").split()


def day_words(d):
    return "%s %d, %d" % (MONTHS[d.month - 1], d.day, d.year)


def strip_fences(text):
    m = re.search(r"```(?:json|python|py)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def final_answer(text):
    """What follows the last 'Answer:' in a reply, or None."""
    found = re.findall(r"answer\s*[:\uff1a]\s*(.+)", text or "", re.I)
    if not found:
        return None
    a = found[-1].strip().strip("*").strip()
    return a.rstrip(".").strip()


def _number(text):
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", (text or "").replace("$", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def same_answer(got, want):
    """Exact-answer grading: numbers by value, words by word."""
    if got is None:
        return False
    try:
        w = float(want)
    except ValueError:
        g = re.sub(r"[^a-z0-9 ]", " ", got.lower()).split()
        return want.lower() in g or " ".join(g) == want.lower()
    n = _number(got)
    return n is not None and abs(n - w) < 1e-6 * max(1.0, abs(w))


# ---------------------------------------------------------------- code --

CODE = []


def _code(fn):
    CODE.append(fn)
    return fn


def _ints(rng, lo, hi, n_lo, n_hi):
    return [rng.randint(lo, hi) for _ in range(rng.randint(n_lo, n_hi))]


def _sentence(rng, n_lo=3, n_hi=12):
    words = [rng.choice(WORDS) for _ in range(rng.randint(n_lo, n_hi))]
    for i in range(len(words)):
        r = rng.random()
        if r < 0.15:
            words[i] += rng.choice(".,!?;:")
        elif r < 0.25:
            words[i] = words[i].capitalize()
    return " ".join(words)


@_code
def _kth(rng):
    k = rng.randint(2, 5)
    return ("every_kth_sum",
            "Write a Python function `every_kth_sum(xs)` that returns the "
            "sum of the elements of the list `xs` whose index is a multiple "
            "of %d (index 0 counts). An empty list sums to 0." % k,
            "def every_kth_sum(xs):\n"
            "    return sum(x for i, x in enumerate(xs) if i %% %d == 0)\n"
            % k,
            [[_ints(rng, -50, 50, 0, 12)] for _ in range(8)])


@_code
def _rotate(rng):
    side = rng.choice(["left", "right"])
    body = ("    return xs[n:] + xs[:n]\n" if side == "left"
            else "    return xs[-n:] + xs[:-n]\n")
    return ("rotate",
            "Write a Python function `rotate(xs, n)` that returns a new list: "
            "`xs` rotated %s by `n` positions. `n` may be zero or larger "
            "than the list; an empty list stays empty." % side,
            "def rotate(xs, n):\n    if not xs:\n        return []\n"
            "    n %= len(xs)\n" + body,
            [[_ints(rng, 0, 9, 0, 8), rng.randint(0, 20)] for _ in range(8)])


@_code
def _long_words(rng):
    n = rng.randint(4, 7)
    return ("count_long",
            "Write a Python function `count_long(text)` that returns how many "
            "words in `text` have at least %d letters. Words are separated by "
            "whitespace; ignore any of the characters .,!?;: at either end "
            "of a word." % n,
            "def count_long(text):\n    return sum(1 for w in text.split() "
            "if len(w.strip('.,!?;:')) >= %d)\n" % n,
            [[_sentence(rng)] for _ in range(8)])


@_code
def _rle(rng):
    first = rng.choice([True, False])
    piece = "str(j - i) + s[i]" if first else "s[i] + str(j - i)"
    example = '"aaab" -> "3a1b"' if first else '"aaab" -> "a3b1"'
    cases = []
    for _ in range(8):
        s = "".join(rng.choice("abc") * rng.randint(1, 4)
                    for _ in range(rng.randint(0, 5)))
        cases.append([s])
    return ("rle",
            "Write a Python function `rle(s)` that run-length encodes a "
            "string: each run of one repeated character becomes %s, as in "
            "%s. The empty string encodes to the empty string." % (
                "its length followed by the character" if first
                else "the character followed by its length", example),
            "def rle(s):\n    out = []\n    i = 0\n    while i < len(s):\n"
            "        j = i\n        while j < len(s) and s[j] == s[i]:\n"
            "            j += 1\n        out.append(%s)\n        i = j\n"
            "    return ''.join(out)\n" % piece,
            cases)


@_code
def _lcg(rng):
    a, b = rng.randint(2, 9), rng.randint(1, 20)
    m, s = rng.choice([97, 101, 251, 1009]), rng.randint(0, 50)
    return ("nth_term",
            "Write a Python function `nth_term(n)` for the sequence "
            "x(0) = %d, x(k+1) = (%d * x(k) + %d) mod %d. Return x(n); "
            "nth_term(0) is %d." % (s, a, b, m, s),
            "def nth_term(n):\n    x = %d\n    for _ in range(n):\n"
            "        x = (%d * x + %d) %% %d\n    return x\n" % (s, a, b, m),
            [[rng.randint(0, 200)] for _ in range(8)])


@_code
def _caesar(rng):
    k, way = rng.randint(1, 25), rng.choice(["forward", "backward"])
    d = k if way == "forward" else -k
    return ("shift",
            "Write a Python function `shift(text)` that moves every ASCII "
            "letter in `text` %d places %s in the alphabet, wrapping around "
            "at the ends and keeping its case. Every other character is "
            "left alone." % (k, way),
            "def shift(text):\n    out = []\n    for c in text:\n"
            "        if 'a' <= c <= 'z':\n"
            "            out.append(chr((ord(c) - 97 + %d) %% 26 + 97))\n"
            "        elif 'A' <= c <= 'Z':\n"
            "            out.append(chr((ord(c) - 65 + %d) %% 26 + 65))\n"
            "        else:\n            out.append(c)\n"
            "    return ''.join(out)\n" % (d, d),
            [[_sentence(rng, 2, 6)] for _ in range(8)])


@_code
def _merge(rng):
    touch = rng.choice([True, False])
    cases = []
    for _ in range(8):
        iv = []
        for _ in range(rng.randint(0, 6)):
            s = rng.randint(0, 20)
            iv.append([s, s + rng.randint(0, 5)])
        if iv and rng.random() < 0.6:          # make one that only touches
            iv.append([iv[0][1], iv[0][1] + rng.randint(1, 4)])
        cases.append([iv])
    return ("merge",
            "Write a Python function `merge(intervals)` that takes a list of "
            "[start, end] integer pairs (start <= end, in any order) and "
            "returns the merged intervals as a list of [start, end] lists "
            "sorted by start. Overlapping intervals merge. Intervals that "
            "only touch, like [1, 3] and [3, 5], %s." % (
                "merge too" if touch else "stay separate"),
            "def merge(intervals):\n    out = []\n"
            "    for s, e in sorted(intervals):\n"
            "        if out and (s < out[-1][1] or (%s and s == out[-1][1])):\n"
            "            out[-1][1] = max(out[-1][1], e)\n"
            "        else:\n            out.append([s, e])\n"
            "    return out\n" % touch,
            cases)


@_code
def _balanced(rng):
    pairs = rng.choice(["()[]", "(){}", "()[]{}", "()<>", "[]{}<>"])
    closing = {pairs[i + 1]: pairs[i] for i in range(0, len(pairs), 2)}
    cases = []
    for _ in range(10):
        if rng.random() < 0.5:                 # build a balanced one
            s, stack = "", []
            for _ in range(rng.randint(0, 8)):
                if stack and rng.random() < 0.5:
                    s += {v: k for k, v in closing.items()}[stack.pop()]
                else:
                    o = rng.choice(list(closing.values()))
                    stack.append(o)
                    s += o
                if rng.random() < 0.3:
                    s += rng.choice("ab ")
            s += "".join({v: k for k, v in closing.items()}[o]
                         for o in reversed(stack))
        else:
            s = "".join(rng.choice(pairs + "ab") for _ in range(rng.randint(1, 10)))
        cases.append([s])
    return ("balanced",
            "Write a Python function `balanced(s)` that returns True if the "
            "brackets in `s` are balanced and properly nested, counting only "
            "these pairs: %s. Every other character is ignored." % " ".join(
                pairs[i:i + 2] for i in range(0, len(pairs), 2)),
            "def balanced(s):\n    pairs = %r\n    opens = set(pairs.values())\n"
            "    stack = []\n    for c in s:\n        if c in opens:\n"
            "            stack.append(c)\n        elif c in pairs:\n"
            "            if not stack or stack.pop() != pairs[c]:\n"
            "                return False\n    return not stack\n" % closing,
            cases)


@_code
def _spiral(rng):
    cases = [[[]]]
    for _ in range(7):
        r, c = rng.randint(1, 5), rng.randint(1, 5)
        cases.append([[[rng.randint(0, 9) for _ in range(c)]
                       for _ in range(r)]])
    return ("spiral",
            "Write a Python function `spiral(m)` that returns the elements of "
            "the matrix `m` (a list of equal-length rows) in clockwise "
            "spiral order, starting at the top-left corner. An empty matrix "
            "gives [].",
            "def spiral(m):\n    out = []\n    m = [list(r) for r in m]\n"
            "    while m and m[0]:\n        out += m.pop(0)\n"
            "        m = [list(r) for r in zip(*m)][::-1]\n    return out\n",
            cases)


@_code
def _run(rng):
    return ("longest_run",
            "Write a Python function `longest_run(xs)` that returns the "
            "length of the longest run of equal adjacent elements in the "
            "list `xs` (0 for an empty list).",
            "def longest_run(xs):\n    best = cur = 0\n    prev = object()\n"
            "    for x in xs:\n        cur = cur + 1 if x == prev else 1\n"
            "        prev = x\n        best = max(best, cur)\n"
            "    return best\n",
            [[_ints(rng, 0, 2, 0, 15)] for _ in range(8)])


@_code
def _root(rng):
    b = rng.choice([3, 5, 7, 8, 12, 16])
    return ("root",
            "Write a Python function `root(n)` for a non-negative integer n: "
            "write n in base %d, add up its digits, and repeat on the sum "
            "until a single base-%d digit is left; return that value as an "
            "int. root(0) is 0." % (b, b),
            "def root(n):\n    while n >= %d:\n        s = 0\n"
            "        while n:\n            s += n %% %d\n"
            "            n //= %d\n        n = s\n    return n\n" % (b, b, b),
            [[rng.randint(0, 10 ** 6)] for _ in range(8)])


@_code
def _top(rng):
    k = rng.randint(2, 4)
    pool = rng.sample(WORDS, 6)
    return ("top_words",
            "Write a Python function `top_words(text)` that returns the %d "
            "most frequent words in `text` as a list, most frequent first. "
            "Words are runs of the letters a-z, compared in lower case. "
            "Break ties alphabetically. With fewer than %d distinct words, "
            "return them all." % (k, k),
            "import re\nfrom collections import Counter\n\n\n"
            "def top_words(text):\n"
            "    c = Counter(re.findall('[a-z]+', text.lower()))\n"
            "    ranked = sorted(c.items(), key=lambda x: (-x[1], x[0]))\n"
            "    return [w for w, _ in ranked[:%d]]\n" % k,
            [[" ".join(rng.choice(pool[:rng.randint(1, 6)]).capitalize()
                       if rng.random() < 0.2 else rng.choice(pool)
                       for _ in range(rng.randint(0, 14)))]
             for _ in range(8)])


@_code
def _second(rng):
    return ("second_largest",
            "Write a Python function `second_largest(xs)` that returns the "
            "second largest distinct value in the list `xs`, or None when "
            "there are fewer than two distinct values.",
            "def second_largest(xs):\n    v = sorted(set(xs))\n"
            "    return v[-2] if len(v) >= 2 else None\n",
            [[_ints(rng, -5, 5, 0, 8)] for _ in range(8)])


@_code
def _base(rng):
    b = rng.choice([2, 3, 5, 7, 9, 11, 13, 16])
    return ("to_base",
            "Write a Python function `to_base(n)` that returns the integer n "
            "written in base %d as a string, using the digits 0-9 and then "
            "lowercase letters, with a leading '-' for negative numbers. "
            "to_base(0) is '0'." % b,
            "def to_base(n):\n    if n == 0:\n        return '0'\n"
            "    d = '0123456789abcdefghijklmnopqrstuvwxyz'\n"
            "    s, neg, n = '', n < 0, abs(n)\n    while n:\n"
            "        s = d[n %% %d] + s\n        n //= %d\n"
            "    return '-' + s if neg else s\n" % (b, b),
            [[rng.randint(-5000, 5000)] for _ in range(7)] + [[0]])


@_code
def _pascal(rng):
    m = rng.choice([2, 3, 5, 7, 10])
    return ("pascal_row",
            "Write a Python function `pascal_row(n)` that returns row n of "
            "Pascal's triangle (row 0 is [1]) with every entry taken modulo "
            "%d." % m,
            "def pascal_row(n):\n    row = [1]\n    for _ in range(n):\n"
            "        row = [1] + [row[i] + row[i + 1] "
            "for i in range(len(row) - 1)] + [1]\n"
            "    return [x %% %d for x in row]\n" % m,
            [[rng.randint(0, 25)] for _ in range(8)])


@_code
def _chunks(rng):
    k = rng.randint(2, 5)
    return ("chunk_sums",
            "Write a Python function `chunk_sums(xs)` that splits the list "
            "`xs` into consecutive chunks of %d (the last one may be "
            "shorter) and returns the list of their sums." % k,
            "def chunk_sums(xs):\n"
            "    return [sum(xs[i:i + %d]) for i in range(0, len(xs), %d)]\n"
            % (k, k),
            [[_ints(rng, -20, 20, 0, 13)] for _ in range(8)])


HARNESS = """import json
import solution

cases = json.loads(@CASES@)
passed = 0
for args, want in cases:
    try:
        got = json.loads(json.dumps(getattr(solution, @NAME@)(*args)))
    except Exception:
        continue
    passed += got == want
print("passed %d of %d" % (passed, len(cases)))
raise SystemExit(0 if passed == len(cases) else 1)
"""


def extract_code(text, name):
    """The block in a reply that defines `name`, else the last block,
    else the reply itself if it defines it."""
    blocks = re.findall(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)```",
                        text or "", re.S | re.I)
    for b in reversed(blocks):
        if re.search(r"def\s+%s\s*\(" % re.escape(name), b):
            return b
    if re.search(r"def\s+%s\s*\(" % re.escape(name), text or ""):
        return text
    return None


def expected(src, name, cases):
    ns = {}
    exec(compile(src, "<reference>", "exec"), ns)       # noqa: S102 - ours
    return [json.loads(json.dumps(ns[name](*copy.deepcopy(a))))
            for a in cases]


def _grade_code(name, cases, want):
    payload = json.dumps([[a, e] for a, e in zip(cases, want, strict=True)])
    main = HARNESS.replace("@CASES@", repr(payload)).replace("@NAME@",
                                                             repr(name))

    def grade(reply, env, world=None):
        src = extract_code(reply.content, name)
        if src is None:
            return 0.0, "no function %s in the reply" % name
        ok, out = env.run_python({"solution.py": src, "main.py": main})
        last = out.strip().splitlines()[-1] if out.strip() else "no output"
        return (1.0 if ok else 0.0), last[:160]
    return grade


def gen_code(rng):
    items, order = [], []
    for i in range(SIZE["code"]):
        if not order:
            order = rng.sample(CODE, len(CODE))
        name, prompt, src, cases = order.pop()(rng)
        want = expected(src, name, cases)
        item = Item("code", "code-%02d-%s" % (i, name),
                    prompt + " Reply with the function in one ```python "
                    "code block; do not include tests or example usage.",
                    _grade_code(name, cases, want))
        item.meta["reference"] = src
        items.append(item)
    return items


# --------------------------------------------------------------- tools --

def _fn(name, desc, props, required=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props,
                       "required": list(required or props)}}}


def _s(d):
    return {"type": "string", "description": d}


def _n(d):
    return {"type": "number", "description": d}


def _i(d):
    return {"type": "integer", "description": d}


TOOLS = {
    "get_weather": _fn("get_weather", "Current weather for a city.", {
        "city": _s("City name"),
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                 "description": "Temperature unit"}}),
    "convert_currency": _fn("convert_currency", "Convert money between "
                            "currencies at today's rate.", {
        "amount": _n("Amount of money"),
        "from_currency": _s("ISO 4217 code, e.g. USD"),
        "to_currency": _s("ISO 4217 code, e.g. EUR")}),
    "search_flights": _fn("search_flights", "Find flights on a date.", {
        "origin": _s("IATA airport code of the departure airport"),
        "destination": _s("IATA airport code of the arrival airport"),
        "date": _s("Departure date, YYYY-MM-DD"),
        "passengers": _i("Number of passengers")}),
    "create_event": _fn("create_event", "Add an event to the calendar.", {
        "title": _s("Event title"), "date": _s("YYYY-MM-DD"),
        "start": _s("Start time, 24-hour HH:MM"),
        "minutes": _i("Duration in minutes")}),
    "send_email": _fn("send_email", "Send an email.", {
        "to": _s("Recipient address"), "subject": _s("Subject line"),
        "body": _s("Message text")}),
    "get_stock_price": _fn("get_stock_price", "Latest price of a stock.", {
        "ticker": _s("Ticker symbol, e.g. AAPL")}),
    "set_timer": _fn("set_timer", "Start a countdown timer.", {
        "minutes": _i("Length in minutes"), "label": _s("What it is for")}),
}

CURRENCIES = [("euros", "EUR"), ("Japanese yen", "JPY"),
              ("US dollars", "USD"), ("British pounds", "GBP"),
              ("Australian dollars", "AUD"), ("Swiss francs", "CHF"),
              ("Canadian dollars", "CAD")]
AIRPORTS = [("Perth", "PER"), ("Singapore", "SIN"), ("Auckland", "AKL"),
            ("Dublin", "DUB"), ("Lisbon", "LIS"), ("Denver", "DEN"),
            ("Helsinki", "HEL"), ("Vienna", "VIE"), ("Brisbane", "BNE")]
COMPANIES = [("Apple", "AAPL"), ("Microsoft", "MSFT"), ("NVIDIA", "NVDA"),
             ("Tesla", "TSLA"), ("Amazon", "AMZN"), ("Netflix", "NFLX"),
             ("Intel", "INTC")]
CITIES = ["Oslo", "Nairobi", "Lima", "Montreal", "Kyoto", "Adelaide",
          "Porto", "Tallinn"]


def _ci(v):
    return lambda x: isinstance(x, str) and x.strip().casefold() == v.casefold()


def _has(v):
    return lambda x: isinstance(x, str) and v.casefold() in x.casefold()


def _num(v):
    def m(x):
        try:
            return abs(float(x) - v) < 1e-6
        except (TypeError, ValueError):
            return False
    return m


def _hhmm(v):
    def m(x):
        got = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(x or ""))
        return bool(got) and "%02d:%s" % (int(got.group(1)),
                                          got.group(2)) == v
    return m


def _date(d):
    return d.isoformat()


def _random_date(rng, lo=2026, hi=2027):
    start = datetime.date(lo, 1, 1)
    return start + datetime.timedelta(days=rng.randint(
        0, (datetime.date(hi, 12, 31) - start).days))


def _single(rng):
    """(prompt, tool, {arg: matcher}) for one-call items."""
    kind = rng.choice(["weather", "currency", "flights", "event", "stock",
                       "timer", "email"])
    if kind == "weather":
        city, unit = rng.choice(CITIES), rng.choice(["celsius", "fahrenheit"])
        return ("How warm is it in %s right now? Give it to me in %s." % (
            city, unit.capitalize()), "get_weather",
            {"city": _has(city), "unit": _ci(unit)})
    if kind == "currency":
        (fa, fc), (ta, tc) = rng.sample(CURRENCIES, 2)
        amount = rng.choice([12, 75, 140, 999, 2500]) + rng.choice([0, 0.5])
        return ("How much is %g %s in %s?" % (amount, fa, ta),
                "convert_currency", {"amount": _num(amount),
                                     "from_currency": _ci(fc),
                                     "to_currency": _ci(tc)})
    if kind == "flights":
        (oa, oc), (da, dc) = rng.sample(AIRPORTS, 2)
        d, n = _random_date(rng), rng.randint(1, 4)
        return ("Find me flights from %s to %s on %s for %d %s." % (
            oa, da, day_words(d), n, "person" if n == 1 else "people"),
            "search_flights", {"origin": _ci(oc), "destination": _ci(dc),
                               "date": _ci(_date(d)), "passengers": _num(n)})
    if kind == "event":
        d = _random_date(rng)
        h, mins = rng.randint(1, 11), rng.choice([0, 15, 30, 45])
        pm = rng.choice([True, False])
        dur = rng.choice([30, 45, 60, 90, 120])
        title = rng.choice(["Dentist", "Team retro", "Piano lesson",
                            "Budget review"])
        long = ("%d hour%s" % (dur // 60, "s" if dur >= 120 else "")
                if dur % 60 == 0 else "%d minutes" % dur)
        return ("Put '%s' on my calendar for %s at %d:%02d %s. It runs %s."
                % (title, day_words(d), h, mins, "pm" if pm else "am", long),
                "create_event", {"title": _has(title), "date": _ci(_date(d)),
                                 "start": _hhmm("%02d:%02d" % (
                                     h + 12 if pm else h, mins)),
                                 "minutes": _num(dur)})
    if kind == "stock":
        name, tick = rng.choice(COMPANIES)
        return ("What is %s's stock trading at?" % name, "get_stock_price",
                {"ticker": _ci(tick)})
    if kind == "timer":
        h, m = rng.randint(0, 2), rng.choice([5, 10, 20, 25, 40])
        label = rng.choice(["pasta", "laundry", "tea", "bread"])
        when = ("%d hour%s and %d minutes" % (h, "" if h == 1 else "s", m)
                if h else "%d minutes" % m)
        return ("Set a timer for %s, and call it %s." % (when, label),
                "set_timer", {"minutes": _num(h * 60 + m),
                              "label": _has(label)})
    who = rng.choice(NAMES).lower()
    subject = rng.choice(["Invoice 4471", "Friday plans", "Server outage"])
    return ("Email %s@example.com with the subject '%s' and tell them I "
            "will call tomorrow." % (who, subject), "send_email",
            {"to": _ci("%s@example.com" % who), "subject": _ci(subject),
             "body": lambda x: isinstance(x, str) and bool(x.strip())})


def _grade_call(name, want):
    def grade(reply, env, world=None):
        if not reply.calls:
            return 0.0, "no tool call"
        c = reply.calls[0]
        if c.name != name:
            return 0.0, "called %s, not %s" % (c.name, name)
        if not isinstance(c.args, dict):
            return 0.0, "arguments are not a JSON object"
        bad = [k for k, m in want.items() if not m(c.args.get(k))]
        if bad:
            return 0.0, "wrong %s: %s" % (", ".join(bad), json.dumps(
                {k: c.args.get(k) for k in bad})[:120])
        return 1.0, "ok"
    return grade


def _grade_no_call(reply, env, world=None):
    if reply.calls:
        return 0.0, "called %s when no tool fits" % reply.calls[0].name
    return (1.0, "ok") if reply.content.strip() else (0.0, "empty reply")


AGENT = ("You are an assistant that acts through the tools you are given. "
         "Look things up with them instead of asking the user.")


class World:
    """A mock the model acts on. call() never raises: bad calls get an
    error back, the way a real API answers."""

    tools = ()

    def __init__(self):
        self.log = []

    def call(self, name, args):
        self.log.append((name, args))
        fn = getattr(self, "t_" + name, None)
        if fn is None or name not in self.tools:
            return {"error": "no such tool: %s" % name}
        if not isinstance(args, dict):
            return {"error": "arguments must be a JSON object"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": "bad arguments: %s" % e}


class Orders(World):
    tools = ("find_customer", "list_orders", "cancel_order")
    schemas = [
        _fn("find_customer", "Look a customer up by email.",
            {"email": _s("Customer email")}),
        _fn("list_orders", "A customer's orders.",
            {"customer_id": _s("Customer id from find_customer")}),
        _fn("cancel_order", "Cancel one order.",
            {"order_id": _s("Order id")})]

    def __init__(self, customers):
        super().__init__()
        self.customers = customers          # email -> (id, [(order, item)])
        self.cancelled = []

    def t_find_customer(self, email):
        c = self.customers.get(str(email).strip().lower())
        return {"customer_id": c[0]} if c else {"error": "no such customer"}

    def t_list_orders(self, customer_id):
        for cid, orders in self.customers.values():
            if cid == customer_id:
                return {"orders": [{"order_id": o, "item": i,
                                    "status": "processing"}
                                   for o, i in orders]}
        return {"error": "no such customer id"}

    def t_cancel_order(self, order_id):
        ids = {o for _, orders in self.customers.values() for o, _ in orders}
        if order_id not in ids:
            return {"error": "no such order"}
        self.cancelled.append(order_id)
        return {"ok": True, "order_id": order_id, "status": "cancelled"}


class Prices(World):
    tools = ("get_price",)
    schemas = [_fn("get_price", "Unit price of an item, in dollars.",
                   {"item": _s("Item name")})]

    def __init__(self, prices):
        super().__init__()
        self.prices = prices

    def t_get_price(self, item):
        p = self.prices.get(str(item).strip().lower().rstrip("s"))
        return {"item": item, "unit_price": p} if p is not None \
            else {"error": "unknown item"}


class Files(World):
    tools = ("list_dir", "read_file")
    schemas = [_fn("list_dir", "Names in a directory.",
                   {"path": _s("Absolute directory path")}),
               _fn("read_file", "Contents of a text file.",
                   {"path": _s("Absolute file path")})]

    def __init__(self, files):
        super().__init__()
        self.files = files                    # path -> text

    def t_list_dir(self, path):
        path = str(path).rstrip("/") + "/"
        names = sorted({p[len(path):].split("/")[0] for p in self.files
                        if p.startswith(path)})
        return {"entries": names} if names else {"error": "no such directory"}

    def t_read_file(self, path):
        text = self.files.get(str(path))
        return {"content": text} if text is not None \
            else {"error": "no such file"}


class Calendar(World):
    tools = ("get_busy", "book")
    schemas = [_fn("get_busy", "Busy blocks on a day, as [start, end] "
                   "24-hour times.", {"date": _s("YYYY-MM-DD")}),
               _fn("book", "Book a meeting.", {
                   "date": _s("YYYY-MM-DD"), "start": _s("24-hour HH:MM"),
                   "minutes": _i("Length in minutes"),
                   "title": _s("Meeting title")})]

    def __init__(self, day, busy):
        super().__init__()
        self.day, self.busy, self.booked = day, busy, []

    def t_get_busy(self, date):
        return {"date": date, "busy": self.busy if date == self.day else []}

    def t_book(self, date, start, minutes, title=""):
        self.booked.append((str(date), str(start), minutes))
        return {"ok": True}


def _mins(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _hm(n):
    return "%02d:%02d" % (n // 60, n % 60)


def _world_orders(rng):
    items = ["standing desk", "coffee grinder", "rain jacket", "desk lamp",
             "backpack", "headphones", "kettle", "yoga mat"]
    customers, used = {}, set()
    for name in rng.sample(NAMES, 3):
        cid = "C-%04d" % rng.randint(1000, 9999)
        orders = []
        for item in rng.sample(items, rng.randint(2, 3)):
            oid = "ORD-%04d" % rng.randint(1000, 9999)
            while oid in used:
                oid = "ORD-%04d" % rng.randint(1000, 9999)
            used.add(oid)
            orders.append((oid, item))
        customers["%s@example.com" % name.lower()] = (cid, orders)
    email = rng.choice(sorted(customers))
    oid, item = rng.choice(customers[email][1])

    def grade(reply, env, world):
        if world.cancelled == [oid]:
            return 1.0, "ok"
        if not world.cancelled:
            return 0.0, "nothing cancelled"
        return 0.0, "cancelled %s, wanted %s" % (world.cancelled, oid)
    return ("Customer %s asked us to cancel their %s order. Please do it."
            % (email, item), lambda: Orders(copy.deepcopy(customers)),
            Orders.schemas, grade)


def _world_prices(rng):
    names = ["pencil", "eraser", "notebook", "stapler", "marker", "folder",
             "ruler"]
    prices = {n: round(rng.randint(40, 900) / 100, 2) for n in names}
    picks = rng.sample(names, 3)
    qty = [rng.randint(2, 12) for _ in picks]
    total = round(sum(q * prices[n] for q, n in zip(qty, picks, strict=True)),
                  2)

    def grade(reply, env, world):
        got = _number(final_answer(reply.content))
        if got is not None and abs(got - total) < 0.005:
            return 1.0, "ok"
        return 0.0, "answered %s, total is %.2f" % (got, total)
    ask = ", ".join("%d %ss" % (q, n) for q, n in zip(qty, picks, strict=True))
    return ("I want %s. What does that come to in total? Check the prices, "
            "then end with a line 'Answer: <total in dollars>'." % ask,
            lambda: Prices(dict(prices)), Prices.schemas, grade)


def _world_files(rng):
    services = rng.sample(["billing", "search", "mailer", "auth", "reports",
                           "uploads"], 3)
    files, ports = {}, {}
    for s in services:
        port = rng.randint(2000, 9800)
        ports[s] = port
        files["/srv/%s/config.ini" % s] = (
            "[server]\nhost = 0.0.0.0\nport = %d\nworkers = %d\n" % (
                port, rng.randint(2, 16)))
        files["/srv/%s/README.md" % s] = "# %s\nSee config.ini.\n" % s
    target = rng.choice(services)

    def grade(reply, env, world):
        got = _number(final_answer(reply.content))
        if got == ports[target]:
            return 1.0, "ok"
        return 0.0, "answered %s, port is %d" % (got, ports[target])
    return ("What port is the %s service configured to listen on? Its files "
            "are somewhere under /srv. End with a line 'Answer: <port>'."
            % target, lambda: Files(dict(files)), Files.schemas, grade)


def _world_calendar(rng):
    while True:
        day = _random_date(rng).isoformat()
        busy, t = [], 9 * 60 + rng.choice([0, 30])
        while t < 16 * 60 and len(busy) < 4:
            length = rng.choice([30, 60, 90])
            busy.append([_hm(t), _hm(min(t + length, 17 * 60))])
            t += length + rng.choice([0, 30, 30, 60, 90])
        need = rng.choice([30, 45, 60, 90])
        want = None
        for start in [9 * 60] + [_mins(e) for _, e in busy]:
            end = start + need
            if end <= 17 * 60 and all(end <= _mins(s) or start >= _mins(e)
                                      for s, e in busy):
                want = start
                break
        if want is not None and busy:
            break

    def grade(reply, env, world):
        got = [(d, s, m) for d, s, m in world.booked]
        ok = (len(got) == 1 and got[0][0] == day and _hhmm(_hm(want))(got[0][1])
              and _num(need)(got[0][2]))
        return (1.0, "ok") if ok else (0.0, "booked %s, wanted %s at %s" % (
            got, day, _hm(want)))
    return ("Book a %d-minute meeting called 'sync' on %s, at the earliest "
            "time I am free between 09:00 and 17:00." % (need, day),
            lambda: Calendar(day, copy.deepcopy(busy)), Calendar.schemas,
            grade)


WORLDS = [_world_orders, _world_prices, _world_files, _world_calendar]


def gen_tools(rng):
    items = []
    n_multi = 12
    n_single = SIZE["tools"] - n_multi
    for i in range(n_single):
        if i % 6 == 5:                          # a request no tool fits
            offered = rng.sample(["get_weather", "get_stock_price",
                                  "set_timer", "search_flights"], 2)
            q = rng.choice([
                "What is %d times %d?" % (rng.randint(12, 99),
                                          rng.randint(12, 99)),
                "Give me a synonym for '%s'." % rng.choice(
                    ["quick", "bright", "calm", "large"]),
                "Translate 'good morning' into %s." % rng.choice(
                    ["French", "Spanish", "German"])])
            items.append(Item("tools", "tools-%02d-none" % i, q,
                              _grade_no_call, AGENT,
                              [TOOLS[t] for t in offered]))
            continue
        prompt, tool, want = _single(rng)
        others = rng.sample([t for t in TOOLS if t != tool], 2)
        offered = [tool] + others
        rng.shuffle(offered)
        items.append(Item("tools", "tools-%02d-%s" % (i, tool), prompt,
                          _grade_call(tool, want), AGENT,
                          [TOOLS[t] for t in offered]))
    for j in range(n_multi):
        maker = WORLDS[j % len(WORLDS)]
        prompt, world, schemas, grade = maker(rng)
        items.append(Item("tools", "tools-%02d-%s" % (
            n_single + j, maker.__name__[7:]), prompt, grade, AGENT,
            schemas, world=world))
    return items


# ------------------------------------------------------------- longctx --

PROJECTS = ("Andromeda Basalt Cygnus Dorado Ember Fornax Gemini Halcyon "
            "Indigo Juniper Kestrel Lyra Meridian Nimbus Obsidian Perseus "
            "Quasar Rigel").split()
STAFF = ("Abernathy Beaumont Castellanos Delacroix Esposito Fairbanks "
         "Galloway Hargreaves Iwasaki Jablonski").split()
FILLER = [
    "The {adj} {noun} near {place} was inspected on {day} and found {state}.",
    "{who} noted that spending on the {noun} rose by {n} percent this quarter.",
    "Minutes from the {day} meeting mention the {adj} {noun} twice.",
    "A shipment of {n} {noun} crates reached {place} ahead of schedule.",
    "Nobody at {place} could say why the {noun} had been painted {color}.",
    "{who} asked for the {adj} {noun} report to be moved to {day}.",
    "The {noun} at {place} has been {state} since the {n}th of {month}.",
    "According to {who}, the {color} {noun} will be replaced in {month}.",
]
FILL = {
    "adj": "north east quiet older newer small large spare".split(),
    "noun": "boiler archive gate ledger printer pump vehicle shelf".split(),
    "place": "the depot, the east annex, the harbour office, the old mill, "
             "the lab, the warehouse".split(", "),
    "day": WEEKDAYS, "month": MONTHS,
    "state": "in order, slightly worn, due for service, missing a panel, "
             "fine".split(", "),
    "color": "green grey orange blue white".split(),
    "who": "Marguerite Tomasz Ingrid Rafael Yusuf Beatrix Leopold".split(),
}


def filler(rng, chars):
    out, n = [], 0
    while n < chars:
        s = rng.choice(FILLER)
        s = re.sub(r"\{(\w+)\}", lambda m: str(rng.randint(2, 97))
                   if m.group(1) == "n" else rng.choice(FILL[m.group(1)]), s)
        out.append(s)
        n += len(s) + 1
    return " ".join(out)


def document(seed, chars, facts):
    """Filler of about `chars` characters with each (depth, sentence)
    fact placed at its depth."""
    rng = random.Random(seed)
    text = filler(rng, chars)
    for depth, sentence in sorted(facts, reverse=True):
        at = text.find(". ", int(len(text) * depth))
        at = len(text) if at < 0 else at + 2
        text = text[:at] + sentence + " " + text[at:]
    return text


def _code_word(rng):
    return "%s%s-%04d" % (rng.choice("BCDFGHJKLMNPRSTVWXZ"),
                          rng.choice("BCDFGHJKLMNPRSTVWXZ"),
                          rng.randint(1000, 9999))


def _contains(want):
    def grade(reply, env, world=None):
        ok = want.lower() in (reply.content or "").lower()
        return (1.0, "ok") if ok else (0.0, "wanted %s, got %.60r" % (
            want, reply.content))
    return grade


def gen_longctx(rng):
    items = []
    for label, target in LONG:
        projects = rng.sample(PROJECTS, 10)
        staff = rng.sample(STAFF, 5)
        facts, questions = [], []
        codes = {p: _code_word(rng) for p in projects[:8]}
        for p, depth in zip(projects[:3], rng.sample([0.1, 0.5, 0.9], 3),
                            strict=True):
            facts.append((depth, "The access code for project %s is %s."
                          % (p, codes[p])))
            questions.append(("What is the access code for project %s? "
                              "Reply with just the code." % p, codes[p]))
        for p in projects[3:8]:                 # distractors
            facts.append((rng.random(), "The access code for project %s is %s."
                          % (p, codes[p])))
        rooms = {s: "room %d-%02d" % (rng.randint(1, 9), rng.randint(1, 60))
                 for s in staff}
        for p, lead in zip(projects[8:10], staff[:2], strict=True):
            a, b = rng.uniform(0.05, 0.45), rng.uniform(0.55, 0.95)
            if rng.random() < 0.5:
                a, b = b, a
            facts.append((a, "Project %s is led by %s." % (p, lead)))
            facts.append((b, "%s works from %s." % (lead, rooms[lead])))
            questions.append(("Which room does the person who leads project "
                              "%s work from? Reply with just the room." % p,
                              rooms[lead]))
        for s in staff[2:]:
            facts.append((rng.random(), "%s works from %s." % (s, rooms[s])))
        doc_seed = rng.random()
        cache = {}

        def text(cpt, _seed=doc_seed, _target=target, _facts=tuple(facts),
                 _cache=cache):
            key = round(cpt, 3)
            if key not in _cache:
                _cache.clear()
                _cache[key] = document(_seed, int(_target * cpt), _facts)
            return _cache[key]

        for k, (q, want) in enumerate(questions):
            item = Item("longctx", "longctx-%s-%d" % (label, k),
                        lambda cpt, q=q, text=text: (
                            "Read the records below, then answer the "
                            "question after them.\n\n" + text(cpt)
                            + "\n\nQuestion: " + q),
                        _contains(want))
            item.meta.update(doc=label, tokens=target, answer=want)
            items.append(item)
    return items


# ------------------------------------------------------------ instruct --

TOPICS = ["the history of lighthouses", "how bees make honey",
          "why the sky is blue", "keeping a sourdough starter alive",
          "how tides work", "planning a small vegetable garden",
          "what a compiler does", "how to tune a guitar by ear",
          "the life of a mayfly", "why cats purr"]
KEYWORDS = ["light", "water", "time", "small", "change"]
PHRASES = ["That is all I have to say.", "Thanks for reading.",
           "Any questions?"]
OPENERS = ["Honestly", "Today", "Picture", "Interestingly"]


def _structure(rng):
    kind = rng.choice(["bullets", "paragraphs", "json", "quotes", "title",
                       None])
    if kind == "bullets":
        n = rng.randint(3, 6)
        return kind, "Use exactly %d bullet points, each on its own line " \
            "starting with '- '." % n, lambda t: len(
                re.findall(r"^\s*[-*\u2022]\s+\S", t, re.M)) == n
    if kind == "paragraphs":
        n = rng.randint(2, 4)
        return kind, "Write exactly %d paragraphs, separated by a line that " \
            "contains only ***." % n, lambda t: "***" in t and len(
                [p for p in re.split(r"^\s*\*\*\*\s*$", t, flags=re.M)
                 if p.strip()]) == n
    if kind == "json":
        keys = rng.sample(["summary", "facts", "question", "rating",
                           "source"], 3)

        def check(t, keys=keys):
            try:
                d = json.loads(strip_fences(t))
            except ValueError:
                return False
            return isinstance(d, dict) and set(d) == set(keys)
        return kind, "Answer with only a JSON object that has exactly the " \
            "keys %s." % ", ".join('"%s"' % k for k in keys), check
    if kind == "quotes":
        return kind, "Wrap your entire answer in double quotation marks.", \
            lambda t: len(t.strip()) > 1 and t.strip()[0] in "\"\u201c" \
            and t.strip()[-1] in "\"\u201d"
    if kind == "title":
        return kind, "Give it a title wrapped in double angular brackets, " \
            "like <<my title>>.", lambda t: bool(re.search(r"<<[^<>\n]+>>", t))
    return None, "", lambda t: True


def _lexical(rng):
    kind = rng.choice(["lower", "nocomma", "keyword", "maxwords", "ends",
                       "starts"])
    if kind == "lower":
        return kind, "Use only lowercase letters; no capital letters at all.", \
            lambda t: t == t.lower() and bool(re.search("[a-z]", t))
    if kind == "nocomma":
        return kind, "Do not use any commas.", lambda t: "," not in t
    if kind == "keyword":
        w, n = rng.choice(KEYWORDS), rng.randint(2, 4)
        return kind, "Use the word '%s' at least %d times." % (w, n), \
            lambda t: len(re.findall(r"\b%s\b" % w, t, re.I)) >= n
    if kind == "maxwords":
        n = rng.choice([40, 60, 80])
        return kind, "Use no more than %d words." % n, \
            lambda t: 0 < len(t.split()) <= n
    if kind == "ends":
        p = rng.choice(PHRASES)
        return kind, "End your answer with the exact sentence: %s" % p, \
            lambda t: t.strip().endswith(p)
    w = rng.choice(OPENERS)
    return kind, "Start your answer with the word '%s'." % w, \
        lambda t: (re.match(r"\W*(\w+)", t) or [None, ""])[1].casefold() \
        == w.casefold()


CLASH = {("json", "ends"), ("json", "starts"), ("json", "nocomma"),
         ("quotes", "ends"), ("quotes", "starts"), ("title", "starts"),
         ("bullets", "starts"), ("json", "maxwords")}


def _grade_rules(rules):
    def grade(reply, env, world=None):
        t = reply.content or ""
        failed = [text for text, check in rules if not check(t)]
        return (0.0, "broke: " + " | ".join(failed)[:160]) if failed \
            else (1.0, "ok")
    return grade


def gen_instruct(rng):
    items = []
    for i in range(SIZE["instruct"]):
        while True:
            sk, stext, scheck = _structure(rng)
            lk, ltext, lcheck = _lexical(rng)
            if (sk, lk) not in CLASH and not (lk == "lower" and sk is None
                                              and rng.random() < 0.5):
                break
        rules = [(ltext, lcheck)] + ([(stext, scheck)] if sk else [])
        prompt = "Write a short piece about %s. %s" % (
            rng.choice(TOPICS), " ".join(r[0] for r in reversed(rules)))
        items.append(Item("instruct", "instruct-%02d-%s%s" % (
            i, sk + "-" if sk else "", lk), prompt, _grade_rules(rules)))
    return items


# -------------------------------------------------------------- reason --

def _change(rng):
    a, b = rng.randint(2, 9), rng.randint(3, 15)
    x, y = rng.randint(2, 12), rng.randint(1, 6)
    cost = a * x + b * y
    paid = (cost // 50 + 1) * 50
    return ("Pens cost $%d each and notebooks $%d each. Sam buys %d pens and "
            "%d notebooks and pays with $%d. How many dollars of change does "
            "Sam get?" % (a, b, x, y, paid), str(paid - cost))


def _powmod(rng):
    a, b, m = rng.randint(2, 30), rng.randint(20, 200), rng.randint(7, 97)
    return ("What is the remainder when %d^%d is divided by %d?" % (a, b, m),
            str(pow(a, b, m)))


def _days(rng):
    d1 = _random_date(rng, 2000, 2030)
    d2 = d1 + datetime.timedelta(days=rng.randint(30, 900))
    return ("How many days after %s is %s?" % (day_words(d1), day_words(d2)),
            str((d2 - d1).days))


def _weekday(rng):
    d = _random_date(rng, 2000, 2030)
    k = rng.randint(10, 400)
    later = d + datetime.timedelta(days=k)
    return ("%s is a %s. What day of the week is it %d days later?" % (
        day_words(d), WEEKDAYS[d.weekday()], k), WEEKDAYS[later.weekday()])


def _order(rng):
    people = rng.sample(NAMES, 5)                 # tallest first
    clues = ["%s is taller than %s." % (people[i], people[i + 1])
             for i in range(4)]
    rng.shuffle(clues)
    rank, word = rng.choice([(1, "second tallest"), (3, "second shortest"),
                             (2, "third tallest"), (4, "shortest")])
    return (" ".join(clues) + " Who is the %s?" % word, people[rank])


def _divisible(rng):
    a = rng.randint(1, 200)
    b = a + rng.randint(100, 900)
    c, d = rng.sample([3, 4, 5, 6, 7, 9, 11], 2)
    n = sum(1 for x in range(a, b + 1) if x % c == 0 or x % d == 0)
    return ("How many integers from %d to %d inclusive are divisible by %d "
            "or by %d?" % (a, b, c, d), str(n))


def _in_base(rng):
    n, b = rng.randint(100, 5000), rng.choice([2, 3, 5, 7, 8])
    digits, m = "", n
    while m:
        digits = str(m % b) + digits
        m //= b
    return ("Write %d in base %d. Answer with the digits only." % (n, b),
            digits)


def _sequence(rng):
    r, d, t = rng.choice([2, 3]), rng.randint(-5, 5), rng.randint(1, 9)
    terms = [t]
    for _ in range(5):
        terms.append(terms[-1] * r + d)
    return ("What is the next term: %s, ...?" % ", ".join(
        str(x) for x in terms[:5]), str(terms[5]))


def _choose(rng):
    n, k = rng.randint(6, 20), rng.randint(2, 5)
    return ("In how many ways can a committee of %d be chosen from %d "
            "people?" % (k, n), str(math.comb(n, k)))


def _average(rng):
    xs = [rng.randint(1, 60) for _ in range(5)]
    while sum(xs) % 5:
        xs[4] += 1
    return ("The average of five numbers is %d. Four of them are %s. What is "
            "the fifth?" % (sum(xs) // 5, ", ".join(map(str, xs[:4]))),
            str(xs[4]))


REASON = [_change, _powmod, _days, _weekday, _order, _divisible, _in_base,
          _sequence, _choose, _average]


def _grade_answer(want):
    def grade(reply, env, world=None):
        got = final_answer(reply.content)
        if same_answer(got, want):
            return 1.0, "ok"
        return 0.0, "answered %r, wanted %s" % ((got or "")[:40], want)
    return grade


def gen_reason(rng):
    items, order = [], []
    for i in range(SIZE["reason"]):
        if not order:
            order = rng.sample(REASON, len(REASON))
        maker = order.pop()
        q, want = maker(rng)
        item = Item("reason", "reason-%02d-%s" % (i, maker.__name__[1:]),
                    q + " End your reply with a line of the form "
                    "'Answer: <answer>'.", _grade_answer(want))
        item.meta["answer"] = want
        items.append(item)
    return items


GENERATORS = {"code": gen_code, "tools": gen_tools, "longctx": gen_longctx,
              "instruct": gen_instruct, "reason": gen_reason}


# -------------------------------------------------------------- custom --

def load_custom(folder):
    """Your own tasks: *.toml files of [[task]] tables in `folder`.

    [[task]]
    id = "logs"                  # optional
    prompt = "..."               # required
    system = "..."               # optional
    domain = "coding"            # coding, agentic, long-context, general,
                                 # reasoning; general by default
    max_tokens = 2048
    check = "contains"           # contains | regex | exact | python
    expect = "..."               # for contains, regex and exact
    case = false                 # contains/exact: match case too
    test = '''assert "x" in reply'''   # for python: `reply` is the answer
    """
    items = []
    for f in sorted(Path(folder).glob("*.toml")) if Path(folder).is_dir() \
            else []:
        try:
            data = tomllib.loads(f.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise ValueError("%s: %s" % (f, e)) from e
        for i, t in enumerate(data.get("task", [])):
            if "prompt" not in t:
                raise ValueError("%s: task %d has no prompt" % (f, i + 1))
            item = Item("custom", "custom-%s-%s" % (f.stem, t.get("id", i)),
                        t["prompt"], _custom_grade(t, f), t.get("system"))
            item.domain = t.get("domain", "general")
            item.max_tokens = int(t.get("max_tokens", CAP["custom"]))
            items.append(item)
    return items


def _custom_grade(t, f):
    check, want = t.get("check", "contains"), t.get("expect", "")
    case = bool(t.get("case", False))
    if check not in ("contains", "regex", "exact", "python"):
        raise ValueError("%s: unknown check %r" % (f, check))
    if check == "python" and not t.get("test"):
        raise ValueError("%s: a python check needs a test" % f)

    def fold(s):
        return s if case else s.casefold()

    def grade(reply, env, world=None):
        got = reply.content or ""
        if check == "contains":
            ok = fold(str(want)) in fold(got)
        elif check == "regex":
            ok = re.search(str(want), got, re.S) is not None
        elif check == "exact":
            ok = fold(got.strip()) == fold(str(want).strip())
        else:
            ok, out = env.run_python({"main.py": "reply = %r\n%s\n" % (
                got, t["test"])})
            return (1.0 if ok else 0.0), (out.strip().splitlines() or [""])[-1][:160]
        return (1.0, "ok") if ok else (0.0, "check %s failed" % check)
    return grade
