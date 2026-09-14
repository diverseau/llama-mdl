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

import itertools
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

SUITE_VERSION = 3
SUITES = ("code", "tools", "longctx", "instruct", "reason")
DOMAIN = {"code": "coding", "tools": "agentic", "longctx": "long-context",
          "instruct": "general", "reason": "reasoning", "custom": "general"}
SIZE = {"code": 40, "tools": 30, "longctx": 24, "instruct": 20,
        "reason": 20}
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


def fingerprint(items):
    """Names the questions themselves, not the seed they came from.

    SUITE_VERSION says a template changed only if someone remembers to
    bump it. This is computed from what was actually asked, so two runs
    that claim to be comparable can be checked rather than trusted.
    """
    h = hashlib.sha256()
    for it in items:
        h.update(it.id.encode())
        h.update(b"\0")
        # a long document is named by its size, not its megabyte of filler
        body = ("doc:%s:%s" % (it.meta.get("doc"), it.meta.get("tokens"))
                if it.meta.get("doc") else it.text(4.0))
        h.update(body.encode("utf-8", "replace"))
        h.update(b"\0")
        h.update(repr(sorted((k, v) for k, v in it.meta.items()
                             if k != "reference")).encode())
        h.update(b"\0")
        h.update(repr([t.get("function", {}).get("name")
                       for t in (it.tools or [])]).encode())
        h.update(b"\n")
    return h.hexdigest()[:12]


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


CLOCK = re.compile(r"(\d{1,2}):(\d{2})")


def same_answer(got, want):
    """Exact-answer grading: numbers by value, words by word."""
    if got is None:
        return False
    if CLOCK.fullmatch(want):        # a time is neither: "9:05" splits
        hit = CLOCK.search(got)      # into two numbers, and 9.05 is not it
        return bool(hit) and (int(hit.group(1)) % 24, int(hit.group(2))) == (
            int(want.split(":")[0]) % 24, int(want.split(":")[1]))
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


# ---- the harder tier -------------------------------------------------
#
# These are not list comprehensions with a twist. Each one needs an idea
# (a monotonic deque, a binary search on the answer, a parser), and the
# hidden tests carry the cases a first draft gets wrong: empty input, one
# element, ties, duplicates, negatives, and one input big enough that a
# quadratic answer runs out of time instead of finishing.

HARD_CODE = []


def _hard_code(fn):
    HARD_CODE.append(fn)
    return fn


CALC_SRC = '''import re


def calc(s):
    toks = re.findall(r"\\d+|[-+*/%^()]", s)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def atom():
        t = take()
        if t == "(":
            v = expr()
            take()
            return v
        if t == "-":
            return -atom()
        return int(t)

    def power():
        b = atom()
        if peek() == "^":
            take()
            return b ** power()
        return b

    def term():
        v = power()
        while peek() in ("*", "/", "%"):
            op = take()
            r = power()
            if op == "*":
                v = v * r
            elif op == "/":
                q = abs(v) // abs(r)
                v = q if (v < 0) == (r < 0) else -q
            else:
                m = abs(v) % abs(r)
                v = m if v >= 0 else -m
        return v

    def expr():
        v = term()
        while peek() in ("+", "-"):
            op = take()
            r = term()
            v = v + r if op == "+" else v - r
        return v

    return expr()
'''


def _calc_fn():
    ns = {}
    exec(compile(CALC_SRC, "<calc>", "exec"), ns)      # noqa: S102 - ours
    return ns["calc"]


def _expr_text(rng, depth=0):
    if depth >= 3 or rng.random() < 0.35:
        t = str(rng.randint(0, 40))
        return "-" + t if rng.random() < 0.2 else t
    op = rng.choice(["+", "-", "*", "/", "%", "^"])
    if op == "^":
        return "%s^%d" % (_expr_text(rng, depth + 2), rng.randint(0, 3))
    left, right = _expr_text(rng, depth + 1), _expr_text(rng, depth + 1)
    out = "%s %s %s" % (left, op, right)
    return "(%s)" % out if rng.random() < 0.5 else out


@_hard_code
def _expr(rng):
    calc = _calc_fn()
    cases = [["2^3^2"], ["-2^2"], ["7/-2"], ["-7%3"], ["0-(3*(4+5))"],
             ["((8))"]]
    while len(cases) < 14:
        text = _expr_text(rng)
        try:
            v = calc(text)
        except (ZeroDivisionError, IndexError, ValueError):
            continue
        if abs(v) < 10 ** 12:
            cases.append([text])
    return ("calc",
            "Write a Python function `calc(s)` that evaluates the arithmetic "
            "expression in the string `s` and returns an int. The operators "
            "are + - * / % ^ and round brackets, on integers. `^` is power "
            "and is right-associative (2^3^2 is 512). Unary minus binds "
            "tighter than `^`, so -2^2 is 4. `/` is division truncated "
            "towards zero (7/-2 is -3), and `%` takes the sign of the "
            "left operand (-7%3 is -1). * / % bind tighter than + -. "
            "There may be spaces anywhere. No division by zero is tested.",
            CALC_SRC, cases)


@_hard_code
def _ship(rng):
    cases = [[[1], 1], [[5, 5, 5], 3], [[3, 2, 2, 4, 1, 4], 3],
             [[1, 2, 3, 1, 1], 4]]
    for _ in range(6):
        n = rng.randint(2, 60)
        cases.append([[rng.randint(1, 500) for _ in range(n)],
                      rng.randint(1, n)])
    big = [rng.randint(1, 10 ** 6) for _ in range(4000)]
    cases.append([big, 900])                   # a linear scan of capacities
    return ("least_capacity",                  # will not finish in time
            "Write a Python function `least_capacity(weights, days)` that "
            "returns the smallest capacity a ship needs to carry every "
            "package within `days` days. Packages are loaded in the order "
            "given, a day's load may not exceed the capacity, and every "
            "package must be shipped. `days` is at least 1 and at most the "
            "number of packages.",
            "def least_capacity(weights, days):\n"
            "    lo, hi = max(weights), sum(weights)\n"
            "    while lo < hi:\n        mid = (lo + hi) // 2\n"
            "        need, load = 1, 0\n        for w in weights:\n"
            "            if load + w > mid:\n                need += 1\n"
            "                load = 0\n            load += w\n"
            "        if need <= days:\n            hi = mid\n"
            "        else:\n            lo = mid + 1\n    return lo\n",
            cases)


@_hard_code
def _lis(rng):
    cases = [[[]], [[7]], [[5, 4, 3, 2]], [[2, 2, 2, 2]],
             [[1, 3, 2, 3, 4, 1, 5]]]
    for _ in range(5):
        cases.append([[rng.randint(-30, 30)
                       for _ in range(rng.randint(0, 40))]])
    cases.append([[rng.randint(-10 ** 6, 10 ** 6) for _ in range(20000)]])
    return ("lis",
            "Write a Python function `lis(xs)` that returns the length of "
            "the longest strictly increasing subsequence of the list `xs` "
            "(the elements need not be adjacent). An empty list gives 0. "
            "One of the hidden tests has twenty thousand elements, so an "
            "O(n^2) answer will not finish.",
            "import bisect\n\n\n"
            "def lis(xs):\n    tails = []\n    for x in xs:\n"
            "        i = bisect.bisect_left(tails, x)\n"
            "        if i == len(tails):\n            tails.append(x)\n"
            "        else:\n            tails[i] = x\n    return len(tails)\n",
            cases)


@_hard_code
def _window(rng):
    cases = [[[], 1], [[4], 1], [[1, 2, 3], 3], [[5, 5, 5, 5], 2],
             [[-1, -3, -5, -2], 2]]
    for _ in range(4):
        n = rng.randint(1, 30)
        cases.append([[rng.randint(-40, 40) for _ in range(n)],
                      rng.randint(1, n)])
    cases.append([[rng.randint(-10 ** 5, 10 ** 5) for _ in range(20000)],
                  300])
    return ("window_max",
            "Write a Python function `window_max(xs, k)` that returns a list "
            "of the largest value in every window of `k` consecutive "
            "elements of `xs`, left to right. When `xs` is empty or shorter "
            "than `k`, return []. One hidden test has twenty thousand "
            "elements and k = 300, so scanning each window will not finish.",
            "from collections import deque\n\n\n"
            "def window_max(xs, k):\n    if not xs or k > len(xs):\n"
            "        return []\n    out, dq = [], deque()\n"
            "    for i, x in enumerate(xs):\n"
            "        while dq and xs[dq[-1]] <= x:\n            dq.pop()\n"
            "        dq.append(i)\n        if dq[0] <= i - k:\n"
            "            dq.popleft()\n        if i >= k - 1:\n"
            "            out.append(xs[dq[0]])\n    return out\n",
            cases)


@_hard_code
def _topo(rng):
    cases = [[3, []], [2, [[0, 1], [1, 0]]], [1, []],
             [4, [[1, 0], [2, 0], [3, 1], [3, 2]]]]
    for _ in range(6):
        n = rng.randint(2, 9)
        edges = []
        order = list(range(n))
        rng.shuffle(order)
        for _ in range(rng.randint(0, n * 2)):
            a, b = sorted(rng.sample(range(n), 2))
            edges.append([order[a], order[b]])
        if rng.random() < 0.25 and n >= 2:      # make a cycle
            edges.append([order[-1], order[0]])
        cases.append([n, edges])
    return ("topo",
            "Write a Python function `topo(n, edges)` for tasks numbered 0 "
            "to n-1. Each edge [a, b] means a must come before b. Return "
            "the order that runs them all, and when several orders are "
            "possible return the one that is smallest if you read it as a "
            "list of numbers (so pick the lowest-numbered runnable task "
            "each time). Return None if no order exists. Duplicate edges "
            "may appear.",
            "import heapq\n\n\n"
            "def topo(n, edges):\n"
            "    adj = {i: set() for i in range(n)}\n"
            "    deg = [0] * n\n    for a, b in edges:\n"
            "        if b not in adj[a]:\n            adj[a].add(b)\n"
            "            deg[b] += 1\n"
            "    heap = [i for i in range(n) if not deg[i]]\n"
            "    heapq.heapify(heap)\n    out = []\n    while heap:\n"
            "        i = heapq.heappop(heap)\n        out.append(i)\n"
            "        for j in sorted(adj[i]):\n            deg[j] -= 1\n"
            "            if not deg[j]:\n                heapq.heappush(heap, j)\n"
            "    return out if len(out) == n else None\n",
            cases)


@_hard_code
def _lru(rng):
    cases = [[1, [["put", 1, 1], ["put", 2, 2], ["get", 1]]],
             [2, [["get", 9]]],
             [2, [["put", 1, 1], ["put", 2, 2], ["get", 1], ["put", 3, 3],
                  ["get", 2], ["get", 3], ["get", 1]]],
             [2, [["put", 1, 1], ["put", 1, 5], ["get", 1]]]]
    for _ in range(6):
        cap = rng.randint(1, 4)
        ops = []
        for _ in range(rng.randint(1, 18)):
            k = rng.randint(1, 6)
            if rng.random() < 0.5:
                ops.append(["put", k, rng.randint(0, 99)])
            else:
                ops.append(["get", k])
        cases.append([cap, ops])
    return ("lru",
            "Write a Python function `lru(capacity, ops)` that runs a "
            "least-recently-used cache and returns the list of results of "
            "the get operations, in order. Each op is either "
            '["put", key, value] or ["get", key]. A get returns the value '
            "or -1 when the key is not there. Both a get that finds the key "
            "and a put count as using that key. When a put would exceed the "
            "capacity, the least recently used key is dropped first; "
            "overwriting an existing key never drops anything.",
            "from collections import OrderedDict\n\n\n"
            "def lru(capacity, ops):\n    cache = OrderedDict()\n"
            "    out = []\n    for op in ops:\n"
            "        if op[0] == 'get':\n            k = op[1]\n"
            "            if k in cache:\n"
            "                cache.move_to_end(k)\n"
            "                out.append(cache[k])\n"
            "            else:\n                out.append(-1)\n"
            "        else:\n            _, k, v = op\n"
            "            if k in cache:\n                cache.move_to_end(k)\n"
            "            cache[k] = v\n"
            "            if len(cache) > capacity:\n"
            "                cache.popitem(last=False)\n    return out\n",
            cases)


@_hard_code
def _roman(rng):
    cases = [[1], [4], [9], [14], [40], [90], [400], [3999], [2024], [3888]]
    for _ in range(4):
        cases.append([rng.randint(1, 3999)])
    return ("roman",
            "Write a Python function `roman(n)` that returns the integer n "
            "(1 to 3999) as a Roman numeral in upper case, using the "
            "subtractive forms IV, IX, XL, XC, CD and CM.",
            "def roman(n):\n"
            "    table = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),\n"
            "             (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),\n"
            "             (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]\n"
            "    out = []\n    for v, sym in table:\n"
            "        while n >= v:\n            out.append(sym)\n"
            "            n -= v\n    return ''.join(out)\n",
            cases)


@_hard_code
def _frac(rng):
    cases = [[[[1, 2], [1, 3]]], [[[1, 2], [-1, 2]]], [[]], [[[3, -4]]],
             [[[2, 4], [2, 4]]], [[[1, 6], [1, 6], [1, 6]]]]
    for _ in range(6):
        cases.append([[[rng.randint(-12, 12), rng.choice(
            [d for d in range(-9, 10) if d])]
            for _ in range(rng.randint(1, 5))]])
    return ("add_fractions",
            "Write a Python function `add_fractions(fracs)` that adds a list "
            "of fractions, each given as [numerator, denominator], and "
            "returns the total as [numerator, denominator] in lowest terms "
            "with a positive denominator. An empty list gives [0, 1], and "
            "zero is [0, 1]. Denominators are never zero but may be "
            "negative. Do not use the fractions module.",
            "from math import gcd\n\n\n"
            "def add_fractions(fracs):\n    n, d = 0, 1\n"
            "    for a, b in fracs:\n        n, d = n * b + a * d, d * b\n"
            "    if n == 0:\n        return [0, 1]\n"
            "    g = gcd(abs(n), abs(d))\n    n, d = n // g, d // g\n"
            "    return [-n, -d] if d < 0 else [n, d]\n",
            cases)


@_hard_code
def _islands(rng):
    cases = [[[]], [["0"]], [["1"]], [["111", "101", "111"]],
             [["1010", "0101", "1010"]]]
    for _ in range(4):
        r, c = rng.randint(1, 9), rng.randint(1, 9)
        cases.append([["".join(rng.choice("01") for _ in range(c))
                       for _ in range(r)]])
    big = ["".join(rng.choice("0011") for _ in range(220))
           for _ in range(220)]
    cases.append([big])
    return ("islands",
            "Write a Python function `islands(grid)` that counts the "
            "connected groups of '1' cells in `grid`, a list of equal-length "
            "strings of '0' and '1'. Cells connect up, down, left and right, "
            "not diagonally. An empty grid has none. One hidden test is "
            "220 by 220, so recursion that goes one cell deep per step may "
            "hit Python's limit.",
            "def islands(grid):\n    if not grid:\n        return 0\n"
            "    rows, cols = len(grid), len(grid[0])\n"
            "    seen = [[False] * cols for _ in range(rows)]\n    n = 0\n"
            "    for r in range(rows):\n        for c in range(cols):\n"
            "            if grid[r][c] != '1' or seen[r][c]:\n"
            "                continue\n            n += 1\n"
            "            stack = [(r, c)]\n            seen[r][c] = True\n"
            "            while stack:\n                y, x = stack.pop()\n"
            "                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):\n"
            "                    ny, nx = y + dy, x + dx\n"
            "                    if 0 <= ny < rows and 0 <= nx < cols and \\\n"
            "                            not seen[ny][nx] and grid[ny][nx] == '1':\n"
            "                        seen[ny][nx] = True\n"
            "                        stack.append((ny, nx))\n    return n\n",
            cases)


@_hard_code
def _csv_row(rng):
    cases = [['a,b,c'], [''], ['"a,b",c'], ['x,"he said ""hi"" to me"'],
             ['a,,b'], ['  a , b  '], ['"", "x"'], ['"a""b"']]
    for _ in range(6):
        fields = []
        for _ in range(rng.randint(1, 4)):
            w = rng.choice(WORDS)
            if rng.random() < 0.4:
                w = '"%s%s"' % (w, rng.choice([",x", '""', " y"]))
            fields.append(w)
        cases.append([",".join(fields)])
    return ("csv_row",
            "Write a Python function `csv_row(line)` that splits one CSV "
            "line into a list of fields. A double quote opens a quoted field "
            "only when it is the first character of that field; such a "
            "field keeps any commas inside it, and a doubled quote inside "
            "it means one quote character. Anywhere else a quote is an "
            "ordinary character, and unquoted fields are taken as they "
            "are, spaces included. The empty line gives ['']. Do not use "
            "the csv module.",
            "def csv_row(line):\n    out, cur, i, n = [], [], 0, len(line)\n"
            "    start = True\n    while i < n:\n        c = line[i]\n"
            "        if c == '\"' and start:\n            i += 1\n"
            "            while i < n:\n"
            "                if line[i] == '\"':\n"
            "                    if i + 1 < n and line[i + 1] == '\"':\n"
            "                        cur.append('\"')\n                        i += 2\n"
            "                        continue\n                    i += 1\n"
            "                    break\n                cur.append(line[i])\n"
            "                i += 1\n            start = False\n"
            "        elif c == ',':\n"
            "            out.append(''.join(cur))\n            cur = []\n"
            "            i += 1\n            start = True\n"
            "        else:\n            cur.append(c)\n"
            "            i += 1\n            start = False\n"
            "    out.append(''.join(cur))\n    return out\n",
            cases)


@_hard_code
def _business(rng):
    holidays = ["2026-01-01", "2026-04-03", "2026-12-25"]
    cases = [["2026-01-02", 1, holidays], ["2026-01-02", 0, holidays],
             ["2026-01-03", 1, holidays], ["2026-12-24", 2, holidays],
             ["2026-01-02", -1, holidays], ["2026-04-02", 1, holidays]]
    for _ in range(6):
        d = _random_date(rng, 2026, 2026)
        cases.append([d.isoformat(), rng.randint(-12, 12), holidays])
    return ("add_business_days",
            "Write a Python function `add_business_days(date, n, holidays)`. "
            "`date` is 'YYYY-MM-DD', `holidays` is a list of such strings, "
            "and a business day is a weekday that is not a holiday. Return "
            "the date `n` business days after `date` as 'YYYY-MM-DD', "
            "counting forwards when n is positive and backwards when it is "
            "negative. n = 0 returns the date unchanged, even if it is a "
            "weekend or a holiday.",
            "import datetime\n\n\n"
            "def add_business_days(date, n, holidays):\n"
            "    d = datetime.date.fromisoformat(date)\n"
            "    if n == 0:\n        return d.isoformat()\n"
            "    step = 1 if n > 0 else -1\n    left = abs(n)\n"
            "    hol = set(holidays)\n    while left:\n"
            "        d += datetime.timedelta(days=step)\n"
            "        if d.weekday() < 5 and d.isoformat() not in hol:\n"
            "            left -= 1\n    return d.isoformat()\n",
            cases)



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
print("@NONCE@ passed %d of %d" % (passed, len(cases)))
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
        # the count is only believed when it carries a word the
        # submission could not have known: the code under test writes to
        # the same stdout the score is read from, and "passed 9 of 9" is
        # four keystrokes to print
        nonce = secrets.token_hex(8)
        ok, out = env.run_python(
            {"solution.py": src, "main.py": main.replace("@NONCE@", nonce)})
        out = out or ""
        said = [ln for ln in out.strip().splitlines() if ln.strip()]
        last = (said[-1] if said else "no output").replace(nonce + " ", "")
        # part marks: a function that handles the ordinary cases and
        # trips on one edge is not the same answer as one that does not
        # run at all, and scoring both zero flattens the whole suite
        hit = re.search(r"%s passed (\d+) of (\d+)" % nonce, out)
        if hit:
            n, total = int(hit.group(1)), int(hit.group(2))
            return (n / total if total else 0.0), last[:160]
        return (1.0 if ok else 0.0), last[:160]
    return grade


@_hard_code
def _charge(rng):
    """A price with five clauses that interact. No name to recognise."""
    k = rng.choice([3, 4, 5, 6])            # bulk threshold
    p = rng.choice([5, 10, 15, 20])         # bulk percent
    g = rng.choice([10, 12, 15])            # gold percent
    s = rng.choice([4, 5, 8])               # silver percent
    fee = rng.choice([150, 200, 250, 300])  # weekend fee
    t = rng.choice([5000, 8000, 10000])     # fee waiver
    cap = rng.choice([1000, 1500, 2000])    # most that can come off
    r = rng.choice([5, 10])                 # rounding step
    m = rng.choice([0, 100, 200])           # floor

    src = ("def charge(items, tier, day):\n"
           "    gross = subtotal = d1 = 0\n"
           "    for unit, qty in items:\n"
           "        line = unit * qty\n"
           "        gross += line\n"
           "        if qty >= %d:\n"
           "            cut = line * %d // 100\n"
           "            d1 += cut\n"
           "            line -= cut\n"
           "        subtotal += line\n"
           "    pct = %d if tier == 'gold' else %d if tier == 'silver' "
           "else 0\n"
           "    d2 = subtotal * pct // 100\n"
           "    total = subtotal - d2\n"
           "    over = d1 + d2 - %d\n"
           "    if over > 0:\n"
           "        total += over\n"
           "    if day in ('sat', 'sun') and gross < %d:\n"
           "        total += %d\n"
           "    total = (total * 2 + %d) // (2 * %d) * %d\n"
           "    return %d if total < %d else total\n"
           % (k, p, g, s, cap, t, fee, r, r, r, m, m))

    prompt = (
        "Write a Python function `charge(items, tier, day)` that returns "
        "an integer number of cents. `items` is a list of `[unit, qty]` "
        "pairs of non-negative integers, `tier` is a string, and `day` is "
        "a three-letter lowercase day name. Apply these steps in this "
        "order.\n"
        "1. A line costs `unit * qty`. A line whose `qty` is at least %d "
        "has %d%% taken off it, and that discount is `line * %d // 100` "
        "(floor division). Let `d1` be the sum of those line discounts "
        "and `subtotal` the sum of the lines after they came off.\n"
        "2. A member discount comes off `subtotal`: %d%% for tier "
        "\"gold\", %d%% for tier \"silver\", nothing for any other "
        "tier. The amount is `subtotal * pct // 100` (floor division). "
        "Call it `d2`.\n"
        "3. The running total is `subtotal - d2`. If `d1 + d2` is more "
        "than %d, the excess `d1 + d2 - %d` is added back to the running "
        "total.\n"
        "4. If `day` is \"sat\" or \"sun\", add %d cents, unless the "
        "sum of the lines BEFORE any discount is %d or more, in which "
        "case add nothing.\n"
        "5. Round the running total to the nearest multiple of %d, with "
        "an exact half rounding up.\n"
        "6. If the result is below %d, return %d instead.\n"
        "Return an int."
        % (k, p, p, g, s, cap, cap, fee, t, r, m, m))

    cases = [
        [[], "gold", "mon"],                       # nothing, then the floor
        [[], "none", "sat"],                       # the fee on an empty cart
        [[[100, 1]], "none", "mon"],
        [[[100, k]], "none", "mon"],               # exactly the threshold
        [[[100, k - 1]], "none", "mon"],           # one short of it
        [[[t, 1]], "none", "sun"],                 # exactly the waiver
        [[[t - 1, 1]], "none", "sun"],             # one short of it
        [[[999, k], [7, 1]], "gold", "sat"],       # floors that are not exact
        [[[50000, 9]], "gold", "sat"],             # the cap has to bind
        [[[50000, 9]], "silver", "wed"],
        [[[3, 1]], "none", "tue"],                 # rounds down to the floor
    ]
    for _ in range(6):
        cases.append([[[rng.randint(0, 4000), rng.randint(0, 8)]
                       for _ in range(rng.randint(0, 6))],
                      rng.choice(["gold", "silver", "none", "bronze"]),
                      rng.choice(["mon", "sat", "sun", "thu"])])
    return ("charge", prompt, src, cases)


@_hard_code
def _check(rng):
    """Five validations with a stated precedence, and a bool that is an
    int. Getting one rule right is not enough; the order is the task."""
    n = rng.choice([6, 7, 8])
    m = rng.choice([12, 16, 20])
    q = rng.choice([50, 99, 250])
    letters = "".join(rng.sample("ABCDEFGHJKLMNPRSTVWXZ", 3))

    src = ("def check(rec):\n"
           "    missing = sorted(k for k in ('code', 'id', 'name', 'qty')\n"
           "                     if k not in rec)\n"
           "    if missing:\n"
           "        return 'missing:' + missing[0]\n"
           "    v = rec['id']\n"
           "    if not isinstance(v, str) or len(v) != %d "
           "or not v.isascii() or not v.isdigit():\n"
           "        return 'bad id'\n"
           "    v = rec['name']\n"
           "    if not isinstance(v, str) or not v or len(v) > %d:\n"
           "        return 'bad name'\n"
           "    v = rec['qty']\n"
           "    if type(v) is not int or v < 1 or v > %d:\n"
           "        return 'bad qty'\n"
           "    v = rec['code']\n"
           "    if (not isinstance(v, str) or len(v) < 2 or v[0] not in %r\n"
           "            or not v[1:].isascii() or not v[1:].isdigit()):\n"
           "        return 'bad code'\n"
           "    return 'ok'\n" % (n, m, q, letters))

    prompt = (
        "Write a Python function `check(rec)` that returns a string. "
        "`rec` is a dict that should have the keys \"id\", \"name\", "
        "\"qty\" and \"code\", and any of them may be absent. Report "
        "only the FIRST failure in this exact order.\n"
        "1. If any of the four keys is absent, return \"missing:\" "
        "followed by whichever absent key comes first alphabetically.\n"
        "2. \"id\" must be a str of exactly %d characters, every one of "
        "them an ASCII digit; otherwise return \"bad id\".\n"
        "3. \"name\" must be a non-empty str of at most %d characters; "
        "otherwise return \"bad name\".\n"
        "4. \"qty\" must be an int from 1 to %d inclusive; a bool is not "
        "an int here, even though Python says it is. Otherwise return "
        "\"bad qty\".\n"
        "5. \"code\" must be a str whose first character is one of "
        "%s and whose remaining characters are one or more ASCII digits; "
        "otherwise return \"bad code\".\n"
        "If every rule passes, return \"ok\"."
        % (n, m, q, ", ".join(letters)))

    ok = {"id": "1" * n, "name": "a" * m, "qty": q,
          "code": letters[0] + "12"}
    cases = [
        [dict(ok)],
        [{}],                                           # four absent
        [{k: v for k, v in ok.items() if k != "id"}],
        [{k: v for k, v in ok.items() if k not in ("id", "code")}],
        [dict(ok, id="1" * (n + 1))],
        [dict(ok, id=int("1" * n))],                    # right shape, an int
        [dict(ok, id="1" * (n - 1) + "x")],
        [dict(ok, name="")],
        [dict(ok, name="a" * (m + 1))],
        [dict(ok, qty=True)],                           # a bool is not an int
        [dict(ok, qty=0)],
        [dict(ok, qty=q + 1)],
        [dict(ok, code=letters[0])],                    # letter, no digits
        [dict(ok, code="0" + "12")],
        [dict(ok, code=letters[2] + "9")],
        [dict(ok, code=letters[1] + "1a")],
        [dict(ok, id="1" * n, name="n", qty=True, code="zz")],  # order
    ]
    return ("check", prompt, src, cases)


@_hard_code
def _machine(rng):
    """A stack machine whose opcodes are named by the seed, so it is not
    the one in the textbook. Operand order and underflow are the task."""
    words = rng.sample(["pl", "mi", "cp", "sw", "dr", "hi", "lo", "tw",
                        "ov", "nx"], 6)
    add, sub, dup, swp, drp, big = words

    src = ("import re\n\n\n"
           "def run(prog):\n"
           "    num = re.compile(r'-?[0-9]+')\n"
           "    st = []\n"
           "    for tok in prog:\n"
           "        if num.fullmatch(tok):\n"
           "            st.append(int(tok))\n"
           "            continue\n"
           "        if tok not in (%r, %r, %r, %r, %r, %r):\n"
           "            return 'bad token'\n"
           "        need = 1 if tok in (%r, %r) else 2\n"
           "        if len(st) < need:\n"
           "            return 'underflow'\n"
           "        if tok == %r:\n"
           "            st.append(st.pop() + st.pop())\n"
           "        elif tok == %r:\n"
           "            a = st.pop()\n"
           "            st.append(st.pop() - a)\n"
           "        elif tok == %r:\n"
           "            st.append(st[-1])\n"
           "        elif tok == %r:\n"
           "            st[-1], st[-2] = st[-2], st[-1]\n"
           "        elif tok == %r:\n"
           "            st.pop()\n"
           "        else:\n"
           "            a = st.pop()\n"
           "            b = st.pop()\n"
           "            st.append(a if a > b else b)\n"
           "    return st\n"
           % (add, sub, dup, swp, drp, big, dup, drp,
              add, sub, dup, swp, drp))

    prompt = (
        "Write a Python function `run(prog)`. `prog` is a list of strings, "
        "executed left to right against a stack of ints that starts "
        "empty.\n"
        "- A token matching `-?[0-9]+` in full is pushed as an int.\n"
        "- \"%s\" pops two values and pushes their sum.\n"
        "- \"%s\" pops two values and pushes the one that was below "
        "minus the one that was on top.\n"
        "- \"%s\" pushes a copy of the top value.\n"
        "- \"%s\" exchanges the top two values.\n"
        "- \"%s\" discards the top value.\n"
        "- \"%s\" pops two values and pushes the larger of them.\n"
        "Any other token: stop at once and return the string "
        "\"bad token\". If an operation needs more values than the "
        "stack holds: stop at once and return the string \"underflow\", "
        "and check this before you check anything else about the "
        "operation. Otherwise return the stack when the program ends, as "
        "a list with the bottom first."
        % (add, sub, dup, swp, drp, big))

    cases = [
        [[]],
        [["1", "2", add]],
        [["1", "2", sub]],                      # 1 - 2, not 2 - 1
        [["-4", "7", sub]],
        [[add]],                                # underflow on an empty stack
        [["1", add]],                           # underflow with one value
        [[dup]],
        [["5", dup, add]],
        [["1", "2", "3", swp, drp]],
        [["-", "1"]],                           # a lone minus is not a number
        [["1x", add]],                          # bad token before underflow
        [["1", "2", "zz", add]],
        [["9", "-9", big]],
        [["0", "0", sub, dup, big]],
        [["007", "08", add]],                   # leading zeroes are fine
    ]
    for _ in range(6):
        prog = []
        for _ in range(rng.randint(0, 12)):
            if rng.random() < 0.55:
                prog.append(str(rng.randint(-20, 20)))
            else:
                prog.append(rng.choice(words[:6]))
        cases.append([prog])
    return ("run", prompt, src, cases)


def gen_code(rng):
    items, order = [], {}
    n_hard = round(SIZE["code"] * HARD_SHARE)
    tiers = ["hard"] * n_hard + ["base"] * (SIZE["code"] - n_hard)
    rng.shuffle(tiers)
    for i, tier in enumerate(tiers):
        pool = HARD_CODE if tier == "hard" else CODE
        if not order.get(tier):
            order[tier] = rng.sample(pool, len(pool))
        name, prompt, src, cases = order[tier].pop()(rng)
        want = expected(src, name, cases)
        item = Item("code", "code-%02d-%s" % (i, name),
                    prompt + " Reply with the function in one ```python "
                    "code block; do not include tests or example usage.",
                    _grade_code(name, cases, want))
        item.meta.update(reference=src, tier=tier)
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


# Tools that are easy to confuse with each other: a hard item is offered
# its near neighbours, not three unrelated tools.
NEAR = {"set_timer": ["create_event"], "create_event": ["set_timer"],
        "get_weather": ["search_flights"], "search_flights": ["create_event"],
        "convert_currency": ["get_stock_price"],
        "get_stock_price": ["convert_currency"],
        "send_email": ["create_event"]}


def _single_hard(rng):
    """(prompt, tool, {arg: matcher}) where no argument can be copied
    straight out of the prompt: each one has to be worked out."""
    kind = rng.choice(["timer_until", "event_end", "next_weekday",
                       "currency_math"])
    if kind == "timer_until":
        now = datetime.datetime(2026, 1, 1, rng.randint(8, 18),
                                rng.choice([0, 5, 10, 20, 35, 40, 50]))
        mins = rng.choice([25, 40, 55, 70, 95, 130])
        label = rng.choice(["pasta", "laundry", "tea", "bread"])
        return ("It is %s. Set a timer called %s that goes off at %s." % (
            now.strftime("%H:%M"), label,
            (now + datetime.timedelta(minutes=mins)).strftime("%H:%M")),
            "set_timer", {"minutes": _num(mins), "label": _has(label)})
    if kind == "event_end":
        d = _random_date(rng)
        dur = rng.choice([30, 45, 60, 90, 120])
        end = datetime.datetime(2026, 1, 1, rng.randint(10, 18),
                                rng.choice([0, 15, 30, 45]))
        title = rng.choice(["Dentist", "Team retro", "Piano lesson"])
        return ("Put '%s' on my calendar on %s. It runs %d minutes and has "
                "to be finished by %s exactly." % (
                    title, day_words(d), dur, end.strftime("%H:%M")),
                "create_event",
                {"title": _has(title), "date": _ci(_date(d)),
                 "start": _hhmm((end - datetime.timedelta(
                     minutes=dur)).strftime("%H:%M")), "minutes": _num(dur)})
    if kind == "next_weekday":
        today = _random_date(rng)
        wd = rng.choice([w for w in range(7) if w != today.weekday()])
        when = today + datetime.timedelta(
            days=(wd - today.weekday()) % 7)
        (oa, oc), (da, dc) = rng.sample(AIRPORTS, 2)
        n = rng.randint(1, 3)
        return ("Today is %s, %s. Find flights from %s to %s next %s for %d "
                "%s." % (WEEKDAYS[today.weekday()], day_words(today), oa, da,
                         WEEKDAYS[wd], n, "person" if n == 1 else "people"),
                "search_flights",
                {"origin": _ci(oc), "destination": _ci(dc),
                 "date": _ci(_date(when)), "passengers": _num(n)})
    (fa, fc), (ta, tc) = rng.sample(CURRENCIES, 2)
    each = rng.choice([12.5, 45.5, 18.25, 99.0, 7.75])
    count = rng.randint(3, 9)
    return ("I bought %d boxes at %g %s each. What is the total in %s?" % (
        count, each, fa, ta), "convert_currency",
        {"amount": _num(round(each * count, 2)), "from_currency": _ci(fc),
         "to_currency": _ci(tc)})


def _ask_item(rng):
    """A request missing something the tool needs. Calling it anyway means
    inventing the missing part; the honest move is to ask."""
    kind = rng.choice(["currency", "flights", "event"])
    if kind == "currency":
        amount, (ta, _) = rng.choice([40, 250, 1200]), rng.choice(CURRENCIES)
        return ("Convert %d into %s for me." % (amount, ta),
                ["convert_currency", "get_stock_price"])
    if kind == "flights":
        (oa, _), (da, _) = rng.sample(AIRPORTS, 2)
        return ("Book me a flight from %s to %s." % (oa, da),
                ["search_flights", "create_event"])
    return ("Put the review on my calendar for Thursday afternoon.",
            ["create_event", "set_timer"])


def _grade_ask(reply, env, world=None):
    if reply.calls:
        c = reply.calls[0]
        return 0.0, "called %s(%s) instead of asking" % (
            c.name, json.dumps(c.args)[:80])
    said = (reply.content or "").strip()
    if "?" not in said:
        return 0.0, "neither called nor asked: %.60r" % said
    return 1.0, "ok"


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


def _plain(rng):
    """A question with one checkable answer and no tool that fits, so
    both halves can be marked: did it stay off the tools, and was it
    right anyway."""
    kind = rng.choice(["times", "odds", "percent", "days"])
    if kind == "times":
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        return "What is %d times %d?" % (a, b), str(a * b)
    if kind == "odds":
        n = rng.randint(7, 40)
        return ("What is the sum of the first %d odd numbers?" % n,
                str(n * n))
    if kind == "percent":
        p, b = rng.choice([5, 12, 15, 20, 25]), rng.randrange(200, 4000, 20)
        return "What is %d%% of %d?" % (p, b), str(p * b // 100)
    y = rng.randint(1996, 2030)
    m = rng.randint(1, 12)
    days = [31, 29 if (y % 4 == 0 and y % 100) or y % 400 == 0 else 28,
            31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    return ("How many days are in %s %d?" % (MONTHS[m - 1], y), str(days))


def _grade_no_call(want):
    """Not calling a tool is half of it. A model that answers "I don't
    know" has also not called a tool, and used to score full marks."""
    def grade(reply, env, world=None):
        if reply.calls:
            return 0.0, "called %s when no tool fits" % reply.calls[0].name
        said = (reply.content or "").strip()
        if not said:
            return 0.0, "empty reply"
        got = final_answer(said) or said
        if same_answer(got, want):
            return 1.0, "ok"
        return 0.0, "answered %r, wanted %s" % (got[-40:], want)
    return grade


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


# ---- worlds that take more than three calls --------------------------
#
# A chain where every step feeds the next, plus the two things that
# separate an agent from a tool-caller: not doing the destructive thing
# when the rules say no, and trying again when a call fails.

class Support(World):
    tools = ("find_order", "get_policy", "refund")
    schemas = [
        _fn("find_order", "Look an order up by its id.",
            {"order_id": _s("Order id, like ORD-1234")}),
        _fn("get_policy", "What the refund policy allows for a reason.",
            {"reason": _s("Reason for the refund request")}),
        _fn("refund", "Refund money against an order.",
            {"order_id": _s("Order id"), "amount": _n("Amount in dollars")})]

    def __init__(self, order, policy):
        super().__init__()
        self.order, self.policy = order, policy
        self.refunds = []

    def t_find_order(self, order_id):
        if str(order_id).strip() != self.order["order_id"]:
            return {"error": "no such order"}
        return dict(self.order)

    def t_get_policy(self, reason):
        return dict(self.policy, reason=reason)

    def t_refund(self, order_id, amount):
        if str(order_id).strip() != self.order["order_id"]:
            return {"error": "no such order"}
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return {"error": "amount must be a number"}
        self.refunds.append((self.order["order_id"], round(amount, 2)))
        return {"ok": True, "refunded": amount}


class Stock(World):
    tools = ("list_warehouses", "get_stock", "create_order")
    schemas = [
        _fn("list_warehouses", "Every warehouse code.", {}),
        _fn("get_stock", "Units of one item held at one warehouse.",
            {"warehouse": _s("Warehouse code"), "sku": _s("Item SKU")}),
        _fn("create_order", "Order more units of an item.",
            {"sku": _s("Item SKU"), "quantity": _i("Units to order")})]

    def __init__(self, stock, sku):
        super().__init__()
        self.stock, self.sku = stock, sku      # {warehouse: units}
        self.orders = []

    def t_list_warehouses(self):
        return {"warehouses": sorted(self.stock)}

    def t_get_stock(self, warehouse, sku):
        if str(sku).strip().upper() != self.sku:
            return {"error": "no such sku"}
        held = self.stock.get(str(warehouse).strip().upper())
        if held is None:
            return {"error": "no such warehouse"}
        return {"warehouse": warehouse, "sku": sku, "units": held}

    def t_create_order(self, sku, quantity):
        if str(sku).strip().upper() != self.sku:
            return {"error": "no such sku"}
        try:
            quantity = int(quantity)
        except (TypeError, ValueError):
            return {"error": "quantity must be an integer"}
        self.orders.append(quantity)
        return {"ok": True, "ordered": quantity}


class Ops(World):
    tools = ("read_log", "lookup_code", "restart_service",
             "rollback_service")
    schemas = [
        _fn("read_log", "The last lines of a service's log.",
            {"service": _s("Service name")}),
        _fn("lookup_code", "What an error code means, and what to do.",
            {"code": _s("Error code, like E409")}),
        _fn("restart_service", "Restart a service.",
            {"service": _s("Service name")}),
        _fn("rollback_service", "Roll a service back to its last release.",
            {"service": _s("Service name")})]

    def __init__(self, service, code, action, log, fix, flaky=True):
        super().__init__()
        self.service, self.code, self.action = service, code, action
        self.log_text, self.fix = log, fix
        self.flaky = flaky                     # the first read fails
        self.reads = 0
        self.done = []

    def t_read_log(self, service):
        if str(service).strip() != self.service:
            return {"error": "no such service"}
        self.reads += 1
        if self.flaky and self.reads == 1:
            return {"error": "log store temporarily unavailable, try again"}
        return {"lines": self.log_text}

    def t_lookup_code(self, code):
        if str(code).strip().upper() != self.code:
            return {"error": "unknown code"}
        return {"code": self.code, "means": self.fix,
                "recommended_action": self.action}

    def t_restart_service(self, service):
        self.done.append(("restart", str(service).strip()))
        return {"ok": True}

    def t_rollback_service(self, service):
        self.done.append(("rollback", str(service).strip()))
        return {"ok": True}


class Team(World):
    tools = ("list_team", "get_busy", "book_meeting")
    schemas = [
        _fn("list_team", "Everyone on the team.", {}),
        _fn("get_busy", "One person's busy blocks on a day.",
            {"person": _s("Person's name"), "date": _s("YYYY-MM-DD")}),
        _fn("book_meeting", "Book a meeting for several people.",
            {"date": _s("YYYY-MM-DD"), "start": _s("Start time HH:MM"),
             "minutes": _i("Length in minutes"),
             "attendees": {"type": "array", "items": {"type": "string"},
                           "description": "Names of the attendees"}})]

    def __init__(self, day, busy):
        super().__init__()
        self.day, self.busy = day, busy        # {person: [[start, end]]}
        self.booked = []

    def t_list_team(self):
        return {"team": sorted(self.busy)}

    def t_get_busy(self, person, date):
        blocks = self.busy.get(str(person).strip())
        if blocks is None:
            return {"error": "no such person"}
        if str(date).strip() != self.day:
            return {"error": "only %s is loaded" % self.day}
        return {"person": person, "date": date, "busy": blocks}

    def t_book_meeting(self, date, start, minutes, attendees):
        self.booked.append((str(date), str(start), minutes,
                            sorted(str(a) for a in (attendees or []))))
        return {"ok": True}


def _world_refund(rng):
    allowed = rng.random() < 0.5
    days = rng.randint(3, 25) if allowed else rng.randint(45, 200)
    amount = round(rng.randint(1500, 24000) / 100, 2)
    order = {"order_id": "ORD-%04d" % rng.randint(1000, 9999),
             "item": rng.choice(["desk lamp", "kettle", "backpack"]),
             "total": amount, "status": "delivered",
             "delivered_days_ago": days}
    policy = {"window_days": 30,
              "rule": "A delivered order may be refunded in full if it was "
                      "delivered no more than 30 days ago. Past that, no "
                      "refund may be issued."}

    def grade(reply, env, world):
        if allowed:
            if len(world.refunds) == 1 and world.refunds[0][1] == amount:
                return 1.0, "ok"
            return 0.0, "refunds %s, wanted one of %.2f" % (world.refunds,
                                                            amount)
        if world.refunds:
            return 0.0, "refunded %s outside the 30-day window" % (
                world.refunds,)
        # refusing because you looked is the answer; refusing because you
        # did nothing only looks like it
        asked = {name for name, _ in world.log}
        missing = {"find_order", "get_policy"} - asked
        if missing:
            return 0.0, "declined without calling %s" % ", ".join(
                sorted(missing))
        said = (reply.content or "").strip()
        return (1.0, "ok") if said else (0.0, "no refund, but said nothing")
    return ("Order %s came back as faulty and the customer is asking for a "
            "refund. Check the order and our policy, and if the policy "
            "allows it refund the full amount. If it does not, do not "
            "refund anything - explain why instead." % order["order_id"],
            lambda: Support(dict(order), dict(policy)), Support.schemas,
            grade)


def _world_restock(rng):
    sku = "SKU-%03d" % rng.randint(100, 999)
    codes = rng.sample(["LDN", "BER", "OSL", "KYO", "LIM"], 3)
    target = rng.choice([40, 50, 75, 100])
    stock = {c: rng.randint(0, target - 5) for c in codes}
    short = sum(target - v for v in stock.values())

    def grade(reply, env, world):
        if world.orders == [short]:
            return 1.0, "ok"
        return 0.0, "ordered %s, shortfall is %d" % (world.orders, short)
    return ("Every warehouse should hold %d units of %s. Work out the total "
            "shortfall across all of them and place a single order for "
            "exactly that many units." % (target, sku),
            lambda: Stock(dict(stock), sku), Stock.schemas, grade)


def _world_incident(rng):
    service = rng.choice(["billing", "search", "mailer", "auth"])
    action, code = rng.choice([("restart", "E%d" % rng.randint(400, 499)),
                               ("rollback", "E%d" % rng.randint(500, 599))])
    fix = ("a worker deadlock that clears on restart" if action == "restart"
           else "a bad release that has to be rolled back")
    log = ["%02d:%02d WARN queue depth rising" % (rng.randint(0, 23),
                                                  rng.randint(0, 59)),
           "%02d:%02d ERROR %s raised in handler" % (rng.randint(0, 23),
                                                     rng.randint(0, 59),
                                                     code),
           "%02d:%02d INFO retrying" % (rng.randint(0, 23),
                                        rng.randint(0, 59))]

    def grade(reply, env, world):
        want = (action, service)
        if world.done == [want]:
            return 1.0, "ok"
        if not world.done:
            return 0.0, "did nothing (log reads: %d)" % world.reads
        return 0.0, "did %s, wanted %s" % (world.done, [want])
    return ("The %s service is paging. Read its log, look up the error code "
            "you find there, and carry out the action the lookup recommends "
            "on that service. The log store is flaky at the moment."
            % service,
            lambda: Ops(service, code, action, list(log), fix),
            Ops.schemas, grade)


def _world_team(rng):
    while True:
        day = _random_date(rng).isoformat()
        people = rng.sample(NAMES, 3)
        busy = {}
        for who in people:
            blocks, t = [], 9 * 60 + rng.choice([0, 30, 60])
            while t < 15 * 60 and len(blocks) < 3:
                length = rng.choice([30, 60, 90])
                blocks.append([_hm(t), _hm(min(t + length, 17 * 60))])
                t += length + rng.choice([30, 60, 90])
            busy[who] = blocks
        need = rng.choice([30, 60])
        want = None
        for start in range(9 * 60, 17 * 60 - need + 1, 15):
            end = start + need
            if all(end <= _mins(s) or start >= _mins(e)
                   for blocks in busy.values() for s, e in blocks):
                want = start
                break
        if want is not None and want > 9 * 60:
            break

    def grade(reply, env, world):
        if len(world.booked) != 1:
            return 0.0, "booked %d meetings, wanted 1" % len(world.booked)
        date, start, minutes, who = world.booked[0]
        if date != day:
            return 0.0, "booked %s, wanted %s" % (date, day)
        if not _hhmm(_hm(want))(start):
            return 0.0, "booked %s, earliest free is %s" % (start, _hm(want))
        if not _num(need)(minutes):
            return 0.0, "booked %s minutes, wanted %d" % (minutes, need)
        if who != sorted(busy):
            return 0.0, "attendees %s, wanted %s" % (who, sorted(busy))
        return 1.0, "ok"
    return ("Book a %d-minute meeting on %s for the whole team, at the "
            "earliest time between 09:00 and 17:00 when every one of them "
            "is free. Put all of them on it." % (need, day),
            lambda: Team(day, copy.deepcopy(busy)), Team.schemas, grade)




class Ledger(World):
    """Eight tools, four of which are beside the point, and a list that
    only arrives a page at a time."""

    tools = ("find_account", "list_transactions", "get_rate",
             "flag_transaction", "get_balance", "list_currencies",
             "get_customer_notes", "export_statement")
    schemas = [
        _fn("find_account", "Look up an account id from a person's name.",
            {"name": _s("The account holder's full name")}),
        _fn("list_transactions",
            "One page of an account's transactions. Pass the page number "
            "from next_page to get the following page.",
            {"account_id": _s("Account id from find_account"),
             "page": _i("Page number, starting at 1")},
            ["account_id"]),
        _fn("get_rate", "Today's rate from a currency to USD.",
            {"currency": _s("Three-letter currency code")}),
        _fn("flag_transaction", "Flag one transaction for review.",
            {"txn_id": _s("Transaction id"),
             "reason": _s("Why it is being flagged")}),
        _fn("get_balance", "An account's current balance.",
            {"account_id": _s("Account id")}),
        _fn("list_currencies", "Every currency code the ledger knows.", {}),
        _fn("get_customer_notes", "Free-text notes about an account.",
            {"account_id": _s("Account id")}),
        _fn("export_statement", "Queue a statement export.",
            {"account_id": _s("Account id"), "format": _s("pdf or csv")})]

    def __init__(self, holder, account, pages, rates, balance):
        super().__init__()
        self.holder = holder
        self.account = account
        self.pages = pages                  # list of lists of txn dicts
        self.rates = rates
        self.balance = balance
        self.flagged = []

    def t_find_account(self, name):
        if str(name).strip().lower() != self.holder.lower():
            return {"error": "no account for %s" % name}
        return {"account_id": self.account, "holder": self.holder}

    def _mine(self, account_id):
        return str(account_id).strip().upper() == self.account

    def t_list_transactions(self, account_id, page=1):
        if not self._mine(account_id):
            return {"error": "no such account: %s" % account_id}
        try:
            n = int(page)
        except (TypeError, ValueError):
            return {"error": "page must be a whole number"}
        if not 1 <= n <= len(self.pages):
            return {"error": "no page %s; there are %d"
                    % (page, len(self.pages))}
        return {"page": n, "pages": len(self.pages),
                "transactions": [dict(t) for t in self.pages[n - 1]],
                "next_page": n + 1 if n < len(self.pages) else None}

    def t_get_rate(self, currency):
        code = str(currency).strip().upper()
        if code not in self.rates:
            return {"error": "unknown currency: %s" % currency}
        return {"currency": code, "rate_to_usd": self.rates[code]}

    def t_flag_transaction(self, txn_id, reason):
        known = {t["txn_id"] for page in self.pages for t in page}
        if str(txn_id).strip().upper() not in known:
            return {"error": "no such transaction: %s" % txn_id}
        self.flagged.append((str(txn_id).strip().upper(), str(reason)))
        return {"flagged": str(txn_id).strip().upper()}

    def t_get_balance(self, account_id):
        if not self._mine(account_id):
            return {"error": "no such account: %s" % account_id}
        return {"balance_usd": self.balance}

    def t_list_currencies(self):
        return {"currencies": sorted(self.rates)}

    def t_get_customer_notes(self, account_id):
        if not self._mine(account_id):
            return {"error": "no such account: %s" % account_id}
        return {"notes": "Account in good standing. Reviewed annually."}

    def t_export_statement(self, account_id, format="pdf"):
        if not self._mine(account_id):
            return {"error": "no such account: %s" % account_id}
        return {"queued": True, "format": str(format)}


MERCHANTS = ("Halden Freight", "Orsova Supplies", "Bright Fen Ltd",
             "Kestrel Media", "Tamarind Foods", "Pallas Hardware",
             "Vellum Press", "Norrland Tools", "Cobalt Rail",
             "Saffron Clinic", "Ironwood Cafe", "Lune Textiles",
             "Aster Logistics", "Quarry House", "Petrel Labs",
             "Mistral Air", "Fenwick Books", "Gannet Marine")


def _world_ledger(rng):
    """Fourteen to eighteen transactions over three or four pages, in
    four currencies, and the limit is in USD."""
    holder = "%s %s" % (rng.choice(NAMES), rng.choice(
        ["Achterberg", "Baptiste", "Corvino", "Dashwood", "Eklund"]))
    account = "AC-%04d" % rng.randint(1000, 9999)
    rates = {"USD": 1.0,
             "EUR": round(rng.uniform(1.02, 1.19), 4),
             "GBP": round(rng.uniform(1.21, 1.38), 4),
             "JPY": round(rng.uniform(0.0059, 0.0074), 5)}
    limit = rng.choice([500, 750, 1000, 1200])
    per_page = 5
    n = rng.randint(14, 18)
    merchants = rng.sample(MERCHANTS, n)
    txns, used = [], set()
    for i in range(n):
        code = rng.choice(["USD", "EUR", "GBP", "JPY", "JPY"])
        over = rng.random() < 0.3
        usd = (rng.uniform(1.25, 4.0) if over else rng.uniform(0.04, 0.78))
        usd *= limit
        if abs(usd - limit) < limit * 0.05:     # never a judgement call
            usd = limit * (1.4 if over else 0.5)
        raw = usd / rates[code]
        amount = round(raw, 0 if code == "JPY" else 2)
        exact = amount * rates[code]
        if abs(exact - limit) < limit * 0.02:
            continue
        txn_id = "TX-%05d" % rng.randint(10000, 99999)
        if txn_id in used:
            continue
        used.add(txn_id)
        txns.append({"txn_id": txn_id, "merchant": merchants[i],
                     "amount": amount, "currency": code,
                     "date": _random_date(rng).isoformat()})
    want = [t["txn_id"] for t in txns
            if t["amount"] * rates[t["currency"]] > limit]
    if len(want) < 2 or len(want) > 6 or len(txns) < 12:
        return _world_ledger(rng)               # unlucky draw, redraw
    pages = [txns[i:i + per_page] for i in range(0, len(txns), per_page)]
    if want[-1] not in {t["txn_id"] for t in pages[-1]}:
        return _world_ledger(rng)               # the last page must matter
    balance = round(sum(t["amount"] * rates[t["currency"]]
                        for t in txns), 2)

    def grade(reply, env, world):
        got = {i for i, _ in world.flagged}
        good = set(want)
        if not got:
            return 0.0, "flagged nothing; %d were over the limit" % len(good)
        hit = got & good
        score = len(hit) / len(got | good)
        bad_reason = [i for i, why in world.flagged
                      if "over limit" not in why.lower()]
        if bad_reason:
            score *= 0.5
        pages_read = {str(a.get("page", 1))
                      for name, a in world.log if name == "list_transactions"}
        why = "flagged %d of %d, %d wrongly, %d page(s) read" % (
            len(hit), len(good), len(got - good), len(pages_read))
        return round(score, 4), "ok" if score == 1.0 else why

    return ("Go through %s's account and flag every transaction worth more "
            "than %d US dollars. Amounts are in several currencies, so "
            "convert with today's rates before you compare. Flag each one "
            "with the reason 'over limit', and flag nothing else. There "
            "are more transactions than fit in one page of results."
            % (holder, limit),
            lambda: Ledger(holder, account, [list(p) for p in pages],
                           dict(rates), balance),
            Ledger.schemas, grade, 20)


HARD_WORLDS = [_world_refund, _world_restock, _world_incident, _world_team,
               _world_ledger]


WORLDS = [_world_orders, _world_prices, _world_files, _world_calendar]


def gen_tools(rng):
    items = []
    n_multi = 12
    n_single = SIZE["tools"] - n_multi
    for i in range(n_single):
        if i % 6 == 5:                          # a request no tool fits
            offered = rng.sample(["get_weather", "get_stock_price",
                                  "set_timer", "search_flights"], 2)
            q, want = _plain(rng)
            items.append(Item("tools", "tools-%02d-none" % i,
                              q + " End your reply with a line of the form "
                              "'Answer: <answer>'.",
                              _grade_no_call(want), AGENT,
                              [TOOLS[t] for t in offered],
                              meta={"tier": "base", "answer": want}))
            continue
        if i % 6 == 2:                          # something the tool needs
            prompt, offered = _ask_item(rng)    # is missing: ask for it
            items.append(Item("tools", "tools-%02d-ask" % i, prompt,
                              _grade_ask, AGENT,
                              [TOOLS[t] for t in offered],
                              meta={"tier": "hard"}))
            continue
        hard = i % 2 == 0
        prompt, tool, want = (_single_hard if hard else _single)(rng)
        pool = NEAR.get(tool, []) if hard else []
        others = pool + rng.sample(
            [t for t in TOOLS if t != tool and t not in pool],
            max(0, 2 - len(pool)))
        offered = [tool] + others
        rng.shuffle(offered)
        items.append(Item("tools", "tools-%02d-%s" % (i, tool), prompt,
                          _grade_call(tool, want), AGENT,
                          [TOOLS[t] for t in offered],
                          meta={"tier": "hard" if hard else "base"}))
    for j in range(n_multi):
        hard = j % 2 == 0
        pool = HARD_WORLDS if hard else WORLDS
        maker = pool[(j // 2) % len(pool)]
        made = maker(rng)
        prompt, world, schemas, grade = made[:4]
        meta = {"tier": "hard" if hard else "base"}
        if len(made) > 4:
            meta["turns"] = made[4]
        items.append(Item("tools", "tools-%02d-%s" % (
            n_single + j, maker.__name__[7:]), prompt, grade, AGENT,
            schemas, world=world, meta=meta))
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


def _said(reply, cap=200):
    """The answer, not the essay around it.

    Every one of these questions says to reply with just the room, just
    the code, just the number. Grading the whole reply by substring
    means a model that quotes a slab of the document back scores for
    whatever happened to be inside it, so read the line it ended on and
    only if it is short enough to be an answer."""
    lines = [ln for ln in (reply.content or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    said = (final_answer(reply.content) or lines[-1]).strip()
    return said if len(said) <= cap else ""


def _floor_answer(want):
    """A bare number is too short to grade by substring: every room name
    in the document holds digits, and one of them will match."""
    def grade(reply, env, world=None):
        said = _said(reply)
        lone = re.fullmatch(r"[^\d-]*?(\d+)\D*", said)
        if lone and lone.group(1).lstrip("0") == want.lstrip("0"):
            return 1.0, "ok"
        named = re.findall(r"floor\s*(?:number\s*)?(\d+)", said, re.I)
        if named and named[-1].lstrip("0") == want.lstrip("0"):
            return 1.0, "ok"
        return 0.0, "wanted floor %s, got %.60r" % (want, said)
    return grade


CODE_SHAPE = re.compile(r"\b[A-Z]{2}-\d{4}\b")
NO_WORDS = re.compile(
    r"(\bnot\b|\bno\b|\bisn'?t\b|\baren'?t\b|\bdoes ?n'?t\b"
    r"|\bcannot\b|\bcan'?t\b|\bnever\b|\babsent\b|\bmissing\b"
    r"|\b(do|does|did)(n'?t| not)\b|\bwithout\b"
    r"|\bnowhere\b|\bunable\b|\bunknown\b|\bunavailable\b"
    r"|\bunspecified\b|\bunlisted\b|\bomitted\b|\black(s|ing)?\b"
    r"|\bnone\b|\bsilent\b|n/a)", re.I)


def _set_answer(want, universe):
    """Overlap, not all-or-nothing: finding two of three is real
    progress and should not score the same as finding none."""
    good = {w.lower() for w in want}
    known = {w.lower() for w in universe}

    def grade(reply, env, world=None):
        # the question asks for just the names, so read the last line
        # rather than any thinking above it that weighed other people
        lines = [ln for ln in (reply.content or "").splitlines() if ln.strip()]
        said = final_answer(reply.content) or (lines[-1] if lines else "")
        got = {w for w in known
               if re.search(r"\b%s\b" % re.escape(w), said.lower())}
        if not got:
            return 0.0, "named none of the %d" % len(good)
        score = len(got & good) / len(got | good)
        return (round(score, 4),
                "ok" if score == 1.0 else "named %d of %d, %d wrongly"
                % (len(got & good), len(good), len(got - good)))
    return grade


ABSENT_FLAT = ("none", "n/a", "na", "unknown", "not given",
               "not recorded", "not listed", "not specified",
               "not stated", "no code", "not provided", "not there")

ABSENT_ABOUT = re.compile(
    r"\b(code|record|records|given|give|recorded|listed|stated|specified|"
    r"provided|available|mention|mentioned|know|find|say|anywhere)\b", re.I)


def _echoed(said, prompt):
    """True when the reply is the prompt handed back. Sixty characters
    of an exact run is far past coincidence, and quoting one line of the
    instructions is the cheapest way to satisfy a rule about them."""
    said, prompt = " ".join(said.split()), " ".join(prompt.split())
    if len(said) < 60:
        return said in prompt and len(said) > 20
    return any(said[i:i + 60] in prompt
               for i in range(0, len(said) - 59, 20))


def _absent_answer(question):
    """The document does not say. Making something up is the failure
    this grades; saying so is the pass.

    Read only the last line, and only a short one: the question asks for
    one word, so a reply that goes on for a paragraph has not answered
    it, and a reply that quotes the question back is not agreeing with
    it. A negative word loose in a long answer proves nothing."""
    def grade(reply, env, world=None):
        lines = [ln for ln in (reply.content or "").splitlines()
                 if ln.strip()]
        if not lines:
            return 0.0, "said nothing"
        said = (final_answer(reply.content) or lines[-1]).strip()
        made_up = CODE_SHAPE.findall(said)
        if made_up:
            return 0.0, "invented a code: %s" % made_up[0]
        if len(said) > 200:
            return 0.0, "did not answer in one line: %.60r" % said
        if _echoed(said, question):
            return 0.0, "read the question back"
        flat = said.strip().strip(".,;:!\"'").lower()
        if flat in ABSENT_FLAT:
            return 1.0, "ok"
        # a negative word on its own is not a refusal: it has to be
        # about the thing that is missing
        if (len(said.split()) <= 25 and NO_WORDS.search(said)
                and ABSENT_ABOUT.search(said)):
            return 1.0, "ok"
        return 0.0, "did not say it is absent: %.60r" % said
    return grade


def _contains(want):
    def grade(reply, env, world=None):
        said = _said(reply)
        ok = want.lower() in said.lower()
        return (1.0, "ok") if ok else (0.0, "wanted %s, got %.60r" % (
            want, said or (reply.content or "")))
    return grade


def gen_longctx(rng):
    items = []
    for label, target in LONG:
        projects = rng.sample(PROJECTS, 12)
        staff = rng.sample(STAFF, 6)
        facts, questions, tiers = [], [], []
        codes = {p: _code_word(rng) for p in projects[:9]}
        p0 = projects[0]                        # plain retrieval, deep
        facts.append((rng.uniform(0.55, 0.95),
                      "The access code for project %s is %s."
                      % (p0, codes[p0])))
        questions.append(("What is the access code for project %s? "
                          "Reply with just the code." % p0, codes[p0],
                          "text"))
        tiers.append("base")
        # the same question, answered twice: whoever stops at the first
        # match in the document gets the stale code
        p1, fresh = projects[1], _code_word(rng)
        a = rng.uniform(0.1, 0.4)
        facts.append((a, "The access code for project %s is %s."
                      % (p1, codes[p1])))
        facts.append((rng.uniform(a + 0.2, 0.95),
                      "Update: the access code for project %s was changed to "
                      "%s, replacing the earlier one." % (p1, fresh)))
        questions.append(("What is the current access code for project %s? "
                          "Reply with just the code." % p1, fresh, "text"))
        tiers.append("hard")
        for p in projects[2:9]:                 # distractors
            facts.append((rng.random(), "The access code for project %s is %s."
                          % (p, codes[p])))
        picked = rng.sample([(a, b) for a in range(1, 10)
                             for b in range(1, 61)], len(staff))
        rooms = {s: "room %d-%02d" % rc
                 for s, rc in zip(staff, picked, strict=True)}
        # three of the six share a floor, so "who is on floor n" has an
        # answer that cannot be finished early
        hot = rng.randint(1, 14)
        others = rng.sample([f for f in range(1, 15) if f != hot], 3)
        share = rng.sample(staff, 3)
        floors = {s: hot for s in share}
        for s, f in zip([x for x in staff if x not in share], others,
                        strict=True):
            floors[s] = f
        p2, lead2 = projects[9], staff[0]       # two hops
        a, b = rng.uniform(0.05, 0.45), rng.uniform(0.55, 0.95)
        if rng.random() < 0.5:
            a, b = b, a
        facts.append((a, "Project %s is led by %s." % (p2, lead2)))
        facts.append((b, "%s works from %s." % (lead2, rooms[lead2])))
        facts.append((rng.random(), "%s is on floor %d." % (
            rooms[lead2].capitalize(), floors[lead2])))
        # graded on the number alone: the question asks for just the
        # room, so a reply of "4-04" is right and "room 4-04" passes too
        questions.append(("Which room does the person who leads project "
                          "%s work from? Reply with just the room." % p2,
                          rooms[lead2].split()[-1], "text"))
        tiers.append("base")
        # three hops, planted in an order that never matches the chain
        for p3, lead3 in zip(projects[10:12], staff[1:3], strict=True):
            depths = sorted(rng.uniform(0.05, 0.95) for _ in range(3))
            chain = ("Project %s is led by %s." % (p3, lead3),
                     "%s works from %s." % (lead3, rooms[lead3]),
                     "%s is on floor %d." % (rooms[lead3].capitalize(),
                                             floors[lead3]))
            for d, f in zip(rng.sample(depths, 3), chain, strict=True):
                facts.append((d, f))
            questions.append(("Which floor does the person who leads project "
                              "%s work on? Reply with just the floor number."
                              % p3, str(floors[lead3]), "floor"))
            tiers.append("hard")
        for s in staff[3:]:                     # distractors for both hops
            facts.append((rng.random(), "%s works from %s." % (s, rooms[s])))
            facts.append((rng.random(), "%s is on floor %d." % (
                rooms[s].capitalize(), floors[s])))
        # nothing to retrieve: the whole document has to be read to count
        questions.append(("How many people does the document say work from "
                          "a room? Reply with just the number.",
                          str(len(staff)), "floor"))
        tiers.append("hard")
        # not one answer but every answer: three of the six people are
        # on that floor, by way of a room named somewhere else again
        questions.append(("Which people work on floor %d? Every one of "
                          "them is named in the records. Reply with just "
                          "their names, separated by commas." % hot,
                          ", ".join(sorted(share)), "set"))
        tiers.append("hard")
        # and nothing to find: the honest answer is that it is not there
        questions.append(("What is the access code for project %s? Reply "
                          "with just the code, or with the single word NONE "
                          "if the records do not give it." % projects[11],
                          "", "absent"))
        tiers.append("hard")
        doc_seed = rng.random()
        cache = {}

        def text(cpt, _seed=doc_seed, _target=target, _facts=tuple(facts),
                 _cache=cache):
            key = round(cpt, 3)
            if key not in _cache:
                _cache.clear()
                _cache[key] = document(_seed, int(_target * cpt), _facts)
            return _cache[key]

        for k, (q, want, kind) in enumerate(questions):
            item = Item("longctx", "longctx-%s-%d" % (label, k),
                        lambda cpt, q=q, text=text: (
                            "Read the records below, then answer the "
                            "question after them.\n\n" + text(cpt)
                            + "\n\nQuestion: " + q),
                        _set_answer(want.split(", "), STAFF)
                        if kind == "set"
                        else _absent_answer(q) if kind == "absent"
                        else _floor_answer(want) if kind == "floor"
                        else _contains(want))
            item.meta.update(doc=label, tokens=target, answer=want,
                             tier=tiers[k])
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


MIN_WORDS = 12
WORDISH = re.compile(r"[A-Za-z0-9']+")


def _wordish(t):
    """Words, counted so that compact JSON is not one of them.

    A valid answer to the JSON rule can be written without a single
    space in it, and splitting on whitespace calls that one word. The
    rules that count words keep their own definition; this is only for
    telling a piece of writing from a reply that is not one."""
    return len(WORDISH.findall(t))


def _grade_rules(rules, prompt=""):
    def grade(reply, env, world=None):
        t = reply.content or ""
        if prompt and _echoed(t, prompt):
            return 0.0, "read the instructions back instead of writing"
        # the task says at least six words, so this is a rule the model
        # was told, not a floor sprung on it afterwards
        if _wordish(t) < 6:
            return 0.0, "fewer than six words: %d" % _wordish(t)
        failed = []
        for text, check in rules:
            try:
                ok = check(t)
            except (ValueError, TypeError, IndexError):
                ok = False
            if not ok:
                failed.append(text)
        if not failed:
            return 1.0, "ok"
        # part marks are for a piece of writing that broke a rule, not
        # for silence: an empty reply keeps "no commas" and "do not use
        # the letter t" for free, and that was worth two thirds of an
        # item to a model that said nothing at all
        if _wordish(t) < MIN_WORDS:
            return 0.0, "nothing written: %d words" % _wordish(t)
        kept = (len(rules) - len(failed)) / len(rules)
        return kept, "broke: " + " | ".join(failed)[:160]
    return grade


def _hard_rule(rng):
    """One more rule on top, of the kind that has to be counted or
    checked letter by letter rather than just remembered."""
    kind = rng.choice(["exact_words", "lipogram", "sentences", "acrostic",
                       "typed_json"])
    if kind == "exact_words":
        n = rng.choice([35, 50, 65])
        return kind, "Use exactly %d words, no more and no fewer." % n, \
            lambda t: len(t.split()) == n
    if kind == "lipogram":
        letter = rng.choice("est")
        return kind, "Do not use the letter '%s' anywhere in your answer." \
            % letter, lambda t, c=letter: c not in t.lower()
    if kind == "sentences":
        n, w = rng.randint(3, 5), rng.choice([8, 10, 12])

        def check(t, n=n, w=w):
            parts = [x for x in re.split(r"(?<=[.!?])\s+", t.strip()) if x]
            return len(parts) == n and all(len(x.split()) <= w for x in parts)
        return kind, "Write exactly %d sentences, each of at most %d " \
            "words." % (n, w), check
    if kind == "acrostic":
        word = rng.choice(["LAMP", "RIVER", "STONE", "CLOUD"])

        def check(t, word=word):
            firsts = [x.strip()[0].upper() for x in
                      re.findall(r"^\s*[-*\u2022]?\s*(.+)$", t, re.M)
                      if x.strip()]
            return len(firsts) == len(word) and "".join(firsts) == word
        return kind, "Write exactly %d lines whose first letters spell " \
            "%s, in order." % (len(word), word), check
    keys = rng.sample(["summary", "facts", "rating", "source"], 2)

    def check(t, keys=keys):
        try:
            d = json.loads(strip_fences(t))
        except ValueError:
            return False
        return (isinstance(d, dict) and set(d) == set(keys + ["tags"])
                and isinstance(d.get("tags"), list) and len(d["tags"]) == 3
                and all(isinstance(x, str) for x in d["tags"]))
    return kind, 'Answer with only a JSON object with the keys %s and ' \
        '"tags", where "tags" is a list of exactly three strings.' % (
            ", ".join('"%s"' % k for k in keys)), check


# A hard rule that cannot be kept alongside the structure it is paired
# with: counting words is impossible inside a fixed JSON shape.
HARD_CLASH = {("json", "exact_words"), ("json", "lipogram"),
              ("json", "sentences"), ("json", "acrostic"),
              ("bullets", "sentences"), ("bullets", "typed_json"),
              ("paragraphs", "typed_json"), ("quotes", "typed_json"),
              ("title", "typed_json"), ("json", "typed_json"),
              ("bullets", "exact_words"), ("paragraphs", "acrostic"),
              ("bullets", "acrostic"), ("title", "sentences"),
              ("paragraphs", "sentences"),
              ("quotes", "acrostic"), ("title", "acrostic"),
              ("quotes", "sentences")}
HARD_LEX_CLASH = {("lower", "acrostic"), ("lower", "typed_json"),
                  ("maxwords", "exact_words"), ("maxwords", "sentences"),
                  ("nocomma", "typed_json"), ("ends", "typed_json"),
                  ("ends", "exact_words"), ("ends", "lipogram"),
                  ("ends", "acrostic"), ("starts", "acrostic"),
                  ("starts", "typed_json"), ("keyword", "lipogram"),
                  ("keyword", "exact_words"), ("ends", "sentences"),
                  ("starts", "lipogram")}


def gen_instruct(rng):
    items = []
    n_hard = round(SIZE["instruct"] * HARD_SHARE)
    tiers = ["hard"] * n_hard + ["base"] * (SIZE["instruct"] - n_hard)
    rng.shuffle(tiers)
    for i, tier in enumerate(tiers):
        while True:
            sk, stext, scheck = _structure(rng)
            lk, ltext, lcheck = _lexical(rng)
            if (sk, lk) in CLASH or (lk == "lower" and sk is None
                                     and rng.random() < 0.5):
                continue
            if tier == "base":
                hk = None
                break
            hk, htext, hcheck = _hard_rule(rng)
            if (sk, hk) not in HARD_CLASH and (lk, hk) not in HARD_LEX_CLASH:
                break
        rules = [(ltext, lcheck)] + ([(stext, scheck)] if sk else [])
        if hk:
            rules.append((htext, hcheck))
        prompt = ("Write a short piece of at least six words about %s. %s"
                  % (rng.choice(TOPICS),
                     " ".join(r[0] for r in reversed(rules))))
        item = Item("instruct", "instruct-%02d-%s%s%s" % (
            i, sk + "-" if sk else "", lk, "-" + hk if hk else ""),
            prompt, _grade_rules(rules, prompt))
        item.meta["tier"] = tier
        items.append(item)
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


# --- harder: several steps, and a plausible wrong turn at each one ---

def _rates(rng):
    a, b = rng.sample([3, 4, 5, 6, 8], 2)
    c = rng.choice([9, 10, 12, 15])          # the drain, slower than either
    rate = 1 / a + 1 / b - 1 / c
    return ("A tank is filled by pipe A alone in %d hours and by pipe B "
            "alone in %d hours. An open drain empties a full tank in %d "
            "hours. Starting empty with both pipes and the drain open, how "
            "many minutes does the tank take to fill? Round to the nearest "
            "minute." % (a, b, c), str(round(60 / rate)))


def _ages(rng):
    while True:
        n, m = rng.randint(3, 7), rng.randint(2, 4)
        k = rng.randint(4, 20)
        if n <= m or k * (m - 1) % (n - m):
            continue
        child = k * (m - 1) // (n - m)
        if 4 <= child <= 30 and n * child <= 90:
            return ("%s is %d times as old as %s. In %d years %s will be "
                    "only %d times as old as %s. How old is %s now?" % (
                        NAMES[0], n, NAMES[1], k, NAMES[0], m, NAMES[1],
                        NAMES[0]), str(n * child))


def _overlap(rng):
    a1, b1, c1 = (rng.randint(20, 80) for _ in range(3))
    ab, ac, bc = (rng.randint(5, 30) for _ in range(3))
    t = rng.randint(3, 15)
    sizes = (a1 + ab + ac + t, b1 + ab + bc + t, c1 + ac + bc + t)
    return ("In a survey, %d people read the Herald, %d read the Gazette "
            "and %d read the Tribune. %d read both the Herald and the "
            "Gazette, %d both the Herald and the Tribune, and %d both the "
            "Gazette and the Tribune. %d read all three. How many read "
            "exactly one of the three papers?" % (
                sizes[0], sizes[1], sizes[2], ab + t, ac + t, bc + t, t),
            str(a1 + b1 + c1))


def _breakeven(rng):
    f1, p1 = rng.randint(0, 20), rng.randint(12, 30)
    f2 = f1 + rng.randint(40, 200)
    p2 = p1 - rng.randint(3, 9)
    units = next(u for u in range(1, 10000)
                 if f2 + p2 * u < f1 + p1 * u)
    return ("Plan A costs $%d a month plus $%d per gigabyte. Plan B costs "
            "$%d a month plus $%d per gigabyte. What is the smallest whole "
            "number of gigabytes in a month for which Plan B costs less "
            "than Plan A?" % (f1, p1, f2, p2), str(units))


def _paths(rng):
    w, h = rng.randint(4, 7), rng.randint(4, 7)
    bx, by = rng.randint(1, w - 1), rng.randint(1, h - 1)
    grid = [[0] * (h + 1) for _ in range(w + 1)]
    grid[0][0] = 1
    for x in range(w + 1):                   # counted, not derived: a
        for y in range(h + 1):               # closed form is easy to fumble
            if (x, y) == (bx, by):
                grid[x][y] = 0
            elif (x, y) != (0, 0):
                grid[x][y] = (grid[x - 1][y] if x else 0) + (
                    grid[x][y - 1] if y else 0)
    return ("On a %dx%d grid of city blocks you walk from the south-west "
            "corner to the north-east corner, moving only north or east "
            "along the streets. The crossing %d blocks east and %d blocks "
            "north of the start is closed. How many different routes are "
            "there?" % (w, h, bx, by), str(grid[w][h]))


def _mixture(rng):
    lo, hi = rng.choice([(10, 40), (15, 45), (20, 50), (5, 35)])
    total = rng.choice([10, 20, 25, 40, 50])
    part = rng.randint(1, total - 1)
    target = (hi * part + lo * (total - part)) / total
    if target != int(target):
        return _mixture(rng)
    return ("A chemist mixes a %d%% acid solution with a %d%% acid solution "
            "to make %d litres of a %d%% solution. How many litres of the "
            "%d%% solution are needed?" % (hi, lo, total, int(target), hi),
            str(part))


def _digitsum(rng):
    lo = rng.randint(100, 4000)
    hi = lo + rng.randint(300, 1500)
    want = rng.randint(9, 18)
    n = sum(1 for x in range(lo, hi + 1)
            if sum(int(d) for d in str(x)) == want)
    return ("How many integers from %d to %d inclusive have digits that add "
            "up to exactly %d?" % (lo, hi, want), str(n))


def _meeting(rng):
    while True:
        u, v = rng.randint(9, 24), rng.randint(9, 24)
        gap, t = rng.choice([20, 30, 45]), rng.randint(40, 200)
        km = (t * (u + v) - gap * v) / 60
        if km == int(km) and km > 5:
            start = datetime.datetime(2026, 1, 1, 9, 0)
            met = start + datetime.timedelta(minutes=t)
            return ("Two towns are %d km apart. A cyclist leaves the first "
                    "at 9:00 riding at %d km/h. Another leaves the second "
                    "%d minutes later riding towards them at %d km/h. At "
                    "what time do they meet? Answer as HH:MM on a 24-hour "
                    "clock." % (int(km), u, gap, v), met.strftime("%H:%M"))




SEATS = ("tea", "cocoa", "cider", "juice", "milk", "water")


def _and_list(words):
    return "%s and %s" % (", ".join(words[:-1]), words[-1])


def _clues(rng, who, what):
    """Every clue is read off the true arrangement, so the set is always
    satisfiable; the pruning below is what makes it unique."""
    n = len(who)
    seat = {name: i for i, name in enumerate(who)}
    drink = {what[i]: i for i in range(n)}
    out = []

    def add(text, fn):
        out.append((text, fn))

    for i in range(n - 1):
        add("%s sits directly to the left of %s." % (who[i], who[i + 1]),
            lambda p, d, a=who[i], b=who[i + 1]: p[b] - p[a] == 1)
    for _ in range(n * 2):
        name = rng.choice(who)
        bad = rng.choice([k for k in range(n) if k != seat[name]])
        add("%s is not in seat %d." % (name, bad + 1),
            lambda p, d, a=name, k=bad: p[a] != k)
    for name in who:
        if seat[name] in (0, n - 1):
            add("%s sits at one of the two ends." % name,
                lambda p, d, a=name, m=n: p[a] in (0, m - 1))
        else:
            add("%s does not sit at either end." % name,
                lambda p, d, a=name, m=n: p[a] not in (0, m - 1))
    for name in who:
        add("%s drinks %s." % (name, what[seat[name]]),
            lambda p, d, a=name, x=what[seat[name]]: d[x] == p[a])
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(drink[what[i]] - drink[what[j]]) == 1:
                add("the %s and the %s are next to each other."
                    % (what[i], what[j]),
                    lambda p, d, x=what[i], y=what[j]:
                    abs(d[x] - d[y]) == 1)
            if drink[what[i]] < seat[who[j]]:
                add("the %s is drunk somewhere to the left of %s."
                    % (what[i], who[j]),
                    lambda p, d, x=what[i], b=who[j]: d[x] < p[b])
    return out


def _solve(who, drinks, clues):
    """Every arrangement that fits, at most two of them: the caller only
    needs to know whether the answer is forced."""
    found = []
    for order in itertools.permutations(who):
        p = {name: i for i, name in enumerate(order)}
        for pour in itertools.permutations(drinks):
            d = {x: i for i, x in enumerate(pour)}
            if all(fn(p, d) for _, fn in clues):
                found.append((order, pour))
                if len(found) > 1:
                    return found
    return found


def _seating(rng):
    n = 4
    who = rng.sample(NAMES, n)
    drinks = rng.sample(SEATS, n)
    pool = _clues(rng, who, drinks)
    rng.shuffle(pool)
    keep = []
    for clue in pool:                       # enough clues to pin it down
        keep.append(clue)
        if len(_solve(who, drinks, keep)) == 1:
            break
    else:                                   # pathological seed: try again
        return _seating(rng)
    for i in range(len(keep) - 1, -1, -1):  # then drop what is not needed
        trimmed = keep[:i] + keep[i + 1:]
        if len(_solve(who, drinks, trimmed)) == 1:
            keep = trimmed
    order = sorted(range(len(keep)), key=lambda i: rng.random())
    lines = "\n".join("%d. %s" % (k + 1, keep[i][0])
                       for k, i in enumerate(order))
    if rng.random() < 0.5:
        target = rng.choice(who)
        q = "Which seat is %s in? Answer with the seat number." % target
        want = str(who.index(target) + 1)
    else:
        target = rng.randrange(n)
        q = ("Who is in seat %d? Answer with the name." % (target + 1))
        want = who[target]
    return ("%s sit in a row of seats numbered 1 to 4 from left to "
            "right, and each drinks a different one of %s. "
            "These statements are all true:\n%s\n%s"
            % (_and_list(sorted(who)), _and_list(sorted(drinks)),
               lines, q), want)


STEP_OPS = (("add %d to a", lambda a, b, c: (a + c, b)),
            ("take %d off a", lambda a, b, c: (a - c, b)),
            ("add %d to b", lambda a, b, c: (a, b + c)),
            ("take %d off b", lambda a, b, c: (a, b - c)),
            ("replace a with a + b", lambda a, b, c: (a + b, b)),
            ("replace b with a - b", lambda a, b, c: (a, a - b)),
            ("swap a and b", lambda a, b, c: (b, a)))


def _machine_run(start_a, start_b, rounds, k, ops, cap):
    a, b = start_a, start_b
    for r in range(1, rounds + 1):
        if r % k == 0:
            op = ops[0]
        elif a > b:
            op = ops[1]
        else:
            op = ops[2]
        a, b = op[0](a, b, op[1])
        if abs(a) > cap or abs(b) > cap:
            return None
    return a, b


def _stepper(rng):
    """Fourteen rounds of three interacting rules. Reading ahead does
    not help; the branch depends on the values, so it has to be run."""
    rounds = rng.choice([12, 14, 16])
    k = rng.choice([3, 4, 5])
    for _ in range(40):
        picks = rng.sample(range(len(STEP_OPS)), 3)
        consts = [rng.randint(2, 9) for _ in picks]
        say, ops = [], []
        for c, i in zip(consts, picks, strict=True):
            text, fn = STEP_OPS[i]
            say.append(text % c if "%d" in text else text)
            ops.append((fn, c))
        a0, b0 = rng.randint(1, 12), rng.randint(1, 12)
        got = _machine_run(a0, b0, rounds, k, ops, 10 ** 6)
        if got is None or got[0] == got[1]:
            continue
        want = rng.choice(["a", "b"])
        return ("Two counters start at a = %d and b = %d. The machine runs "
                "for %d rounds, numbered 1 to %d. In each round exactly "
                "one thing happens: if the round number is a multiple of "
                "%d, %s; otherwise, if a is greater than b, %s; otherwise, "
                "%s. What is %s after round %d?"
                % (a0, b0, rounds, rounds, k, say[0], say[1], say[2],
                   want, rounds),
                str(got[0] if want == "a" else got[1]))
    return _stepper(rng)                    # pathological seed: try again


HARD = [_rates, _ages, _overlap, _breakeven, _paths, _mixture, _digitsum,
        _meeting, _seating, _stepper]


REASON = [_change, _powmod, _days, _weekday, _order, _divisible, _in_base,
          _sequence, _choose, _average]


def _grade_answer(want):
    def grade(reply, env, world=None):
        got = final_answer(reply.content)
        if same_answer(got, want):
            return 1.0, "ok"
        return 0.0, "answered %r, wanted %s" % ((got or "")[:40], want)
    return grade


# Most of the suite is the harder tier: the single-step templates alone
# put every competent model at the ceiling, where nothing can be told
# apart. HARD_SHARE of the items are multi-step.
HARD_SHARE = 0.6


def gen_reason(rng):
    items, order = [], {}
    n_hard = round(SIZE["reason"] * HARD_SHARE)
    tiers = ["hard"] * n_hard + ["base"] * (SIZE["reason"] - n_hard)
    rng.shuffle(tiers)
    for i, tier in enumerate(tiers):
        pool = HARD if tier == "hard" else REASON
        if not order.get(tier):
            order[tier] = rng.sample(pool, len(pool))
        maker = order[tier].pop()
        q, want = maker(rng)
        item = Item("reason", "reason-%02d-%s" % (i, maker.__name__[1:]),
                    q + " End your reply with a line of the form "
                    "'Answer: <answer>'.", _grade_answer(want))
        item.meta.update(answer=want, tier=tier)
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
