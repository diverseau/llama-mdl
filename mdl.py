#!/usr/bin/env python3
"""mdl - run local llama.cpp servers from a config file, one or several."""

import ctypes
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
try:
    import tomllib
except ImportError:                      # 3.10 and older
    raise SystemExit("mdl: needs Python 3.11 or newer (tomllib)") from None
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

def _base(env, *fallback):
    """XDG dir if the variable is set, else the conventional path."""
    root = os.environ.get(env)
    return Path(root) if root else Path.home().joinpath(*fallback)


CONFIG_DIR = _base("XDG_CONFIG_HOME", ".config") / "mdl"
CONFIG = CONFIG_DIR / "models.toml"
STATE_DIR = _base("XDG_STATE_HOME", ".local", "state") / "mdl"
VERSION = "0.10.0"
DEFAULT_BIN = "llama-server"
CONFIG_DATA = {}          # last parsed config, for UI-only settings
DEFAULT_PORT = 8080
# Wording differs across llama.cpp builds: older ones say "server is listening on
# http://...", build 10424+ says "llama_server: listening on http://...".
# Matched only to colour log lines; readiness is decided by server_ready().
READY = re.compile(r"listening on http|server is listening|HTTP server listening"
                   r"|starting the main loop")
READY_TIMEOUT = 300      # seconds; ready_timeout in the config overrides

KEEP_LOGS = 3            # <name>.log plus .1 .. .N-1
KNOWN = {"model", "mmproj", "ngl", "n_cpu_moe", "ctx", "flash_attn",
         "kv_type", "parallel", "port", "args", "group", "llama_server"}
SIMPLE = (("ngl", "-ngl"), ("n_cpu_moe", "--n-cpu-moe"), ("ctx", "-c"),
          ("parallel", "-np"), ("port", "--port"))

USAGE = ("usage: mdl {init|config [--path|--undo|--history]|"
         "add <model.gguf>|check|list|"
         "doctor [--json] [name]|run <name> [--port N]|"
         "stop [<name>|--all]|ps [--json]|logs [-f] [name]|"
         "ui [--no-open] [--port N]|tui [--no-fx]|snapshot|"
         "fit <gguf|hf:repo|name> [--help]|eval <name> [--help]|"
         "find [--help]|pull <org/repo[:quant]> [--run]|manifest <name>|"
         "lab {run|report|compare} [--help]|"
         "catalog {pull|build|tree|search|stats}} [--version]")

# The model path mdl init leaves behind. check knows to treat it as a
# to-do rather than a fault; tests keep the two in step.
PLACEHOLDER = "/path/to/your-model.gguf"

STARTER = '''# mdl config. One table per model; the table name is what you
# pass to `mdl run`. Use forward slashes in paths on Windows - TOML
# treats a backslash as an escape character.

# Where llama-server lives. $MDL_LLAMA_SERVER overrides this.
llama_server = "%s"

# Rename this, point it at a .gguf, and run: mdl run example
[example]
model = "/path/to/your-model.gguf"
ngl = 99          # layers on the GPU; 99 means all of them
ctx = 8192        # context window
flash_attn = true
kv_type = "q8_0"  # quantised KV cache, needs flash_attn
parallel = 1
# mmproj = "/path/to/mmproj-F16.gguf"   # for a vision model
port = 8080
args = ["--metrics"]   # extra flags, passed through as-is
'''


class MdlError(Exception):
    """Anything the user caused. main() prints it; the TUI shows it in a modal."""


def die(msg):
    raise MdlError(msg)


def load_config():
    try:
        data = tomllib.loads(CONFIG.read_text())
    except FileNotFoundError:
        die(f"no config at {CONFIG}; run 'mdl init' to create one")
    except (OSError, tomllib.TOMLDecodeError) as e:
        die(f"cannot read {CONFIG}: {e}")
    global CONFIG_DATA
    CONFIG_DATA = data
    models = {k: v for k, v in data.items() if isinstance(v, dict)}
    binary = (os.environ.get("MDL_LLAMA_SERVER")
              or data.get("llama_server") or DEFAULT_BIN)
    return models, str(binary)


# A model's name is a TOML table name and a file name (its state and its
# log), so it is held to what is safe as both: letters, digits, - _ and
# ., starting with a letter or digit. A dot needs quoting in TOML, which
# toml_key does; a slash or a space is never allowed.
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def check_name(name):
    if not isinstance(name, str) or not NAME.match(name):
        die(f"bad model name {name!r}: use letters, digits, - _ and ., "
            f"starting with a letter or digit")
    return name


def toml_key(name):
    """A table name as TOML text: bare when it can be, quoted otherwise."""
    return name if re.fullmatch(r"[A-Za-z0-9_-]+", name) else toml_value(name)


def toml_path(path):
    """A filesystem path as a TOML basic string.

    Backslashes become forward slashes rather than escapes, because
    that is what the config asks people to write. A quote is legal in
    a POSIX filename and would otherwise close the string early.
    """
    return str(path).replace(chr(92), "/").replace(chr(34),
                                                   chr(92) + chr(34))


def toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    # Backslashes first, then quotes - reversing the order would escape
    # the backslash this line just added. A value like the JSON that
    # --chat-template-kwargs takes is nothing but quotes, and unescaped
    # they close the string early and leave the args array open.
    return chr(34) + (str(v).replace(chr(92), chr(92) * 2)
                            .replace(chr(34), chr(92) + chr(34))) + chr(34)


KEY_LINE = re.compile(r'\s*([A-Za-z0-9_-]+|"[^"]*")\s*=')


def _parses(text):
    try:
        tomllib.loads(text)
        return True
    except tomllib.TOMLDecodeError:
        return False


def _header(line):
    """The table a line opens - "[demo]", "[ demo ]  # note", '["demo"]' -
    or None. tomllib decides, so anything it accepts as a header is one."""
    s = line.strip()
    if not s.startswith("[") or s.startswith("[["):
        return None
    try:
        t = tomllib.loads(s + chr(10))
    except tomllib.TOMLDecodeError:
        return None
    if len(t) == 1:
        (key, value), = t.items()
        if value == {}:
            return key
    return ""                       # a header, but a dotted one: not ours


def _value_end(lines, i, eq):
    """Index of the last line of the value that starts after `eq` on
    line i. An array can run over several lines; tomllib says when it
    has closed."""
    for j in range(i, min(len(lines), i + 500)):
        text = chr(10).join([lines[i][eq:]] + lines[i + 1:j + 1])
        if _parses("v =" + text):
            return j
    return i


def _comment(value):
    """A trailing comment on a one-line value, kept when it is rewritten."""
    for hit in re.finditer("#", value):
        head = value[:hit.start()]
        if head.strip() and _parses("v =" + head):
            return "  " + value[hit.start():].strip()
    return ""


def plan_params(name, cfg, path=None, drop=()):
    """Return (old, proposed) text making [name] exactly `cfg`, without writing.

    Only key lines are touched, so comments, ordering and blank lines
    survive - which a dump-and-rewrite through tomllib would not. A key
    left out of `cfg` is removed: that is how the dashboard clears a
    field. To change some keys and keep the rest, use patch_params.

    A header or key the way people write them by hand - a comment after
    "[demo]", indentation, a quoted name, an args array over several
    lines - is found as tomllib reads it. The result is parsed before it
    replaces the file, so a write that would break the config is refused.
    """
    path = path or CONFIG
    text = path.read_text(encoding="utf-8")
    lines = text.split(chr(10))
    head = next((i for i, ln in enumerate(lines) if _header(ln) == name),
                None)
    if head is None:
        die(f"no [{name}] table in {path} to update")
    body, seen, insert_at = [], set(), 0
    i, tail = head + 1, len(lines)
    while i < len(lines):
        line = lines[i]
        if _header(line) is not None:
            tail = i
            break
        m = KEY_LINE.match(line)
        if not m:
            body.append(line)
            i += 1
            continue
        key = m.group(1).strip(chr(34))
        end = _value_end(lines, i, m.end())
        seen.add(key)
        if key in cfg and key not in drop:
            one_line = end == i
            note = _comment(line[m.end():]) if one_line else ""
            indent = line[:len(line) - len(line.lstrip())]
            body.append("%s%s = %s%s" % (indent, key, toml_value(cfg[key]),
                                         note))
            insert_at = len(body)
        i = end + 1
    for key in cfg:
        if key not in seen and key not in drop:
            body.insert(insert_at, "%s = %s" % (key, toml_value(cfg[key])))
            insert_at += 1
    lines[head + 1:tail] = body
    out = chr(10).join(lines)
    try:
        got = tomllib.loads(out).get(name, {})
    except tomllib.TOMLDecodeError as e:
        die(f"not writing {path}: the edit would not parse ({e})")
    want = {k for k in cfg if k not in drop}
    if set(got) != want:
        die(f"not writing {path}: [{name}] would have "
            f"{', '.join(sorted(set(got) ^ want))} wrong")
    return text, out


def plan_append_table(current, block, name, path=None):
    """Return (old, proposed) text, refusing an invalid new [name] table."""
    path = path or CONFIG
    out = current + block
    try:
        got = tomllib.loads(out)
    except tomllib.TOMLDecodeError as e:
        die(f"not writing {path}: the new [{name}] would not parse ({e})")
    if not isinstance(got.get(name), dict):
        die(f"not writing {path}: [{name}] would not read back as a table")
    return current, out


def plan_patch_params(name, changes, path=None, drop=()):
    """Plan `changes` in [name] and remove `drop`, keeping every other key.

    Return (old, proposed) text without writing either file.

    write_params makes the table exactly what it is given, so `mdl fit
    --apply` - which knows about the flags it tuned and nothing else -
    once deleted model, port, group and llama_server along the way.
    """
    path = path or CONFIG
    try:
        current = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        die(f"cannot read {path}: {e}")
    table = current.get(name)
    if not isinstance(table, dict):
        die(f"no [{name}] table in {path} to update")
    merged = {k: v for k, v in table.items() if k not in drop}
    merged.update({k: v for k, v in changes.items() if k not in drop})
    return plan_params(name, merged, path=path)


def write_params(name, cfg, path=None, drop=()):
    """Replace the table, including clearing fields omitted by the dashboard."""
    _, text = plan_params(name, cfg, path, drop)
    write_atomic(path or CONFIG, text, keep_backup=True)


def patch_params(name, changes, path=None, drop=()):
    """Write a validated patch while preserving unrelated keys."""
    _, text = plan_patch_params(name, changes, path, drop)
    write_atomic(path or CONFIG, text, keep_backup=True)


def append_table(current, block, name, path=None):
    """Write a validated new table."""
    _, text = plan_append_table(current, block, name, path)
    write_atomic(path or CONFIG, text, keep_backup=True)


def check_port(value, where=""):
    """A port as an int in 1..65535, or one line saying why not. A string
    port used to reach socket code as a TypeError, 70000 an OverflowError."""
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) \
            or not 1 <= value <= 65535:
        die(f"{where}port must be a number from 1 to 65535, not {value!r}")
    return value


# What each key may hold. ngl takes -1/"all"/"auto" the way llama.cpp does.
def _is_count(v, lo=0):
    return isinstance(v, int) and not isinstance(v, bool) and v >= lo


CHECKS = {
    "model": (lambda v: isinstance(v, str) and v.strip() != "",
              "a path to a .gguf"),
    "mmproj": (lambda v: isinstance(v, str) and v.strip() != "",
               "a path to a .gguf"),
    "llama_server": (lambda v: isinstance(v, str) and v.strip() != "",
                     "a non-empty string"),
    "ngl": (lambda v: _is_count(v, -1) or v in ("all", "auto"),
            "a layer count, or \"all\""),
    "n_cpu_moe": (lambda v: _is_count(v), "a count, 0 or more"),
    "ctx": (lambda v: _is_count(v), "a token count, 0 or more"),
    "parallel": (lambda v: _is_count(v, 1), "a count, 1 or more"),
    # a number, not a string: a string reached the socket code from
    # spawn() however it had been checked
    "port": (lambda v: _is_count(v, 1) and v <= 65535,
             "a number from 1 to 65535"),
    "flash_attn": (lambda v: isinstance(v, bool), "true or false"),
    "kv_type": (lambda v: isinstance(v, str) and re.fullmatch(
        r"[A-Za-z0-9_]+", v) is not None, "a cache type such as q8_0"),
    "group": (lambda v: isinstance(v, str), "a string"),
    "args": (lambda v: isinstance(v, list)
             and all(isinstance(a, str) for a in v), "a list of strings"),
}

# Flags that decide where the server is and what it serves. mdl reads
# model and port from their keys for its state, its pre-flight and its
# health check; the same flag in args would win at llama-server and send
# all three looking in the wrong place.
OWNED = {"-m": "model", "--model": "model", "--port": "port",
         "-mu": "model", "--model-url": "model",
         "-hf": "model", "-hfr": "model", "--hf-repo": "model"}


def check_cfg(name, cfg):
    unknown = sorted(set(cfg) - KNOWN)
    if unknown:
        die(f"model '{name}': unknown key(s): {', '.join(unknown)}")
    if "model" not in cfg:
        die(f"model '{name}': missing required key 'model'")
    for key, (ok, what) in CHECKS.items():
        if key in cfg and not ok(cfg[key]):
            die(f"model '{name}': '{key}' must be {what}")
    for a in cfg.get("args", []):
        flag = a.split("=", 1)[0]
        if flag in OWNED:
            die(f"model '{name}': '{flag}' in args would override its "
                f"'{OWNED[flag]}' key, which mdl reads; set the key instead")


def build_argv(name, cfg, binary):
    check_name(name)
    check_cfg(name, cfg)
    # A model that needs its own build - a fork with a quant type upstream
    # does not load yet - names it, and that beats MDL_LLAMA_SERVER: the
    # environment says what to use by default, the model says what it needs.
    if "llama_server" in cfg:
        binary = cfg["llama_server"]
    argv = [binary, "-m", str(cfg["model"])]
    if "mmproj" in cfg:          # the vision half of a multimodal model
        argv += ["--mmproj", str(cfg["mmproj"])]
    for key, flag in SIMPLE:
        if key in cfg:
            argv += [flag, str(cfg[key])]
    if "flash_attn" in cfg:     # false is "off", not "whatever the default is"
        argv += ["-fa", "on" if cfg["flash_attn"] else "off"]
    if "kv_type" in cfg:
        kv = str(cfg["kv_type"])
        argv += ["--cache-type-k", kv, "--cache-type-v", kv]
    return argv + list(cfg.get("args", []))


def ready_timeout():
    try:
        return float(CONFIG_DATA.get("ready_timeout", READY_TIMEOUT))
    except (TypeError, ValueError):
        return READY_TIMEOUT


def server_ready(port):
    """True once the server answers /health. Log wording changes between
    llama.cpp builds; this contract does not."""
    try:
        with urllib.request.urlopen(
                "http://127.0.0.1:%d/health" % port, timeout=1) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def proc_started(pid):
    """OS process creation time, or None where we cannot cheaply get it.

    Guards against pid reuse: a recycled pid is alive but was created at
    a different instant, so the state file is stale even though the pid
    looks fine. macOS has no /proc, so it opts out rather than guess.
    """
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            times = [ctypes.c_ulonglong() for _ in range(4)]
            if not kernel32.GetProcessTimes(
                    handle, *[ctypes.byref(x) for x in times]):
                return None
            return times[0].value            # creation time
        finally:
            kernel32.CloseHandle(handle)
    try:                                     # Linux: starttime, field 22
        stat = Path("/proc/%d/stat" % pid).read_text()
        return int(stat.rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def port_busy(port):
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


if os.name == "nt":
    def alive(pid):
        """True if pid is a live process. os.kill(pid, 0) lies on Windows."""
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
else:
    def alive(pid):
        """True if pid is a live process, zombies excluded.

        A server we started ourselves is our child, so once it dies it
        stays a zombie until someone reaps it - and a zombie still
        answers kill(pid, 0). That only bites the in-process callers,
        mdl ui above all, which would show a stopped server as running
        for as long as it stayed open.
        """
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                return False
        except ChildProcessError:
            pass                         # not ours; nothing to reap
        except OSError:
            pass
        try:
            os.kill(pid, 0)
        except PermissionError:
            return True
        except OSError:
            return False
        return True


def run_dir():
    """Where the per-server state files live.

    A function, not a constant, so it follows STATE_DIR - which the tests
    point at a temp directory.
    """
    return STATE_DIR / "run"


def state_path(name):
    return run_dir() / f"{check_name(name)}.json"


def live_state(path):
    """One server's state, or None - clearing the file if it is stale."""
    try:
        state = json.loads(path.read_text())
        pid = state["pid"]
    except (OSError, ValueError, KeyError, TypeError):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(pid, int) or not running(state):
        path.unlink(missing_ok=True)
        return None
    return state


def running(state):
    """True while anything this server started is still alive.

    The pid we launched is only the start of it: if llama_server is a
    wrapper, the wrapper can exit and leave the real server behind, still
    holding the port and the GPU. So a dead leader is not the end - its
    process group (POSIX) or its descendants (Windows) are checked too.
    """
    pid = state["pid"]
    if alive(pid):
        born = state.get("born")
        # a live pid created at another instant is a recycled pid: the
        # server is gone, and whatever group that pid leads is not ours
        return born is None or proc_started(pid) in (None, born)
    return bool(survivors(state))


def survivors(state):
    """What is left of a server whose leader has exited, or [].

    POSIX: spawn() made the leader a session leader, so the group id is
    its pid, and the kernel hands out no pid that is still in use as a
    group id - a group with members is the one we started. Windows keeps
    a dead parent's pid on its children, so they are found by parent pid,
    and a child created before the leader was is a recycled pid, not ours.
    """
    if os.name == "nt":
        return _descendants_nt(state["pid"], state.get("born"))
    pgid = state.get("pgid")
    if not isinstance(pgid, int) or pgid <= 1:
        return []
    try:
        os.killpg(pgid, 0)
    except PermissionError:
        return [pgid]
    except OSError:
        return []
    return [pgid]


def _descendants_nt(pid, born):
    """Live descendants of pid created after it was, via a toolhelp
    snapshot. Empty when we cannot tell - never someone else's."""
    if born is None:
        return []

    class Entry(ctypes.Structure):
        _fields_ = [("size", ctypes.c_ulong), ("usage", ctypes.c_ulong),
                    ("pid", ctypes.c_ulong), ("heap", ctypes.c_size_t),
                    ("module", ctypes.c_ulong), ("threads", ctypes.c_ulong),
                    ("ppid", ctypes.c_ulong), ("base", ctypes.c_long),
                    ("flags", ctypes.c_ulong), ("exe", ctypes.c_char * 260)]

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    snap = kernel32.CreateToolhelp32Snapshot(0x2, 0)   # TH32CS_SNAPPROCESS
    if not snap or snap == ctypes.c_void_p(-1).value:
        return []
    kids = {}
    try:
        e = Entry()
        e.size = ctypes.sizeof(Entry)
        ok = kernel32.Process32First(ctypes.c_void_p(snap), ctypes.byref(e))
        while ok:
            kids.setdefault(e.ppid, []).append(e.pid)
            ok = kernel32.Process32Next(ctypes.c_void_p(snap), ctypes.byref(e))
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(snap))
    out, todo = [], [pid]
    while todo:
        for kid in kids.get(todo.pop(), []):
            t = proc_started(kid)
            if kid not in out and kid != pid and t is not None and t >= born:
                out.append(kid)
                todo.append(kid)
    return out


def read_states(read_only=False):
    """Every server we started that is still alive, keyed by name.

    One file per server rather than one file listing them: two `mdl run`
    calls at the same moment would otherwise read, modify and write the
    same file, and one of them would lose.

    read_only includes stale records for diagnosis, without removing
    files. Its caller must check liveness itself.
    """
    out = {}
    try:
        paths = sorted(run_dir().glob("*.json"))
    except OSError:
        return out
    for path in paths:
        if read_only:
            # Diagnosis must leave stale files in place.
            try:
                state = json.loads(path.read_text())
                if not isinstance(state, dict) or not isinstance(
                        state.get("name", path.stem), str):
                    continue
            except (OSError, ValueError):
                continue
        else:
            state = live_state(path)
        if state:
            name = state.get("name", path.stem)
            if not read_only or name not in out:
                out[name] = state
    return out


def read_state(name=None):
    """One server's state.

    Given a name, that server or None. Given nothing, the only running
    server - which is what keeps every caller written when there could
    only be one working unchanged - or None if there are none. Several
    running and no name is genuinely ambiguous, so it says so instead of
    picking one.
    """
    states = read_states()
    if name is not None:
        return states.get(name)
    if not states:
        return None
    if len(states) > 1:
        die("several servers are running (%s); name the one you mean"
            % ", ".join(sorted(states)))
    return next(iter(states.values()))


def uptime(seconds):
    s = max(0, int(seconds))
    h, m, s = s // 3600, s // 60 % 60, s % 60
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _drain(fh, pending):
    """Echo whole lines from the log, returning the partial remainder."""
    pending += fh.read()
    while chr(10) in pending:
        line, pending = pending.split(chr(10), 1)
        print(line, flush=True)
    return pending


def tail_until_ready(proc, log, name, port):
    """Echo the log until /health answers. Fatal if it dies or times out."""
    limit = ready_timeout()
    deadline = time.monotonic() + limit
    with open(log, "r", errors="replace") as fh:
        pending = ""
        while True:
            pending = _drain(fh, pending)
            if server_ready(port):
                time.sleep(0.2)          # let the last writes land, then show them
                pending = _drain(fh, pending)
                if pending:
                    print(pending, flush=True)
                return
            if proc.poll() is not None:
                pending = _drain(fh, pending)
                if pending:
                    print(pending, flush=True)
                state_path(name).unlink(missing_ok=True)
                die(f"{name} exited with status {proc.returncode} "
                    f"during startup; see {log}")
            if time.monotonic() > deadline:
                die(f"{name} not ready after {limit:g}s; it may still be "
                    f"loading - see {log}, or run 'mdl stop {name}'")
            time.sleep(0.2)


def terminate(pid, sig, state=None):
    """Signal the whole process tree, not just the pid we launched.

    If llama_server is a wrapper script - setting LD_LIBRARY_PATH, say -
    the recorded pid is the wrapper and the real server is its child.
    Signalling only the wrapper orphans the server and leaves the port
    held. spawn() puts it in its own session, so the group is the tree;
    given the state, the group recorded there is signalled even once the
    wrapper has gone, which getpgid() on a dead pid cannot find.
    """
    if os.name == "nt":
        pids = [pid] + (survivors(state) if state else [])
        for p in pids:
            if p == pid or alive(p):
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(p)],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               creationflags=getattr(subprocess,
                                                     "CREATE_NO_WINDOW", 0))
        return
    pgid = (state or {}).get("pgid")
    if isinstance(pgid, int) and pgid > 1:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass                         # nothing left in it
        return
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        os.kill(pid, sig)                # not a group leader after all


def _sibling_tmp(path, tag=""):
    """A temp name beside `path` that no other writer has. The pid alone
    was shared by threads: two saving one file at once each removed the
    other's temp file, and the second rename failed."""
    return path.with_name("%s%s.tmp%d-%s" % (path.name, tag, os.getpid(),
                                              os.urandom(4).hex()))


def _replace(src, dst):
    """os.replace, waiting out Windows refusing a rename onto a file
    another rename is replacing that instant (Access is denied)."""
    for wait in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if os.name == "nt" else ():
        try:
            return os.replace(src, dst)
        except PermissionError:
            time.sleep(wait)
    return os.replace(src, dst)


def write_atomic(path, text, keep_backup=False):
    """Replace a file in one step, never leaving it half written.

    models.toml is hand-edited and lives in nobody's git, so a write cut
    short by a crash, a full disk or a Ctrl-C has to leave the old file
    untouched rather than truncated. Writing a sibling temp file and
    renaming it over gives that: the rename is atomic, so a reader sees
    either the whole old file or the whole new one.
    """
    if keep_backup and path.exists():
        backups = config_backups(path)
        for i in range(4, 0, -1):
            if backups[i - 1].exists():
                os.replace(backups[i - 1], backups[i])
            else:
                backups[i].unlink(missing_ok=True)
        # Never move the current config: a failed rotation leaves it intact.
        tmp_backup = _sibling_tmp(path, ".bak")
        try:
            shutil.copy2(path, tmp_backup)
            os.replace(tmp_backup, backups[0])
        finally:
            tmp_backup.unlink(missing_ok=True)
    tmp = _sibling_tmp(path)
    try:
        # "x": a name another writer already holds is an error, never shared
        mode = "xb" if isinstance(text, bytes) else "x"
        kwargs = {} if isinstance(text, bytes) else {"encoding": "utf-8"}
        with open(tmp, mode, **kwargs) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())    # the rename is no use if the data is not down
        _replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # nothing half-written left lying about


def rotate(log, keep=KEEP_LOGS):
    """Shuffle <name>.log along to .1, .2, ... so a crash stays readable."""
    if not log.exists():
        return
    for i in range(keep - 1, 0, -1):
        older = log.with_name(log.name + ".%d" % (i + 1))
        current = log.with_name(log.name + ".%d" % i)
        if current.exists():
            current.replace(older)
    for _ in range(10):                  # a just-killed writer can linger
        try:
            return log.replace(log.with_name(log.name + ".1"))
        except OSError:
            time.sleep(0.05)
    print(f"mdl: cannot rotate {log}; overwriting it", file=sys.stderr)


class file_lock:
    """One holder at a time, across processes: a file made with O_EXCL
    holding the holder's pid. One left by a process that died is taken
    over once its pid is gone. `tries` tenths of a second, then `busy`."""

    def __init__(self, path, busy, tries=50):
        self.path, self.busy, self.tries = Path(path), busy, tries

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(self.tries):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    holder = int(self.path.read_text() or 0)
                except (OSError, ValueError):
                    holder = 0
                if holder and not alive(holder) and self._take_over(holder):
                    continue                    # its holder died; try again
                time.sleep(0.1)
                continue
            with os.fdopen(fd, "w") as fh:
                fh.write(str(os.getpid()))
            return self
        die(self.busy)

    def _take_over(self, dead):
        """Remove a lock left by `dead` - only while it is still that one.

        Two waiters could both see the dead pid; the first removed the
        lock and took a new one, and the second then removed that. So a
        takeover is itself taken, with O_EXCL on a guard file, and the
        lock is read again under it. A guard a crash left behind goes
        once it is older than any takeover takes.
        """
        guard = self.path.with_name(self.path.name + ".takeover")
        try:
            fd = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - guard.stat().st_mtime > 10:
                    guard.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        os.close(fd)
        try:
            try:
                now = int(self.path.read_text() or 0)
            except FileNotFoundError:
                return True                     # gone already: go and take it
            except (OSError, ValueError):
                return False
            if now != dead:
                return False                    # someone took it meanwhile
            self.path.unlink(missing_ok=True)
            return True
        finally:
            guard.unlink(missing_ok=True)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        return False


class launch_lock(file_lock):
    """One launch of <name> at a time. Two `mdl run demo` at the same
    moment both saw nothing running, both started a server, and the
    second state file hid the first server for good."""

    def __init__(self, name):
        super().__init__(run_dir() / f"{check_name(name)}.lock",
                         f"'{name}' is being started by another mdl; "
                         f"try again in a moment")


class port_lock(file_lock):
    """One launch onto <port> at a time. Two models sharing a port, started
    at once, both found it free - a server takes seconds to bind - and the
    second died at load with the port taken. Named with a leading _,
    which no model name can have, so it never meets a launch_lock."""

    def __init__(self, port):
        super().__init__(run_dir() / f"_port-{int(port)}.lock",
                         f"another mdl is starting a server on port {port}; "
                         f"try again in a moment")


def spawn(name, models, binary, port=None):
    """Launch <name> detached, write its state file, return (proc, log, port).

    Shared by the CLI and the TUI so there is one way to start a server.
    A port here overrides the config, for `mdl run <name> --port N`; the
    value that reaches the state file is the one we actually launched
    with, so ps, the readiness probe and the next pre-flight agree.

    Under a per-name lock, with the running check made again inside it,
    and a server whose state cannot be written is stopped rather than
    left running where nothing can find it.
    """
    with launch_lock(name):
        running = read_state(name)
        if running:
            die(f"'{name}' is already running (pid {running['pid']}, "
                f"port {running['port']}); run 'mdl stop {name}' first")
        launched = _spawn(name, models, binary, port)
    _record()
    return launched


def _record():
    """Book what the servers serve while any is up, for mdl ui's history:
    one quiet recorder process, started here if none is. Never fails a
    launch."""
    try:
        from mdl_web import usage
    except ImportError:
        return
    usage.ensure()


def _spawn(name, models, binary, port):
    cfg = dict(models[name])
    if port is not None:
        cfg["port"] = port
    argv = build_argv(name, cfg, binary)
    binary = argv[0]                    # the model's own, if it names one
    port = cfg.get("port", DEFAULT_PORT)
    if not shutil.which(binary) and not Path(binary).is_file():
        die(f"llama-server not found: {binary}")
    if not Path(models[name]["model"]).is_file():
        die(f"model file not found: {models[name]['model']}")
    with port_lock(port):
        return _launch(name, argv, binary, port)


def _launch(name, argv, binary, port):
    # a server we started but that is still loading holds the port without
    # listening on it yet: its state file says so before the socket does
    owner = next((s["name"] for s in read_states().values()
                  if s.get("port") == port and s.get("name") != name), None)
    if owner:
        die(f"port {port} is already serving '{owner}'; give {name} its "
            f"own port, or run 'mdl stop {owner}'")
    if port_busy(port):
        die(f"port {port} is already in use")
    try:
        run_dir().mkdir(parents=True, exist_ok=True)
        log = STATE_DIR / f"{check_name(name)}.log"
        rotate(log)
        handle = open(log, "wb")
    except OSError as e:
        die(f"cannot open log in {STATE_DIR}: {e}")
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=handle,
                                stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as e:
        die(f"cannot start {binary}: {e}")
    finally:
        handle.close()
    state = {"name": name, "pid": proc.pid, "port": port,
             "started": time.time(), "log": str(log),
             "born": proc_started(proc.pid),
             # start_new_session: the pid is the group id, and the group
             # outlives a wrapper that exits and leaves the server behind
             "pgid": proc.pid if os.name != "nt" else None,
             # what actually ran, for eval to record and check against
             # the config it reads later
             "argv": argv,
             # which build: a rebuilt llama-server at the same path is a
             # different runtime, and stat is all a launch can afford
             "binary": file_id(shutil.which(binary) or binary),
             "model_ids": model_ids(argv)}
    try:
        write_atomic(state_path(name), json.dumps(state))
    except OSError as e:
        terminate(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM), state)
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            pass
        die(f"cannot record {name}'s state in {run_dir()} ({e}); "
            f"stopped it rather than leave it untracked")
    return proc, log, port


def cmd_run(args):
    port = None
    if len(args) == 3 and args[1] == "--port":
        port = check_port(args[2])
        args = args[:1]
    if len(args) != 1:
        die("usage: mdl run <name> [--port N]")
    name = args[0]
    models, binary = load_config()
    if name not in models:
        die(f"no model named '{name}' in {CONFIG}")
    # spawn() says if it is already running, under the launch lock
    proc, log, port = spawn(name, models, binary, port)
    print(f"starting {name} (pid {proc.pid}), log {log}", flush=True)
    tail_until_ready(proc, log, name, port)
    print(f"ready: {name} on http://127.0.0.1:{port} (pid {proc.pid})")
    learn_from_log(name, models[name], binary, log)


def learn_from_log(name, cfg, binary, log):
    """Passive calibration: whatever buffer sizes the load log printed
    are booked against this config, so `mdl fit` improves from normal
    use. Never allowed to fail a launch."""
    try:
        from mdl_fit import calib
        calib.passive(name, build_argv(name, cfg, binary), log)
    except Exception:            # noqa: BLE001
        pass


def cmd_fit(args):
    from mdl_fit import cli
    try:
        cli.main(args)
    except OSError as e:           # a file or the config, not a bug
        die("cannot fit: %s" % " ".join(str(e).splitlines()))


def cmd_eval(args):
    from mdl_fit import evalrun
    evalrun.main(args)


def cmd_catalog(args):
    from mdl_fit import catalog
    catalog.main(args)


def cmd_find(args):
    from mdl_fit import find
    find.main(args)


def cmd_pull(args):
    from mdl_fit import pull
    pull.main(args)


def cmd_manifest(args):
    from mdl_fit import manifest
    manifest.main(args)


def cmd_lab(args):
    from mdl_fit import lab
    lab.main(args)


def file_id(path):
    """{path, size, mtime} for a file, or {path, missing}."""
    try:
        st = os.stat(path)
    except (OSError, TypeError, ValueError):
        return {"path": str(path), "missing": True}
    return {"path": str(Path(path).resolve()), "size": st.st_size,
            "mtime": int(st.st_mtime), "mtime_ns": st.st_mtime_ns}


def model_ids(argv):
    """file_id of every model file argv loads, taken at launch: what
    manifest compares later to tell a file replaced under a running
    server, which still holds the one it opened."""
    from mdl_fit.manifest import shards
    paths = [argv[i + 1] for i, a in enumerate(argv[:-1])
             if a in ("-m", "--model", "-mm", "--mmproj")]
    return [file_id(s) for p in paths for s in shards(p)]


def stop_one(name, state):
    """SIGTERM, wait, SIGKILL. True if it is gone afterwards.

    Gone means everything it started, not the pid we launched: a wrapper
    that exits on SIGTERM while its server ignores it used to count as a
    clean stop, and the server kept the port with nothing listed in ps.
    """
    pid = state["pid"]
    try:
        terminate(pid, signal.SIGTERM, state)
    except OSError as e:
        print(f"mdl: cannot signal {name} (pid {pid}): {e}", file=sys.stderr)
        return False
    for _ in range(100):
        if not running(state):
            break
        time.sleep(0.1)
    else:
        try:
            terminate(pid, getattr(signal, "SIGKILL", signal.SIGTERM), state)
        except OSError:
            pass
        for _ in range(30):
            if not running(state):
                break
            time.sleep(0.1)
    if running(state):
        print(f"mdl: {name} (pid {pid}) would not die", file=sys.stderr)
        return False
    state_path(name).unlink(missing_ok=True)
    port = state.get("port")
    for _ in range(30):
        if not isinstance(port, int) or not port_busy(port):
            break
        time.sleep(0.1)
    else:
        print(f"mdl: {name} (pid {pid}) is gone, but port {port} is still "
              f"held - something it started may have outlived it",
              file=sys.stderr)
        return False
    print(f"stopped {name} (pid {pid})")
    return True


def cmd_stop(args):
    """mdl stop [<name>|--all]

    Bare `stop` still means "the one that is running", so it keeps doing
    what it always did. It only asks which when there is a real choice.
    """
    states = read_states()
    if args == ["--all"]:
        targets = sorted(states)
    elif len(args) == 1 and not args[0].startswith("-"):
        if args[0] not in states:
            die(f"'{args[0]}' is not running")
        targets = [args[0]]
    elif args:
        die("usage: mdl stop [<name>|--all]")
    elif len(states) > 1:
        die("several servers are running (%s); name one, or 'mdl stop --all'"
            % ", ".join(sorted(states)))
    else:
        targets = sorted(states)
    if not targets:
        print("nothing running")
        return
    # Try every one of them, say what happened to each, and fail if any
    # survived - stopping two of three is not success.
    failed = [n for n in targets if not stop_one(n, states[n])]
    if failed:
        die(f"{len(failed)} of {len(targets)} did not stop: " + ", ".join(failed))


def cmd_ps(args):
    as_json = args == ["--json"]
    if args and not as_json:
        die("usage: mdl ps [--json]")
    rows = [dict(state, uptime=round(time.time() - state.get("started", 0)))
            for _, state in sorted(read_states().items())]
    if as_json:
        print(json.dumps(rows))          # always a list, [] when idle
        return
    if not rows:
        print("nothing running")
        return
    width = max(len(r["name"]) for r in rows)
    for r in rows:
        print(f"{r['name'].ljust(width)}  pid {r['pid']}  port {r['port']}  "
              f"up {uptime(r['uptime'])}")


def cmd_list(args):
    if args:
        die("usage: mdl list")
    models, _ = load_config()
    if not models:
        die(f"no models defined in {CONFIG}")
    width = max(len(n) for n in models)
    for name in sorted(models):
        print(f"{name.ljust(width)}  {models[name].get('model', '(no model path)')}")


def gguf_layers(path, window=32 << 20):
    """Highest blk.N index in the tensor table, i.e. the layer count.

    A full GGUF parser would be a hundred lines; the tensor names sit
    near the front of the file and are all we need.
    """
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return None
            head = fh.read(window)
    except OSError:
        return None
    blocks = re.findall(rb"blk\.(\d+)\.", head)
    return max(int(b) for b in blocks) + 1 if blocks else None


def find_mmproj(model):
    """The vision projector sitting beside a model, if there is exactly one.

    Multimodal repos ship it as mmproj-<something>.gguf next to the
    weights, and without it llama-server loads the text half and says
    nothing about the missing eyes. Two candidates is a choice, not a
    default, so it declines to guess.
    """
    try:
        found = sorted(p for p in model.parent.glob("*.gguf")
                       if p.name.lower().startswith("mmproj"))
    except OSError:
        return None
    return found[0] if len(found) == 1 else None


def human_size(nbytes):
    for unit, div in (("T", 1 << 40), ("G", 1 << 30),
                      ("M", 1 << 20), ("K", 1 << 10)):
        if nbytes >= div:
            return f"{nbytes / div:.1f}{unit}"
    return f"{nbytes}B"


def config_backups(path):
    """Newest first, with five slots shared by every config writer."""
    return [path.with_name(path.name + ".bak" + (".%d" % i if i else ""))
            for i in range(5)]


def config_history(undo=False):
    """Undo swaps current and .bak; older slots stay in place.

    The replaced current is the next undo target, so two undos are an
    identity, not a walk backwards through history. Ordinary writes push
    that target into .bak.1 and retain at most five previous versions.
    """
    backups = config_backups(CONFIG)
    try:
        if undo:
            backup = backups[0]
            if not backup.is_file():
                die("no config backup to undo")
            previous = backup.read_bytes()
            tomllib.loads(previous.decode("utf-8"))
            stamp = time.ctime(backup.stat().st_mtime)
            current = CONFIG.read_bytes()
            # Save the displaced bytes before replacing the current file.
            write_atomic(backup, current)
            try:
                write_atomic(CONFIG, previous)
            except (OSError, KeyboardInterrupt):
                write_atomic(backup, previous)
                raise
            print("restored config from %s; %d backups remain" % (
                stamp, sum(p.is_file() for p in backups)))
            return
        current = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
        for i, backup in enumerate(backups):
            if not backup.is_file():
                continue
            stat = backup.stat()
            try:
                old = tomllib.loads(backup.read_text(encoding="utf-8"))
                names = sorted(k for k in current.keys() | old.keys()
                               if (isinstance(current.get(k), dict)
                                   or isinstance(old.get(k), dict))
                               and current.get(k) != old.get(k))
                detail = ", ".join(names) or "(none)"
            except (ValueError, UnicodeError):
                detail = "(invalid TOML)"
            print("%d  %s  %d bytes  differs: %s" % (
                i, time.ctime(stat.st_mtime), stat.st_size, detail))
    except (OSError, ValueError, UnicodeError) as e:
        die("cannot %s config: %s" % (
            "undo" if undo else "read", " ".join(str(e).splitlines())))


def cmd_config(args):
    if args in (["--undo"], ["--history"]):
        return config_history(undo=args == ["--undo"])
    if args == ["--path"]:
        print(CONFIG.resolve())
        return
    if args:
        die("usage: mdl config [--path|--undo|--history]")
    if not CONFIG.is_file():
        die("No config found. Run 'mdl init' first.")
    editor = (os.environ.get("VISUAL") or os.environ.get("EDITOR")
              or ("notepad" if os.name == "nt" else "vi"))
    try:
        # Windows paths need their backslashes preserved. Pass argv directly;
        # editor settings are commands with arguments, not shell programs.
        argv = shlex.split(editor, posix=os.name != "nt")
        if os.name == "nt":
            argv = [a[1:-1] if a.startswith('"') and a.endswith('"') else a
                    for a in argv]
        if not argv:
            die("VISUAL or EDITOR must name an editor")
        result = subprocess.run([*argv, str(CONFIG.resolve())])
    except (OSError, ValueError) as e:
        die(f"cannot open config editor: {e}")
    if result.returncode:
        die(f"config editor exited with status {result.returncode}")


def cmd_init(args):
    if args:
        die("usage: mdl init")
    if CONFIG.exists():
        die(f"config already exists at {CONFIG}")
    found = shutil.which(DEFAULT_BIN) or "/path/to/llama-server"
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        write_atomic(CONFIG, STARTER % found.replace(chr(92), "/"))
    except OSError as e:
        die(f"cannot write {CONFIG}: {e}")
    print(f"wrote {CONFIG}")
    if not shutil.which(DEFAULT_BIN):
        print("set llama_server in it: llama-server is not on your PATH")
    print("edit it, then run: mdl list")


def cmd_add(args):
    """mdl add <model.gguf> [name] [port]"""
    if not args or len(args) > 3:
        die("usage: mdl add <model.gguf> [name] [port]")
    path = Path(args[0]).expanduser()
    if not path.is_file():
        die(f"no such file: {path}")
    # absolute: a relative path only works from the directory it was
    # added in, and `mdl run` is run from anywhere
    path = Path(os.path.abspath(path))   # not resolve(): keep their links
    if len(args) > 1:
        name = check_name(args[1])
    else:                       # Foo-Bar-Q4_K_M.gguf -> foo-bar
        stem = re.sub(r"[-_.]?(q\d+[_0-9a-z]*|f16|f32|bf16)$", "", path.stem,
                      flags=re.I)
        name = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-").lower()
        name = name[:64].strip("-") or "model"
    port = check_port(args[2]) if len(args) > 2 else DEFAULT_PORT
    models, _ = load_config()
    if name in models:
        die(f"{name} is already in {CONFIG}; pick another name")
    layers = gguf_layers(path)
    mmproj = find_mmproj(path)
    block = (
        f"{chr(10)}[{toml_key(name)}]{chr(10)}"
        f'model = "{toml_path(path)}"{chr(10)}'
        + (f'mmproj = "{toml_path(mmproj)}"{chr(10)}' if mmproj else "")
        + f"ngl = 99{chr(10)}ctx = 8192{chr(10)}flash_attn = true{chr(10)}"
        f'kv_type = "q8_0"{chr(10)}parallel = 1{chr(10)}port = {port}{chr(10)}'
        f'args = ["--metrics"]{chr(10)}')
    try:
        current = CONFIG.read_text(encoding="utf-8")
        append_table(current, block, name)
    except OSError as e:
        die(f"cannot write {CONFIG}: {e}")
    print(f"added [{name}] to {CONFIG}")
    detail = human_size(path.stat().st_size)
    if layers:
        detail += f", {layers} layers"
    print(f"  {path.name} ({detail})")
    if mmproj:
        print(f"  vision: {mmproj.name}")
    print(f"  run it with: mdl run {name}")


def cmd_check(args):
    """Validate every model in the config without launching anything."""
    if args:
        die("usage: mdl check")
    models, binary = load_config()
    if not models:
        die(f"no models defined in {CONFIG}")
    problems = 0
    ports = {}
    if not shutil.which(binary) and not Path(binary).is_file():
        print(f"llama_server: not found: {binary}")
        problems += 1
    for name in sorted(models):
        notes = []
        cfg = models[name]
        try:
            argv = build_argv(name, cfg, binary)
            if "llama_server" in cfg and not (shutil.which(argv[0])
                                                or Path(argv[0]).is_file()):
                notes.append(f"llama_server: not found: {argv[0]}")
        except MdlError as e:
            notes.append(str(e).split(': ', 1)[-1])
        raw = str(cfg.get("model", ""))
        blank = raw == PLACEHOLDER   # straight out of mdl init
        model = Path(raw)            # compare raw: Path flips the slashes
        if not model.is_file() and not blank:
            notes.append("model file not found")
        else:
            layers = gguf_layers(model)
            if layers and isinstance(cfg.get("ngl"), int) and 0 < cfg["ngl"] < layers:
                notes.append(f"ngl {cfg['ngl']} < {layers} layers, partial offload")
        if "mmproj" in cfg and not Path(str(cfg["mmproj"])).is_file():
            notes.append("mmproj file not found")
        problems += len(notes)
        ports.setdefault(cfg.get("port", DEFAULT_PORT), []).append(name)
        if blank:            # a to-do, so it must not fail the check
            notes.append("not filled in yet; edit it or delete it")
        status = "ok" if not notes else "; ".join(notes)
        print(f"{name.ljust(max(len(n) for n in models))}  {status}")
    for port, sharing in sorted(ports.items()):
        if len(sharing) > 1:      # legal; only one of them can be up at once
            print("note: %s share port %d; only one at a time"
                  % (", ".join(sharing), port))
    if problems:
        die(f"{problems} problem(s) found")


def _doctor_note(notes, level, check, message):
    notes.append({"level": level, "check": check,
                  "message": " ".join(str(message).split())})


def _doctor_probe(binary, flag):
    try:
        result = subprocess.run(
            [binary, flag], stdin=subprocess.DEVNULL, capture_output=True,
            text=True, errors="replace", timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            return None, "exit status %d" % result.returncode
        return result.stdout + "\n" + result.stderr, None
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        return None, str(e)


def _doctor_model(name, cfg, binary, states, help_cache):
    notes = []

    def note(level, check, message):
        _doctor_note(notes, level, check, message)

    argv = None
    try:
        argv = build_argv(name, cfg, binary)
        note("ok", "config", "config valid")
    except MdlError as e:
        note("fail", "config", e)
    if argv is not None:
        binary = argv[0]
        try:
            found = shutil.which(binary) or Path(binary).is_file()
        except (OSError, ValueError):
            found = False
        if not found:
            note("fail", "binary", "llama-server not found: %s" % binary)
        else:
            note("ok", "binary", "llama-server found: %s" % binary)
            text, error = _doctor_probe(binary, "--version")
            if error:
                note("warn", "binary", "cannot run --version: %s" % error)
            else:
                build = re.search(r"version: [^\r\n]+", text)
                note("ok" if build else "warn", "binary",
                     build.group() if build else "no build line in --version")
            if binary not in help_cache:
                help_cache[binary] = _doctor_probe(binary, "--help")
            text, error = help_cache[binary]
            if error:
                note("warn", "flags", "flags skipped; cannot run --help: %s"
                     % error)
            else:
                flags = dict.fromkeys(
                    a.split("=", 1)[0] for a in argv[1:] if a.startswith("-")
                    and not re.fullmatch(r"-[\d.]+", a))
                missing = [f for f in flags if not re.search(
                    r"(?<![\w-])%s(?![\w-])" % re.escape(f), text)]
                for flag in missing:
                    note("warn", "flags", "flag not listed in --help: %s" % flag)
                if not missing:
                    note("ok", "flags", "all flags listed in --help")
    for key in ("model", "mmproj"):
        if key not in cfg or not isinstance(cfg[key], str):
            continue                    # check_cfg explains missing keys and types
        raw = cfg[key]
        if raw == PLACEHOLDER:
            note("warn", key, "%s not filled in yet; edit it or delete it" % key)
            continue
        try:
            path = Path(raw)
            if not path.is_file():
                note("fail", key, "%s file not found: %s" % (key, raw))
                continue
            with path.open("rb") as fh:
                valid = fh.read(4) == b"GGUF"
            note("ok" if valid else "fail", key,
                 "%s %s: %s" % (key, "GGUF header valid" if valid
                                 else "file is not GGUF", raw))
        except (OSError, ValueError) as e:
            note("fail", key, "cannot read %s: %s" % (key, e))
    try:
        state = states.get(name)
        port = check_port(state.get("port") if state else
                          cfg.get("port", DEFAULT_PORT))
        if state:
            pid = state.get("pid")
            if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
                note("warn", "runtime", "state has an invalid launcher pid")
            else:
                if not running(state):
                    note("warn", "runtime", "stale state: the server is gone "
                         "(the next mdl ps clears it)")
                elif not alive(pid):
                    note("warn", "runtime", "launcher pid is gone, but the "
                         "server it started survives")
                ready = server_ready(port)
                note("ok" if ready else "warn", "runtime",
                     "port %d: %s" % (port, "/health answers" if ready
                                       else "/health does not answer"))
        elif port_busy(port):
            note("warn", "runtime", "something else is listening on port %d"
                 % port)
        else:
            note("ok", "runtime", "not running; port %d is free" % port)
    except (MdlError, OSError, ValueError, TypeError, OverflowError) as e:
        note("warn", "runtime", "cannot check runtime: %s" % e)
    return notes


def _doctor_print(report, as_json):
    findings = report["global"] + [f for notes in report["models"].values()
                                   for f in notes]
    fails = sum(f["level"] == "fail" for f in findings)
    warns = sum(f["level"] == "warn" for f in findings)
    summary = "%d fail, %d warn" % (fails, warns)
    if as_json:
        print(json.dumps(report, indent=1))
    else:
        for name, notes in [("environment", report["global"]),
                            *report["models"].items()]:
            print(" ".join(name.split()) + ":")
            for f in notes:
                message = f["message"]
                if len(message) > 90:
                    message = message[:87] + "..."
                print("  %-4s  %s" % (f["level"], message))
        print(summary)
    if fails:
        die(summary)


def cmd_doctor(args):
    """Diagnose presets without launching servers or cleaning up their state."""
    as_json = "--json" in args
    rest = [a for a in args if a != "--json"]
    if args.count("--json") > 1 or len(rest) > 1 or any(
            a.startswith("-") for a in rest):
        die("usage: mdl doctor [--json] [name]")
    report = {"global": [], "models": {}}
    notes = report["global"]
    try:
        models, binary = load_config()
    except (MdlError, UnicodeError) as e:
        _doctor_note(notes, "fail", "config", e)
        _doctor_print(report, as_json)
        return
    if rest and rest[0] not in models:
        die("no model named %r in %s" % (rest[0], CONFIG))
    _doctor_note(notes, "ok", "config", "config parses: %s" % CONFIG)
    for directory in (STATE_DIR, run_dir()):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=directory):
                pass
            _doctor_note(notes, "ok", "writable", "writable: %s" % directory)
        except OSError as e:
            _doctor_note(notes, "fail", "writable", e)
    ports = {}
    for name, cfg in models.items():
        port = cfg.get("port", DEFAULT_PORT)
        if _is_count(port, 1) and port <= 65535:
            ports.setdefault(port, []).append(name)
    for port, sharing in sorted(ports.items()):
        if len(sharing) > 1:
            _doctor_note(notes, "warn", "ports",
                         "%s share port %d; only one at a time"
                         % (", ".join(sorted(sharing)), port))
    try:
        for path in sorted(run_dir().glob("*.lock")):
            try:
                pid = int(path.read_text())
                if pid <= 0 or not alive(pid):
                    _doctor_note(notes, "warn", "locks",
                                 "leftover lock; pid is not alive: %s" % path.name)
            except (OSError, ValueError, OverflowError) as e:
                _doctor_note(notes, "warn", "locks",
                             "cannot check lock %s: %s" % (path.name, e))
        states = read_states(read_only=True)
    except (OSError, ValueError) as e:
        _doctor_note(notes, "warn", "runtime", "cannot read state: %s" % e)
        states = {}
    help_cache = {}
    for name in rest or sorted(models):
        report["models"][name] = _doctor_model(
            name, models[name], binary, states, help_cache)
    _doctor_print(report, as_json)


def cmd_logs(args):
    follow = "-f" in args
    rest = [a for a in args if a != "-f"]
    if len(rest) > 1:
        die("usage: mdl logs [-f] [name]")
    if rest:
        log = STATE_DIR / (check_name(rest[0]) + ".log")
    else:
        state = read_state()          # errors if several are running
        if not state:
            die("nothing running; pass a model name")
        log = Path(state["log"])
    if not log.is_file():
        die(f"no log at {log}")
    with open(log, "r", errors="replace") as fh:
        while True:
            chunk = fh.read()
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            elif not follow:
                return
            else:
                time.sleep(0.3)


def _launch_ui(fx=None):
    try:
        from mdl_ui import run_ui
    except ImportError as e:
        die("mdl tui needs textual: pip install \"llama-mdl[ui]\" ({})"
            .format(e))
    run_ui(fx)


def cmd_tui(args):
    """The terminal dashboard (Textual)."""
    if args == ["--no-fx"]:
        _launch_ui("off")
        return
    if args:
        die("usage: mdl tui [--no-fx]")
    _launch_ui()


def cmd_ui(args):
    """The web UI: the page in an app window, served on 127.0.0.1."""
    if args and args[0] == "--tui":
        # the terminal dashboard's old name, kept for one release
        print("mdl: `mdl ui --tui` is now `mdl tui`; the old spelling goes "
              "in the next release", file=sys.stderr)
        cmd_tui(args[1:])
        return
    from mdl_web import server
    server.main(args)


def cmd_snapshot(args):
    """What the web UI draws, as JSON: the config's models, which run and
    how fast, the GPUs. One reading, so speeds are the servers' averages."""
    if args not in ([], ["--json"]):
        die("usage: mdl snapshot [--json]")
    from mdl_web import snapshot
    print(json.dumps(snapshot.build(), indent=1))


COMMANDS = {"init": cmd_init, "config": cmd_config,
            "add": cmd_add, "check": cmd_check, "doctor": cmd_doctor,
            "ui": cmd_ui, "tui": cmd_tui, "snapshot": cmd_snapshot,
            "run": cmd_run, "stop": cmd_stop, "ps": cmd_ps, "list": cmd_list,
            "logs": cmd_logs, "fit": cmd_fit, "eval": cmd_eval,
            "catalog": cmd_catalog, "find": cmd_find, "pull": cmd_pull,
            "manifest": cmd_manifest, "lab": cmd_lab}


def _dispatch():
    if len(sys.argv) < 2:
        # Bare `mdl` opens the UI; without textual it prints usage as before.
        try:
            from mdl_ui import run_ui
        except ImportError:
            print(USAGE)
            sys.exit(2)
        run_ui()
        return
    if sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        return
    if sys.argv[1] in ("-V", "--version"):
        print("mdl " + VERSION)
        return
    command = COMMANDS.get(sys.argv[1])
    if not command:
        die("unknown command '{}'".format(sys.argv[1]) + chr(10) + USAGE)
    command(sys.argv[2:])


def safe_streams():
    """Never die on a character the output cannot encode.

    Piped or redirected on Windows, stdout is cp1252, and the tables use
    → · ✓ ⚑: `mdl find | more` ended in a UnicodeEncodeError traceback. A
    console gets UTF-8 either way; a pipe now gets '?' for what its
    encoding lacks, instead of nothing at all.
    """
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if enc != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def main():
    """Entry point. Owns error reporting so console_scripts behaves too."""
    safe_streams()
    try:
        _dispatch()
    except MdlError as e:
        sys.stdout.flush()               # keep order when stdout is a pipe
        print(f"mdl: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    # mdl_fit reaches back with `import mdl`; without this alias that loads
    # a second copy whose MdlError the handler above cannot catch
    sys.modules.setdefault("mdl", sys.modules[__name__])
    main()
