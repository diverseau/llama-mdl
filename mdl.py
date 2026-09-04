#!/usr/bin/env python3
"""mdl - run one local llama.cpp server from a config file."""

import ctypes
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
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
STATE = STATE_DIR / "state.json"
VERSION = "0.5.1"
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
         "kv_type", "parallel", "port", "args", "group"}
SIMPLE = (("ngl", "-ngl"), ("n_cpu_moe", "--n-cpu-moe"), ("ctx", "-c"),
          ("parallel", "-np"), ("port", "--port"))

USAGE = ("usage: mdl {init|add <model.gguf>|check|list|run <name> [--port N]|"
         "stop [<name>|--all]|ps [--json]|logs [-f] [name]|ui [--no-fx]} "
         "[--version]")

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


def toml_path(path):
    """A filesystem path as a TOML basic string.

    Backslashes become forward slashes rather than escapes, because
    that is what the config asks people to write. A quote is legal in
    a POSIX filename and would otherwise close the string early.
    """
    return str(path).replace(chr(92), "/").replace(chr(34),
                                                   chr(92) + chr(34))


def build_argv(name, cfg, binary):
    unknown = sorted(set(cfg) - KNOWN)
    if unknown:
        die(f"model '{name}': unknown key(s): {', '.join(unknown)}")
    if "model" not in cfg:
        die(f"model '{name}': missing required key 'model'")
    argv = [binary, "-m", str(cfg["model"])]
    if "mmproj" in cfg:          # the vision half of a multimodal model
        argv += ["--mmproj", str(cfg["mmproj"])]
    for key, flag in SIMPLE:
        if key in cfg:
            argv += [flag, str(cfg[key])]
    if cfg.get("flash_attn"):
        argv += ["-fa", "on"]
    if "kv_type" in cfg:
        kv = str(cfg["kv_type"])
        argv += ["--cache-type-k", kv, "--cache-type-v", kv]
    extra = cfg.get("args", [])
    if not isinstance(extra, list):
        die(f"model '{name}': 'args' must be a list of strings")
    return argv + [str(a) for a in extra]


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
    return run_dir() / f"{name}.json"


def migrate_state():
    """Move a pre-0.3 single state.json into the per-server directory.

    Someone upgrading with a server up should keep control of it rather
    than be told nothing is running. This, STATE, and the test that
    covers them can go in 0.5: by then nobody is stopping a server they
    started two minor versions ago.
    """
    if not STATE.exists():
        return
    try:
        old = json.loads(STATE.read_text())
        name = old["name"]
    except (OSError, ValueError, KeyError, TypeError):
        STATE.unlink(missing_ok=True)
        return
    try:
        run_dir().mkdir(parents=True, exist_ok=True)
        if not state_path(name).exists():
            write_atomic(state_path(name), json.dumps(old))
    except OSError:
        return                           # try again next time; nothing is lost
    STATE.unlink(missing_ok=True)


def live_state(path):
    """One server's state, or None - clearing the file if it is stale."""
    try:
        state = json.loads(path.read_text())
        pid = state["pid"]
    except (OSError, ValueError, KeyError, TypeError):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(pid, int) or not alive(pid):
        path.unlink(missing_ok=True)
        return None
    born = state.get("born")
    if born is not None and proc_started(pid) not in (None, born):
        path.unlink(missing_ok=True)     # pid recycled onto someone else
        return None
    return state


def read_states():
    """Every server we started that is still alive, keyed by name.

    One file per server rather than one file listing them: two `mdl run`
    calls at the same moment would otherwise read, modify and write the
    same file, and one of them would lose.
    """
    migrate_state()
    out = {}
    try:
        paths = sorted(run_dir().glob("*.json"))
    except OSError:
        return out
    for path in paths:
        state = live_state(path)
        if state:
            out[state.get("name", path.stem)] = state
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


def terminate(pid, sig):
    """Signal the whole process tree, not just the pid we launched.

    If llama_server is a wrapper script - setting LD_LIBRARY_PATH, say -
    the recorded pid is the wrapper and the real server is its child.
    Signalling only the wrapper orphans the server and leaves the port
    held. spawn() puts it in its own session, so the group is the tree.
    """
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        os.kill(pid, sig)                # not a group leader after all


def write_atomic(path, text, keep_backup=False):
    """Replace a file in one step, never leaving it half written.

    models.toml is hand-edited and lives in nobody's git, so a write cut
    short by a crash, a full disk or a Ctrl-C has to leave the old file
    untouched rather than truncated. Writing a sibling temp file and
    renaming it over gives that: the rename is atomic, so a reader sees
    either the whole old file or the whole new one.
    """
    if keep_backup and path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    tmp = path.with_name(path.name + ".tmp%d" % os.getpid())
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())    # the rename is no use if the data is not down
        os.replace(tmp, path)
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


def spawn(name, models, binary, port=None):
    """Launch <name> detached, write its state file, return (proc, log, port).

    Shared by the CLI and the TUI so there is one way to start a server.
    A port here overrides the config, for `mdl run <name> --port N`; the
    value that reaches the state file is the one we actually launched
    with, so ps, the readiness probe and the next pre-flight agree.
    """
    cfg = dict(models[name])
    if port is not None:
        cfg["port"] = port
    argv = build_argv(name, cfg, binary)
    port = cfg.get("port", DEFAULT_PORT)
    if not shutil.which(binary) and not Path(binary).is_file():
        die(f"llama-server not found: {binary}")
    if not Path(models[name]["model"]).is_file():
        die(f"model file not found: {models[name]['model']}")
    if port_busy(port):
        owner = next((s["name"] for s in read_states().values()
                      if s.get("port") == port), None)
        if owner:
            die(f"port {port} is already serving '{owner}'; give {name} its "
                f"own port, or run 'mdl stop {owner}'")
        die(f"port {port} is already in use")
    try:
        run_dir().mkdir(parents=True, exist_ok=True)
        log = STATE_DIR / f"{name}.log"
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
    write_atomic(state_path(name), json.dumps(
        {"name": name, "pid": proc.pid, "port": port, "started": time.time(),
         "log": str(log), "born": proc_started(proc.pid)}))
    return proc, log, port


def cmd_run(args):
    port = None
    if len(args) == 3 and args[1] == "--port":
        try:
            port = int(args[2])
        except ValueError:
            die(f"port must be a number, not {args[2]!r}")
        args = args[:1]
    if len(args) != 1:
        die("usage: mdl run <name> [--port N]")
    name = args[0]
    models, binary = load_config()
    if name not in models:
        die(f"no model named '{name}' in {CONFIG}")
    running = read_state(name)
    if running:
        die(f"'{name}' is already running (pid {running['pid']}, "
            f"port {running['port']}); run 'mdl stop {name}' first")

    proc, log, port = spawn(name, models, binary, port)
    print(f"starting {name} (pid {proc.pid}), log {log}", flush=True)
    tail_until_ready(proc, log, name, port)
    print(f"ready: {name} on http://127.0.0.1:{port} (pid {proc.pid})")


def stop_one(name, state):
    """SIGTERM, wait, SIGKILL. True if it is gone afterwards."""
    pid = state["pid"]
    try:
        terminate(pid, signal.SIGTERM)
    except OSError as e:
        print(f"mdl: cannot signal {name} (pid {pid}): {e}", file=sys.stderr)
        return False
    for _ in range(100):
        if not alive(pid):
            break
        time.sleep(0.1)
    else:
        try:
            terminate(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except OSError:
            pass
        time.sleep(0.5)
    if alive(pid):
        print(f"mdl: {name} (pid {pid}) would not die", file=sys.stderr)
        return False
    state_path(name).unlink(missing_ok=True)
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
    if len(args) > 1:
        name = args[1]
    else:                       # Foo-Bar-Q4_K_M.gguf -> foo-bar
        stem = re.sub(r"[-_.]?(q\d+[_0-9a-z]*|f16|f32|bf16)$", "", path.stem,
                      flags=re.I)
        name = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-").lower()
    try:
        port = int(args[2]) if len(args) > 2 else DEFAULT_PORT
    except ValueError:
        die(f"port must be a number, not {args[2]!r}")
    models, _ = load_config()
    if name in models:
        die(f"{name} is already in {CONFIG}; pick another name")
    layers = gguf_layers(path)
    mmproj = find_mmproj(path)
    block = (
        f"{chr(10)}[{name}]{chr(10)}"
        f'model = "{toml_path(path)}"{chr(10)}'
        + (f'mmproj = "{toml_path(mmproj)}"{chr(10)}' if mmproj else "")
        + f"ngl = 99{chr(10)}ctx = 8192{chr(10)}flash_attn = true{chr(10)}"
        f'kv_type = "q8_0"{chr(10)}parallel = 1{chr(10)}port = {port}{chr(10)}'
        f'args = ["--metrics"]{chr(10)}')
    try:
        current = CONFIG.read_text(encoding="utf-8")
        write_atomic(CONFIG, current + block, keep_backup=True)
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
            build_argv(name, cfg, binary)
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


def cmd_logs(args):
    follow = "-f" in args
    rest = [a for a in args if a != "-f"]
    if len(rest) > 1:
        die("usage: mdl logs [-f] [name]")
    if rest:
        log = STATE_DIR / (rest[0] + ".log")
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
        die("mdl ui needs textual: pip install \"llama-mdl[ui]\" ({})".format(e))
    run_ui(fx)


def cmd_ui(args):
    if args == ["--no-fx"]:
        _launch_ui("off")
        return
    if args:
        die("usage: mdl ui [--no-fx]")
    _launch_ui()


COMMANDS = {"init": cmd_init, "add": cmd_add, "check": cmd_check, "ui": cmd_ui,
            "run": cmd_run, "stop": cmd_stop, "ps": cmd_ps, "list": cmd_list,
            "logs": cmd_logs}


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


def main():
    """Entry point. Owns error reporting so console_scripts behaves too."""
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
    main()
