"""Opening a coding agent on a running model, in a folder, in a terminal.

The agent and folder a model opens with are settings: the last choice is
the next default, the last six folders are offered again, and a running
model keeps its own until it stops. Opening writes what the agent needs -
its environment, or a config file for agents that read one - under
mdl's state directory, readable only by you, and changes nothing in the
agent's own config. The key is the server's --api-key when it has one.

How each agent is pointed at an OpenAI- or Anthropic-style endpoint is
0xSero's, from omarchy-local-ai's backend (MIT).
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

AGENTS = ("pi", "claude", "codex", "opencode", "omp", "crush", "grok",
          "copilot", "hermes")
FOLDERS_KEPT = 6


def ui_dir():
    import mdl
    return mdl.STATE_DIR / "ui"


def _read(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path, data):
    import mdl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mdl.write_atomic(path, json.dumps(data, indent=1))


def settings():
    return _read(ui_dir() / "settings.json")


def binary(agent):
    """Where an agent is, preferring ~/.local/bin, which a desktop
    session's PATH may not have."""
    home = Path.home() / ".local" / "bin"
    for name in (agent, agent + ".exe", agent + ".cmd"):
        p = home / name
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return shutil.which(agent)


def installed():
    return [a for a in AGENTS if binary(a)]


def default_agent(saved=None):
    have = installed()
    for a in (saved, "pi") + tuple(have):
        if a and a in have:
            return a
    return None


def default_folder(saved=None):
    for f in (saved, str(Path.home() / "Work"), str(Path.home())):
        if f and Path(f).is_dir():
            return f
    return str(Path.home())


def defaults():
    s = settings()
    return {"agent": default_agent(s.get("agent")),
            "folder": default_folder(s.get("folder")),
            "folders": [f for f in s.get("folders", []) if Path(f).is_dir()]}


def run_config(name, state):
    """A running model's agent and folder: its own, once chosen, while
    this server runs; the defaults until then. And its tailnet URL, if
    it is shared."""
    own = _read(ui_dir() / "run" / ("%s.json" % name))
    d = defaults()
    if own.get("pid") != state.get("pid"):
        own = {}
    return {"agent": default_agent(own.get("agent") or d["agent"]),
            "folder": default_folder(own.get("folder") or d["folder"]),
            "shared": own.get("shared")}


def remember(name, state, **kv):
    """Keep something about one running server, until it stops."""
    path = ui_dir() / "run" / ("%s.json" % name)
    own = _read(path)
    if own.get("pid") != state.get("pid"):
        own = {"pid": state.get("pid")}
    own.update(kv)
    _write(path, own)


def choose(key, value, name=None, state=None):
    """Set the agent or folder: the default, and a running model's."""
    import mdl
    if key == "agent":
        if value not in AGENTS:
            raise mdl.MdlError("unknown agent %r" % value)
        if not binary(value):
            raise mdl.MdlError("%s is not installed" % value)
    elif key == "folder":
        p = Path(os.path.expanduser(value))
        if not p.is_dir():
            raise mdl.MdlError("no folder %s" % value)
        value = str(p.resolve())
    else:
        raise mdl.MdlError("set agent or folder")
    s = settings()
    s[key] = value
    if key == "folder":
        s["folders"] = ([value] + [f for f in s.get("folders", [])
                                   if f != value])[:FOLDERS_KEPT]
    _write(ui_dir() / "settings.json", s)
    if name and state:
        remember(name, state, **{key: value})


# ------------------------------------------------------------- the argv --

def argv(agent, exe, endpoint, model, ctx, vision, home, key):
    """(argv, env, files) that point `agent` at `endpoint`. files is
    {path: text} to write first; the key stays in env or in those."""
    inp = ["text", "image"] if vision else ["text"]
    v1 = endpoint + "/v1"
    if agent == "claude":
        return ([exe, "--model", model],
                {"ANTHROPIC_BASE_URL": endpoint, "ANTHROPIC_AUTH_TOKEN": key,
                 "ANTHROPIC_MODEL": model,
                 "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
                 "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                 "ANTHROPIC_DEFAULT_HAIKU_MODEL": model}, {})
    if agent == "codex":
        return ([exe, "-c", "model_providers.mdl.name=mdl",
                 "-c", "model_providers.mdl.base_url=" + v1,
                 "-c", "model_providers.mdl.wire_api=responses",
                 "-c", "model_providers.mdl.env_key=MDL_API_KEY",
                 "-c", "model_provider=mdl", "-c", "model=" + model,
                 "-c", "model_context_window=%d" % ctx],
                {"MDL_API_KEY": key}, {})
    if agent == "opencode":
        cfg = {"$schema": "https://opencode.ai/config.json",
               "model": "mdl/" + model, "small_model": "mdl/" + model,
               "provider": {"mdl": {
                   "npm": "@ai-sdk/openai-compatible", "name": "mdl",
                   "options": {"baseURL": v1, "apiKey": "{env:MDL_API_KEY}"},
                   "models": {model: {"name": model,
                                      "limit": {"context": ctx,
                                                "output": 8192},
                                      "modalities": {"input": inp,
                                                     "output": ["text"]}}}}}}
        return ([exe, "--model", "mdl/" + model],
                {"OPENCODE_CONFIG_CONTENT": json.dumps(cfg),
                 "MDL_API_KEY": key}, {})
    if agent in ("pi", "omp"):
        models = {"providers": {"mdl": {
            "baseUrl": v1, "apiKey": key, "api": "openai-completions",
            "models": [{"id": model, "name": model + " · local",
                        "reasoning": True, "contextWindow": ctx,
                        "maxTokens": 32768, "input": inp,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0,
                                 "cacheWrite": 0},
                        "compat": {"supportsDeveloperRole": False}}]}}}
        files = {home / "models.json": json.dumps(models)}
        if agent == "omp":
            files[home / "models.yml"] = json.dumps(models)
            files[home / "config.yml"] = ("modelRoles:\n  default: mdl/%s\n"
                                          "setupVersion: 2\n" % model)
        return ([exe, "--provider", "mdl", "--model", model],
                {"PI_CODING_AGENT_DIR": str(home),
                 "OMP_CODING_AGENT_DIR": str(home)}, files)
    if agent == "crush":
        cfg = {"providers": {"mdl": {
            "type": "openai", "name": "mdl", "base_url": v1, "api_key": key,
            "models": [{"id": model, "name": model, "context_window": ctx,
                        "default_max_tokens": 8192,
                        "supports_attachments": bool(vision)}]}},
            "models": {"large": {"provider": "mdl", "model": model},
                       "small": {"provider": "mdl", "model": model}}}
        return ([exe], {"XDG_CONFIG_HOME": str(home),
                        "XDG_DATA_HOME": str(home)},
                {home / "crush" / "crush.json": json.dumps(cfg)})
    if agent == "copilot":
        return ([exe, "--model", model],
                {"COPILOT_PROVIDER_BASE_URL": v1,
                 "COPILOT_PROVIDER_API_KEY": key}, {})
    if agent == "grok":
        toml = ('[models]\ndefault = "mdl"\n[features]\nremote_fetch = false\n'
                'telemetry = false\n[model.mdl]\nmodel = %s\nbase_url = %s\n'
                'context_window = %d\nenv_key = "XAI_API_KEY"\n'
                % (json.dumps(model), json.dumps(v1), ctx))
        return ([exe, "--model", "mdl"],
                {"GROK_HOME": str(home), "XAI_API_KEY": key},
                {home / "config.toml": toml})
    return ([exe], {"OPENAI_BASE_URL": v1, "OPENAI_API_BASE": v1,
                    "OPENAI_MODEL": model, "OPENAI_API_KEY": key}, {})


# ---------------------------------------------------------- the terminal --

LAUNCH = '''"""mdl's agent launcher: a folder, an environment, a command."""
import json, os, subprocess, sys
spec = json.load(open(sys.argv[1], encoding="utf-8"))
os.chdir(spec["cwd"])
os.environ.update(spec["env"])
print("mdl: %s on %s" % (spec["agent"], spec["endpoint"]), flush=True)
sys.exit(subprocess.call(spec["argv"]))
'''


def terminal(cwd, command):
    """argv that opens `command` in a new terminal window in `cwd`."""
    if os.name == "nt":
        wt = shutil.which("wt")
        if wt:
            return [wt, "-w", "new", "-d", cwd] + command
        return ["cmd", "/c", "start", "mdl", "/D", cwd, "cmd", "/k"] + command
    if sys.platform == "darwin":
        script = ui_dir() / "agents" / "open.command"
        script.write_text("#!/bin/sh\ncd %s && exec %s\n"
                          % (shlex.quote(cwd), shlex.join(command)))
        script.chmod(0o700)
        return ["open", "-a", "Terminal", str(script)]
    for term, flag in ((os.environ.get("TERMINAL"), "-e"),
                       ("x-terminal-emulator", "-e"), ("kitty", "--"),
                       ("alacritty", "-e"), ("foot", ""), ("wezterm", "start --"),
                       ("gnome-terminal", "--"), ("konsole", "-e"),
                       ("xfce4-terminal", "-x"), ("xterm", "-e")):
        if term and shutil.which(term):
            return [term] + flag.split() + command
    return None


def open_agent(name, state, cfg, run):
    """Open the model's agent in its folder. Raises MdlError with why not."""
    import mdl
    agent, folder = run["agent"], run["folder"]
    if not agent:
        raise mdl.MdlError("no coding agent installed (pi, claude, codex, "
                           "opencode...)")
    exe = binary(agent)
    if not exe:
        raise mdl.MdlError("%s is not installed" % agent)
    if not Path(folder).is_dir():
        raise mdl.MdlError("folder %s is gone" % folder)
    from . import snapshot
    key = snapshot.api_key(state.get("argv")) or "mdl"
    endpoint = "http://127.0.0.1:%d" % state["port"]
    home = ui_dir() / "agents" / agent
    home.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(ui_dir() / "agents", 0o700)
    except OSError:
        pass
    command, env, files = argv(agent, exe, endpoint, name,
                               int(cfg.get("ctx") or 32768),
                               bool(cfg.get("mmproj")), home, key)
    for path, text in files.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
        _private(path)
    spec = home / "launch.json"
    spec.write_text(json.dumps({"cwd": folder, "env": env, "argv": command,
                                "agent": agent, "endpoint": endpoint}),
                    encoding="utf-8")
    _private(spec)
    launcher = ui_dir() / "agents" / "launch.py"
    launcher.write_text(LAUNCH, encoding="utf-8")
    how = terminal(folder, [sys.executable, str(launcher), str(spec)])
    if not how:
        raise mdl.MdlError("no terminal found to open %s in; set $TERMINAL"
                           % agent)
    subprocess.Popen(how, cwd=folder, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     **({"start_new_session": True} if os.name != "nt"
                        else {}))
    return agent


def _private(path):
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
