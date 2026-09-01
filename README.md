# mdl

A small CLI for running one local [llama.cpp](https://github.com/ggml-org/llama.cpp)
server at a time from a config file, instead of pasting flag soup into your shell.

`mdl.py` is a single file, Python 3.11+ (needs `tomllib`), standard library
only. Runs on Linux, macOS and Windows.

`mdl_ui.py` adds an optional terminal dashboard (`mdl ui`). It is the only
part that needs a dependency — [Textual](https://textual.textualize.io/) —
and the CLI never imports it, so the four commands stay dependency-free.

## Install

Linux / macOS:

```sh
git clone <this repo> ~/src/mdl
chmod +x ~/src/mdl/mdl.py
sudo ln -s ~/src/mdl/mdl.py /usr/local/bin/mdl
```

Windows has no symlink worth relying on, so drop a `mdl.cmd` shim somewhere on
your PATH:

```
@echo off
python "%USERPROFILE%\src\mdl\mdl.py" %*
```

For the optional dashboard:

```sh
pip install textual
```

Then write a config:

```sh
mkdir -p ~/.config/mdl && $EDITOR ~/.config/mdl/models.toml
```

On Windows that is `%USERPROFILE%\.config\mdl\models.toml`.

## Config

`~/.config/mdl/models.toml`. One table per model; the table name is what you
pass to `mdl run`.

```toml
# Optional. Defaults to "llama-server" on $PATH.
# $MDL_LLAMA_SERVER overrides this.
llama_server = "/opt/llama.cpp/build/bin/llama-server"

[ornith]
model = "/srv/models/Ornith-1.5-35B-A3B-Q4_K_M.gguf"
ngl = 99
n_cpu_moe = 24
ctx = 65536
flash_attn = true
kv_type = "q8_0"
parallel = 1
port = 8080

[qwen-small]
model = "/srv/models/Qwen3-8B-Q5_K_M.gguf"
ngl = 99
ctx = 16384
port = 8080
```

On Windows, write paths with forward slashes (`C:/models/foo.gguf`) or double
the backslashes, since TOML treats `\` as an escape character.

### Keys

| Key | llama-server flag | Notes |
| --- | --- | --- |
| `model` | `-m` | Required. |
| `ngl` | `-ngl` | |
| `n_cpu_moe` | `--n-cpu-moe` | |
| `ctx` | `-c` | |
| `flash_attn` | `-fa on` | Only emitted when `true`. |
| `kv_type` | `--cache-type-k` and `--cache-type-v` | Both get the same value. |
| `parallel` | `-np` | |
| `port` | `--port` | Defaults to 8080. |
| `args` | passed through verbatim | Array of strings, appended last. |

Anything else in a model table is an error, so a typo like `flash_atn` tells you
instead of silently doing nothing.

## Commands

```
mdl run <name>   Start <name> in the background, then tail its log until the
                 server reports it is listening, and exit. The server keeps
                 running after mdl exits.
mdl stop         SIGTERM the running server, SIGKILL after 10s, clean up.
mdl ps           name, pid, port and uptime, or "nothing running".
mdl list         The models defined in the config.
mdl ui           The dashboard. Bare `mdl` opens it too.
```

Without textual installed, `mdl ui` fails with one line and bare `mdl` prints
the usage string, exactly as it always did.

```console
$ mdl list
ornith      /srv/models/Ornith-1.5-35B-A3B-Q4_K_M.gguf
qwen-small  /srv/models/Qwen3-8B-Q5_K_M.gguf

$ mdl run ornith
starting ornith (pid 48812), log /home/leon/.local/state/mdl/ornith.log
load_tensors: offloaded 43/43 layers to GPU
llama_context: n_ctx = 65536
main: server is listening on http://127.0.0.1:8080
ready: ornith on http://127.0.0.1:8080 (pid 48812)

$ mdl ps
ornith  pid 48812  port 8080  up 1h04m

$ mdl stop
stopped ornith (pid 48812)
```

## The UI

`mdl ui` (or just `mdl`) opens a dashboard over the same config and the same
state file. Anything you do in it is visible to the CLI and vice versa.

Idle, it lists your models with a status dot, shows the selected model's
parameters, and previews the exact `llama-server` command it would run.
Running, it swaps in live telemetry: VRAM, KV-cache use, a tokens/sec
sparkline, busy slots, and a colour-coded log tail.

```
 key          does
 up/down, j k select a model
 enter, r     run the selected model
 s            stop the running server
 R            restart
 e            edit ngl / ctx / kv_type / port for this session
 c            copy the llama-server command
 p            prompt the running model without leaving the UI
 l            focus the log, / filters it
 g            reload the config
 ?            help
 q            quit the UI - the server keeps running
```

Quitting never stops a server; `s` is the only thing that does.

### Animation

The wordmark drifts its gradient slowly and continuously by default. Set
`ui_fx` at the top level of the config to change that:

```toml
ui_fx = "always"   # slow continuous drift (default)
ui_fx = "sweep"    # one ~1.2s pass at launch, then it parks
ui_fx = "off"      # painted flat, no timer at all
```

`$MDL_UI_FX` overrides the config, and `mdl ui --no-fx` overrides both.

The telemetry panels need llama.cpp's metrics endpoint, so add `--metrics`
to a model's `args` to light them up:

```toml
args = ["--metrics"]
```

Without it the dashboard still works, and those panels say `metrics off`
rather than failing. While a model is loading they say `loading` instead,
since nothing is listening yet.

## Files

```
~/.config/mdl/models.toml       your config
~/.local/state/mdl/state.json   name, pid, port and start time of the server
~/.local/state/mdl/<name>.log   server stdout+stderr, truncated on each run
~/.local/state/mdl/ui-marks.json which models the UI has seen start or fail
```

Same layout on Windows, under `%USERPROFILE%`.

## Behaviour notes

- **One server at a time.** `run` while something is up is an error telling you
  to `stop` first.
- **Stale state self-heals.** If the pid in `state.json` is gone (crash, reboot,
  `kill -9`) the file is removed and `ps` reports nothing running.
- **If the server exits during startup,** `run` reports its exit status, removes
  the state file, and exits 1. The log has the reason.
- **If it does not report ready within 300s,** `run` exits 1 but leaves the
  server running, since it may still be loading. Check the log, or `mdl stop`.
- **Pid reuse is not guarded against.** If the OS recycles a dead server's pid,
  `ps` will report it as still running. Rare, and the cure costs more than the
  disease.
- Errors are one line on stderr and a non-zero exit. No tracebacks.

## Deliberately not included

No daemon, no web UI, no model downloading, no hot-swap, no auto-unload,
no multiple concurrent servers.

## License

MIT. See [LICENSE](LICENSE).
