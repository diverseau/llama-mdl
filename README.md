# mdl

A small CLI for running one local [llama.cpp](https://github.com/ggml-org/llama.cpp)
server at a time from a config file, instead of pasting flag soup into your shell.

Single file, Python 3.11+ (needs `tomllib`), standard library only. POSIX only —
it uses process groups and signals.

## Install

```sh
git clone <this repo> ~/src/mdl
chmod +x ~/src/mdl/mdl.py
sudo ln -s ~/src/mdl/mdl.py /usr/local/bin/mdl
```

Then write a config:

```sh
mkdir -p ~/.config/mdl && $EDITOR ~/.config/mdl/models.toml
```

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
```

```console
$ mdl list
ornith      /srv/models/Ornith-1.5-35B-A3B-Q4_K_M.gguf
qwen-small  /srv/models/Qwen3-8B-Q5_K_M.gguf

$ mdl run ornith
starting ornith (pid 48812), log /home/leon/.local/state/mdl/ornith.log
load_tensors: offloading 99 repeating layers to GPU
...
main: server is listening on http://0.0.0.0:8080 - starting the main loop
ready: ornith on http://127.0.0.1:8080 (pid 48812)

$ mdl ps
ornith  pid 48812  port 8080  up 1h04m

$ mdl stop
stopped ornith (pid 48812)
```

## Files

```
~/.config/mdl/models.toml       your config
~/.local/state/mdl/state.json   name, pid, port and start time of the server
~/.local/state/mdl/<name>.log   server stdout+stderr, truncated on each run
```

## Behaviour notes

- **One server at a time.** `run` while something is up is an error telling you
  to `stop` first.
- **Stale state self-heals.** If the pid in `state.json` is gone — crash, reboot,
  `kill -9` — the file is removed and `ps` reports nothing running.
- **If the server exits during startup,** `run` reports its exit status, removes
  the state file, and exits 1. The log has the reason.
- **If it doesn't report ready within 300s,** `run` exits 1 but leaves the server
  running, since it may still be loading. Check the log, or `mdl stop`.
- Errors are one line on stderr and a non-zero exit. No tracebacks.

## Deliberately not included

No daemon, no web UI, no model downloading, no hot-swap, no auto-unload,
no multiple concurrent servers.

## License

MIT. See [LICENSE](LICENSE).
