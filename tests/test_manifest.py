"""mdl manifest: what a server is running as, read from the server and
its state rather than from a config that may have changed since."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support                                        # noqa: E402
from support import mdl, run, sandbox, teardown       # noqa: E402

from mdl_fit import manifest                          # noqa: E402

t = support.Tally("test_manifest")
check = t.check

# ------------------------------------------------------ not running ----
root, port = sandbox(extra='args = ["--jinja", "--api-key", "s3cret"]\n')
man = manifest.build("demo", probe=False)
check("a model that is not running is described from its preset, and says so",
      (man["source"], man["argv"][1:3]),
      ("config", ["-m", str(support.FAKE).replace("\\", "/")]))
check("its model file is named by size and hash, not read whole",
      (man["model"][0]["name"], man["model"][0]["size"],
       len(man["model"][0]["hash"])),
      (support.FAKE.name, support.FAKE.stat().st_size, 16))
check("the flags it would run with", (man["flags"]["ctk"], man["flags"]["np"]),
      ("q8_0", 1))
_, err, code = run(manifest.build, "nope")
check("an unknown name is one line", ("no model named 'nope'" in err, code),
      (True, 1))

# ---------------------------------------------------------- running ----
run(mdl.cmd_run, ["demo"])
state = mdl.read_state("demo")
check("spawn records the binary it launched, by stat",
      sorted(state["binary"]), ["mtime", "path", "size"])
live = manifest.build("demo", probe=False)
check("a running model is described from its state",
      (live["source"], live["argv"], live["port"]),
      ("running", state["argv"], port))
before = live["identity"]
# the config changes under a running server: the manifest does not
mdl.CONFIG.write_text(mdl.CONFIG.read_text(encoding="utf-8").replace(
    "ctx = 4096", "ctx = 8192"), encoding="utf-8")
again = manifest.build("demo", probe=False)
check("editing the config does not change what the running server is",
      (again["flags"]["ctx"], again["identity"]), (4096, before))
run(mdl.cmd_stop, ["demo"])
after = manifest.build("demo", probe=False)
check("once stopped, the preset's new settings are a new identity",
      (after["flags"]["ctx"], after["identity"] != before), (8192, True))

# ---------------------------------------------------------- identity ----
moved = dict(live, argv=[a if a != str(port) else "9999" for a in live["argv"]])
check("a port is not part of the identity", manifest.identity(moved), before)
rebuilt = dict(live, build="b9999-abcdef0")
check("another build is", manifest.identity(rebuilt) != before, True)
other = dict(live, model=[dict(live["model"][0], hash="0" * 16)])
check("and so are other model bytes", manifest.identity(other) != before, True)

# ------------------------------------------------------------ redact ----
red = manifest.redact(live)
text = json.dumps(red)
check("redact blanks a secret flag's value",
      ("s3cret" in text, "<redacted>" in red["argv"]), (False, True))
check("and cuts paths to file names",
      (str(Path.home()) in text, red["argv"][0], "path" in red["model"][0]),
      (False, Path(live["argv"][0]).name, False))
check("--api-key=value is blanked too",
      manifest.redact(dict(live, argv=["ls", "--api-key=abc"]))["argv"],
      ["ls", "--api-key=<redacted>"])

# ---------------------------------------------------------- the log ----
log = root / "state" / "x.log"
log.write_text("llama_server: build: 6432 (1a2b3c4d) with cc\n",
               encoding="utf-8")
check("the build is read off the load log",
      manifest.build_from_log(log), (6432, "1a2b3c4d"))
check("no log, no build", manifest.build_from_log(root / "none.log"), None)
check("split models list every shard",
      [p.name for p in manifest.shards(Path("m-00002-of-00003.gguf"))],
      ["m-00001-of-00003.gguf", "m-00002-of-00003.gguf",
       "m-00003-of-00003.gguf"])

out, err, code = run(mdl.cmd_manifest, ["demo", "--redact", "--no-probe"])
check("mdl manifest prints JSON", (code, json.loads(out)["name"]), (0, "demo"))
_, err, code = run(mdl.cmd_manifest, ["demo", "--what"])
check("an unknown option is one line", ("usage: mdl manifest" in err, code),
      (True, 1))
teardown(root)

sys.exit(t.done())
