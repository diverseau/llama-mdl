// view.js, checked without a browser: snapshot in, page out. Run by
// test_web.py when node is on PATH; prints PASS/FAIL lines like the rest.
"use strict";

const path = require("path");
const View = require(path.join(__dirname, "..", "mdl_web", "static", "view.js"));

let fails = 0;
function check(label, got, want) {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  console.log((ok ? "PASS " : "FAIL ") + label);
  if (!ok) {
    console.log("      got:  " + JSON.stringify(got));
    console.log("      want: " + JSON.stringify(want));
    fails++;
  }
}

const GB = Math.pow(2, 30);
const snap = {
  schema: 1, version: "0.11.0",
  gpus: [{ name: "NVIDIA GeForce RTX 3060", used: 6 * GB, total: 12 * GB, temp: 51 }],
  ram: { total: 16 * GB, free: 5 * GB },
  models: [
    { name: "qwen", state: "ready", quant: "Q4_K_M", ctx: 32768, size: 5 * GB, port: 8080,
      run: { n_ctx: 32768, up: 125, url: "http://127.0.0.1:8080/v1", api_key: true,
             series: [[1, 0], [2, 50]],
             metrics: { tps: 41.26, decode_avg: 40, prefill_avg: 1500, tokens: 12345,
                        kv: 0.5 } } },
    { name: "plain", state: "ready", ctx: 4096, port: 8081, run: { up: 3, metrics: null } },
    { name: "gemma", state: "stopped", group: "google", quant: "Q8_0", ctx: 8192, port: 8082,
      ngl: 99, flash_attn: true, file: "/m/gemma.gguf" },
    { name: "llama", state: "failed", error: "missing tensor", ctx: 4096, port: 8083 },
  ],
};

check("no snapshot yet is a wait", View.build(null, {}).page, "wait");

const home = View.build(snap, { page: "home" });
check("home: a card per running model", home.cards.map(c => c.name), ["qwen", "plain"]);
check("home: the card's speed and tokens", [home.cards[0].tps, home.cards[0].tokens],
      ["41.3", "12.3K"]);
check("home: a server without --metrics says so",
      [home.cards[1].noMetrics, home.cards[1].tps], [true, null]);
check("home: stopped and failed models are listed, grouped, ungrouped last",
      home.groups.map(g => [g.name, g.rows.map(r => r.name)]),
      [["google", ["gemma"]], ["", ["llama"]]]);
check("home: a failed start is shown with why", home.failed,
      [{ name: "llama", error: "missing tensor" }]);
check("home: a row runs its model", home.groups[0].rows[0].act, "run|gemma");
check("home: the summary counts the running", home.summary.mid, "2 running");
check("home: the GPU's name is short", home.gpus[0].name, "RTX 3060");
check("home: memory as used of total", home.gpus[0].text, "6 / 12 GB");
check("home: RAM free of total", home.ram, { free: "5 GB", total: "16 GB" });

const live = View.build(snap, { page: "model", name: "qwen" });
check("model: running has stats, stopped settings do not show",
      [!!live.stats, live.settings === undefined], [true, true]);
check("model: context fill as a share of n_ctx",
      live.stats.filter(s => s.label === "context")[0].value, "50% of 32K");
check("model: reach names the url and whether a key is needed", live.reach,
      { url: "http://127.0.0.1:8080/v1", key: true });
check("model: its actions are logs and stop",
      live.actions.map(a => a.act), ["page|log|qwen", "stop|qwen"]);

const idle = View.build(snap, { page: "model", name: "gemma" });
check("model: stopped offers Run as the one primary action",
      idle.actions.map(a => [a.act, a.primary]), [["run|gemma", true]]);
check("model: stopped shows its settings",
      idle.settings.slice(0, 3), [["file", "/m/gemma.gguf"], ["context", "8K"],
                                   ["gpu layers", "99"]]);
check("model: a name the snapshot no longer has is gone",
      View.build(snap, { page: "model", name: "nope" }).page, "gone");

check("numbers: k", [View.k(0), View.k(999), View.k(1500), View.k(2.5e6)],
      ["0", "999", "1.5K", "2.5M"]);
check("numbers: dur", [View.dur(5), View.dur(125), View.dur(3900)],
      ["5s", "2m", "1:05h"]);

console.log("\nweb_view: " + fails + " failure(s)");
process.exit(fails ? 1 : 0);
