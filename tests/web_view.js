// view.js, checked without a browser: snapshot in, rows out. Run by
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
const types = v => v.rows.map(r => r.type);
const byType = (v, t) => v.rows.filter(r => r.type === t);

// a Thursday at noon, UTC; 20 weeks of days ending with it
const AT = Date.UTC(2026, 8, 24, 12) / 1000;
const days = new Array(140).fill(0);
days[136] = 12000; days[130] = 3000;
const qwen = { id: "qwen", name: "qwen", family: "qwen", format: "GGUF · Q4_K_M", ctx: 32768, sizeGb: 18.6,
               caps: { vision: false }, weights: [{ repository: "unsloth/Qwen3-GGUF", revision: "abc" }],
               port: 8080 };
const gemma = { id: "gemma", name: "gemma", family: "", format: "GGUF · Q8_0", ctx: 8192, sizeGb: 12,
                caps: { vision: true }, weights: [], port: 8082 };
const gpu = key => ({ key: key, name: "RTX 3090", usedMiB: key === "0" ? 18432 : 400, vramGb: 24, tempC: 51 });
const snap = {
  schema: 2, version: "0.11.0", at: AT, home: "/home/me",
  gpus: [gpu("0"), gpu("1"), gpu("2")],
  kinds: [{ hw: "RTX 3090", keys: ["0", "1", "2"], free: ["1", "2"], taken: [], models: [qwen, gemma], groups: [] }],
  deployments: [
    { id: "qwen", name: "qwen", family: "qwen", keys: ["0"], state: "ready", port: 8080, api_key: true,
      startedAt: new Date((AT - 3900) * 1000).toISOString(), agent: "pi", folder: "/home/me/Work",
      format: "GGUF · Q4_K_M", ctx: 32768, caps: { vision: false }, weights: qwen.weights,
      session: { tokens: 1500, all: { tokens: 12345, requests: 7, decode: 41, prefill: 1500, ttft: 250,
                                      line: [0, 5000, 12345], since: "Sep 20", last: AT - 60 } } },
    { id: "llama", name: "llama", family: "", keys: ["2"], state: "error", error: "missing tensor", port: 8083,
      session: { tokens: 0, all: {} } },
  ],
  total: 15000, week: 15000,
  life: { requests: 7, days: days, start: AT - 136 * 86400, today: 136, since: "Sep 20" },
  agents: ["pi", "claude"], defaults: { agent: "pi", folder: "/home/me/Work" },
  folders: ["/home/me/Work", "/home/me/src"], tailnet: true,
};

// -- home ------------------------------------------------------------------
const home = View.build(snap, { view: "home" });
check("home: lifetime, the running card, then what is available",
      types(home), ["life", "run", "sec", "slot", "slot", "field"]);
const life = home.rows[0];
check("home: the lifetime's totals", [life.tokens, life.requests, life.since],
      ["15K tokens", "7 requests", "since Sep 20"]);
check("home: a day a cell, shaded against the busiest, the future blank",
      [life.cells.length, life.cells[136], life.cells[130], life.cells[0], life.cells[137]],
      [140, 4, 1, 0, -1]);
check("home: a hovered day says its date and tokens", life.labels[136], "Thu Sep 24  12K tokens");
const card = home.rows[1];
check("home: the card: name, line, card and memory",
      [card.name, card.line, card.gpu, card.mem], ["qwen", [0, 5000, 12345], "RTX 3090", "18 / 24 GB"]);
check("home: the card's figures and its agent",
      [card.chips.map(c => c.text), card.primary, card.more],
      [["41 tok/s", "12.3K"], { label: "Open pi", action: "open|qwen" }, "more|qwen"]);
const [free, crashed] = byType(home, "slot");
check("home: a free card runs the first model on it",
      [free.label, free.run.label, free.run.action], ["RTX 3090", "run qwen ›", "run|qwen|1"]);
check("home: a crashed start, to run again or dismiss",
      [crashed.crashed, crashed.hint, crashed.run.action, crashed.dismiss],
      [true, "crashed", "again|llama|2", "stop|llama"]);
check("home: a card already running is one 'all GPUs' away",
      [home.rows[5].label, home.rows[5].value, home.rows[5].action], ["all GPUs", "3", "gpus"]);
check("home: the window's mark is the worst of it", home.mark, "failed");

const opened = View.build(snap, { view: "home", open: "gpu:2" });
const links = byType(opened, "links")[0];
check("home: an opened crash says why, with its buttons",
      [links.note, links.items.map(i => i.label)], ["missing tensor", ["Run again ›", "View logs", "Config"]]);
check("home: its chips are the card's memory and heat",
      links.chips.map(c => [c.icon, c.text]), [["memory", "0.4 / 24 GB"], ["temp", "51°"]]);

const quiet = Object.assign({}, snap, { life: Object.assign({}, snap.life, { requests: 0 }) });
check("home: no lifetime until there is one", types(View.build(quiet, {}))[0], "run");
const starting = Object.assign({}, snap, { deployments: [Object.assign({}, snap.deployments[0],
  { state: "starting", detail: "loading", percent: 40 })] });
const sc = View.build(starting, {}).rows.filter(r => r.type === "run")[0];
check("home: a load in progress: how far, and Stop",
      [sc.progress, sc.sub, sc.primary.label, sc.mem], [40, "loading · 40%", "Stop model", ""]);

const pulling = Object.assign({}, snap, { deployments: [{ id: "tiny", name: "tiny", family: "", keys: ["1"],
  state: "download", detail: "3 of 14 GB", percent: 21, session: { tokens: 0, all: {} } }] });
const pc = View.build(pulling, {}).rows.filter(r => r.type === "run")[0];
check("home: a download: how much of how much, the bar, and Stop",
      [pc.sub, pc.progress, pc.primary.label, pc.gpu], ["3 of 14 GB", 21, "Stop model", "RTX 3090"]);
check("home: the window's mark while it downloads", View.build(pulling, {}).mark, "busy");

// -- nothing to run ----------------------------------------------------------
const soon = View.build(Object.assign({}, snap, { kinds: [], deployments: [] }), {});
check("an empty config: what is missing, and where to read how",
      soon.rows.map(r => [r.type, r.head, r.action]),
      [["soon", "No models in your config yet", "url|https://github.com/diverseau/llama-mdl#readme"]]);
check("no snapshot yet: the title, nothing else", View.build(null, {}).rows, []);

// -- a running model's page ---------------------------------------------------
const run = View.build(snap, { view: "run", id: "qwen" });
check("run: name and what it is", [run.hero.name, run.hero.chips.map(c => c.text)],
      ["qwen", ["GGUF Q4_K_M", "1 × RTX 3090", "32K"]]);
check("run: its line's scale and span",
      [run.hero.top, run.hero.mid, run.hero.since, run.hero.now], ["12.3K tokens", "6.2K", "Sep 20", "now"]);
check("run: the figures", byType(run, "grid")[0].cells.map(c => [c.v, c.k]),
      [["41", "decode avg"], ["1.5K", "prefill avg"], ["0.3", "first token"], ["1.5K", "session"],
       ["15K", "week"], ["1:05h", "up"]]);
check("run: the sections, in order", byType(run, "sec").map(r => r.label),
      ["GPUS", "OPENS WITH", "WEIGHTS", "REACH"]);
check("run: what opens it, the folder short",
      byType(run, "field").slice(0, 2).map(r => [r.label, r.value]), [["agent", "pi"], ["folder", "~/Work"]]);
check("run: its weights link to the revision it runs",
      byType(run, "field")[2].action, "url|https://huggingface.co/unsloth/Qwen3-GGUF/tree/abc");
check("run: reach: here, and the tailnet with a key",
      byType(run, "field").slice(3).map(r => [r.label, r.value, r.action || ""]),
      [["this machine", "127.0.0.1:8080", ""], ["tailnet", "share", "share|qwen"]]);
check("run: logs and stop", byType(run, "acts")[0].items.map(i => i.action), ["log|qwen", "stop|qwen"]);

// how fast it runs as the context fills, once there is some of it
const fast = Object.assign({}, snap, { deployments: [Object.assign({}, snap.deployments[0], { ctxMax: 32768,
  session: Object.assign({}, snap.deployments[0].session, {
    speed: { n_ctx: 32768, decode: [[2000, 41.2], [9000, 33.9], [16000, 28.1]], prefill: [[1000, 850], [5000, 700]] },
    lab: { decode: [[4200, 39.0]], prefill: [] } }) })] });
const fh = View.build(fast, { view: "run", id: "qwen" }).hero;
check("run: speed against context: its scale, its span, the lab's dots",
      [fh.line, fh.curve.xmax, fh.curve.ymax, fh.curve.lab, fh.top, fh.mid, fh.since, fh.now, fh.note],
      [undefined, 32768, 50, [[4200, 39.0]], "50 tok/s decode", "", "0", "32K context", "● mdl lab"]);
check("run: a click shows prefill", fh.toggle, "curve|prefill");
const ph = View.build(fast, { view: "run", id: "qwen", curve: "prefill" }).hero;
check("run: prefill on its own scale, a click back",
      [ph.curve.points.length, ph.curve.ymax, ph.top, ph.toggle, ph.note], [2, 1000, "1K tok/s prefill", "curve|decode", ""]);
check("run: until then, the token line and a word on what is coming",
      [run.hero.curve, run.hero.note], [undefined, "speed by context after a few requests"]);
check("numbers: nice", [View.nice(41.2), View.nice(850), View.nice(9.1), View.nice(0)], [50, 1000, 10, 1]);

const shared = Object.assign({}, snap, { deployments: [Object.assign({}, snap.deployments[0],
  { shared: "https://box.tail.ts.net:8080" })] });
const sf = byType(View.build(shared, { view: "run", id: "qwen" }), "field").pop();
check("run: shared, the address hidden and copied",
      [sf.value, sf.secret, sf.action], ["https://box.tail.ts.net:8080", true, "copy|https://box.tail.ts.net:8080"]);
const nokey = Object.assign({}, snap, { deployments: [Object.assign({}, snap.deployments[0],
  { api_key: false, metrics: false })] });
const nk = View.build(nokey, { view: "run", id: "qwen" });
check("run: no key, no share", byType(nk, "field").pop().value, "needs an --api-key");
check("run: no --metrics says what to add", byType(nk, "error")[0].label.indexOf("--metrics") >= 0, true);
check("run: no tailnet, no row",
      byType(View.build(Object.assign({}, snap, { tailnet: false }), { view: "run", id: "qwen" }), "field")
        .map(r => r.label).indexOf("tailnet"), -1);

const picking = View.build(snap, { view: "run", id: "qwen", open: "agent" });
check("run: the agent's choices, the one in use on",
      byType(picking, "opt").map(r => [r.label, r.on, r.action]),
      [["pi", true, "set|agent|pi|qwen"], ["claude", false, "set|agent|claude|qwen"]]);
const folders = View.build(snap, { view: "run", id: "qwen", open: "folder" });
check("run: the folders offered, then a path to type",
      [byType(folders, "opt").map(r => r.label), types(folders).indexOf("path") > 0], [["~/Work", "~/src"], true]);
check("run: a model not running is not a page", View.build(snap, { view: "run", id: "llama" }).title, "MDL");

// -- a card's Config ------------------------------------------------------------
const kind = View.build(snap, { view: "kind", id: "RTX 3090", key: "1" });
check("config: every model, the first chosen",
      byType(kind, "opt").map(r => [r.label, r.value, r.on]),
      [["qwen", "GGUF Q4_K_M  32K", true], ["gemma", "GGUF Q8_0  8K", false]]);
check("config: Run on the card it was opened from", byType(kind, "acts")[0].items[0].action, "run|qwen|1");
const other = View.build(snap, { view: "kind", id: "RTX 3090", key: "1", model: "gemma" });
check("config: another model chosen, its facts and Run",
      [other.hero.chips.map(c => c.icon || c.text), byType(other, "acts")[0].items[0].action],
      [["GGUF Q8_0", "gpu", "context", "weights", "vision"], "run|gemma|1"]);
// a model find picked, not on this machine yet: marked in the list, and Run says it downloads
const picked = JSON.parse(JSON.stringify(snap));
picked.kinds[0].models[1].pull = true;
const pk = View.build(picked, { view: "kind", id: "RTX 3090", key: "1", model: "gemma" });
check("config: a pick carries the mark, a model you have does not",
      byType(pk, "opt").map(r => [r.label, r.mark]), [["qwen", ""], ["gemma", "download"]]);
check("config: its Run says it downloads, and how much",
      byType(pk, "acts")[0].items[0].label, "Download 12 GB and run ›");
picked.kinds[0].models.reverse();
const pickHome = View.build(picked, { view: "home", open: "gpu:1" });
check("home: a free card whose first model is a pick shows the mark, and says so on Run",
      [byType(pickHome, "slot")[0].run.mark, byType(pickHome, "links")[0].items[0].label],
      ["download", "Download 12 GB and run ›"]);
check("the wording lives in one place", View.PICK.run({ sizeGb: 4.2 }), "Download 4.2 GB and run ›");
const busy = View.build(snap, { view: "kind", id: "RTX 3090", key: "0" });
check("config: a card that is busy has no Run", byType(busy, "acts")[0].items[0].action, "");

// -- editing a model's config ---------------------------------------------------------
const form = { ngl: "99", n_cpu_moe: "", ctx: "32768", kv_type: "q8_0", parallel: "1", port: "8080", group: "",
               mmproj: "", flash_attn: "on", args: "--metrics" };
const conf = JSON.parse(JSON.stringify(snap));
conf.configs = { qwen: form, gemma: Object.assign({}, form, { kv_type: "", flash_attn: "" }) };
check("config: a model in the config has Edit, a pick does not",
      [byType(View.build(conf, { view: "kind", id: "RTX 3090", key: "1" }), "acts")[0].items.map(i => i.action),
       byType(View.build(Object.assign({}, conf, { configs: {} }), { view: "kind", id: "RTX 3090", key: "1" }), "acts")[0]
         .items.length],
      [["run|qwen|1", "edit|qwen"], 1]);
check("run: logs, edit and stop",
      byType(View.build(conf, { view: "run", id: "qwen" }), "acts")[0].items.map(i => i.action),
      ["log|qwen", "edit|qwen", "stop|qwen"]);
conf.deployments[0].stale = true;
const stale = byType(View.build(conf, { view: "run", id: "qwen" }), "links");
check("run: a config changed since it started offers the restart",
      stale.map(r => [r.note, r.items[0].action]), [["its config has changed since it started", "again|qwen|0"]]);
conf.deployments[0].stale = false;
const ed = View.build(conf, { view: "edit", id: "qwen" });
check("edit: its fields under their headings, typed or picked",
      ed.rows.filter(r => r.type === "sec" || r.type === "input" || r.type === "field").map(r => r.label),
      ["SERVER", "ngl", "n_cpu_moe", "ctx", "kv_type", "flash_attn", "parallel", "port", "MORE", "mmproj", "group", "args"]);
check("edit: what the config says, and what unset means",
      [byType(ed, "input")[0].value, byType(ed, "input")[1].hint, byType(ed, "field").map(r => r.value)],
      ["99", "llama.cpp default", ["q8_0", "on"]]);
check("edit: a running model saves and restarts, or only saves",
      byType(ed, "acts")[0].items.map(i => [i.label, i.action]),
      [["Save and restart ›", "save|qwen|restart"], ["Save", "save|qwen|"], ["Cancel", "back"]]);
const edGemma = View.build(conf, { view: "edit", id: "gemma", open: "kv_type", draft: { ctx: "4096" } });
check("edit: one not running only saves",
      byType(edGemma, "acts")[0].items.map(i => i.label), ["Save", "Cancel"]);
check("edit: the picker offers llama.cpp's default and the cache types",
      byType(edGemma, "opt").map(r => [r.label, r.on]).slice(0, 3),
      [["llama.cpp default", true], ["f16", false], ["bf16", false]]);
check("edit: what was typed is shown, and lit as a change",
      byType(edGemma, "input").filter(r => r.key === "ctx").map(r => [r.value, r.changed]), [["4096", true]]);
check("edit: a value the list lacks is still offered",
      byType(View.build(conf, { view: "edit", id: "qwen", open: "kv_type", draft: { kv_type: "turbo3" } }), "opt")
        .filter(r => r.on).map(r => r.label), ["turbo3"]);
check("edit: a refused save says why on the page",
      View.build(conf, { view: "edit", id: "qwen", problem: "port must be a whole number" }).rows[0],
      { type: "error", label: "port must be a whole number" });
check("edit: a model not in the config is not a page", View.build(conf, { view: "edit", id: "nope" }).title, "MDL");

// -- every card, and a log ----------------------------------------------------------
const gpus = View.build(snap, { view: "gpus" });
check("gpus: every card, running ones too",
      [types(gpus), byType(gpus, "slot").map(r => r.note || r.hint || r.run.label)],
      [["sec", "slot", "slot", "slot"], ["run qwen ›", "running qwen", "crashed"]]);
check("log: its own page", types(View.build(snap, { view: "log", id: "qwen" })), ["sec", "log"]);

// -- tones: each reaches the contrast it must, on white-on-black -------------------------
const hex = c => "#" + [c.r, c.g, c.b].map(x => ("0" + Math.round(x * 255).toString(16)).slice(-2)).join("");
const white = { r: 1, g: 1, b: 1 }, black = { r: 0, g: 0, b: 0 };
const t = View.tones(white, black, { r: 1, g: 1, b: 1, a: 0.06 }, { r: 0.64, g: 0.64, b: 0.64 });
check("tones: the panel's own on white-on-black",
      [t.ink, t.value, t.label, t.rule].map(hex), ["#ffffff", "#d4d4d4", "#b2b2b2", "#535353"]);
const card2 = View.over({ r: 1, g: 1, b: 1, a: 0.06 }, black);
check("tones: every text tone reaches its Lc",
      ["ink", "value", "label"].map(n => Math.abs(View.apca(t[n], card2)) >= View.LC[n] - 0.5), [true, true, true]);

// -- numbers -------------------------------------------------------------------------------
check("numbers: k", [View.k(0), View.k(999), View.k(1500), View.k(2.5e6)], ["0", "999", "1.5K", "2.5M"]);
check("numbers: dur", [View.dur(5), View.dur(125), View.dur(3900)], ["0m", "2m", "1:05h"]);
check("numbers: ctx, and gb whole from ten", [View.ctx(32768), View.ctx(512), View.gb(8.64), View.gb(18.6)],
      ["32K", "512", "8.6 GB", "19 GB"]);
check("numbers: ago", [View.ago(AT - 60, AT), View.ago(AT - 1800, AT), View.ago(AT - 36000, AT),
                       View.ago(AT - 3 * 86400, AT)], ["now", "30m ago", "10h ago", "3d ago"]);

console.log("\nweb_view: " + fails + " failure(s)");
process.exit(fails ? 1 : 0);
