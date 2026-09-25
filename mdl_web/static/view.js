// What the page shows, as data: mdl's snapshot and the page's ui state in,
// a view out. app.js draws it. No DOM, no side effects, so every page is a
// function of state and can be checked without a browser.
//
// The layout and its restraint - one model per card, its tokens as a line,
// labels a tone below values, one primary action - follow 0xSero's Local AI
// panel for Omarchy (github.com/0xSero/omarchy-local-ai, MIT).

"use strict";

var View = (function () {
  function k(n) {
    if (n === null || n === undefined) return "-";
    return n >= 1e6 ? (Math.round(n / 1e5) / 10) + "M"
      : n >= 1e3 ? (Math.round(n / 100) / 10) + "K" : String(Math.round(n));
  }
  function gb(bytes) {
    if (!bytes) return "-";
    var g = bytes / Math.pow(2, 30);
    if (g < 1) return Math.max(1, Math.round(bytes / Math.pow(2, 20))) + " MB";
    return (g >= 10 ? Math.round(g) : Math.round(g * 10) / 10) + " GB";
  }
  function ctx(n) {
    if (!n) return "-";
    return n >= 1024 ? Math.round(n / 1024) + "K" : String(n);
  }
  function dur(s) {
    s = Math.max(0, Math.round(s || 0));
    if (s < 60) return s + "s";
    if (s < 3600) return Math.floor(s / 60) + "m";
    return Math.floor(s / 3600) + ":" + ("0" + Math.floor(s % 3600 / 60)).slice(-2) + "h";
  }
  function rate(x) { return x === null || x === undefined ? "-" : (x >= 100 ? Math.round(x) : Math.round(x * 10) / 10) + ""; }
  function pct(x) { return x === null || x === undefined ? "-" : Math.round(x * 100) + "%"; }

  function byName(snap, name) {
    return (snap.models || []).filter(function (m) { return m.name === name; })[0] || null;
  }
  function live(m) { return m.state === "ready" || m.state === "loading"; }

  // one line under a model's name: what it is, in the fewest words
  function spec(m) {
    var out = [];
    if (m.quant) out.push({ icon: "weights", text: m.quant });
    var n = m.run && m.run.n_ctx ? m.run.n_ctx : m.ctx;
    if (n) out.push({ icon: "context", text: ctx(n) });
    if (m.size) out.push({ icon: "memory", text: gb(m.size) });
    if (m.vision) out.push({ icon: "vision", text: "" });
    return out;
  }

  function gpuLine(g) {
    return { name: g.name.replace(/^NVIDIA GeForce /, "").replace(/^NVIDIA /, ""),
      used: g.used, total: g.total, temp: g.temp,
      frac: g.total ? Math.min(1, (g.used || 0) / g.total) : null,
      text: gb(g.used).replace(" GB", "") + " / " + gb(g.total) };
  }

  function card(snap, m) {
    var r = m.run || {}, x = r.metrics || null;
    var gpu = (snap.gpus || [])[0];
    return {
      name: m.name, state: m.state, series: r.series || [],
      sub: gpu ? gpuLine(gpu) : null,
      loading: m.state === "loading" ? "loading " + dur(r.up) : null,
      tps: x ? rate(x.tps) : null, tokens: x ? k(x.tokens) : null,
      noMetrics: m.state === "ready" && !x,
      actions: [
        { label: "More", act: "page|model|" + m.name, primary: false },
        { label: "Stop", act: "stop|" + m.name, primary: false },
      ],
    };
  }

  function home(snap) {
    var models = snap.models || [];
    var running = models.filter(live);
    var failed = models.filter(function (m) { return m.state === "failed"; });
    var idle = models.filter(function (m) { return m.state === "stopped" || m.state === "failed"; });
    var tokens = running.reduce(function (a, m) {
      return a + ((m.run && m.run.metrics && m.run.metrics.tokens) || 0); }, 0);
    var groups = [];
    idle.forEach(function (m) {
      var g = m.group || "";
      var at = groups.filter(function (x) { return x.name === g; })[0];
      if (!at) groups.push(at = { name: g, rows: [] });
      at.rows.push({ name: m.name, quant: m.quant || "", ctx: ctx(m.ctx),
        failed: m.state === "failed", error: m.error || null,
        act: "run|" + m.name, open: "page|model|" + m.name });
    });
    groups.sort(function (a, b) { return a.name === "" ? 1 : b.name === "" ? -1 : a.name < b.name ? -1 : 1; });
    return {
      page: "home",
      head: { title: "MDL", version: snap.version || "" },
      summary: running.length
        ? { left: k(tokens) + " tokens", mid: running.length + " running", right: "this session" }
        : { left: "nothing running", mid: "", right: "" },
      cards: running.map(function (m) { return card(snap, m); }),
      failed: failed.map(function (m) { return { name: m.name, error: m.error }; }),
      groups: groups,
      gpus: (snap.gpus || []).map(gpuLine),
      ram: snap.ram && snap.ram.total ? { free: gb(snap.ram.free), total: gb(snap.ram.total) } : null,
      error: snap.error || null,
    };
  }

  function modelPage(snap, name) {
    var m = byName(snap, name);
    if (!m) return { page: "gone", name: name };
    var r = m.run || {}, x = r.metrics || null;
    var v = { page: "model", name: m.name, state: m.state, spec: spec(m), error: m.error || null };
    if (live(m)) {
      v.chart = { series: r.series || [], total: x ? k(x.tokens) : null,
        from: r.series && r.series.length ? "since mdl ui started" : "" };
      v.stats = x ? [
        { value: rate(x.tps) + " tok/s", label: "now" },
        { value: rate(x.decode_avg) + " tok/s", label: "decode avg" },
        { value: k(x.prefill_avg) + " tok/s", label: "prefill avg" },
        { value: k(x.tokens), label: "session" },
        { value: x.kv === null || x.kv === undefined ? ctx(r.n_ctx) : pct(x.kv) + " of " + ctx(r.n_ctx), label: "context" },
        { value: dur(r.up), label: "up" },
      ] : null;
      v.noMetrics = m.state === "ready" && !x;
      v.loading = m.state === "loading" ? "loading " + dur(r.up) : null;
      v.gpus = (snap.gpus || []).map(gpuLine);
      v.reach = r.url ? { url: r.url, key: r.api_key } : null;
      v.actions = [
        { label: "View logs", act: "page|log|" + m.name, primary: false },
        { label: "Stop model", act: "stop|" + m.name, primary: false, quiet: true },
      ];
    } else {
      v.settings = [
        ["file", m.file || "-"],
        ["context", ctx(m.ctx)],
        ["gpu layers", m.ngl === undefined ? "-" : String(m.ngl)],
      ];
      if (m.n_cpu_moe !== undefined) v.settings.push(["experts on cpu", String(m.n_cpu_moe)]);
      if (m.kv_type) v.settings.push(["kv cache", m.kv_type]);
      if (m.flash_attn !== undefined) v.settings.push(["flash attention", m.flash_attn ? "on" : "off"]);
      if (m.parallel) v.settings.push(["parallel", String(m.parallel)]);
      v.settings.push(["port", String(m.port)]);
      if (m.group) v.settings.push(["group", m.group]);
      if (m.own_server) v.settings.push(["llama-server", "its own build"]);
      v.gpus = (snap.gpus || []).map(gpuLine);
      v.actions = [{ label: "Run", act: "run|" + m.name, primary: true }];
    }
    return v;
  }

  function build(snap, ui) {
    if (!snap) return { page: "wait" };
    ui = ui || { page: "home" };
    if (ui.page === "model") return modelPage(snap, ui.name);
    if (ui.page === "log") return { page: "log", name: ui.name };
    return home(snap);
  }

  return { build: build, k: k, gb: gb, ctx: ctx, dur: dur };
})();

if (typeof module !== "undefined") module.exports = View;
