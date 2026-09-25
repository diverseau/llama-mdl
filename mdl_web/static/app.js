// Draws View.build(snapshot, ui) and turns its actions ("verb|arg|arg")
// into requests. The snapshot arrives over server-sent events as it
// changes; nothing here polls it.

"use strict";

(function () {
  var snap = null;
  var ui = { page: "home", name: null };
  var busy = {};              // name -> the verb sent and not yet seen through
  var logTimer = null;
  var root = document.getElementById("app");
  var toastEl = document.getElementById("toast");

  // -- a few icons, drawn as paths so the page needs no font of glyphs ------
  var PATHS = {
    back: "M15 5l-7 7 7 7",
    chev: "M9 5l7 7-7 7",
    weights: "M6 8h12l2 12H4L6 8zm3 0a3 3 0 016 0",
    context: "M4 6h16M4 10h16M4 14h10M4 18h7",
    memory: "M7 4h10v16H7zM4 8h3M4 12h3M4 16h3M17 8h3M17 12h3M17 16h3",
    vision: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zm10 3a3 3 0 100-6 3 3 0 000 6z",
    speed: "M13 2L4 14h7l-1 8 9-12h-7l1-8z",
    sum: "M18 4H6l7 8-7 8h12",
    gpu: "M3 7h18v10H3zM7 17v3M17 17v3M7 10h4v4H7z",
    machine: "M3 5h18v11H3zM9 20h6M12 16v4",
    key: "M14 10a4 4 0 11-2.8-3.8L21 4v4h-3v3h-3l-1.2 1.2",
    ram: "M4 8h16v8H4zM7 16v3M11 16v3M15 16v3M7 11h2M11 11h2M15 11h2",
    model: "M12 2l8 4.5v9L12 20l-8-4.5v-9L12 2zm0 0v18M4 6.5l8 4.5 8-4.5",
  };
  function icon(name, cls) {
    var ns = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("class", "icon" + (cls ? " " + cls : ""));
    svg.setAttribute("aria-hidden", "true");
    var p = document.createElementNS(ns, "path");
    p.setAttribute("d", PATHS[name] || "");
    svg.appendChild(p);
    return svg;
  }

  // -- a tiny element builder: el("div.card", child, "text", ...) ----------
  function el(spec) {
    var parts = spec.split("."), node = document.createElement(parts[0] || "div");
    if (parts.length > 1) node.className = parts.slice(1).join(" ");
    for (var i = 1; i < arguments.length; i++) {
      var c = arguments[i];
      if (c === null || c === undefined || c === false) continue;
      if (typeof c === "string" || typeof c === "number") node.appendChild(document.createTextNode(String(c)));
      else if (Array.isArray(c)) c.forEach(function (x) { if (x) node.appendChild(x); });
      else node.appendChild(c);
    }
    return node;
  }
  function button(a) {
    var b = el("button.btn" + (a.primary ? ".primary" : "") + (a.quiet ? ".quiet" : ""), a.label,
      a.primary ? icon("chev", "after") : null);
    var verb = a.act.split("|")[0], name = a.act.split("|").pop();
    if ((verb === "run" || verb === "stop") && busy[name]) {
      b.disabled = true;
      b.textContent = busy[name] === "run" ? "Starting..." : "Stopping...";
    }
    b.addEventListener("click", function () { act(a.act); });
    return b;
  }
  function link(text, action, cls) {
    var a = el("button.link" + (cls ? "." + cls : ""), text);
    a.addEventListener("click", function () { act(action); });
    return a;
  }
  function toast(text, bad) {
    toastEl.textContent = text;
    toastEl.className = "toast show" + (bad ? " bad" : "");
    clearTimeout(toast.t);
    toast.t = setTimeout(function () { toastEl.className = "toast"; }, bad ? 7000 : 3000);
  }

  // -- the tokens line ---------------------------------------------------------
  function chart(series, tall) {
    var c = el("canvas.chart" + (tall ? ".tall" : ""));
    requestAnimationFrame(function () { drawLine(c, series); });
    return c;
  }
  function drawLine(c, series) {
    var dpr = window.devicePixelRatio || 1;
    var w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    var g = c.getContext("2d");
    g.scale(dpr, dpr);
    var css = getComputedStyle(document.documentElement);
    var ink = css.getPropertyValue("--line").trim(), fill = css.getPropertyValue("--fill").trim();
    var pts = series || [];
    if (pts.length < 2) {
      g.strokeStyle = ink; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(0, h - 1.5); g.lineTo(w, h - 1.5); g.stroke();
      return;
    }
    var t0 = pts[0][0], t1 = pts[pts.length - 1][0];
    var v0 = pts[0][1], v1 = pts[pts.length - 1][1];
    var span = Math.max(t1 - t0, 1), rise = Math.max(v1 - v0, 1);
    function X(t) { return (t - t0) / span * w; }
    function Y(v) { return h - 2 - (v - v0) / rise * (h * 0.78); }
    g.beginPath();
    g.moveTo(0, h);
    pts.forEach(function (p) { g.lineTo(X(p[0]), Y(p[1])); });
    g.lineTo(w, h);
    g.closePath();
    g.fillStyle = fill; g.fill();
    g.beginPath();
    pts.forEach(function (p, i) { if (i) g.lineTo(X(p[0]), Y(p[1])); else g.moveTo(X(p[0]), Y(p[1])); });
    g.strokeStyle = ink; g.lineWidth = 1.5; g.lineJoin = "round"; g.stroke();
  }

  // -- pages -------------------------------------------------------------
  function specLine(items) {
    return el("div.spec", items.map(function (s) {
      return el("span.item", icon(s.icon), s.text ? el("span", s.text) : null);
    }));
  }
  function gpuRows(gpus) {
    if (!gpus || !gpus.length) return null;
    return el("section", el("h2", "GPUS"), gpus.map(function (g) {
      var bar = el("span.bar", el("span.fill"));
      if (g.frac !== null) bar.firstChild.style.width = Math.round(g.frac * 100) + "%";
      return el("div.row", el("span.value", g.name), el("span.grow"), bar,
        el("span.label", g.text), g.temp !== null ? el("span.label", Math.round(g.temp) + "°") : null);
    }));
  }

  function renderHome(v) {
    var out = [];
    out.push(el("header", el("span.title", v.head.title), el("span.version", v.head.version)));
    if (v.error) out.push(el("div.alert", v.error));
    out.push(el("div.summary", el("span.value", v.summary.left), el("span.label", v.summary.mid),
      el("span.grow"), el("span.label", v.summary.right)));
    v.cards.forEach(function (c) {
      var foot = el("div.foot",
        el("div.actions", c.actions.map(button)),
        el("span.grow"),
        c.loading ? el("span.label.pulse", c.loading) : null,
        c.noMetrics ? el("span.label", "add --metrics for live speed") : null,
        c.tps !== null ? el("span.stat", icon("speed"), c.tps + " tok/s") : null,
        c.tokens !== null ? el("span.stat", icon("sum"), c.tokens) : null);
      var head = el("div.cardhead", icon("model", "big"), el("span.name", c.name));
      var sub = c.sub ? el("div.sub", el("span.value", c.sub.name), el("span.label", c.sub.text)) : null;
      var box = el("div.card" + (c.state === "loading" ? ".loading" : ""), head, sub, chart(c.series), foot);
      head.addEventListener("click", function () { act("page|model|" + c.name); });
      out.push(box);
    });
    v.failed.forEach(function (f) {
      out.push(el("div.alert", el("span.value", f.name + " did not start"), el("div.label", f.error || "")));
    });
    if (v.groups.length) {
      var sec = el("section", el("h2", "MODELS"));
      v.groups.forEach(function (g) {
        if (g.name) sec.appendChild(el("div.group", g.name));
        g.rows.forEach(function (r) {
          var row = el("div.row.model" + (r.failed ? ".failed" : ""),
            link(r.name, r.open, "value"),
            el("span.label", r.quant), el("span.label", r.ctx), el("span.grow"),
            busy[r.name] ? el("span.label.pulse", busy[r.name] === "run" ? "starting" : "stopping")
              : link("run", r.act, "go"));
          sec.appendChild(row);
        });
      });
      out.push(sec);
    }
    var sys = [];
    v.gpus.forEach(function (g) {
      sys.push(el("div.row", icon("gpu"), el("span.value", g.name), el("span.grow"),
        el("span.label", g.text), g.temp !== null ? el("span.label", Math.round(g.temp) + "°") : null));
    });
    if (v.ram) sys.push(el("div.row", icon("ram"), el("span.value", "RAM"), el("span.grow"),
      el("span.label", v.ram.free + " free of " + v.ram.total)));
    if (sys.length) out.push(el("section", el("h2", "THIS MACHINE"), sys));
    return out;
  }

  function renderModel(v) {
    var out = [el("nav", link("‹ home", "page|home"))];
    out.push(el("div.titleline", icon("model", "big"), el("span.name", v.name)));
    out.push(specLine(v.spec));
    if (v.error) out.push(el("div.alert", el("span.value", "did not start"), el("div.label", v.error)));
    if (v.chart) {
      var box = el("div.chartbox", chart(v.chart.series, true),
        el("div.overlay.top", v.chart.total ? v.chart.total + " tokens" : ""),
        el("div.overlay.bottom", el("span", v.chart.from)));
      out.push(box);
    }
    if (v.loading) out.push(el("div.row", el("span.label.pulse", v.loading)));
    if (v.noMetrics) out.push(el("div.row", el("span.label", "started without --metrics: no counters to show")));
    if (v.stats) out.push(el("div.grid", v.stats.map(function (s) {
      return el("div.cell", el("div.value", s.value), el("div.label", s.label)); })));
    if (v.settings) out.push(el("section", el("h2", "SETTINGS"), v.settings.map(function (kv) {
      return el("div.row", el("span.label", kv[0]), el("span.grow"), el("span.value.clip", kv[1]));
    })));
    var g = gpuRows(v.gpus);
    if (g) out.push(g);
    if (v.reach) {
      var copy = link(v.reach.url, "copy|" + v.reach.url, "value");
      out.push(el("section", el("h2", "REACH"),
        el("div.row", icon("machine"), el("span.label", "this machine"), el("span.grow"), copy),
        el("div.row", icon("key"), el("span.label", "api key"), el("span.grow"),
          el("span.value", v.reach.key ? "required" : "none"))));
    }
    if (v.actions) out.push(el("div.actions.bottom", v.actions.map(button)));
    return out;
  }

  function renderLog(v) {
    var pre = el("pre.log", "reading...");
    fetchLog(v.name, pre);
    clearInterval(logTimer);
    logTimer = setInterval(function () { fetchLog(v.name, pre); }, 2000);
    return [el("nav", link("‹ " + v.name, "page|model|" + v.name)), el("h2", "LOG"), pre];
  }
  function fetchLog(name, pre) {
    fetch("/api/log?name=" + encodeURIComponent(name), { credentials: "same-origin" })
      .then(function (r) { return r.ok ? r.text() : "no log yet"; })
      .then(function (t) {
        var atEnd = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
        pre.textContent = t;
        if (atEnd) pre.scrollTop = pre.scrollHeight;
      });
  }

  function render() {
    var v = View.build(snap, ui);
    if (v.page !== "log") clearInterval(logTimer);
    var kids = v.page === "wait" ? [el("div.label.pulse", "reading mdl...")]
      : v.page === "home" ? renderHome(v)
      : v.page === "model" ? renderModel(v)
      : v.page === "log" ? renderLog(v)
      : [el("nav", link("‹ home", "page|home")), el("div.label", v.name + " is no longer in the config")];
    // a log page redraws itself; a new snapshot must not reset its scroll
    if (v.page === "log" && root.dataset.page === "log" && root.dataset.name === v.name) return;
    root.dataset.page = v.page; root.dataset.name = v.name || "";
    root.replaceChildren.apply(root, kids);
  }

  // -- actions ---------------------------------------------------------------
  function act(action) {
    var a = action.split("|");
    if (a[0] === "page") {
      // the page lives in the hash, so reload and the back button work
      var hash = a[1] === "home" ? "" : "#" + a[1] + "/" + encodeURIComponent(a[2] || "");
      if (location.hash !== hash) history.pushState(null, "", hash || location.pathname);
      fromHash();
      window.scrollTo(0, 0);
      return;
    }
    if (a[0] === "copy") {
      navigator.clipboard.writeText(a[1]).then(function () { toast("copied " + a[1]); },
        function () { toast("could not copy", true); });
      return;
    }
    if (a[0] === "run" || a[0] === "stop") {
      busy[a[1]] = a[0];
      render();
      fetch("/api/action", {
        method: "POST", credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ verb: a[0], name: a[1] }),
      }).then(function (r) { return r.json().catch(function () { return { ok: false, error: "HTTP " + r.status }; }); })
        .then(function (res) {
          delete busy[a[1]];
          if (!res.ok) toast(res.error || "failed", true);
          render();
        }, function () { delete busy[a[1]]; toast("mdl ui is not answering", true); render(); });
    }
  }

  // -- the snapshot stream -------------------------------------------------
  function connect() {
    var es = new EventSource("/api/events");
    es.addEventListener("snapshot", function (e) {
      try { snap = JSON.parse(e.data); } catch (err) { return; }
      document.body.classList.remove("offline");
      render();
    });
    es.onerror = function () { document.body.classList.add("offline"); };
  }
  function fromHash() {
    var m = /^#(model|log)\/(.+)$/.exec(location.hash);
    ui = m ? { page: m[1], name: decodeURIComponent(m[2]) } : { page: "home", name: null };
    root.dataset.page = "";
    render();
  }
  window.addEventListener("popstate", fromHash);
  window.addEventListener("resize", function () { root.dataset.page = ""; render(); });
  connect();
  fromHash();
})();
