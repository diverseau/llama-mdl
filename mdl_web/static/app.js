// Draws View.build(snapshot, ui) and turns its actions ("verb|arg|arg") into requests: the panel from
// omarchy-local-ai (Panel.qml, MIT, see NOTICE) as a page. The snapshot arrives over server-sent events as it
// changes; nothing here polls it.

"use strict";

(function () {
  var U = 1.4615                        // the panel's Style.space(1), in px, as its screenshots measure it
  var G = 20 * U                        // the gutter
  var root = document.getElementById("app")
  var snap = null, ui = { view: "home", id: "", open: "", key: "", problem: "", model: "" }
  var copied = false, revealed = false, copiedTimer = 0, logTimer = 0
  var T = View.tones({ r: 1, g: 1, b: 1, a: 1 }, { r: 0, g: 0, b: 0, a: 1 }, { r: 1, g: 1, b: 1, a: 0.06 },
    { r: 0xa4 / 255, g: 0xa4 / 255, b: 0xa4 / 255, a: 1 })
  function css(c, a) { return "rgba(" + [c.r, c.g, c.b].map(function(v) { return Math.round(v * 255) }).join(",") + "," + (a === undefined ? 1 : a) + ")" }
  ;[["--ink", T.ink], ["--value", T.value], ["--label", T.label], ["--rule", T.rule], ["--alert", T.alert], ["--alert-rule", T.alertRule]]
    .forEach(function(p) { document.documentElement.style.setProperty(p[0], css(p[1])) })

  // -- pieces ------------------------------------------------------------------------------------------------
  function el(tag, cls, kids) {
    var n = document.createElement(tag)
    if (cls) n.className = cls
    ;(kids || []).forEach(function(k) { if (k !== null && k !== undefined && k !== false) n.appendChild(typeof k === "string" ? document.createTextNode(k) : k) })
    return n
  }
  function label(text, cls, size) {
    var n = el("span", "l" + (cls ? " " + cls : ""), [text || ""])
    if (size) n.style.fontSize = size + "px"
    return n
  }
  // Nerd Font glyphs for the icon names View uses, the panel's own from the same font, each with how far its ink
  // reaches in the font's 1000 units. Where the ink runs past the advance, the panel's label for it is an em wide
  // (as its screenshots measure), so the text after one keeps clear of it
  var GLYPHS = { gpu: [0xf08ae, 832], memory: [0xf035b, 750], temp: [0xf050f, 600], context: [0xf09aa, 668],
    weights: [0xf01a7, 750], vision: [0xf06d0, 918], speed: [0xf140c, 600], tokens: [0xf04a0, 600], agent: [0xf07b7, 771],
    folder: [0xf0256, 832], machine: [0xf0379, 918], tailnet: [0xf0317, 832], check: [0xf012c, 729], down: [0xf0140, 600] }
  function glyph(name) { return GLYPHS[name] ? String.fromCodePoint(GLYPHS[name][0]) : "" }
  function icon(name, size) {
    var n = label(glyph(name), "icon", size)
    if (GLYPHS[name]) n.style.width = (GLYPHS[name][1] > 600 ? 1 : 0.6) * (size || CAPTION) + "px"
    return n
  }
  // only the logos shipped beside this file; any other family shows none and takes no space
  function logo(family, size) {
    if (["qwen", "lfm", "hf"].indexOf(family) < 0) return null
    var i = el("img", "logo")
    i.src = (window.MDL_STATIC || "/static/") + family + ".svg"
    i.alt = ""
    i.style.width = i.style.height = size * U + "px"
    return i
  }
  function click(node, action) {
    if (!action) return node
    node.classList.add("click")
    node.addEventListener("click", function(e) { e.stopPropagation(); activate(action) })
    return node
  }
  // Facts as small icon-and-text pairs, spaced instead of joined with dots
  function chips(items, tone, size) {
    size = size || CAPTION
    return el("div", "chips", (items || []).map(function(c) {
      return el("span", "chip", [c.icon ? icon(c.icon, size) : null, c.text ? label(c.text, tone, size) : null])
    }))
  }

  // The label is centered optically, not on its advance: first on its ink (a trailing "›" carries empty space on
  // its right), then nudged right by a sixth of the space and chevron, since a thin chevron weighs less than the
  // letters and would otherwise leave "Open pi ›" sitting left.
  var measure = document.createElement("canvas").getContext("2d")
  var CAPTION = 14.4                     // the panel's caption, as its text measures
  function ink(text, size) {
    measure.font = (size || CAPTION) + "px " + getComputedStyle(document.body).fontFamily
    var m = measure.measureText(text)
    return { x: -m.actualBoundingBoxLeft, w: m.actualBoundingBoxLeft + m.actualBoundingBoxRight, adv: m.width }
  }
  function btn(o) {
    var text = o.label, chevron = /\s›$/.test(text), g = ink(text), words = ink(text.replace(/\s›$/, ""))
    var weight = chevron ? (g.x + g.w - words.x - words.w) / 6 : 0
    var b = el("button", "btn" + (o.primary ? " primary" : "") + (o.danger ? " danger" : "") + (o.action ? "" : " off"), [label(text)])
    b.style.width = (Math.ceil(g.w) + 24 * U) + "px"
    b.style.height = "33px"                // as the panel draws it at this size
    b.firstChild.style.marginLeft = (g.adv / 2 - (g.x + g.w / 2) + weight) + "px"
    if (o.action) click(b, o.action)
    return b
  }
  // Text that must fit gives way in its middle, as the panel's long values do, so both ends of a long model name
  // stay readable; `room` is the width it may take
  function squeeze(node, room, size) {
    var full = node.dataset.full || node.textContent
    node.dataset.full = full
    if (room <= 0 || ink(full, size).adv <= room) { node.textContent = full; return }
    for (var n = full.length - 1; n > 1; n--) {
      var cut = full.slice(0, Math.ceil(n / 2)) + "…" + full.slice(full.length - Math.floor(n / 2))
      if (ink(cut, size).adv <= room) { node.textContent = cut; return }
    }
  }
  // where a node starts, from its row's left edge
  function leftIn(node, row) { return node.getBoundingClientRect().left - row.getBoundingClientRect().left }

  function right(node, margin) {
    node.classList.add("right")
    if (margin !== undefined) node.style.right = margin + "px"
    return node
  }

  // Tokens over time, cumulative, rising to the right: a dim line over a faint area, so text over it keeps its contrast
  function line(values) {
    var c = el("canvas")
    requestAnimationFrame(function() {
      var w = c.clientWidth, h = c.clientHeight, dpr = window.devicePixelRatio || 1
      if (!w || !h) return
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr)
      var g = c.getContext("2d"), v = values || [], n = v.length, top = Math.max.apply(null, v.concat([1]))
      g.scale(dpr, dpr)
      if (n < 2 || top <= 1) return
      g.beginPath()
      for (var i = 0; i < n; i++) {
        var x = i / (n - 1) * w, y = h - 4 * U - v[i] / top * (h * 0.8)
        if (i) g.lineTo(x, y)
        else g.moveTo(x, y)
      }
      g.strokeStyle = css(T.ink, 0.25)
      g.lineWidth = 1.2 * U
      g.stroke()
      g.lineTo(w, h)
      g.lineTo(0, h)
      g.closePath()
      g.fillStyle = css(T.ink, 0.06)
      g.fill()
    })
    return c
  }

  // Speed against context depth: the same dim line over a faint area, from the first depth seen to the last, on
  // the whole context's width so how much of it use has reached shows; what mdl lab measured as dots
  function curve(c0) {
    var c = el("canvas")
    requestAnimationFrame(function() {
      var w = c.clientWidth, h = c.clientHeight, dpr = window.devicePixelRatio || 1
      if (!w || !h) return
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr)
      var g = c.getContext("2d"), pts = c0.points || []
      g.scale(dpr, dpr)
      function X(d) { return Math.min(1, d / (c0.xmax || 1)) * w }
      // its top a little lower than the token line's: the labels sit where this line is highest
      function Y(v) { return h - 4 * U - v / (c0.ymax || 1) * (h * 0.66) }
      if (pts.length > 1) {
        g.beginPath()
        pts.forEach(function(p, i) { if (i) g.lineTo(X(p[0]), Y(p[1])); else g.moveTo(X(p[0]), Y(p[1])) })
        g.strokeStyle = css(T.ink, 0.25)
        g.lineWidth = 1.2 * U
        g.stroke()
        g.lineTo(X(pts[pts.length - 1][0]), h)
        g.lineTo(X(pts[0][0]), h)
        g.closePath()
        g.fillStyle = css(T.ink, 0.06)
        g.fill()
      }
      ;(c0.lab || []).forEach(function(p) {
        g.beginPath()
        g.arc(X(p[0]), Y(p[1]), 2 * U, 0, 2 * Math.PI)
        g.fillStyle = css(T.ink, 0.5)
        g.fill()
      })
    })
    return c
  }

  // -- rows --------------------------------------------------------------------------------------------------
  function lifeRow(r) {
    var days = r.cells || [], cols = Math.ceil(days.length / 7)
    var cell = Math.min(12 * U, (493 - 2 * G - (cols - 1) * 3 * U) / cols)
    // the panel's squares land on whole pixels: 17 wide, 22 apart
    var size = Math.floor(cell), pitch = Math.round(cell + 3 * U)
    var hover = -1, since = label(r.since, "label")
    var top = el("div", "top", [label(r.tokens, "ink"), label(r.requests, "label"), right(since)])
    var grid = el("div", "cells")
    grid.style.height = (7 * pitch - (pitch - size)) + "px"
    days.forEach(function(level, i) {
      var sq = el("i")
      sq.style.left = Math.floor(i / 7) * pitch + "px"
      sq.style.top = (i % 7) * pitch + "px"
      sq.style.width = sq.style.height = size + "px"
      sq.style.background = level < 0 ? "transparent" : "rgba(255,255,255," + [0.07, 0.25, 0.45, 0.7, 0.95][level] + ")"
      if (level >= 0) {
        sq.addEventListener("mouseenter", function() {
          hover = i; sq.classList.add("hover")
          since.textContent = (r.labels || [])[i] || ""; since.className = "l ink right"
        })
        sq.addEventListener("mouseleave", function() {
          if (hover !== i) return
          hover = -1; sq.classList.remove("hover")
          since.textContent = r.since; since.className = "l label right"
        })
      }
      grid.appendChild(sq)
    })
    var months = el("div", "months", (r.months || []).map(function(m) {
      var l = label(m.label)
      l.style.left = (G + m.col * pitch) + "px"
      return l
    }))
    return el("div", "life", [top, grid, months])
  }

  function runRow(r) {
    var col = el("div", "col", [
      el("div", "name", [logo(r.family, 18), label(r.name)]),
      el("div", "on", [label(r.gpu, "label"), r.mem ? label(r.mem, "label dim") : null]),
      el("div", "spacer"),
      r.sub ? el("div", "l sub value", [r.sub]) : null,
      r.progress >= 0 ? el("div", "bar2", [(function() { var i = el("i"); i.style.width = r.progress + "%"; return i })()]) : null])
    var dim = col.querySelector(".dim")
    if (dim) dim.style.opacity = 0.6
    // speed and tokens, small, in the bottom-right corner, level with the buttons
    var corner = chips(r.chips || [], "label", 9 * U)
    corner.classList.add("corner")
    return el("div", "run breathes", [line(r.line), col, corner,
      el("div", "btns", [btn({ label: r.primary.label + (r.primary.quiet ? "" : " ›"), action: r.primary.action, primary: !r.primary.quiet,
        danger: !!r.primary.quiet }), btn({ label: "More", action: r.more })])])
  }

  function slotRow(r) {
    var row = el("div", "row slot" + (r.crashed ? " crashed" : ""))
    if (r.crashed) row.appendChild(el("div", "dash"))
    click(row, r.toggle)
    row.appendChild(el("div", "left", [label(r.label, r.open ? "ink" : "value"), r.hint ? label(r.hint, "alert") : null]))
    if (r.run) {
      var runLabel = label(r.run.label, "ink")
      var go = right(click(el("div", "go", [logo(r.run.family, 12), runLabel]), r.run.action))
      row.appendChild(go)
      requestAnimationFrame(function() {
        var left = row.querySelector(".left")
        var room = row.clientWidth - G - (leftIn(left, row) + left.offsetWidth + 16 * U) - (go.offsetWidth - runLabel.offsetWidth)
        squeeze(runLabel, room)
      })
      if (r.dismiss) {
        var dis = click(label("dismiss", "dismiss"), r.dismiss)
        row.appendChild(dis)
        requestAnimationFrame(function() { dis.style.right = (G + go.offsetWidth + 16 * U) + "px" })
      }
    } else {
      row.appendChild(right(label(r.note || "", r.warn ? "alert" : "label")))
    }
    return row
  }

  function linksRow(r) {
    var kids = []
    if ((r.chips || []).length) { var c = chips(r.chips, "label"); c.classList.add("at-gutter"); kids.push(c) }
    if (r.note) kids.push(el("div", "l note at-gutter", [r.note]))
    if ((r.items || []).length) kids.push(el("div", "flow at-gutter", r.items.map(btn)))
    return el("div", "links", kids)
  }

  function soonRow(r) {
    var wave = el("div", "wave"), c = el("canvas")
    wave.appendChild(c)
    requestAnimationFrame(function() { drawWave(c) })
    return el("div", "soon", [wave, label(r.head, "value"), btn({ label: r.button, action: r.action })])
  }
  // a square wave drifting left, thin and quiet, fading out at both ends
  var waveX = 0
  function drawWave(c) {
    var w = Math.round(140 * U), h = Math.round(18 * U), dpr = window.devicePixelRatio || 1, period = 28 * U, stroke = 1.5
    c.width = w * dpr; c.height = h * dpr; c.style.width = w + "px"; c.style.height = h + "px"
    var g = c.getContext("2d")
    function frame() {
      if (!c.isConnected) return
      g.setTransform(dpr, 0, 0, dpr, 0, 0)
      g.clearRect(0, 0, w, h)
      g.strokeStyle = css(T.label); g.lineWidth = stroke
      g.beginPath()
      for (var x = waveX - period; x < w + period; x += period) {
        g.moveTo(x, h - 2); g.lineTo(x, 2); g.lineTo(x + period / 2, 2); g.lineTo(x + period / 2, h - 2); g.lineTo(x + period, h - 2)
      }
      g.stroke()
      var fade = g.createLinearGradient(0, 0, w, 0)
      fade.addColorStop(0, "#000"); fade.addColorStop(0.25, "rgba(0,0,0,0)"); fade.addColorStop(0.75, "rgba(0,0,0,0)"); fade.addColorStop(1, "#000")
      g.fillStyle = fade; g.fillRect(0, 0, w, h)
      waveX = (waveX - period * 50 / 2400) % period
      setTimeout(function() { requestAnimationFrame(frame) }, 50)
    }
    frame()
  }

  function textRow(r) {
    return r.type === "sec" ? el("div", "l sec", [r.label]) : el("div", "l error", [r.label || ""])
  }

  function gridRow(r) {
    return el("div", "grid", r.cells.map(function(c) {
      return el("div", "cell", [el("div", "fig", [label(c.v), c.u ? label(c.u, "label") : null]), label(c.k, "label")])
    }))
  }

  function gpuRow(r) {
    var row = el("div", "row gpu" + (r.status ? " status" : ""))
    row.appendChild(el("div", "names", [label(r.name), r.status ? label(r.status, "label") : null]))
    var mem = right(label(r.mem + (r.temp ? "  " + r.temp : "")))
    row.appendChild(mem)
    if (r.bar) {
      var meter = el("div", "meter"), fill = el("i")
      fill.style.width = r.pct + "%"
      meter.appendChild(fill)
      row.appendChild(meter)
      requestAnimationFrame(function() {
        var x = G + 104 * U
        meter.style.width = Math.max(0, row.clientWidth - G - mem.offsetWidth - x - 12 * U) + "px"
      })
    }
    return row
  }

  function fieldRow(r) {
    var row = click(el("div", "row field"), r.secret ? "" : r.action)
    var left = el("div", "left", [r.icon ? el("span", "glyph", [icon(r.icon)]) : null, logo(r.logo, 12), label(r.label, "label")])
    row.appendChild(left)
    var text = r.secret ? (copied ? "copied" : "copy") : r.value
    var val = el("span", "l val " + (r.open ? "ink" : "value"), [text])
    if (!r.secret && r.drop) val.appendChild(document.createTextNode("  " + glyph("down")))
    else if (!r.secret && r.action) val.appendChild(document.createTextNode(" ›"))
    right(val)
    row.appendChild(val)
    // a long value (a weights repository) gives way in its middle rather than run over the label
    if (!r.secret) requestAnimationFrame(function() {
      squeeze(val, row.clientWidth - left.offsetLeft - left.offsetWidth - G - 16 * U)
    })
    if (r.secret) {
      click(val, r.action)
      var hid = label(revealed ? r.value : r.value.replace(/[^.:\/]+/g, "•••"), "secret")
      hid.style.opacity = revealed ? 1 : 0.55
      hid.addEventListener("click", function(e) { e.stopPropagation(); revealed = !revealed; render() })
      row.appendChild(hid)
      requestAnimationFrame(function() { hid.style.right = (G + val.offsetWidth + 10 * U) + "px" })
    }
    return row
  }

  function optRow(r) {
    var row = click(el("div", "row opt"), r.action), name = label(r.label, r.on ? "ink" : "value")
    var val = r.value ? right(label(r.value, "label")) : null
    row.appendChild(el("div", "left", [el("span", "glyph", [r.on ? icon("check") : null]), name]))
    if (val) row.appendChild(val)
    requestAnimationFrame(function() {
      squeeze(name, row.clientWidth - G - (val ? val.offsetWidth + 16 * U : 0) - leftIn(name, row))
    })
    return row
  }

  function pathRow(r) {
    var input = el("input")
    input.placeholder = "or type a path"
    input.spellcheck = false
    input.addEventListener("keydown", function(e) {
      if (e.key === "Enter" && input.value.trim()) activate("set|folder|" + input.value.trim() + "|" + r.id)
    })
    return el("div", "row path", [input])
  }

  function logRow(r) {
    var pre = el("pre", "log", ["reading..."])
    function load() {
      fetch("/api/log?name=" + encodeURIComponent(r.id), { credentials: "same-origin" })
        .then(function(x) { return x.ok ? x.text() : "no log yet" })
        .then(function(t) {
          var end = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8
          pre.textContent = t
          if (end) pre.scrollTop = pre.scrollHeight
        })
    }
    load()
    clearInterval(logTimer)
    logTimer = setInterval(function() { if (pre.isConnected) load(); else clearInterval(logTimer) }, 2000)
    return pre
  }

  function hero(h) {
    var who = el("div", "who", [el("div", "name", [logo(h.family, 14), label(h.name)]), chips(h.chips, "value")])
    var box = el("div", "hero", [who])
    if (h.line || h.curve) {
      var surface = el("div", "hero-line breathes", [h.curve ? curve(h.curve) : line(h.line)])
      if (h.toggle) click(surface, h.toggle)
      if (h.note) {
        var note = label(h.note, "label note")
        note.style.top = (8 * U - 3) + "px"
        surface.appendChild(note)
      }
      var t = label(h.top, "label"), m = label(h.mid, "label"), since = label(h.since, "label"), now = label(h.now, "label")
      t.style.top = (8 * U - 3) + "px"
      m.style.top = "50%"; m.style.transform = "translateY(-50%)"
      since.style.bottom = (8 * U) + "px"
      now.style.bottom = (8 * U) + "px"; now.style.left = "auto"; now.style.right = (6 * U) + "px"
      ;[t, m, since, now].forEach(function(x) { surface.appendChild(x) })
      box.appendChild(surface)
    }
    return box
  }

  var ROWS = { life: lifeRow, run: runRow, slot: slotRow, links: linksRow, soon: soonRow, grid: gridRow, gpu: gpuRow,
    field: fieldRow, opt: optRow, path: pathRow, acts: linksRow, log: logRow }

  // The space above row i: a group opens a gap, a surface follows a surface closely, rows in a group touch
  function gapBefore(v, i) {
    var rows = v.rows || [], t = rows[i].type
    if (i === 0 && !v.hero && t !== "sec") return 12 * U
    if (t === "sec" || t === "acts" || t === "error") return 20 * U
    if (t === "run" || t === "grid") return i === 0 && !v.hero ? 12 * U : 8 * U
    return i === 0 ? 12 * U : 0
  }

  // The window's icon is the panel's mark: nine dots, faint when idle, lit when a model is ready, urgent when one
  // failed, a diagonal ripple while working
  var markState = null, ripple = 0, rippling = null
  function markIcon() {
    var link = document.querySelector("link[rel=icon]"), dots = ""
    if (!link) return
    for (var i = 0; i < 9; i++) {
      var on = markState === "busy" ? (i % 3 + Math.floor(i / 3)) === ripple % 5 : !!markState
      dots += "<circle cx='" + (6 + 10 * (i % 3)) + "' cy='" + (6 + 10 * Math.floor(i / 3)) + "' r='3' fill='" +
        (markState === "failed" ? "#a4a4a4" : "#fff") + "' fill-opacity='" + (markState === "failed" || on ? 1 : 0.3) + "'/>"
    }
    link.href = "data:image/svg+xml," + encodeURIComponent("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>" +
      "<rect width='32' height='32' rx='7' fill='#000'/>" + dots + "</svg>")
  }
  function setMark(m) {
    if (m === markState) return
    markState = m
    clearInterval(rippling)
    rippling = m === "busy" ? setInterval(function() { ripple = (ripple + 1) % 5; markIcon() }, 160) : null
    markIcon()
  }

  function render() {
    if (!snap) { root.replaceChildren(); return }
    var v
    try {
      v = View.build(snap, ui)
    } catch (e) {
      v = { title: "MDL", mark: "failed", rows: [{ type: "error", label: "could not read mdl's answer: " + e.message }] }
    }
    var kids = []
    var head = el("div", "head", [click(label(v.back ? "‹ home" : v.title, v.back ? "value" : "label"), v.back ? "home" : ""),
      !v.back && v.version ? label(v.version, "version") : null])
    kids.push(head)
    if (v.hero) {
      var h = hero(v.hero)
      h.style.marginTop = (12 * U + 1) + "px"
      kids.push(h)
    }
    ;(v.rows || []).forEach(function(r, i) {
      var node = (ROWS[r.type] || textRow)(r)
      node.style.marginTop = gapBefore(v, i) + "px"
      kids.push(node)
    })
    var y = window.scrollY
    root.replaceChildren.apply(root, kids)
    window.scrollTo(0, y)
    document.title = "MDL" + (v.mark === "failed" ? " · failed" : v.mark === "busy" ? " · working" : "")
    setMark(v.mark)
  }

  // -- actions -----------------------------------------------------------------------------------------------
  // a new view starts at its top with nothing chosen; within a view, a chosen model stays chosen
  function nav(patch) {
    var moved = patch.view !== undefined || patch.id !== undefined
    ui = Object.assign({ view: ui.view, id: ui.id, open: "", key: ui.key, problem: "", model: moved ? "" : ui.model || "",
      curve: ui.curve || "" }, patch)
    revealed = false
    if (moved) {
      var hash = ui.view === "home" ? "" : "#" + [ui.view, ui.id, ui.key].map(encodeURIComponent).join("/")
      if (location.hash !== hash) history.pushState(null, "", hash || location.pathname)
      window.scrollTo(0, 0)
    }
    render()
  }
  function goHome() { nav({ view: "home", id: "", key: "" }) }
  function post(body) {
    return fetch("/api/action", { method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) })
      .then(function(r) { return r.json().catch(function() { return { ok: false, error: "HTTP " + r.status } }) },
        function() { return { ok: false, error: "mdl ui is not answering" } })
      .then(function(res) {
        // a verb that fails says why, on the home page, as the panel does
        if (!res.ok) nav({ view: "home", id: "", key: "", problem: res.error || "that did not work (see the log)" })
        return res
      })
  }
  function activate(action) {
    var a = (action || "").split("|")
    switch (a[0]) {
    case "run": post({ verb: "run", name: a[1], keys: a[2] || "" }); goHome(); break
    case "again": post({ verb: "again", name: a[1], keys: a[2] || "" }); goHome(); break
    case "stop": post({ verb: "stop", name: a[1] }); goHome(); break
    case "open": post({ verb: "open", name: a[1] }); break
    case "share": post({ verb: "share", name: a[1] }); break
    case "set": post({ verb: "set", key: a[1], value: a.slice(2, a.length - 1).join("|"), name: a[a.length - 1] }); nav({ open: "" }); break
    case "more": nav({ view: "run", id: a[1] }); break
    case "kind": nav({ view: "kind", id: a[1], key: a[2] || "" }); break
    case "group": nav({ view: "group", id: a[1], key: a[2] }); break
    case "model": nav({ model: a[1] }); break
    case "curve": nav({ curve: a[1] }); break
    case "gpus": nav({ view: "gpus", id: "" }); break
    case "pick": nav({ open: ui.open === a[1] ? "" : a[1] }); break
    case "home": goHome(); break
    case "log": nav({ view: "log", id: a[1] }); break
    case "url": post({ verb: "url", url: a.slice(1).join("|") }); break
    case "copy":
      navigator.clipboard.writeText(a.slice(1).join("|")).catch(function() {})
      copied = true; render()
      clearTimeout(copiedTimer); copiedTimer = setTimeout(function() { copied = false; render() }, 1500)
      break
    }
  }
  function fromHash() {
    var p = location.hash.slice(1).split("/").map(decodeURIComponent)
    ui = Object.assign({}, ui, { view: p[0] || "home", id: p[1] || "", key: p[2] || "", open: "", model: "", problem: "" })
    render()
  }
  window.addEventListener("popstate", fromHash)
  window.addEventListener("keydown", function(e) {
    if (e.key === "Escape" && document.activeElement.tagName !== "INPUT") goHome()
  })

  // the page is drawn once its font is in: buttons are sized by the ink of their labels
  var ready = document.fonts.load(CAPTION + 'px "mdl mono"', "A" + glyph("gpu")).catch(function() {})

  // -- the snapshot stream -----------------------------------------------------------------------------------
  // the fixture hook: a page that sets window.MDL_SNAPSHOT draws that and connects to nothing
  if (window.MDL_SNAPSHOT) {
    snap = window.MDL_SNAPSHOT
    if (window.MDL_UI) ui = Object.assign(ui, window.MDL_UI)
    ready.then(render)
    return
  }
  var es = new EventSource("/api/events")
  es.addEventListener("snapshot", function(e) {
    var s = View.parse(e.data)
    if (!s) return
    snap = s
    document.body.classList.remove("offline")
    // a row being typed in is not redrawn under the cursor
    if (document.activeElement && document.activeElement.tagName === "INPUT") return
    render()
  })
  es.onerror = function() { document.body.classList.add("offline") }
  ready.then(fromHash)
})()
