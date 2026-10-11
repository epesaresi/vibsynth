/* VChart: small dependency-free interactive chart for vibsynth.
 *  - drag: zoom to box            - wheel: zoom (Shift: x only, Alt: y only)
 *  - right/middle drag or Shift+drag: pan     - double click: reset
 *  - hover: crosshair and values  - legend: click to show / hide a series
 *  - toolbar: x log, y log, reset, PNG, CSV, enlarge
 * Series: {name, x, y, color, width, dash, mode:'line'|'points'|'both'|'step', axis:'y'|'y2'}
 * Image:  {x0,x1,y0,y1,z:[[...]] (rows = y), zmin, zmax, label}
 */
(function () {
  const PALETTE = ["#1f5a8a", "#c0504d", "#4f8f3a", "#8a5ab0", "#d08a1a", "#2a9d9b", "#7a7a7a", "#b5487f"];
  const css = (n, d) => (getComputedStyle(document.documentElement).getPropertyValue(n) || d).trim() || d;
  function fmt(v) {
    if (v === null || v === undefined || !isFinite(v)) return "–";
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(3);
    return (+v.toPrecision(5)).toString();
  }
  function niceTicks(a, b, n) {
    const span = b - a; if (!(span > 0)) return [a];
    const step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const r = step0 / mag, step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
    const out = []; for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }
  function logTicks(a, b) { // a,b in log10 units
    const out = [], lo = Math.floor(a), hi = Math.ceil(b), dec = hi - lo;
    for (let e = lo; e <= hi; e++) {
      const mults = dec <= 2 ? [1, 2, 3, 5] : dec <= 5 ? [1, 2, 5] : [1];
      for (const m of mults) { const v = Math.log10(m) + e; if (v >= a - 1e-9 && v <= b + 1e-9) out.push({ v, major: m === 1 }); }
    }
    return out;
  }
  function viridis(t) {
    const c = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
    t = Math.min(1, Math.max(0, t)) * 4; const i = Math.min(3, Math.floor(t)), f = t - i;
    return c[i].map((v, k) => Math.round(v + (c[i + 1][k] - v) * f));
  }

  class VChart {
    constructor(el, o = {}) {
      this.el = typeof el === "string" ? document.querySelector(el) : el;
      this.o = Object.assign({ title: "", xlabel: "", ylabel: "", y2label: "", xlog: false, ylog: false,
        height: 300, legend: true, xpad: 0, symmetricY: false }, o);
      this.series = []; this.image = null; this.view = null; this.hidden = new Set();
      this.el.classList.add("vchart");
      this.el.innerHTML = `<div class="vc-head"><span class="vc-title"></span><span class="vc-tools">
        <button data-a="xlog" title="x axis linear / log">x log</button><button data-a="ylog" title="y axis linear / log">y log</button>
        <button data-a="reset" title="reset zoom (double click)">reset</button><button data-a="png" title="save image">PNG</button>
        <button data-a="csv" title="save visible data">CSV</button><button data-a="big" title="enlarge">⤢</button></span></div>
        <div class="vc-box"><canvas></canvas><div class="vc-tip" hidden></div></div><div class="vc-legend"></div>`;
      this.cv = this.el.querySelector("canvas"); this.tip = this.el.querySelector(".vc-tip");
      this.box = this.el.querySelector(".vc-box");
      this.el.querySelector(".vc-title").textContent = this.o.title;
      this.el.querySelectorAll(".vc-tools button").forEach(b => b.onclick = () => this.action(b.dataset.a));
      this.bind(); this.ro = new ResizeObserver(() => this.draw()); this.ro.observe(this.box);
      this.syncTools();
    }
    action(a) {
      if (a === "xlog") { this.o.xlog = !this.o.xlog; this.view = null; }
      if (a === "ylog") { this.o.ylog = !this.o.ylog; this.view = null; }
      if (a === "reset") { this.view = null; this.emitView(); }
      if (a === "png") { const l = document.createElement("a"); l.download = (this.o.title || "chart") + ".png"; l.href = this.snapshot(); l.click(); return; }
      if (a === "csv") return this.csv();
      if (a === "big") { this.el.classList.toggle("vc-big"); }
      this.syncTools(); this.draw();
    }
    syncTools() {
      const t = this.el.querySelectorAll(".vc-tools button");
      t[0].classList.toggle("on", this.o.xlog); t[1].classList.toggle("on", this.o.ylog);
    }
    setTitle(t) { this.o.title = t; this.el.querySelector(".vc-title").textContent = t; }
    setData(series, opts = {}) {
      Object.assign(this.o, opts);
      this.series = (series || []).filter(s => s && s.x && s.y).map((s, i) => Object.assign({ color: PALETTE[i % PALETTE.length], width: 1.4, mode: "line", axis: "y" }, s));
      this.image = opts.image || null;
      if (!opts.keepView) this.view = null;
      this.el.querySelector(".vc-title").textContent = this.o.title || "";
      this.syncTools(); this.legend(); this.draw();
    }
    replaceSeries(i, s) { this.series[i] = Object.assign({}, this.series[i], s); this.draw(); }
    legend() {
      const L = this.el.querySelector(".vc-legend"); L.innerHTML = "";
      if (!this.o.legend) return;
      this.series.forEach(s => {
        if (!s.name) return;
        const b = document.createElement("span"); b.className = "vc-leg" + (this.hidden.has(s.name) ? " off" : "");
        b.innerHTML = `<i style="background:${s.color}"></i>${s.name}`;
        b.onclick = () => { this.hidden.has(s.name) ? this.hidden.delete(s.name) : this.hidden.add(s.name); this.legend(); this.draw(); };
        L.appendChild(b);
      });
      if (this.image && this.image.label) { const s = document.createElement("span"); s.className = "vc-leg"; s.textContent = "colour: " + this.image.label; L.appendChild(s); }
    }
    tx(v, log) { return log ? (v > 0 ? Math.log10(v) : NaN) : v; }
    vis() { return this.series.filter(s => !this.hidden.has(s.name)); }
    autoRange() {
      let xs = [Infinity, -Infinity], ys = [Infinity, -Infinity], y2 = [Infinity, -Infinity];
      for (const s of this.vis()) {
        const r = s.axis === "y2" ? y2 : ys;
        for (let i = 0; i < s.x.length; i++) {
          const X = this.tx(s.x[i], this.o.xlog), Y = this.tx(s.y[i], s.axis === "y2" ? false : this.o.ylog);
          if (!isFinite(X) || !isFinite(Y) || s.y[i] === null) continue;
          if (X < xs[0]) xs[0] = X; if (X > xs[1]) xs[1] = X; if (Y < r[0]) r[0] = Y; if (Y > r[1]) r[1] = Y;
        }
      }
      if (this.image) { const im = this.image; xs = [Math.min(xs[0], im.x0), Math.max(xs[1], im.x1)]; ys = [Math.min(ys[0], this.tx(im.y0, this.o.ylog)), Math.max(ys[1], this.tx(im.y1, this.o.ylog))]; }
      const pad = (r, f = 0.05) => { if (!isFinite(r[0])) return [0, 1]; if (r[0] === r[1]) return [r[0] - 1, r[1] + 1]; const d = (r[1] - r[0]) * f; return [r[0] - d, r[1] + d]; };
      let Y = pad(ys); if (this.o.symmetricY && !this.o.ylog) { const m = Math.max(Math.abs(Y[0]), Math.abs(Y[1])); Y = [-m, m]; }
      if (this.o.ymin !== undefined && !this.o.ylog) Y[0] = this.o.ymin;
      return { x: pad(xs, this.image ? 0 : 0.01), y: Y, y2: this.o.y2range || pad(y2) };
    }
    layout() {
      const r = this.box.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
      const W = Math.max(r.width, 200), H = this.el.classList.contains("vc-big") ? Math.max(300, window.innerHeight - 170) : this.o.height;
      this.cv.width = W * dpr; this.cv.height = H * dpr; this.cv.style.height = H + "px";
      const hasY2 = this.series.some(s => s.axis === "y2");
      return { W, H, dpr, L: 62, R: hasY2 ? 58 : 14, T: 10, B: 40, hasY2 };
    }
    draw() {
      if (!this.box.isConnected) return;
      const g = this.layout(), c = this.cv.getContext("2d"); this.g = g;
      c.setTransform(g.dpr, 0, 0, g.dpr, 0, 0);
      const ink = css("--ink", "#1d2430"), muted = css("--muted", "#5b6575"), grid = css("--grid", "#e7eaef"), bg = css("--panel", "#fff");
      c.fillStyle = bg; c.fillRect(0, 0, g.W, g.H);
      const v = this.view || this.autoRange(); this.cur = v;
      const pw = g.W - g.L - g.R, ph = g.H - g.T - g.B;
      const X = x => g.L + (x - v.x[0]) / (v.x[1] - v.x[0]) * pw;
      const Y = y => g.T + ph - (y - v.y[0]) / (v.y[1] - v.y[0]) * ph;
      const Y2 = y => g.T + ph - (y - v.y2[0]) / (v.y2[1] - v.y2[0]) * ph;
      this.map = { X, Y, Y2, pw, ph, v };
      c.font = "11px system-ui, sans-serif"; c.lineWidth = 1;
      // image layer
      if (this.image) this.drawImage(c, v, X, Y);
      // grid + ticks
      const xt = this.o.xlog ? logTicks(v.x[0], v.x[1]) : niceTicks(v.x[0], v.x[1], Math.max(3, pw / 90)).map(t => ({ v: t, major: true }));
      const yt = this.o.ylog ? logTicks(v.y[0], v.y[1]) : niceTicks(v.y[0], v.y[1], Math.max(3, ph / 45)).map(t => ({ v: t, major: true }));
      c.strokeStyle = grid; c.fillStyle = muted; c.textAlign = "center"; c.textBaseline = "top";
      const lab = (t, log) => {
        if (!log) return fmt(t);
        const e = Math.round(t); if (Math.abs(t - e) < 1e-9 && (e > 4 || e < -3)) return "1e" + e;
        return fmt(Math.pow(10, t));
      };
      c.globalAlpha = this.image ? 0.25 : 1;
      const nx = xt.filter(t => t.major).length, showMinor = nx < 3, stepX = Math.max(1, Math.ceil(nx * 60 / pw));
      let kx = 0;
      for (const t of xt) { const p = X(t.v); c.save(); c.globalAlpha *= t.major ? 1 : 0.5; c.beginPath(); c.moveTo(p, g.T); c.lineTo(p, g.T + ph); c.stroke(); c.restore();
        if ((t.major && (kx++ % stepX === 0)) || (!t.major && showMinor)) { c.save(); c.globalAlpha = 1; c.fillText(lab(t.v, this.o.xlog), p, g.T + ph + 4); c.restore(); } }
      c.textAlign = "right"; c.textBaseline = "middle";
      const ny = yt.filter(t => t.major).length, showMinorY = ny < 3, stepY = Math.max(1, Math.ceil(ny * 18 / ph));
      let ky = 0;
      for (const t of yt) { const p = Y(t.v); c.save(); c.globalAlpha *= t.major ? 1 : 0.5; c.beginPath(); c.moveTo(g.L, p); c.lineTo(g.L + pw, p); c.stroke(); c.restore();
        if ((t.major && (ky++ % stepY === 0)) || (!t.major && showMinorY)) { c.save(); c.globalAlpha = 1; c.fillText(lab(t.v, this.o.ylog), g.L - 5, p); c.restore(); } }
      c.globalAlpha = 1;
      if (g.hasY2) { c.textAlign = "left"; for (const t of niceTicks(v.y2[0], v.y2[1], Math.max(3, ph / 45))) c.fillText(fmt(t), g.L + pw + 5, Y2(t)); }
      c.strokeStyle = muted; c.strokeRect(g.L, g.T, pw, ph);
      c.fillStyle = ink; c.textAlign = "center"; c.textBaseline = "bottom"; c.fillText(this.o.xlabel, g.L + pw / 2, g.H - 2);
      c.save(); c.translate(12, g.T + ph / 2); c.rotate(-Math.PI / 2); c.textBaseline = "middle"; c.fillText(this.o.ylabel, 0, 0); c.restore();
      if (g.hasY2) { c.save(); c.translate(g.W - 8, g.T + ph / 2); c.rotate(Math.PI / 2); c.textBaseline = "middle"; c.fillText(this.o.y2label, 0, 0); c.restore(); }
      // series
      c.save(); c.beginPath(); c.rect(g.L, g.T, pw, ph); c.clip();
      for (const s of this.vis()) {
        const yy = s.axis === "y2" ? Y2 : Y, ylog = s.axis === "y2" ? false : this.o.ylog;
        c.strokeStyle = s.color; c.fillStyle = s.color; c.lineWidth = s.width; c.setLineDash(s.dash || []);
        if (s.mode !== "points") {
          c.beginPath(); let pen = false, px = null;
          const n = s.x.length, stride = n > 6 * pw ? Math.floor(n / (3 * pw)) : 1;
          for (let i = 0; i < n; i += stride) {
            const a = this.tx(s.x[i], this.o.xlog), b = this.tx(s.y[i], ylog);
            if (!isFinite(a) || !isFinite(b) || s.y[i] === null) { pen = false; continue; }
            const xp = X(a), yp = yy(b);
            if (!pen) { c.moveTo(xp, yp); pen = true; } else if (s.mode === "step") { c.lineTo(xp, c._ly); c.lineTo(xp, yp); } else c.lineTo(xp, yp);
            c._ly = yp; px = xp;
          }
          c.stroke();
        }
        if (s.mode === "points" || s.mode === "both") {
          for (let i = 0; i < s.x.length; i++) {
            const a = this.tx(s.x[i], this.o.xlog), b = this.tx(s.y[i], ylog);
            if (!isFinite(a) || !isFinite(b)) continue; c.beginPath(); c.arc(X(a), yy(b), 2.3, 0, 7); c.fill();
          }
        }
      }
      c.setLineDash([]); c.restore();
      if (this.sel) { c.fillStyle = "rgba(31,90,138,0.12)"; c.strokeStyle = "rgba(31,90,138,0.6)"; const s = this.sel; c.fillRect(s.x0, s.y0, s.x1 - s.x0, s.y1 - s.y0); c.strokeRect(s.x0, s.y0, s.x1 - s.x0, s.y1 - s.y0); }
      if (this.hover) { c.strokeStyle = muted; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(this.hover.x, g.T); c.lineTo(this.hover.x, g.T + ph); c.stroke(); c.setLineDash([]); }
    }
    drawImage(c, v, X, Y) {
      const im = this.image, ny = im.z.length, nx = im.z[0].length;
      if (!this._img || this._img.src !== im) {
        const off = document.createElement("canvas"); off.width = nx; off.height = ny;
        const ctx = off.getContext("2d"), d = ctx.createImageData(nx, ny);
        const zmin = im.zmin ?? Math.min(...im.z.flat().filter(isFinite)), zmax = im.zmax ?? Math.max(...im.z.flat().filter(isFinite));
        for (let r = 0; r < ny; r++) for (let k = 0; k < nx; k++) {
          const col = viridis((im.z[r][k] - zmin) / (zmax - zmin || 1)), p = ((ny - 1 - r) * nx + k) * 4;
          d.data[p] = col[0]; d.data[p + 1] = col[1]; d.data[p + 2] = col[2]; d.data[p + 3] = 255;
        }
        ctx.putImageData(d, 0, 0); this._img = { src: im, off };
      }
      const y0 = this.tx(im.y0, this.o.ylog), y1 = this.tx(im.y1, this.o.ylog);
      c.save(); c.imageSmoothingEnabled = false;
      c.drawImage(this._img.off, X(im.x0), Y(y1), X(im.x1) - X(im.x0), Y(y0) - Y(y1)); c.restore();
    }
    toData(px, py) { const m = this.map, v = m.v, g = this.g; return { x: v.x[0] + (px - g.L) / m.pw * (v.x[1] - v.x[0]), y: v.y[0] + (g.T + m.ph - py) / m.ph * (v.y[1] - v.y[0]), y2: v.y2[0] + (g.T + m.ph - py) / m.ph * (v.y2[1] - v.y2[0]) }; }
    emitView() { if (this.o.onView) { const v = this.view || this.autoRange(); const f = t => this.o.xlog ? Math.pow(10, t) : t; this.o.onView(f(v.x[0]), f(v.x[1]), !this.view); } }
    bind() {
      const cv = this.cv; let drag = null;
      const pos = e => { const r = cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
      cv.addEventListener("contextmenu", e => e.preventDefault());
      cv.addEventListener("mousedown", e => {
        const p = pos(e); drag = { p, view: JSON.parse(JSON.stringify(this.view || this.cur)), pan: e.button !== 0 || e.shiftKey };
      });
      window.addEventListener("mousemove", e => {
        if (!this.map) return; const p = pos(e);
        if (drag) {
          if (drag.pan) {
            const m = this.map, v = drag.view, dx = (p.x - drag.p.x) / m.pw * (v.x[1] - v.x[0]), dy = (p.y - drag.p.y) / m.ph * (v.y[1] - v.y[0]), dy2 = (p.y - drag.p.y) / m.ph * (v.y2[1] - v.y2[0]);
            this.view = { x: [v.x[0] - dx, v.x[1] - dx], y: [v.y[0] + dy, v.y[1] + dy], y2: [v.y2[0] + dy2, v.y2[1] + dy2] };
          } else this.sel = { x0: Math.min(drag.p.x, p.x), x1: Math.max(drag.p.x, p.x), y0: Math.min(drag.p.y, p.y), y1: Math.max(drag.p.y, p.y) };
          this.draw(); return;
        }
        const g = this.g; if (p.x < g.L || p.x > g.W - g.R || p.y < g.T || p.y > g.H - g.B) { if (this.hover) { this.hover = null; this.tip.hidden = true; this.draw(); } return; }
        this.hover = p; this.showTip(p); this.draw();
      });
      window.addEventListener("mouseup", e => {
        if (!drag) return; const p = pos(e);
        if (!drag.pan && this.sel && (this.sel.x1 - this.sel.x0 > 6 || this.sel.y1 - this.sel.y0 > 6)) {
          const a = this.toData(this.sel.x0, this.sel.y1), b = this.toData(this.sel.x1, this.sel.y0), v = this.view || this.cur;
          const wide = this.sel.x1 - this.sel.x0 > 6, tall = this.sel.y1 - this.sel.y0 > 6;
          this.view = { x: wide ? [a.x, b.x] : v.x, y: tall ? [a.y, b.y] : v.y, y2: tall ? [a.y2, b.y2] : v.y2 };
        }
        const moved = drag.pan; drag = null; this.sel = null; this.draw(); if (moved || this.view) this.emitView();
      });
      cv.addEventListener("dblclick", () => { this.view = null; this.draw(); this.emitView(); });
      cv.addEventListener("wheel", e => {
        e.preventDefault(); const p = pos(e), d = this.toData(p.x, p.y), v = this.view || this.cur, k = e.deltaY > 0 ? 1.25 : 0.8;
        const zx = !e.altKey, zy = !e.shiftKey;
        const sc = (r, c0) => [c0 + (r[0] - c0) * k, c0 + (r[1] - c0) * k];
        this.view = { x: zx ? sc(v.x, d.x) : v.x, y: zy ? sc(v.y, d.y) : v.y, y2: zy ? sc(v.y2, d.y2) : v.y2 };
        this.draw(); clearTimeout(this._wt); this._wt = setTimeout(() => this.emitView(), 250);
      }, { passive: false });
      cv.addEventListener("mouseleave", () => { this.hover = null; this.tip.hidden = true; this.draw(); });
    }
    showTip(p) {
      const d = this.toData(p.x, p.y), xv = this.o.xlog ? Math.pow(10, d.x) : d.x;
      let h = `<b>${this.o.xlabel || "x"}: ${fmt(xv)}</b>`;
      for (const s of this.vis()) {
        let lo = 0, hi = s.x.length - 1; if (hi < 0) continue;
        while (hi - lo > 1) { const m = (lo + hi) >> 1; (s.x[m] < xv) ? lo = m : hi = m; }
        const i = Math.abs(s.x[lo] - xv) < Math.abs(s.x[hi] - xv) ? lo : hi;
        h += `<br><i style="background:${s.color}"></i>${s.name || ""}: ${fmt(s.y[i])}`;
      }
      if (this.image) {
        const im = this.image, yv = this.o.ylog ? Math.pow(10, d.y) : d.y;
        const k = Math.floor((xv - im.x0) / (im.x1 - im.x0) * im.z[0].length), r = Math.floor((yv - im.y0) / (im.y1 - im.y0) * im.z.length);
        if (r >= 0 && r < im.z.length && k >= 0 && k < im.z[0].length) h += `<br>${this.o.ylabel}: ${fmt(yv)}<br>${im.label || "z"}: ${fmt(im.z[r][k])}`;
      }
      this.tip.innerHTML = h; this.tip.hidden = false;
      const g = this.g; this.tip.style.left = (p.x > g.W * 0.6 ? p.x - this.tip.offsetWidth - 12 : p.x + 12) + "px"; this.tip.style.top = (g.T + 6) + "px";
    }
    snapshot() {
      const g = this.g, out = document.createElement("canvas"), legend = this.vis().filter(s => s.name);
      out.width = this.cv.width; out.height = this.cv.height + (24 + 16 * Math.ceil(legend.length / 3)) * g.dpr;
      const c = out.getContext("2d"); c.fillStyle = css("--panel", "#fff"); c.fillRect(0, 0, out.width, out.height);
      c.scale(g.dpr, g.dpr); c.fillStyle = css("--ink", "#000"); c.font = "bold 13px system-ui"; c.fillText(this.o.title || "", 8, 16);
      c.drawImage(this.cv, 0, 22, g.W, g.H); c.font = "11px system-ui";
      legend.forEach((s, i) => { const x = 10 + (i % 3) * (g.W / 3), y = 22 + g.H + 14 + 16 * Math.floor(i / 3); c.fillStyle = s.color; c.fillRect(x, y - 8, 12, 4); c.fillStyle = css("--ink", "#000"); c.fillText(s.name, x + 16, y); });
      return out.toDataURL("image/png");
    }
    csv() {
      const v = this.view || this.autoRange(), f = t => this.o.xlog ? Math.pow(10, t) : t, x0 = f(v.x[0]), x1 = f(v.x[1]);
      let txt = "";
      for (const s of this.vis()) {
        txt += `# ${s.name || "series"}\n${this.o.xlabel || "x"},${s.name || "y"}\n`;
        for (let i = 0; i < s.x.length; i++) if (s.x[i] >= x0 && s.x[i] <= x1) txt += `${s.x[i]},${s.y[i] ?? ""}\n`;
        txt += "\n";
      }
      const l = document.createElement("a"); l.download = (this.o.title || "chart") + ".csv";
      l.href = URL.createObjectURL(new Blob([txt], { type: "text/csv" })); l.click();
    }
  }
  window.VChart = VChart; window.VChart.palette = PALETTE; window.VChart.fmt = fmt;
})();
