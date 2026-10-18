/* vibsynth web interface */
"use strict";
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const F = (v, d = 4) => (v === null || v === undefined || !isFinite(v)) ? "–" : (+(+v).toPrecision(d)).toString();
const S = { state: null, rec: null, res: null, charts: {}, jobWatchers: [], mission: null, analysed: {} };

async function api(path, opt = {}) {
  const o = { method: opt.method || (opt.body !== undefined || opt.raw ? "POST" : "GET"), headers: {} };
  if (opt.raw) o.body = opt.raw;
  else if (opt.body !== undefined) { o.body = JSON.stringify(opt.body); o.headers["Content-Type"] = "application/json"; }
  const r = await fetch(path, o);
  const j = await r.json().catch(() => ({}));
  if (!r.ok || j.error) { toast(j.error || r.statusText, true); throw new Error(j.error || r.statusText); }
  return j;
}
function toast(msg, err) { const t = $("#toast"); t.textContent = msg; t.className = err ? "e" : ""; t.hidden = false; clearTimeout(t._t); t._t = setTimeout(() => t.hidden = true, err ? 7000 : 3000); }
function chart(id, o) { if (!S.charts[id]) S.charts[id] = new VChart("#" + id, o); else Object.assign(S.charts[id].o, o); return S.charts[id]; }
function download(url) { const a = document.createElement("a"); a.href = url; a.click(); }
const num = v => (v === "" || v === null || v === undefined) ? null : +v;
const nums = s => s.trim().split(/[\s,;]+/).filter(Boolean).map(Number);
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
const kindBadge = k => `<span class="badge ${k === "result" ? "result" : k.includes("spec") ? "spec" : ""}">${{ record: "record", result: "result", psd_spec: "PSD spec", sine_spec: "sine spec" }[k] || k}</span>`;

/* ------------------------------------------------------------------ state */
async function refresh() {
  S.state = await api("/api/state");
  $("#ver").textContent = "v" + S.state.version;
  renderModel(); renderItems(); renderPickers(); renderSpecs(); renderResults(); renderMissionList();
  $("#methods").innerHTML = Object.entries(S.state.methods).map(([k, v]) => `<b>${k}</b>: ${esc(v)}`).join("<br>");
  return S.state;
}
const items = (kinds) => S.state.items.filter(i => !kinds || kinds.includes(i.kind));
const item = id => S.state.items.find(i => i.id === id);

/* ------------------------------------------------------------------ tabs */
$$("#tabs button").forEach(b => b.onclick = () => {
  $$("#tabs button").forEach(x => x.classList.toggle("on", x === b));
  $$(".tab").forEach(t => t.hidden = t.id !== "tab-" + b.dataset.tab);
  Object.values(S.charts).forEach(c => c.draw());
});

/* ----------------------------------------------------------- damage model */
$("#modelchip").onclick = () => $("#model").hidden = !$("#model").hidden;
function renderModel() {
  const s = S.state.settings;
  $$("[data-s]").forEach(e => { const v = s[e.dataset.s]; e.value = v === null ? "" : String(v); });
  $("#modelchip").textContent = `Damage model: ${s.fmin}–${s.fmax || "fs/2.5"} Hz, ${s.scale === "octave" ? "1/" + s.ppo + " oct" : s.resolution + " Hz"}, Q ${s.Q}, b ${s.b}, ${s.fds_method}`;
}
$("#savemodel").onclick = async () => {
  const d = {};
  $$("[data-s]").forEach(e => { const k = e.dataset.s, v = e.value.trim();
    d[k] = ["scale", "prefilter", "fds_method"].includes(k) ? (k === "prefilter" && v !== "auto" ? v === "true" : v) : (v === "" ? null : +v); });
  await api("/api/settings", { body: d }); await refresh(); toast("damage model updated"); $("#model").hidden = true;
};

/* -------------------------------------------------------------------- jobs */
let pollT = null;
function watchJob(onDone) {
  S.jobWatchers.push(onDone); $("#jobbar").hidden = false;
  if (!pollT) poll();
}
async function poll() {
  const j = await api("/api/job"); pollT = null;
  $("#job_title").textContent = j.title + (j.running ? " …" : j.error ? " – " + j.error : " – done");
  $("#job_bar").style.width = (100 * (j.running ? Math.max(j.progress, 0.03) : 1)) + "%";
  $("#job_last").textContent = j.log.length ? j.log[j.log.length - 1] : "";
  $("#job_full").textContent = j.log.join("\n"); $("#job_full").scrollTop = 1e9;
  if (j.running) { pollT = setTimeout(poll, 700); return; }
  const r = await api("/api/job/result"), w = S.jobWatchers; S.jobWatchers = [];
  if (r.error) toast(r.error, true);
  for (const f of w) await f(r.result, r.error);
  setTimeout(() => { if (!S.jobWatchers.length) $("#jobbar").hidden = true; }, 4000);
}
$("#job_cancel").onclick = () => api("/api/job/cancel", { body: {} });
$("#job_log").onclick = () => $("#job_full").hidden = !$("#job_full").hidden;
async function runJob(path, body, onDone) { await api(path, { body }); watchJob(onDone); }

/* ----------------------------------------------------------------- import */
$("#btn_import").onclick = async () => {
  const files = [...$("#files").files]; if (!files.length) return toast("choose one or more files", true);
  for (const f of files) {
    const info = await api("/api/upload?name=" + encodeURIComponent(f.name), { raw: await f.arrayBuffer() });
    const v = info.variables || [];
    const needs = v.length > 1 || (v[0] && v[0].shape.length === 2 && Math.min(...v[0].shape) > 2);
    if (needs) importDialog(info); else await doImport(info, v[0] && v[0].name, null);
  }
};
function importDialog(info) {
  const box = document.createElement("div"); box.className = "card";
  const opts = info.variables.map(v => `<option value="${esc(v.name)}">${esc(v.name)} [${v.shape.join("×")}]</option>`).join("");
  box.innerHTML = `<h3>${esc(info.name)}</h3><div class="row wrap"><label class="inline">variable<select>${opts}</select></label>
    <label class="inline">columns (1 = first data column, empty = all)<input placeholder="e.g. 1 3"></label><button>Import</button><button class="sec">Cancel</button></div>`;
  const [sel, cols] = box.querySelectorAll("select,input"), [ok, no] = box.querySelectorAll("button");
  ok.onclick = async () => { await doImport(info, sel.value, cols.value.trim() ? nums(cols.value).map(c => c - 1) : null); box.remove(); };
  no.onclick = () => box.remove(); $("#importdlg").appendChild(box);
}
async function doImport(info, variable, columns) {
  const tc = $("#imp_time").value;
  const r = await api("/api/import", { body: { upload_id: info.upload_id, variable, columns, fs: num($("#imp_fs").value),
    units: $("#imp_units").value, time_column: tc === "auto" ? "auto" : tc === "yes" } });
  await refresh(); toast(`${r.ids.length} record(s) imported`); selectRecord(r.ids[0]);
}

/* -------------------------------------------------------------- workspace */
function renderItems() {
  const rows = S.state.items.map(i => {
    const st = i.stats || {};
    return `<tr data-id="${i.id}" class="${S.rec === i.id ? "sel" : ""}"><td class="l"><span class="nm">${esc(i.name)}</span></td><td>${kindBadge(i.kind)}</td>
      <td>${i.fs ? F(i.fs, 6) : "–"}</td><td>${i.duration ? F(i.duration, 5) : "–"}</td><td>${F(st.rms ?? i.rms)}</td><td>${F(st.kurtosis)}</td><td>${F(st.crest_factor)}</td>
      <td><input class="life" data-id="${i.id}" size="6" value="${i.meta.life_h ?? ""}" placeholder="${i.duration ? F(i.duration / 3600, 3) : 1}"></td>
      <td><button class="sec ren" data-id="${i.id}">rename</button> ${i.duration ? `<select class="xf" data-id="${i.id}"><option value="">export…</option><option>mat</option><option>csv</option><option>txt</option><option>npy</option><option>wav</option><option value="psd-csv">PSD csv</option><option value="psd-xlsx">PSD xlsx</option><option value="stats-csv">stats csv</option></select>` : ""}
      <button class="sec del" data-id="${i.id}">✕</button></td></tr>`;
  }).join("");
  $("#itemtable").innerHTML = `<tr><th class="l">name</th><th>kind</th><th>fs [Hz]</th><th>duration [s]</th><th>RMS</th><th>kurtosis</th><th>crest</th><th>life [h]</th><th></th></tr>` + (rows || `<tr><td class="l" colspan="9">No records yet: import a file.</td></tr>`);
  $$("#itemtable tr[data-id]").forEach(tr => tr.onclick = e => { if (!["INPUT", "SELECT", "BUTTON"].includes(e.target.tagName)) { const it = item(tr.dataset.id); if (it.duration) selectRecord(it.id); } });
  $$("#itemtable .life").forEach(inp => inp.onchange = async () => { await api(`/api/item/${inp.dataset.id}/meta`, { body: { life_h: num(inp.value) } }); await refresh(); });
  $$("#itemtable .ren").forEach(b => b.onclick = async () => { const n = prompt("New name", item(b.dataset.id).name); if (n) { await api(`/api/item/${b.dataset.id}/rename`, { body: { name: n } }); refresh(); } });
  $$("#itemtable .del").forEach(b => b.onclick = async () => { if (confirm("Delete " + item(b.dataset.id).name + "?")) { await api(`/api/item/${b.dataset.id}/delete`, { body: {} }); if (S.rec === b.dataset.id) { S.rec = null; $("#recdetail").hidden = true; } refresh(); } });
  $$("#itemtable .xf").forEach(s => s.onchange = () => { const v = s.value; if (!v) return; const [what, fmt] = v.includes("-") ? v.split("-") : ["series", v]; download(`/api/export?what=${what}&fmt=${fmt}&id=${s.dataset.id}`); s.value = ""; });
}
async function seriesChart(id, itemId, title) {
  const c = chart(id, { title, xlabel: "time [s]", ylabel: "acceleration", height: 260 });
  const d = await api(`/api/item/${itemId}/series?n=3000`);
  c.setData([{ name: item(itemId).name, x: d.t, y: d.y, width: 0.8 }], { xlog: false, ylog: false });
  c.o.onView = async (a, b, full) => {
    const r = await api(`/api/item/${itemId}/series?n=3000&t0=${full ? 0 : a}&t1=${full ? 1e12 : b}`);
    c.replaceSeries(0, { x: r.t, y: r.y });
  };
  return c;
}
async function selectRecord(id) {
  S.rec = id; renderItems(); const it = item(id);
  $("#recdetail").hidden = false; $("#rec_title").textContent = it.name + (it.meta.history ? "  ·  " + it.meta.history.join(", ") : "");
  await seriesChart("ch_rec", id, "Time history");
  $("#rec_audio").innerHTML = `<audio controls preload="none" src="/api/item/${id}/audio.wav"></audio> <span class="hint">first 60 s, re-sampled to at least 8 kHz</span>`;
  $("#pp_name").value = ""; $("#sh_out").textContent = "";
}
const zoomRange = c => { const v = c.view; if (!v) return null; return v.x; };
$("#pp_zoom").onclick = () => { const r = zoomRange(S.charts.ch_rec); if (!r) return toast("zoom the chart first", true); $("#pp_t0").value = F(Math.max(r[0], 0), 6); $("#pp_t1").value = F(r[1], 6); };
$("#sp_zoom").onclick = () => { const r = zoomRange(S.charts.ch_rec); if (!r) return toast("zoom the chart first", true); $("#sp_edges").value = ($("#sp_edges").value + " " + F(Math.max(r[0], 0), 6) + " " + F(r[1], 6)).trim(); };
$("#pp_go").onclick = async () => {
  const b = { t0: num($("#pp_t0").value), t1: num($("#pp_t1").value), f_low: num($("#pp_hp").value), f_high: num($("#pp_lp").value),
    resample_fs: num($("#pp_fs").value), gain: num($("#pp_gain").value), detrend: $("#pp_dt").checked, name: $("#pp_name").value || null };
  const r = await api(`/api/item/${S.rec}/preprocess`, { body: b }); S.state = r.state; await refresh(); selectRecord(r.id); toast("new record created");
};
$("#sp_go").onclick = async () => { const r = await api(`/api/item/${S.rec}/split`, { body: { edges: nums($("#sp_edges").value) } }); await refresh(); toast(`${r.ids.length} situations created`); };
function shakerHtml(r, lim) {
  const ok = k => r.ok[k] ? `<span class="ok">ok</span>` : `<span class="err">exceeds</span>`;
  return `peak acceleration ${F(r.peak_acc)} m/s² ${lim.acc_limit ? ok("acc") : ""}<br>peak velocity ${F(r.peak_vel)} m/s ${lim.vel_limit ? ok("vel") : ""}<br>
    displacement p-p ${F(r.peak_to_peak_disp * 1000)} mm ${lim.disp_limit ? ok("disp") : ""}<br><span class="hint">integration high-pass ${F(r.hp, 3)} Hz</span>` + (r.suggested_hp ? `<br>a high-pass at <b>${F(r.suggested_hp, 3)} Hz</b> keeps the stroke within the limit (Pre-processing).` : "");
}
async function shaker(id, pre) {
  const lim = { acc_limit: num($(`#${pre}_a`).value), vel_limit: num($(`#${pre}_v`).value), disp_limit: num($(`#${pre}_d`).value), hp: num($(`#${pre}_hp`).value) };
  const r = await api(`/api/item/${id}/shaker`, { body: lim }); return shakerHtml(r, lim);
}
$("#sh_go").onclick = async () => $("#sh_out").innerHTML = await shaker(S.rec, "sh");

/* ---------------------------------------------------------------- analysis */
function targetsList() {
  const out = [];
  for (const i of S.state.items) out.push({ id: i.id, name: i.name, kind: i.kind });
  for (const m of S.state.missions) out.push({ id: m.id, name: m.name || m.id, kind: "mission" });
  return out;
}
function renderPickers() {
  const prev = new Set($$("#an_pick input:checked").map(i => i.value));
  $("#an_pick").innerHTML = targetsList().map(t => `<label class="inline"><input type="checkbox" value="${t.id}" ${prev.has(t.id) ? "checked" : ""}>${esc(t.name)} ${kindBadge(t.kind)}</label>`).join("") || `<span class="hint">nothing to analyse yet</span>`;
  const sig = items(["record", "result"]);
  const keep = $("#dg_item").value;
  $("#dg_item").innerHTML = sig.map(i => `<option value="${i.id}">${esc(i.name)}</option>`).join(""); if (keep) $("#dg_item").value = keep;
  const tsel = $("#sy_target").value;
  const grp = (lab, arr) => arr.length ? `<optgroup label="${lab}">${arr.map(t => `<option value="${t.id}">${esc(t.name)}</option>`).join("")}</optgroup>` : "";
  $("#sy_target").innerHTML = grp("Records", items(["record"])) + grp("Specifications", items(["psd_spec", "sine_spec"])) + grp("Missions", S.state.missions.map(m => ({ id: m.id, name: m.name || m.id }))) + grp("Results", items(["result"]));
  if (tsel) $("#sy_target").value = tsel;
  if (!$("#sy_method").options.length) {
    $("#sy_method").innerHTML = Object.entries(S.state.methods).map(([k, v]) => `<option value="${k}">${k.toUpperCase()}: ${esc(v.split(":").slice(-1)[0].trim())}</option>`).join("");
    $("#sy_method").value = "kfds"; $("#sy_method").onchange();
    $("#sy_multi").innerHTML = Object.keys(S.state.methods).map(k => `<label><input type="checkbox" value="${k}">${k.toUpperCase()}</label>`).join("");
    $$("#sy_multi input").forEach(i => i.onchange = showMM); showMM();
  }
}
const mmSelected = () => $("#sy_method").value === "mmvsd" || $$("#sy_multi input:checked").some(i => i.value === "mmvsd");
const edSelected = () => $("#sy_method").value === "edit" || $$("#sy_multi input:checked").some(i => i.value === "edit");
function showMM() {
  const on = mmSelected(); $("#sy_mm").hidden = !on; if (on) $("#sy_mm").open = true;
  const ed = edSelected(); $("#sy_ed").hidden = !ed; if (ed) $("#sy_ed").open = true;
}
$("#sy_method").onchange = () => { $("#sy_mdesc").textContent = S.state ? S.state.methods[$("#sy_method").value] : ""; showMM(); };
$("#an_go").onclick = async () => {
  const ids = $$("#an_pick input:checked").map(i => i.value); if (!ids.length) return toast("tick at least one signal, specification or mission", true);
  await runJob("/api/analyze", { ids }, res => { if (res) { Object.assign(S.analysed, res); drawAnalysis(ids.map(i => res[i]).filter(Boolean)); } });
};
function drawAnalysis(list) {
  const P = VChart.palette;
  chart("ch_fds", { title: "Fatigue Damage Spectrum (over the life)", xlabel: "natural frequency [Hz]", ylabel: "damage", height: 300 })
    .setData(list.map((t, i) => ({ name: `${t.name} (${F(t.life_h, 3)} h)`, x: t.fn, y: t.fds, color: P[i % 8], mode: "both" })), { xlog: true, ylog: true });
  chart("ch_mrs", { title: "Maximum Response Spectrum", xlabel: "natural frequency [Hz]", ylabel: "pseudo-acceleration", height: 300 })
    .setData(list.map((t, i) => ({ name: t.name, x: t.fn, y: t.mrs, color: P[i % 8], mode: "both" })), { xlog: true, ylog: true });
  chart("ch_psd", { title: "PSD", xlabel: "frequency [Hz]", ylabel: "PSD [(m/s²)²/Hz]", height: 300 })
    .setData(list.map((t, i) => ({ name: t.name, x: t.psd_f, y: t.psd, color: P[i % 8] })), { xlog: false, ylog: true });
  const kz = list.filter(t => t.kz && t.kz.length);
  if (kz.length) chart("ch_resp", { title: "Kurtosis of the oscillator responses k_z (3 = Gaussian)", xlabel: "natural frequency [Hz]", ylabel: "kurtosis", height: 300 })
    .setData(kz.map((t, i) => ({ name: t.name, x: t.fn, y: t.kz, color: P[list.indexOf(t) % 8], mode: "both" })), { xlog: true, ylog: false });
  $("#an_exports").innerHTML = `<label class="inline">Save FDS / MRS / PSD of<select id="an_xid">${list.map(t => `<option value="${t.id}">${esc(t.name)}</option>`).join("")}</select></label>
    <select id="an_xf"><option value="csv">csv (zip)</option><option>xlsx</option><option>mat</option><option>json</option></select><button class="sec" id="an_xgo">Download</button>`;
  $("#an_xgo").onclick = () => download(`/api/export?what=target&id=${$("#an_xid").value}&fmt=${$("#an_xf").value}`);
}
$("#an_resp").onclick = async () => {
  const ids = $$("#an_pick input:checked").map(i => i.value).filter(i => item(i) && item(i).duration);
  if (!ids.length) return toast("tick at least one time history", true);
  await runJob("/api/response_stats", { ids }, res => {
    if (!res) return; const L = Object.values(res), P = VChart.palette;
    chart("ch_resp", { title: "Kurtosis (solid) and std (dashed, right) of the SDOF response", xlabel: "natural frequency [Hz]", ylabel: "kurtosis", y2label: "std", height: 300 })
      .setData(L.flatMap((t, i) => [{ name: t.name + " kurtosis", x: t.fn, y: t.kurtosis, color: P[i % 8] }, { name: t.name + " std", x: t.fn, y: t.std, color: P[i % 8], dash: [5, 4], axis: "y2" }]), { xlog: true, ylog: false });
  });
};
$("#an_comp").onclick = async () => {
  const ids = $$("#an_pick input:checked").map(i => i.value).filter(i => item(i) && item(i).duration);
  if (!ids.length) return toast("tick at least one time history", true);
  await runJob("/api/compressibility", { ids, window_s: num($("#an_cwin").value) }, res => {
    if (!res) return; const L = Object.values(res), P = VChart.palette;
    const ser = L.flatMap((t, i) => [{ name: t.name + " (mean over fn)", x: t.q, y: t.mean, color: P[i % 8], width: 2 },
      { name: t.name + " (worst fn)", x: t.q, y: t.worst, color: P[i % 8], dash: [5, 4] }]);
    ser.push({ name: "stationary Gaussian", x: [0, 1], y: [0, 1], dash: [2, 3], color: "#bbb" });
    chart("ch_lorenz", { title: "Damage Lorenz curve: share of the damage in the most damaging fraction of the time", xlabel: "fraction of the time kept (= 1/alpha)", ylabel: "share of the damage", height: 300 })
      .setData(ser, { xlog: false, ylog: false });
    $("#an_comp_out").innerHTML = `<table class="list"><tr><th class="l">record</th><th>kurtosis</th><th>window</th><th>alpha<sub>max</sub> 80 %</th><th>90 %</th><th>95 %</th><th>lognormal 90 %</th></tr>`
      + L.map(t => `<tr><td class="l">${esc(t.name)}</td><td>${F(t.kurtosis, 3)}</td><td>${F(t.window_s, 2)} s</td>`
        + ["80", "90", "95"].map(k => `<td>${F(t.alpha_max[k].worst, 2)} <span class="hint">(${F(t.alpha_max[k].mean, 2)})</span></td>`).join("")
        + `<td>${F(t.lognormal_alpha90, 2)}</td></tr>`).join("")
      + `</table><p class="hint">alpha<sub>max</sub>: largest time compression at which the worst oscillator still keeps the given share of the damage by editing alone (in brackets: mean over the oscillators). Beyond it a test can only be shortened by raising the amplitude, which makes it depend on the S-N exponent. The last column is the closed form for a lognormal envelope with the record's kurtosis, an upper estimate (theory manual, Section 12).</p>`;
  });
};
$("#dg_go").onclick = async () => {
  const id = $("#dg_item").value; if (!id) return;
  const [p, r, sp, sk, kg, sd] = await Promise.all([api(`/api/item/${id}/pdf`), api(`/api/item/${id}/running?window=${$("#dg_win").value}`),
    api(`/api/item/${id}/spectrogram`), api(`/api/item/${id}/sk`), api(`/api/item/${id}/kurtogram`), api(`/api/item/${id}/sdof?fn=${$("#dg_fn").value}`)]);
  chart("ch_pdf", { title: `PDF (kurtosis ${F(p.stats.kurtosis, 3)}, skewness ${F(p.stats.skewness, 3)})`, xlabel: "acceleration", ylabel: "probability density", height: 280 })
    .setData([{ name: "measured", x: p.x, y: p.pdf, mode: "step" }, { name: "Gaussian, same std", x: p.x, y: p.gauss, dash: [5, 4] }], { ylog: true });
  chart("ch_run", { title: "Running RMS and kurtosis", xlabel: "time [s]", ylabel: "RMS", y2label: "kurtosis", height: 280 })
    .setData([{ name: "RMS", x: r.t, y: r.rms, mode: "step" }, { name: "kurtosis", x: r.t, y: r.kurtosis, axis: "y2", mode: "step" }]);
  const t = sp.t, f = sp.f;
  chart("ch_spg", { title: "Spectrogram", xlabel: "time [s]", ylabel: "frequency [Hz]", height: 280, legend: true })
    .setData([], { image: { x0: t[0], x1: t[t.length - 1], y0: f[0], y1: f[f.length - 1], z: sp.z, label: "PSD [dB]" } });
  chart("ch_sk", { title: "Spectral kurtosis (0 for a Gaussian process)", xlabel: "frequency [Hz]", ylabel: "SK", height: 280 })
    .setData([{ name: "SK", x: sk.f, y: sk.sk }]);
  const lv = kg.nwin.map(n => Math.log2(n));
  chart("ch_kg", { title: "Kurtogram (STFT): spectral kurtosis vs window length", xlabel: "frequency [Hz]", ylabel: "log2(window length)", height: 280 })
    .setData([], { image: { x0: kg.f[0], x1: kg.f[kg.f.length - 1], y0: lv[0] - 0.5, y1: lv[lv.length - 1] + 0.5, z: kg.z, label: "SK" } });
  chart("ch_sdof", { title: `SDOF response at ${$("#dg_fn").value} Hz (kurtosis ${F(sd.stats.kurtosis, 3)})`, xlabel: "time [s]", ylabel: "pseudo-acceleration", height: 280 })
    .setData([{ name: "response", x: sd.t, y: sd.y, width: 0.8 }]);
};

/* ----------------------------------------------------------- specifications */
const pts = s => { const a = s.trim().split(/\n+/).map(l => nums(l)).filter(r => r.length >= 2); return [a.map(r => r[0]), a.map(r => r[1])]; };
$("#ps_prev").onclick = async () => {
  const [f, g] = pts($("#ps_pts").value); const r = await api("/api/spec/preview", { body: { type: "psd", f, psd: g } });
  chart("ch_ps", { title: "PSD specification", xlabel: "frequency [Hz]", ylabel: "PSD [(m/s²)²/Hz]", height: 260 }).setData([{ name: "spec", x: r.f, y: r.y }, { name: "breakpoints", x: f, y: g, mode: "points" }], { xlog: true, ylog: true });
  $("#ps_rms").textContent = `RMS ${F(r.rms)} m/s² (${F(r.rms / 9.81)} g)`;
};
$("#ps_add").onclick = async () => { const [f, g] = pts($("#ps_pts").value); await api("/api/spec", { body: { type: "psd", f, psd: g, name: $("#ps_name").value } }); await refresh(); toast("specification added"); };
$("#ss_prev").onclick = async () => {
  const [f, a] = pts($("#ss_pts").value);
  const r = await api("/api/spec/preview", { body: { type: "sine", f, amplitude: a, rate: +$("#ss_rate").value, mode: $("#ss_mode").value, sweeps: +$("#ss_n").value } });
  chart("ch_ss", { title: "Swept sine amplitude", xlabel: "frequency [Hz]", ylabel: "peak acceleration [m/s²]", height: 260 }).setData([{ name: "amplitude", x: r.f, y: r.y }], { xlog: true, ylog: true });
  $("#ss_dur").textContent = `duration ${F(r.duration)} s`;
};
$("#ss_add").onclick = async () => { const [f, a] = pts($("#ss_pts").value); await api("/api/spec", { body: { type: "sine", f, amplitude: a, rate: +$("#ss_rate").value, mode: $("#ss_mode").value, sweeps: +$("#ss_n").value, name: $("#ss_name").value } }); await refresh(); toast("specification added"); };
function renderSpecs() {
  const sp = items(["psd_spec", "sine_spec"]);
  $("#spectable").innerHTML = `<tr><th class="l">name</th><th>kind</th><th>band [Hz]</th><th>RMS / duration</th><th class="l">generate time history</th></tr>` + sp.map(s => `<tr><td class="l">${esc(s.name)}</td><td>${kindBadge(s.kind)}</td><td>${F(s.f[0])}–${F(s.f[s.f.length - 1])}</td>
    <td>${s.kind === "psd_spec" ? "RMS " + F(s.rms) : F(s.duration) + " s"}</td><td class="l"><input size="6" placeholder="fs" class="gfs"> <input size="6" placeholder="${s.kind === "psd_spec" ? "60 s" : "all"}" class="gdur"> <button class="sec gen" data-id="${s.id}">Generate</button></td></tr>`).join("");
  $$("#spectable .gen").forEach(b => b.onclick = async () => { const tr = b.closest("tr"); await api(`/api/item/${b.dataset.id}/generate`, { body: { fs: num(tr.querySelector(".gfs").value), duration: num(tr.querySelector(".gdur").value) } }); await refresh(); toast("time history created (Records)"); });
}

/* ------------------------------------------------------------------ missions */
function newMission() { return { name: "Mission " + (S.state.missions.length + 1), root: { type: "group", kind: "series", children: [] } }; }
function renderMissionList() {
  const cur = S.mission && S.mission.id;
  $("#ms_pick").innerHTML = `<option value="">(new, unsaved)</option>` + S.state.missions.map(m => `<option value="${m.id}">${esc(m.name || m.id)}</option>`).join("");
  $("#ms_pick").value = cur || "";
  if (!S.mission) { S.mission = newMission(); }
  renderTree();
}
$("#ms_pick").onchange = () => { const m = S.state.missions.find(x => x.id === $("#ms_pick").value); S.mission = m ? JSON.parse(JSON.stringify(m)) : newMission(); renderTree(); };
$("#ms_name").onchange = () => { S.mission.name = $("#ms_name").value; };
$("#ms_new").onclick = () => { S.mission = newMission(); $("#ms_pick").value = ""; renderTree(); };
function renderTree() {
  $("#ms_name").value = S.mission.name || "";
  const opts = items(["record", "psd_spec", "sine_spec", "result"]);
  const html = (n, path) => {
    if (n.type === "situation") {
      return `<div class="sit" data-p="${path}"><span>⦿</span><select class="m-item">${opts.map(o => `<option value="${o.id}" ${o.id === n.item ? "selected" : ""}>${esc(o.name)}</option>`).join("")}</select>
        <label class="inline">life [h]<input class="m-life" size="7" value="${n.life_h ?? ""}"></label><label class="inline">name<input class="m-name" size="14" value="${esc(n.name || "")}"></label><button class="sec m-del">✕</button></div>`;
    }
    return `<div class="grp ${n.kind}" data-p="${path}"><div class="head"><select class="m-kind"><option value="series" ${n.kind === "series" ? "selected" : ""}>SERIES</option><option value="parallel" ${n.kind === "parallel" ? "selected" : ""}>PARALLEL</option></select>
      <input class="m-gname" size="14" placeholder="group name" value="${esc(n.name || "")}"><button class="sec m-addsit">+ situation</button><button class="sec m-addgrp">+ group</button>${path ? `<button class="sec m-del">✕</button>` : ""}</div>
      <div class="kids">${(n.children || []).map((c, i) => html(c, path + "/" + i)).join("") || `<span class="hint">empty</span>`}</div></div>`;
  };
  $("#ms_tree").className = "tree"; $("#ms_tree").innerHTML = html(S.mission.root, "");
  const node = p => p.split("/").filter(Boolean).reduce((n, i) => n.children[+i], S.mission.root);
  const parent = p => { const a = p.split("/").filter(Boolean); a.pop(); return [node(a.length ? "/" + a.join("/") : ""), +p.split("/").pop()]; };
  $$("#ms_tree [data-p]").forEach(el => {
    const p = el.dataset.p, n = node(p), q = s => n.type === "situation" ? el.querySelector(s) : el.querySelector(":scope > .head > " + s);
    if (n.type === "situation") {
      if (!n.item && opts[0]) n.item = opts[0].id;
      q(".m-item").onchange = e => n.item = e.target.value; q(".m-life").onchange = e => n.life_h = num(e.target.value); q(".m-name").onchange = e => n.name = e.target.value;
    } else {
      q(".m-kind").onchange = e => { n.kind = e.target.value; renderTree(); }; q(".m-gname").onchange = e => n.name = e.target.value;
      q(".m-addsit").onclick = () => { n.children.push({ type: "situation", item: opts[0] && opts[0].id, life_h: 1 }); renderTree(); };
      q(".m-addgrp").onclick = () => { n.children.push({ type: "group", kind: n.kind === "series" ? "parallel" : "series", children: [] }); renderTree(); };
    }
    const d = q(".m-del"); if (d) d.onclick = () => { const [par, i] = parent(p); par.children.splice(i, 1); renderTree(); };
  });
}
async function saveMission() {
  S.mission.name = $("#ms_name").value || S.mission.name;
  const r = await api("/api/mission", { body: S.mission }); S.mission.id = r.id; await refresh(); $("#ms_pick").value = r.id; return r.id;
}
$("#ms_save").onclick = async () => { await saveMission(); toast("mission saved"); };
$("#ms_del").onclick = async () => { if (S.mission.id) { await api(`/api/mission/${S.mission.id}/delete`, { body: {} }); S.mission = null; await refresh(); } };
$("#ms_go").onclick = async () => {
  const id = await saveMission();
  await runJob("/api/analyze", { ids: [id] }, res => {
    const t = res && res[id]; if (!t) return; $("#ms_res").hidden = false; const P = VChart.palette;
    $("#ms_stats").innerHTML = `life ${F(t.life_h)} h · RMS ${F(t.stats.rms)} · kurtosis ${F(t.stats.kurtosis)} · synthesis fs ${F(t.fs)} Hz · ${esc((t.notes || []).join("; "))}`;
    chart("ch_msfds", { title: "Mission FDS and situations", xlabel: "natural frequency [Hz]", ylabel: "damage", height: 300 })
      .setData([{ name: "mission", x: t.fn, y: t.fds, width: 2.4, color: "#1c2330" }, ...t.breakdown.map((b, i) => ({ name: `${b.name} (${F(b.life_h, 3)} h)`, x: t.fn, y: b.fds, color: P[(i + 1) % 8], dash: [4, 3] }))], { xlog: true, ylog: true });
    chart("ch_msmrs", { title: "Mission MRS", xlabel: "natural frequency [Hz]", ylabel: "pseudo-acceleration", height: 300 })
      .setData([{ name: "mission", x: t.fn, y: t.mrs, width: 2.4, color: "#1c2330" }, ...t.breakdown.map((b, i) => ({ name: b.name, x: t.fn, y: b.mrs, color: P[(i + 1) % 8], dash: [4, 3] }))], { xlog: true, ylog: true });
    chart("ch_mspsd", { title: "Mission PSD", xlabel: "frequency [Hz]", ylabel: "PSD", height: 300 }).setData([{ name: "mission", x: t.psd_f, y: t.psd }], { ylog: true });
    const tot = t.fds;
    chart("ch_mscontrib", { title: "Share of the damage per situation [%]", xlabel: "natural frequency [Hz]", ylabel: "%", height: 300 })
      .setData(t.breakdown.map((b, i) => ({ name: b.name, x: t.fn, y: b.fds.map((v, k) => 100 * v / tot[k]), color: P[(i + 1) % 8] })), { xlog: true, ylog: false });
    $("#ms_exports").innerHTML = "Export targets: " + ["csv", "xlsx", "mat", "json"].map(f => `<button class="sec" data-f="${f}">${f}</button>`).join(" ");
    $$("#ms_exports button").forEach(b => b.onclick = () => download(`/api/export?what=target&id=${id}&fmt=${b.dataset.f}`));
  });
};

/* ----------------------------------------------------------------- synthesis */
$("#sy_go").onclick = async () => {
  const unit = +$("#sy_unit").value, seeds = nums($("#sy_seeds").value), methods = $$("#sy_multi input:checked").map(i => i.value);
  const b = { target: $("#sy_target").value, life_h: num($("#sy_life").value), method: $("#sy_method").value, duration: +$("#sy_dur").value * unit,
    kurtosis: num($("#sy_k").value), rms: num($("#sy_rms").value), pattern: $("#sy_pat").value, window: num($("#sy_win").value), seed: $("#sy_seed").value,
    iterations: num($("#sy_it").value), tol: num($("#sy_tol").value), fs: num($("#sy_fs").value), prominence_db: num($("#sy_prom").value), drift: num($("#sy_drift").value),
    seeds: seeds.length ? seeds : null, methods: methods.length ? methods : null, name: $("#sy_name").value || null };
  if (mmSelected()) Object.assign(b, { modulation: $("#sy_mm_mod").value, mapping: $("#sy_mm_map").value, lines: $("#sy_mm_lines").value,
    block_s: num($("#sy_mm_block").value), freq_corr: num($("#sy_mm_fc").value), time_corr: num($("#sy_mm_tc").value),
    max_excess: num($("#sy_mm_exc").value), cl_anchor: num($("#sy_mm_anchor").value) });
  Object.assign(b, { ranking: $("#sy_ed_rank").value, gain: $("#sy_ed_gain").value, gain_max: num($("#sy_ed_gmax").value),
    margin: num($("#sy_ed_margin").value), fade_s: num($("#sy_ed_fade").value), retention: num($("#sy_ed_ret").value),
    compress_kurtosis: $("#sy_ck").checked });
  if (!b.target) return toast("choose a target", true);
  await runJob("/api/synth", b, async res => { await refresh(); if (res && res.ids && res.ids.length) showResult(res.ids[res.ids.length - 1]); });
};
S.resCache = {};
async function resSummary(id) { if (!S.resCache[id]) S.resCache[id] = await api("/api/result/" + id); return S.resCache[id]; }
async function renderResults() {
  const rs = items(["result"]);
  const rows = [];
  for (const r of rs) {
    const v = await resSummary(r.id).catch(() => null); if (!v) continue; const s = v.summary;
    rows.push(`<tr data-id="${r.id}" class="${S.res === r.id ? "sel" : ""}"><td class="l"><input type="checkbox" class="rc" value="${r.id}"> ${esc(r.name)}</td><td>${v.method}</td><td>${F(s.fds_max, 3)} %</td><td>${F(s.fds_median, 2)} %</td>
      <td>${F(s.kurtosis, 3)}</td><td>${F(s.rms)}</td><td>${F(s.crest, 3)}</td><td>${F(s.mrs_max, 3)}</td><td>${F(s.psd_db, 2)} dB</td><td>${s.kz_med === null || s.kz_med === undefined ? "–" : F(s.kz_med, 2)}</td><td>${v.warnings.length ? "⚠ " + v.warnings.length : ""}</td></tr>`);
  }
  $("#restable").innerHTML = `<tr><th class="l">result</th><th>method</th><th>FDS max err</th><th>median</th><th>kurtosis</th><th>RMS</th><th>crest</th><th>MRS max ratio</th><th>PSD dev</th><th title="median |error| of the kurtosis of the oscillator responses">k<sub>z</sub> err</th><th></th></tr>` + (rows.join("") || `<tr><td class="l" colspan="11">No results yet.</td></tr>`);
  $$("#restable tr[data-id]").forEach(tr => tr.onclick = e => { if (e.target.tagName !== "INPUT") showResult(tr.dataset.id); });
}
async function showResult(id) {
  S.res = id; delete S.resCache[id]; const v = await resSummary(id); renderResults();
  $("#rs_detail").hidden = false; $("#rs_title").textContent = `${v.name}  →  target: ${v.target_name}`;
  const mm = v.mmvsd, ln = mm && mm.lines;
  $("#rs_warn").innerHTML = v.warnings.map(w => `<div class="warn">${esc(w)}</div>`).join("") + (v.tones && v.tones.length ? `<div class="hint">tones: ${v.tones.map(t => F(t[0]) + " Hz / " + F(t[1])).join(", ")}</div>` : "")
    + (ln && ln.tracks && ln.tracks.length ? `<div class="hint">sine-on-random blocks reproduced as lines (${esc(ln.mapping || "")}): ${ln.tracks.map(t => `${F(t.f_min, 3)}–${F(t.f_max, 3)} Hz ×${F(t.gain, 3)}`).join(", ")}</div>` : "");
  $("#rs_tab").innerHTML = `<tr><td></td><td><b>target</b></td><td><b>result</b></td></tr>` + v.table.map(r => `<tr><td>${esc(r[0])}</td><td>${esc(r[1])}</td><td>${esc(r[2])}</td></tr>`).join("");
  const best = v.history.reduce((a, h) => (a.push(a.length ? Math.min(a[a.length - 1], h) : h), a), []);
  chart("ch_hist", { title: "Closed-loop convergence", xlabel: "iteration", ylabel: "max |error| [%]", height: 200 })
    .setData([{ name: "max error", x: v.history.map((_, i) => i), y: v.history.map(h => 100 * h), mode: "both" },
      { name: "best so far", x: best.map((_, i) => i), y: best.map(h => 100 * h), color: "#c0504d", dash: [4, 3] }], { ylog: true });
  const c = await seriesChart("ch_rs_time", id, "Time history" + (v.envelope ? " and envelope" : ""));
  if (v.envelope) { const s = c.series; s.push({ name: "envelope × RMS", x: v.envelope.t, y: v.envelope.y.map(y => y * v.summary.rms), color: "#c0504d", width: 1.2 }); c.setData(s, { keepView: true }); }
  $("#rs_audio").innerHTML = `<audio controls preload="none" src="/api/item/${id}/audio.wav"></audio>`;
  const ratio = v.fds_ratio, fdsRes = v.fds_target.map((t, i) => t * ratio[i]);
  chart("ch_rs_fds", { title: "FDS: target and result (error % on the right)", xlabel: "natural frequency [Hz]", ylabel: "damage", y2label: "error [%]", height: 300 })
    .setData([{ name: "target", x: v.fn, y: v.fds_target, width: 2 }, { name: "result (rainflow)", x: v.fn, y: fdsRes, mode: "points" }, { name: "error %", x: v.fn, y: ratio.map(r => 100 * (r - 1)), axis: "y2", color: "#8a8a8a", width: 1 }], { xlog: true, ylog: true });
  chart("ch_rs_mrs", { title: "MRS ratio result / target (above 1: higher peak stresses)", xlabel: "natural frequency [Hz]", ylabel: "ratio", height: 300 })
    .setData([{ name: "MRS ratio", x: v.fn, y: v.mrs_ratio, mode: "both" }, { name: "1", x: [v.fn[0], v.fn[v.fn.length - 1]], y: [1, 1], dash: [4, 4], color: "#999" }], { xlog: true, ylog: false });
  chart("ch_rs_psd", { title: "PSD", xlabel: "frequency [Hz]", ylabel: "PSD", height: 300 })
    .setData([{ name: "target", x: v.psd_ref_f, y: v.psd_ref }, { name: "result", x: v.psd_f, y: v.psd }], { ylog: true });
  const hist = (arr, edges) => { const h = new Array(edges.length - 1).fill(0); for (const a of arr) { let k = Math.min(edges.length - 2, Math.max(0, Math.floor((a - edges[0]) / (edges[1] - edges[0])))); h[k]++; } return h.map(x => x / arr.length); };
  const all = [...v.rms_syn, ...(v.rms_ref || [])], mx = Math.max(...all) * 1.05, ed = Array.from({ length: 41 }, (_, i) => i * mx / 40), mid = ed.slice(0, -1).map((e, i) => e + mx / 80);
  const ser = [{ name: "result", x: mid, y: hist(v.rms_syn, ed), mode: "step", color: "#c0504d" }]; if (v.rms_ref) ser.unshift({ name: "target record", x: mid, y: hist(v.rms_ref, ed), mode: "step" });
  chart("ch_rs_rms", { title: "Distribution of the 1-second RMS (burst structure)", xlabel: "1-s RMS", ylabel: "fraction of windows", height: 300 }).setData(ser);
  const kz = v.kz || {}, hasKz = kz.result && kz.result.length;
  $("#rs_kzrow").hidden = !hasKz && !mm;
  if (hasKz) {
    const ks = [];
    if (kz.target && kz.target.length) ks.push({ name: "target", x: v.fn, y: kz.target, width: 2 });
    ks.push({ name: "result", x: v.fn, y: kz.result, mode: "points" });
    chart("ch_rs_kz", { title: "Kurtosis of the oscillator responses k_z (3 = Gaussian)", xlabel: "natural frequency [Hz]", ylabel: "kurtosis", height: 300 }).setData(ks, { xlog: true, ylog: false });
  }
  $("#ch_rs_mod").hidden = !mm;
  if (mm) {
    chart("ch_rs_mod", { title: "MMVSD modulation m(f, t): local power / mean power", xlabel: "time [s]", ylabel: "frequency [Hz]", height: 300, legend: true })
      .setData([], { image: { x0: mm.x0, x1: mm.x1, y0: mm.y0, y1: mm.y1, z: mm.z, label: "dB" }, ylog: mm.ylog, xlog: false });
    const tc = mm.time_corr_s;
    $("#rs_mminfo").innerHTML = `frames ${esc(mm.frames || "")} · burst width ${F(mm.freq_corr_oct, 3)} oct · burst duration ${tc ? F(tc[0], 3) + "–" + F(tc[1], 3) + " s" : "–"}`
      + (mm.pattern_mapping ? ` · record pattern (${esc(mm.pattern_mapping)}), correlation ${F(mm.tf_corr, 2)}` : " · random field");
  } else $("#rs_mminfo").textContent = "";
  const edv = v.edit;
  $("#rs_edrow").hidden = !edv;
  if (edv) {
    const q = edv.lorenz_q, ser = [
      { name: "mean over fn", x: q, y: edv.lorenz_mean, width: 2 },
      { name: "worst fn (sets the limit)", x: q, y: edv.lorenz_worst, color: "#c0504d", width: 1.5 },
      { name: "lognormal closed form", x: q, y: edv.lorenz_lognormal, dash: [5, 4], color: "#888" },
      { name: "stationary Gaussian", x: [0, 1], y: [0, 1], dash: [2, 3], color: "#bbb" },
      { name: `requested 1/alpha = ${F(1 / edv.alpha_requested, 3)}`, x: [1 / edv.alpha_requested, 1 / edv.alpha_requested], y: [0, 1], color: "#2e8b57", width: 1 },
      { name: `${F(100 * edv.retention, 2)} % retention`, x: [0, 1], y: [edv.retention, edv.retention], dash: [4, 4], color: "#999" }];
    chart("ch_rs_lorenz", { title: "Damage Lorenz curve: share of the damage in the most damaging fraction of the time", xlabel: "fraction of the time kept (= 1/alpha)", ylabel: "share of the damage", height: 300 })
      .setData(ser, { xlog: false, ylog: false });
    chart("ch_rs_gain", { title: "Spectral gain applied after the editing (1 = editing alone)", xlabel: "natural frequency [Hz]", ylabel: "gain", height: 300 })
      .setData([{ name: "gain", x: v.fn, y: edv.gain_fn, mode: "both" }, { name: "1", x: [v.fn[0], v.fn[v.fn.length - 1]], y: [1, 1], dash: [4, 4], color: "#999" }], { xlog: true, ylog: false });
    const am = edv.alpha_max || {};
    $("#rs_edinfo").innerHTML = `compression requested ${F(edv.alpha_requested, 3)}, realised ${F(edv.alpha_realised, 3)} · editing alone keeps ${F(100 * edv.retention, 2)} % of the damage at every f<sub>n</sub> up to alpha ${F(edv.alpha_limit, 3)}`
      + ` (80 / 90 / 95 %: ${["80", "90", "95"].map(k => am[k] ? F(am[k].worst, 2) : "–").join(" / ")}; mean over f<sub>n</sub>: ${["80", "90", "95"].map(k => am[k] ? F(am[k].mean, 2) : "–").join(" / ")})`
      + ` · ${edv.n_segments} stretches kept, window ${F(edv.window_s, 2)} s + ring-down ${F(edv.margin_s, 2)} s, ranking ${esc(edv.ranking)}`
      + ` · windows hold ${F(100 * edv.kept_share, 2)} % of the damage (worst f<sub>n</sub> ${F(100 * edv.kept_share_min, 2)} %)`
      + ` · kurtosis of the record ${F(edv.kurtosis_record, 3)}, of the edited record ${F(edv.kurtosis_edited, 3)}`
      + (edv.gain_on ? ` · spectral gain x${F(Math.min(...edv.gain_fn), 3)} to x${F(Math.max(...edv.gain_fn), 3)} in ${edv.gain_iterations} steps` : " · spectral gain off")
      + (Array.isArray(edv.segments_s) && edv.segments_s.length ? `<br>kept stretches of the record [s]: ${edv.segments_s.map(s => `${(+s[0]).toFixed(1)} to ${(+s[1]).toFixed(1)}`).join(", ")}` : "");
  } else $("#rs_edinfo").textContent = "";
  $("#rsh_out").textContent = "";
}
$("#rs_dl").onclick = () => download(`/api/export?what=series&fmt=${$("#rs_fmt").value}&id=${S.res}`);
$("#rs_cdl").onclick = () => download(`/api/export?what=compare&fmt=${$("#rs_cfmt").value}&id=${S.res}`);
$("#rs_pdf").onclick = () => runJob(`/api/result/${S.res}/matchpdf`, {}, async r => { await refresh(); if (r) showResult(r.ids[0]); });
$("#rsh_go").onclick = async () => $("#rsh_out").innerHTML = await shaker(S.res, "rsh");
$("#se_go").onclick = () => runJob(`/api/result/${S.res}/sensitivity`, { b_values: nums($("#se_b").value), Q_values: nums($("#se_q").value) }, res => {
  if (!res) return; const P = VChart.palette;
  chart("ch_sens", { title: "FDS error with other damage parameters [%]", xlabel: "natural frequency [Hz]", ylabel: "error [%]", height: 260 })
    .setData(res.map((r, i) => ({ name: `b ${r.b}, Q ${r.Q}`, x: r.fn, y: r.ratio.map(x => 100 * (x - 1)), color: P[i % 8] })), { xlog: true });
});
$("#rs_cmp").onclick = async () => {
  const ids = $$("#restable .rc:checked").map(i => i.value); if (!ids.length) return toast("tick results to compare", true);
  const vs = await Promise.all(ids.map(resSummary)), P = VChart.palette;
  chart("ch_cmp", { title: "FDS error of the selected results [%]", xlabel: "natural frequency [Hz]", ylabel: "error [%]", height: 280 })
    .setData(vs.map((v, i) => ({ name: v.name, x: v.fn, y: v.fds_ratio.map(r => 100 * (r - 1)), color: P[i % 8], mode: "both" })), { xlog: true });
};

/* ------------------------------------------------------------------- project */
$("#pj_save").onclick = () => download("/api/project/save?name=" + encodeURIComponent($("#pj_name").value || "project"));
$("#pj_load").onclick = async () => { const f = $("#pj_file").files[0]; if (!f) return toast("choose a .vsproj file", true); await api("/api/project/load", { raw: await f.arrayBuffer() }); S.resCache = {}; S.mission = null; await refresh(); toast("project opened"); };
$("#pj_new").onclick = async () => { if (confirm("Clear the workspace?")) { await api("/api/project/new", { body: {} }); S.resCache = {}; S.mission = null; await refresh(); } };
$("#lg_show").onclick = async () => { $("#lg_txt").textContent = await (await fetch("/api/log")).text(); };
$("#lg_dl").onclick = () => download("/api/log");

refresh().then(s => { if (s.job.running) watchJob(() => refresh()); });
