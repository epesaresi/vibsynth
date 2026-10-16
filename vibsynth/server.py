"""Local web server for the vibsynth interface (standard library only).

`python -m vibsynth gui` starts it at http://127.0.0.1:8765 and opens the
browser. One workspace per server; long computations run as a cancellable
background job whose log and progress the page polls.
"""
from __future__ import annotations

import io
import json
import mimetypes
import os
import tempfile
import threading
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from scipy.signal import resample_poly

from . import __version__
from . import diagnostics as DG
from . import export as EX
from .io import load_signal, list_mat_variables
from .signals import welch_psd, statistics
from .spec import PSDSpec, SineSweepSpec
from .synthesis import METHODS
from .workspace import Workspace

WEB = Path(__file__).with_name("web")


class Cancelled(Exception):
    pass


class App:
    def __init__(self):
        self.ws = Workspace()
        self.uploads: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.job = {"id": None, "running": False, "log": [], "progress": 0.0, "error": None,
                    "result": None, "cancel": False, "title": ""}
        self.analysis: dict[str, dict] = {}       # last computed curves per target id

    # ---------------------------------------------------------------- jobs
    def log(self, msg):
        with self.lock:
            self.job["log"].append(str(msg))
        if self.job["cancel"]:
            raise Cancelled()

    def progress(self, f):
        self.job["progress"] = float(f)
        if self.job["cancel"]:
            raise Cancelled()

    def start(self, title, fn):
        if self.job["running"]:
            raise RuntimeError("a computation is already running")
        self.job = {"id": uuid.uuid4().hex[:6], "running": True, "log": [], "progress": 0.0,
                    "error": None, "result": None, "cancel": False, "title": title}

        def run():
            try:
                self.job["result"] = fn()
                self.log("done")
            except Cancelled:
                self.job["error"] = "cancelled"
            except Exception as exc:  # noqa: BLE001
                self.job["error"] = str(exc)
                with self.lock:
                    self.job["log"].append(traceback.format_exc(limit=3))
            finally:
                self.job["running"] = False
                self.job["progress"] = 1.0
        threading.Thread(target=run, daemon=True).start()
        return {"job": self.job["id"]}

    # ---------------------------------------------------------- serializers
    def state(self):
        ws = self.ws
        return {"version": __version__, "settings": ws.settings,
                "items": [i.summary() for i in ws.items.values()],
                "missions": list(ws.missions.values()), "methods": METHODS,
                "job": self.job_public()}

    def job_public(self):
        j = self.job
        return {k: j[k] for k in ("id", "running", "progress", "error", "title")} | {"log": j["log"][-300:]}

    def target_curves(self, ref_id, life_h=None):
        ref = self.ws.target(ref_id, life_h, progress=self.progress)
        out = {"id": ref_id, "name": ref.name, "kind": ref.kind, "fn": ref.params.fn, "fds": ref.fds,
               "mrs": ref.mrs, "psd_f": ref.psd_f, "psd": ref.psd, "life_h": ref.life / 3600,
               "stats": ref.stats, "fs": ref.fs, "notes": ref.notes, "repeats": ref.repeats,
               "params": {"Q": ref.params.Q, "b": ref.params.b},
               "kz": ref.response_kurtosis}
        if ref_id in self.ws.missions:
            out["breakdown"] = self.ws.mission_breakdown(ref_id)
        self.analysis[ref_id] = out
        return out

    def result_view(self, rid):
        it = self.ws.get(rid)
        if "eval" not in it.meta:
            self.ws.reevaluate(rid)
        ev = it.meta["eval"]
        ref = self.ws.target(it.meta["target"], it.meta.get("life_h_override"))
        sc = ev.get("target_scale", 1.0)
        rows = [("Duration [s]", f"{ref.duration:.1f} (life {ref.life / 3600:.4g} h)",
                 f"{it.x.size / it.fs:.1f} x {it.meta.get('repeats', 1)}")]
        for k, lab in (("rms", "RMS"), ("kurtosis", "Kurtosis"), ("skewness", "Skewness"),
                       ("crest_factor", "Crest factor"), ("peak", "Peak")):
            rv = ref.stats.get(k, float("nan"))
            rows.append((lab, f"{rv:.4g}" if np.isfinite(rv) else "\u2013", f"{ev['stats'][k]:.4g}"))
        rows += [("FDS error max / median", "", f"{ev['fds_max_error_pct']:.1f} % / {ev['fds_median_error_pct']:.1f} %"),
                 ("FDS nodes within 10 %", "", f"{100 * ev['fds_within_10pct']:.0f} %"),
                 ("MRS ratio max", "", f"{np.max(ev['mrs_ratio']):.2f}"),
                 ("PSD deviation in band (rms)", "", f"{ev['psd_rms_db_dev']:.1f} dB")]
        kz_med = ev.get("kz_median_abs_err")
        if kz_med is not None:
            rows.append(("Response kurtosis k_z: median |error|", "", f"{kz_med:.2f}"))
        res = it.meta.get("result", {})
        env = it.meta.get("envelope")
        t = np.arange(it.x.size) / it.fs
        envd = None
        if env is not None and np.size(env) == it.x.size:
            te, ee = DG.decimate_minmax(t, env, 2000)
            envd = {"t": te, "y": ee}
        rr = DG.running_stats(it.x, it.fs, 1.0)
        rf = DG.running_stats(ref.x, ref.fs, 1.0) if ref.x is not None else None
        return {"id": rid, "name": it.name, "method": it.meta.get("method"), "target": it.meta.get("target"),
                "target_name": ref.name, "table": rows, "warnings": it.meta.get("warnings", []),
                "fn": ref.params.fn, "fds_target": ref.fds * sc, "fds_ratio": ev["fds_ratio"],
                "mrs_target": ref.mrs, "mrs_ratio": ev["mrs_ratio"], "psd_ref_f": ref.psd_f,
                "psd_ref": ref.psd, "psd_f": ev["psd_f"], "psd": ev["psd"],
                "history": res.get("fds_error_history", []), "envelope": envd,
                "rms_syn": rr[1], "rms_ref": rf[1] if rf else None, "fs": it.fs,
                "ref_has_series": ref.x is not None, "ref_id": it.meta.get("target"),
                "summary": {"fds_max": ev["fds_max_error_pct"], "fds_median": ev["fds_median_error_pct"],
                            "kurtosis": ev["stats"]["kurtosis"], "rms": ev["stats"]["rms"],
                            "crest": ev["stats"]["crest_factor"], "mrs_max": float(np.max(ev["mrs_ratio"])),
                            "psd_db": ev["psd_rms_db_dev"], "kz_med": kz_med},
                "tones": res.get("tones"),
                "kz": {"target": ev.get("kz_target"), "result": ev.get("k_response"),
                       "model": res.get("k_model")},
                "mmvsd": self._mmvsd_view(it, res, ref)}

    @staticmethod
    def _mmvsd_view(it, res, ref):
        """Modulation field and settings of an MMVSD result (None for other methods)."""
        if res.get("modulation") is None:
            return None
        m = np.asarray(res["modulation"], float)
        fn = ref.params.fn
        if m.ndim != 2 or m.shape[0] != fn.size:
            return None
        T = it.x.size / it.fs
        z = np.clip(10 * np.log10(np.maximum(m, 1e-6)), -20, 15)
        # rows on a uniform axis (log f for a 1/n-octave grid, f for a linear one),
        # so that the image rows sit at their frequencies
        d = np.diff(np.log(fn)) if fn.size > 2 else np.ones(1)
        ylog = bool(fn.size > 2 and np.allclose(d[:-1], d[0], rtol=1e-3))
        u = np.log(fn) if ylog else fn
        grid = np.linspace(u[0], u[-1], fn.size)
        z = np.array([np.interp(grid, u, col) for col in z.T]).T
        h = 0.5 * (grid[1] - grid[0]) if fn.size > 1 else 0.5
        y0, y1 = (np.exp(grid[0] - h), np.exp(grid[-1] + h)) if ylog else (grid[0] - h, grid[-1] + h)
        tc = res.get("time_corr_s")
        return {"z": z, "x0": 0.0, "x1": T,
                "y0": y0, "y1": y1, "ylog": ylog, "frames": res.get("frames"),
                "freq_corr_oct": res.get("freq_corr_oct"),
                "time_corr_s": None if tc is None else [float(np.min(tc)), float(np.max(tc))],
                "pattern_mapping": res.get("pattern_mapping"), "tf_corr": res.get("tf_corr"),
                "lines": res.get("lines"), "excess": res.get("excess")}


APP = App()


def _clean(o):
    """JSON with NaN/inf -> null and arrays -> lists."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    # ---------------------------------------------------------------- utils
    def send(self, code, body=None, ctype="application/json", headers=None):
        if body is None:
            body = {}
        if isinstance(body, (dict, list)):
            body = json.dumps(_clean(body)).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def file(self, data: bytes, name: str):
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        self.send(200, data, ctype, {"Content-Disposition": f'attachment; filename="{name}"'})

    def body(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def jbody(self):
        b = self.body()
        return json.loads(b) if b else {}

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        parts = [unquote(p) for p in u.path.strip("/").split("/")]
        try:
            if u.path in ("/", "/index.html"):
                return self.send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if parts[0] == "static":
                f = WEB / parts[1]
                if f.is_file() and f.parent == WEB:
                    return self.send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "text/plain")
                return self.send(404, {"error": "not found"})
            if u.path == "/api/state":
                return self.send(200, APP.state())
            if u.path == "/api/job":
                return self.send(200, APP.job_public())
            if u.path == "/api/job/result":
                return self.send(200, {"result": APP.job["result"], "error": APP.job["error"]})
            if u.path == "/api/log":
                return self.file(APP.ws.logfile().encode(), "vibsynth_log.txt")
            if u.path == "/api/project/save":
                return self.file(APP.ws.save_project(), q.get("name", "project") + ".vsproj")
            if parts[:2] == ["api", "item"] and len(parts) >= 4:
                return self.get_item(parts[2], parts[3], q)
            if parts[:2] == ["api", "result"] and len(parts) == 3:
                return self.send(200, APP.result_view(parts[2]))
            if u.path == "/api/analysis":
                return self.send(200, APP.analysis.get(q["id"]) or {})
            if u.path == "/api/export":
                return self.export(q)
            return self.send(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            return self.send(400, {"error": str(exc), "trace": traceback.format_exc(limit=2)})

    def get_item(self, iid, what, q):
        it = APP.ws.get(iid)
        if what == "series":
            if it.x is None:
                return self.send(200, {"t": [], "y": []})
            t0 = float(q.get("t0", 0))
            t1 = float(q.get("t1", it.x.size / it.fs))
            i0 = max(int(t0 * it.fs) - 1, 0)
            i1 = min(int(t1 * it.fs) + 2, it.x.size)
            t = np.arange(i0, i1) / it.fs
            tt, yy = DG.decimate_minmax(t, it.x[i0:i1], int(q.get("n", 3000)))
            return self.send(200, {"t": tt, "y": yy, "full": i1 - i0 <= int(q.get("n", 3000))})
        if what == "audio.wav":
            x = it.x[: int(min(it.x.size, 60 * it.fs))]
            fs = it.fs
            if fs < 8000:
                up = int(np.ceil(8000 / fs))
                x, fs = resample_poly(x, up, 1), fs * up
            return self.send(200, EX.series_bytes(x, fs, "wav"), "audio/wav")
        if what == "psd":
            f, P = welch_psd(it.x, it.fs, resolution=float(q.get("res", APP.ws.settings["psd_resolution"])))
            return self.send(200, {"f": f, "psd": P})
        if what == "pdf":
            c, h, g = DG.pdf(it.x, int(q.get("bins", 121)))
            return self.send(200, {"x": c, "pdf": h, "gauss": g, "stats": statistics(it.x)})
        if what == "running":
            t, r, k = DG.running_stats(it.x, it.fs, float(q.get("window", 1.0)))
            return self.send(200, {"t": t, "rms": r, "kurtosis": k})
        if what == "spectrogram":
            f, t, S = DG.spectrogram(it.x, it.fs, int(q["nperseg"]) if q.get("nperseg") else None)
            return self.send(200, {"f": f, "t": t, "z": S})
        if what == "sk":
            f, sk = DG.spectral_kurtosis(it.x, it.fs, int(q["nfft"]) if q.get("nfft") else None)
            return self.send(200, {"f": f, "sk": sk})
        if what == "kurtogram":
            f, nw, Z = DG.kurtogram(it.x, it.fs)
            return self.send(200, {"f": f, "nwin": nw, "z": Z})
        if what == "sdof":
            fn = float(q["fn"])
            Q = float(q.get("Q", APP.ws.settings["Q"]))
            z = DG.sdof_response(it.x, it.fs, fn, Q, q.get("quantity", "pseudo-acceleration"))
            t = np.arange(z.size) / it.fs
            tt, yy = DG.decimate_minmax(t, z, 4000)
            return self.send(200, {"t": tt, "y": yy, "stats": statistics(z)})
        return self.send(404, {"error": "unknown view"})

    def export(self, q):
        what, fmt = q.get("what", "series"), q.get("fmt", "csv")
        safe = lambda s: "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:60]
        if what == "target":
            ref = APP.ws.target(q["id"], float(q["life_h"]) if q.get("life_h") else None)
            nm = safe(ref.name)
            files = {}
            meta = {"name": ref.name, "life_h": ref.life / 3600, "Q": ref.params.Q, "b": ref.params.b,
                    "C": ref.params.C, "K": ref.params.K}
            fd = {"fn_Hz": ref.params.fn, "FDS": ref.fds, "MRS": ref.mrs}
            ps = {"f_Hz": ref.psd_f, "PSD": ref.psd}
            if fmt in ("xlsx", "mat", "json"):
                both = dict(fd, **{"psd_" + k: v for k, v in ps.items()})
                return self.file(EX.curves_bytes(both, fmt, meta, "FDS_MRS_PSD"), f"{nm}_FDS_MRS_PSD.{fmt}")
            files[f"{nm}_FDS_MRS.csv"] = EX.curves_bytes(fd, "csv", meta)
            files[f"{nm}_PSD.csv"] = EX.curves_bytes(ps, "csv", meta)
            return self.file(EX.zip_bytes(files), f"{nm}_curves.zip")
        it = APP.ws.get(q["id"])
        nm = safe(it.name)
        if what == "series":
            data = EX.series_bytes(it.x, it.fs, fmt, {"name": it.name})
            suffix = f"_fullscale_{np.max(np.abs(it.x)):.4g}" if fmt == "wav" else ""
            return self.file(data, f"{nm}{suffix}.{fmt}")
        if what == "psd":
            f, P = welch_psd(it.x, it.fs, resolution=float(q.get("res", APP.ws.settings["psd_resolution"])))
            return self.file(EX.curves_bytes({"f_Hz": f, "PSD": P}, fmt, {"name": it.name}), f"{nm}_PSD.{fmt}")
        if what == "stats":
            st = statistics(it.x)
            return self.file(EX.curves_bytes({k: [v] for k, v in st.items()}, fmt, {"name": it.name}),
                             f"{nm}_stats.{fmt}")
        if what == "compare":
            v = APP.result_view(it.id)
            cur = {"fn_Hz": v["fn"], "FDS_target": v["fds_target"],
                   "FDS_result": np.asarray(v["fds_target"]) * np.asarray(v["fds_ratio"]),
                   "FDS_ratio": v["fds_ratio"], "MRS_target": v["mrs_target"],
                   "MRS_ratio": v["mrs_ratio"]}
            return self.file(EX.curves_bytes(cur, fmt, {"result": it.name, "target": v["target_name"]}),
                             f"{nm}_comparison.{fmt}")
        return self.send(400, {"error": "unknown export"})

    # ----------------------------------------------------------------- POST
    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        parts = [unquote(p) for p in u.path.strip("/").split("/")]
        ws = APP.ws
        try:
            if u.path == "/api/settings":
                d = self.jbody()
                for k, v in d.items():
                    if k in ws.settings:
                        ws.settings[k] = v if v not in ("", None) or k != "fmax" else None
                return self.send(200, {"settings": ws.settings})
            if u.path == "/api/upload":
                return self.send(200, self.upload(q))
            if u.path == "/api/import":
                return self.send(200, self.do_import(self.jbody()))
            if u.path == "/api/job/cancel":
                APP.job["cancel"] = True
                return self.send(200, {})
            if u.path == "/api/project/load":
                APP.ws = Workspace.load_project(self.body())
                APP.analysis = {}
                return self.send(200, APP.state())
            if u.path == "/api/project/new":
                APP.ws = Workspace()
                APP.analysis = {}
                return self.send(200, APP.state())
            if parts[:2] == ["api", "item"]:
                return self.post_item(parts[2], parts[3], self.jbody())
            if u.path == "/api/spec":
                d = self.jbody()
                if d["type"] == "psd":
                    sid = ws.add_spec(PSDSpec(d["f"], d["psd"], d.get("name") or "PSD specification"))
                else:
                    sid = ws.add_spec(SineSweepSpec(d["f"], d["amplitude"], float(d.get("rate", 1)),
                                                    d.get("mode", "log"), float(d.get("sweeps", 1)),
                                                    d.get("name") or "Swept sine"))
                return self.send(200, {"id": sid})
            if u.path == "/api/spec/preview":
                d = self.jbody()
                if d["type"] == "psd":
                    s = PSDSpec(d["f"], d["psd"])
                    f = np.geomspace(s.f[0], s.f[-1], 400)
                    return self.send(200, {"f": f, "y": s.psd_on(f), "rms": s.rms()})
                s = SineSweepSpec(d["f"], d["amplitude"], float(d.get("rate", 1)), d.get("mode", "log"),
                                  float(d.get("sweeps", 1)))
                f = np.geomspace(s.f[0], s.f[-1], 400)
                from .spec import loglog_interp
                return self.send(200, {"f": f, "y": loglog_interp(s.f, s.amplitude, f),
                                       "duration": s.duration})
            if u.path == "/api/mission":
                d = self.jbody()
                return self.send(200, {"id": ws.set_mission(d)})
            if parts[:2] == ["api", "mission"] and len(parts) == 4 and parts[3] == "delete":
                ws.missions.pop(parts[2], None)
                return self.send(200, {})
            if u.path == "/api/analyze":
                d = self.jbody()
                ids = d["ids"]
                life = d.get("life_h") or {}

                def job():
                    out = {}
                    for k, i in enumerate(ids):
                        APP.log(f"analysing {ws._label(i)} ({k + 1}/{len(ids)})")
                        out[i] = APP.target_curves(i, life.get(i))
                    return out
                return self.send(200, APP.start("analysis", job))
            if u.path == "/api/response_stats":
                d = self.jbody()

                def job():
                    out = {}
                    for i in d["ids"]:
                        it = ws.get(i)
                        p = ws.params_for([it.fs])
                        APP.log(f"response kurtosis / std: {it.name}")
                        k, s = DG.response_stats(it.x, it.fs, p.fn, p.Q, p.prefilter, progress=APP.progress)
                        out[i] = {"fn": p.fn, "kurtosis": k, "std": s, "name": it.name}
                    return out
                return self.send(200, APP.start("response statistics", job))
            if u.path == "/api/synth":
                return self.send(200, self.synth(self.jbody()))
            if parts[:2] == ["api", "result"] and len(parts) == 4:
                return self.post_result(parts[2], parts[3], self.jbody())
            return self.send(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            return self.send(400, {"error": str(exc), "trace": traceback.format_exc(limit=2)})

    def upload(self, q):
        data = self.body()
        name = q.get("name", "record.mat")
        ext = os.path.splitext(name)[1].lower()
        d = tempfile.mkdtemp(prefix="vibsynth_")
        path = os.path.join(d, "upload" + ext)
        with open(path, "wb") as f:
            f.write(data)
        uid = uuid.uuid4().hex[:8]
        info = {"upload_id": uid, "name": name, "ext": ext}
        if ext == ".mat":
            from scipy.io import loadmat
            try:
                m = loadmat(path)
                vars_ = [{"name": k, "shape": list(np.shape(v))} for k, v in m.items()
                         if not k.startswith("__") and np.size(v) > 10 and np.asarray(v).dtype.kind in "fiu"]
                fsm = [float(np.ravel(v)[0]) for k, v in m.items() if k.lower() in ("fs", "fsamp")]
            except NotImplementedError:
                vars_ = [{"name": v, "shape": []} for v in list_mat_variables(path)]
                fsm = []
            info.update(variables=vars_, fs_meta=fsm[0] if fsm else None)
        elif ext in (".csv", ".txt", ".dat", ".npy"):
            if ext == ".npy":
                a = np.load(path, allow_pickle=False)
            else:
                try:
                    a = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
                except ValueError:
                    a = np.loadtxt(path, delimiter=",", skiprows=1)
            info.update(variables=[{"name": "data", "shape": list(a.shape)}])
        elif ext == ".vsproj":
            raise ValueError("load projects from the Project tab")
        else:
            raise ValueError(f"unsupported file type {ext}")
        APP.uploads[uid] = {"path": path, **info}
        return info

    def do_import(self, d):
        up = APP.uploads[d["upload_id"]]
        path, name = up["path"], d.get("name") or os.path.splitext(up["name"])[0]
        fs = float(d["fs"]) if d.get("fs") else None
        units = d.get("units", "m/s^2")
        var = d.get("variable")
        if up["ext"] == ".mat":
            from scipy.io import loadmat
            try:
                a = np.asarray(loadmat(path)[var or up["variables"][0]["name"]], float).squeeze()
            except NotImplementedError:
                import h5py
                with h5py.File(path, "r") as h:
                    a = np.array(h[var or up["variables"][0]["name"]]).T.squeeze()
            fs = fs or up.get("fs_meta")
        elif up["ext"] == ".npy":
            a = np.load(path, allow_pickle=False)
        else:
            try:
                a = np.loadtxt(path, delimiter="," if up["ext"] == ".csv" else None)
            except ValueError:
                a = np.loadtxt(path, delimiter=",", skiprows=1)
        a = np.asarray(a, float)
        if a.ndim == 2 and a.shape[0] < a.shape[1]:
            a = a.T
        time_col = d.get("time_column", "auto")
        cols = []
        if a.ndim == 1:
            cols = [(name, a)]
        else:
            t = a[:, 0]
            dt = np.diff(t)
            has_time = (time_col is True) or (time_col == "auto" and np.all(dt > 0)
                                               and np.std(dt) < 1e-3 * np.mean(dt))
            if has_time:
                fs = fs or 1.0 / np.mean(dt)
                data = a[:, 1:]
            else:
                data = a
            sel = d.get("columns") or list(range(data.shape[1]))
            for c in sel:
                cols.append((name if data.shape[1] == 1 else f"{name} ch{c + 1}", data[:, int(c)]))
        if not fs:
            raise ValueError("no time vector found: give the sampling frequency")
        ids = []
        for nm, x in cols:
            x = np.asarray(x, float) * (9.81 if units == "g" else 1.0)
            ids.append(APP.ws.add_record(x, fs, nm, {"source": up["name"], "units_in": units,
                                                     "variable": var}))
        return {"ids": ids, "state": APP.state()}

    def post_item(self, iid, action, d):
        ws = APP.ws
        if action == "rename":
            ws.rename(iid, d["name"])
        elif action == "delete":
            ws.delete(iid)
        elif action == "meta":
            it = ws.get(iid)
            it.meta.update({k: v for k, v in d.items() if k in ("life_h", "note")})
            ws._cache.clear()
        elif action == "preprocess":
            nid = ws.preprocess(iid, **{k: (float(v) if k != "name" and k != "detrend" else v)
                                        for k, v in d.items() if v not in ("", None, False)})
            return self.send(200, {"id": nid, "state": APP.state()})
        elif action == "split":
            ids = ws.split(iid, [float(e) for e in d["edges"]])
            return self.send(200, {"ids": ids, "state": APP.state()})
        elif action == "shaker":
            it = ws.get(iid)
            lim = {k: float(d[k]) if d.get(k) not in (None, "") else None
                   for k in ("acc_limit", "vel_limit", "disp_limit", "hp")}
            return self.send(200, DG.shaker_check(it.x, it.fs, **lim))
        elif action == "generate":
            it = ws.get(iid)
            fs = float(d.get("fs") or ws.settings["spec_fs"])
            if it.kind == "psd_spec":
                x = it.spec.signal(fs, float(d.get("duration", 60)), seed=d.get("seed"))
            else:
                x = it.spec.signal(fs, float(d["duration"]) if d.get("duration") else None)
            nid = ws.add_record(x, fs, f"{it.name} signal", {"source": "specification " + it.name})
            return self.send(200, {"id": nid, "state": APP.state()})
        else:
            return self.send(404, {"error": "unknown action"})
        return self.send(200, APP.state())

    def synth(self, d):
        ws = APP.ws
        target = d["target"]
        life_h = float(d["life_h"]) if d.get("life_h") else None
        methods = d.get("methods") or [d["method"]]
        seeds = d.get("seeds") or [None if d.get("seed") in (None, "") else int(d["seed"])]
        opts = {}
        for k in ("kurtosis", "rms", "window", "fs", "tol", "prominence_db", "drift"):
            if d.get(k) not in (None, ""):
                opts[k] = float(d[k])
        if d.get("iterations") not in (None, ""):
            opts["iterations"] = int(d["iterations"])
        if d.get("pattern"):
            opts["pattern"] = d["pattern"]
        if d.get("tones"):
            opts["tones"] = [tuple(map(float, t)) for t in d["tones"]]
        for k in ("block_s", "max_excess", "freq_corr", "time_corr", "cl_anchor"):   # MMVSD
            if d.get(k) not in (None, ""):
                opts[k] = float(d[k])
        for k in ("modulation", "mapping", "lines"):
            if d.get(k):
                opts[k] = str(d[k])
        dur = float(d["duration"])

        def job():
            ids = []
            n = len(methods) * len(seeds)
            k = 0
            for m in methods:
                for s in seeds:
                    k += 1
                    APP.log(f"[{k}/{n}] {METHODS.get(m, m)}" + (f", seed {s}" if s is not None else ""))
                    nm = d.get("name") if n == 1 else None
                    ids.append(ws.synthesize(target, m, dur, life_h=life_h, name=nm, log=APP.log, seed=s, **opts))
            return {"ids": ids}
        return APP.start("synthesis", job)

    def post_result(self, rid, action, d):
        ws = APP.ws
        if action == "sensitivity":
            bs = [float(v) for v in d.get("b_values", [])]
            qs = [float(v) for v in d.get("Q_values", [])]
            return self.send(200, APP.start("sensitivity", lambda: ws.sensitivity(rid, bs, qs, APP.progress)))
        if action == "matchpdf":
            it = ws.get(rid)
            ref = ws.target(it.meta["target"], it.meta.get("life_h_override"))
            src = ref.x if ref.x is not None else np.concatenate([s[0] for s in ref.segments])
            y = DG.match_pdf(it.x, src * (np.std(it.x) / np.std(src)))

            def job():
                nid = ws.add_record(y, it.fs, it.name + " [PDF matched]", dict(it.meta))
                ws.items[nid].kind = "result"
                ws.items[nid].meta.pop("envelope", None)
                ws.items[nid].meta.pop("eval", None)
                APP.log("re-measuring the FDS after PDF matching")
                ws.reevaluate(nid)
                return {"ids": [nid]}
            return self.send(200, APP.start("PDF matching", job))
        return self.send(404, {"error": "unknown action"})


def serve(port=8765, open_browser=True):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"vibsynth {__version__} running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
