"""Session state of the application: records, specifications, missions, results.

Everything the interface does goes through a Workspace, so it can also be
scripted, saved to a project file and reopened.
"""
from __future__ import annotations

import io
import json
import time
import uuid
import zipfile
from dataclasses import dataclass, field

import numpy as np

from . import __version__
from . import preprocess as PP
from .analysis import Reference, analyze_reference, evaluate, response_statistics
from .fds import FDSParams, fds_time
from .mission import Group, Situation, combine, spec_target
from .pipeline import run_method
from .signals import natural_frequencies, statistics
from .spec import PSDSpec, SineSweepSpec

__all__ = ["Workspace", "Item", "DEFAULT_SETTINGS"]

DEFAULT_SETTINGS = {
    "fmin": 5.0, "fmax": None, "scale": "octave", "ppo": 12, "resolution": 1.0,
    "Q": 10.0, "b": 8.0, "C": 1.0, "K": 1.0, "prefilter": "auto", "fds_method": "time",
    "psd_resolution": 0.5, "spec_fs": 2000.0,
}


@dataclass
class Item:
    id: str
    name: str
    kind: str                         # record | psd_spec | sine_spec | result
    x: np.ndarray | None = None
    fs: float | None = None
    spec: object = None
    meta: dict = field(default_factory=dict)

    def summary(self):
        d = {"id": self.id, "name": self.name, "kind": self.kind, "meta": _jsonable(self.meta)}
        if self.x is not None:
            st = statistics(self.x)
            d.update(fs=self.fs, n=int(self.x.size), duration=self.x.size / self.fs,
                     stats={k: round(v, 6) for k, v in st.items()})
        if self.kind == "psd_spec":
            d.update(f=list(map(float, self.spec.f)), psd=list(map(float, self.spec.psd)),
                     rms=self.spec.rms())
        if self.kind == "sine_spec":
            s = self.spec
            d.update(f=list(map(float, s.f)), amplitude=list(map(float, s.amplitude)), rate=s.rate,
                     mode=s.mode, sweeps=s.sweeps, duration=s.duration)
        return d


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 5000 else f"<array {o.shape}>"
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


class Workspace:
    def __init__(self):
        self.items: dict[str, Item] = {}
        self.missions: dict[str, dict] = {}
        self.settings = dict(DEFAULT_SETTINGS)
        self.log: list[tuple[str, str]] = []
        self._cache: dict = {}

    # ------------------------------------------------------------------ log
    def note(self, text):
        self.log.append((time.strftime("%Y-%m-%d %H:%M:%S"), str(text)))

    def logfile(self) -> str:
        head = [f"vibsynth {__version__} log", "settings: " + json.dumps(self.settings), ""]
        return "\n".join(head + [f"[{t}] {m}" for t, m in self.log]) + "\n"

    # ---------------------------------------------------------------- items
    def _new_id(self):
        return uuid.uuid4().hex[:8]

    def _unique(self, name):
        names = {i.name for i in self.items.values()}
        if name not in names:
            return name
        k = 2
        while f"{name} ({k})" in names:
            k += 1
        return f"{name} ({k})"

    def add_record(self, x, fs, name, meta=None) -> str:
        i = Item(self._new_id(), self._unique(name), "record", np.asarray(x, float), float(fs),
                 meta=dict(meta or {}))
        self.items[i.id] = i
        self.note(f"record added: {i.name} ({x.size} samples, {fs:g} Hz)")
        return i.id

    def add_spec(self, spec, name=None) -> str:
        kind = "psd_spec" if isinstance(spec, PSDSpec) else "sine_spec"
        i = Item(self._new_id(), self._unique(name or spec.name), kind, spec=spec)
        self.items[i.id] = i
        self.note(f"specification added: {i.name}")
        return i.id

    def get(self, item_id) -> Item:
        if item_id not in self.items:
            raise KeyError(f"unknown item {item_id}")
        return self.items[item_id]

    def rename(self, item_id, name):
        self.get(item_id).name = self._unique(name)

    def delete(self, item_id):
        self.items.pop(item_id, None)
        self._cache = {k: v for k, v in self._cache.items() if k[0] != item_id}

    def preprocess(self, item_id, *, t0=None, t1=None, f_low=None, f_high=None, detrend=False,
                   resample_fs=None, gain=None, name=None) -> str:
        it = self.get(item_id)
        if it.x is None:
            raise ValueError("only time histories can be pre-processed")
        x, fs, hist = it.x, it.fs, list(it.meta.get("history", []))
        if t0 is not None or t1 is not None:
            x = PP.crop(x, fs, t0, t1)
            hist.append(f"crop {t0 or 0}-{t1 or 'end'} s")
        if detrend:
            x = PP.detrend(x)
            hist.append("mean removed")
        if f_low or f_high:
            x = PP.bandpass(x, fs, f_low, f_high)
            hist.append(f"filter {f_low or 0}-{f_high or 'fs/2'} Hz")
        if resample_fs:
            x, fs = PP.resample(x, fs, resample_fs)
            hist.append(f"resampled to {fs:g} Hz")
        if gain:
            x = PP.scale(x, gain)
            hist.append(f"scaled x{gain}")
        meta = dict(it.meta, history=hist, parent=it.name)
        return self.add_record(x, fs, name or f"{it.name} [{', '.join(hist[-3:])}]", meta)

    def split(self, item_id, edges) -> list[str]:
        it = self.get(item_id)
        pieces = PP.split(it.x, it.fs, edges)
        ids = []
        t = 0.0
        for k, piece in enumerate(pieces, 1):
            ids.append(self.add_record(piece, it.fs, f"{it.name} situation {k}",
                                       dict(it.meta, parent=it.name, history=it.meta.get("history", []) +
                                            [f"situation {k} from {t:.2f} s"])))
            t += piece.size / it.fs
        return ids

    # ------------------------------------------------------------- analysis
    def params_for(self, fs_list) -> FDSParams:
        s = self.settings
        fmax = s["fmax"] or min(fs_list) / 2.5
        fn = natural_frequencies(float(s["fmin"]), float(fmax), s["scale"], int(s["ppo"]),
                                 float(s["resolution"]))
        return FDSParams(fn=fn, Q=float(s["Q"]), b=float(s["b"]), C=float(s["C"]), K=float(s["K"]),
                         prefilter=s["prefilter"])

    def _key(self, *extra):
        return json.dumps([self.settings, *extra], sort_keys=True, default=str)

    def target(self, ref_id, life_h=None, settings_override=None, progress=None) -> Reference:
        """Target of a record, a specification or a mission (cached)."""
        saved = None
        if settings_override:
            saved = dict(self.settings)
            self.settings.update(settings_override)
        try:
            key = (ref_id, self._key(life_h))
            if key in self._cache:
                return self._cache[key]
            if ref_id in self.missions:
                ref = self._mission_target(ref_id)
            else:
                ref = self._item_target(self.get(ref_id), life_h, progress)
            self._cache[key] = ref
            return ref
        finally:
            if saved is not None:
                self.settings = saved

    def _fmax_common(self, fs_values):
        return self.settings["fmax"] or min(fs_values) / 2.5

    def _item_target(self, it: Item, life_h=None, progress=None, fmax=None, spec_fs=None) -> Reference:
        s = self.settings
        life = life_h * 3600 if life_h else (it.meta.get("life_h") or 0) * 3600 or None
        if it.kind in ("record", "result"):
            return analyze_reference(it.x, it.fs, fmin=float(s["fmin"]),
                                     fmax=float(fmax or s["fmax"] or it.fs / 2.5),
                                     scale=s["scale"], points_per_octave=int(s["ppo"]),
                                     resolution=float(s["resolution"]), Q=float(s["Q"]), b=float(s["b"]),
                                     C=float(s["C"]), K=float(s["K"]), life=life, prefilter=s["prefilter"],
                                     psd_resolution=float(s["psd_resolution"]), progress=progress,
                                     fds_method=s["fds_method"], name=it.name)
        fs = float(spec_fs or s["spec_fs"])
        p = self.params_for([fs]) if not fmax else FDSParams(
            fn=natural_frequencies(float(s["fmin"]), float(fmax), s["scale"], int(s["ppo"]),
                                   float(s["resolution"])),
            Q=float(s["Q"]), b=float(s["b"]), C=float(s["C"]), K=float(s["K"]), prefilter=s["prefilter"])
        spec_life = life or (it.spec.duration if it.kind == "sine_spec" else 3600.0)
        return spec_target(it.spec, p, spec_life, fs, name=it.name)

    # ------------------------------------------------------------- missions
    def set_mission(self, mission: dict) -> str:
        mid = mission.get("id") or "m" + self._new_id()
        mission["id"] = mid
        self._check_node(mission["root"])
        self.missions[mid] = mission
        self._cache = {k: v for k, v in self._cache.items() if k[0] != mid}
        self.note(f"mission defined: {mission.get('name', mid)}")
        return mid

    def _check_node(self, node):
        if node["type"] == "situation":
            self.get(node["item"])
            if float(node.get("life_h", 0)) <= 0:
                raise ValueError("every situation needs a positive life [h]")
        else:
            if node.get("kind") not in ("series", "parallel"):
                raise ValueError("group kind must be series or parallel")
            for c in node.get("children", []):
                self._check_node(c)

    def _leaves(self, node):
        if node["type"] == "situation":
            return [node]
        return [l for c in node.get("children", []) for l in self._leaves(c)]

    def _mission_target(self, mid) -> Reference:
        m = self.missions[mid]
        leaves = self._leaves(m["root"])
        rec_fs = [self.get(lf["item"]).fs for lf in leaves if self.get(lf["item"]).fs]
        spec_fs = max(rec_fs) if rec_fs else float(self.settings["spec_fs"])
        fs_vals = rec_fs or [spec_fs]
        fmax = self._fmax_common(fs_vals)

        def build(node):
            if node["type"] == "situation":
                it = self.get(node["item"])
                t = self._item_target(it, float(node["life_h"]), fmax=fmax, spec_fs=spec_fs)
                t.name = node.get("name") or it.name
                return Situation(t.name, t)
            return Group(node["kind"], [build(c) for c in node.get("children", [])], node.get("name", ""))
        ref = combine(build(m["root"]))
        ref.name = m.get("name", mid)
        return ref

    def mission_breakdown(self, mid):
        """FDS of every situation (over its life) for the contribution plot."""
        m = self.missions[mid]
        leaves = self._leaves(m["root"])
        rec_fs = [self.get(l["item"]).fs for l in leaves if self.get(l["item"]).fs]
        spec_fs = max(rec_fs) if rec_fs else float(self.settings["spec_fs"])
        fmax = self._fmax_common(rec_fs or [spec_fs])
        out = []
        for lf in leaves:
            t = self._item_target(self.get(lf["item"]), float(lf["life_h"]), fmax=fmax, spec_fs=spec_fs)
            out.append({"name": lf.get("name") or self.get(lf["item"]).name, "fds": t.fds, "mrs": t.mrs,
                        "life_h": float(lf["life_h"])})
        return out

    # ------------------------------------------------------------ synthesis
    def synthesize(self, ref_id, method, duration, *, life_h=None, name=None, log=None, **opts) -> str:
        ref = self.target(ref_id, life_h)
        res = run_method(ref, method, float(duration), log=log, **opts)
        scale = res.info.get("fds_target_scale", 1.0)
        if ref.response_kurtosis is None and res.info.get("k_response") is not None:
            try:                                     # target k_z for the comparison (one pass)
                response_statistics(ref)
            except ValueError:
                pass
        ev = evaluate(res.x, res.fs, ref, repeats=res.repeats, fds=res.info.get("fds_measured"),
                      mrs=res.info.get("mrs_measured"), target_scale=scale,
                      k_response=res.info.get("k_response"))
        label = name or f"{res.method} {duration:g}s <- {self._label(ref_id)}"
        meta = {"method": res.method, "target": ref_id, "target_life_h": ref.life / 3600,
                "duration": float(duration), "repeats": res.repeats, "options": opts,
                "warnings": res.warnings, "settings": dict(self.settings), "life_h_override": life_h}
        rid = self.add_record(res.x, res.fs, label, meta)
        it = self.items[rid]
        it.kind = "result"
        it.meta["result"] = {k: v for k, v in res.info.items()
                             if k in ("fds_measured", "mrs_measured", "fds_error_history", "alpha",
                                      "window_s", "pattern", "t_burst_s", "n_bursts", "tones",
                                      "fds_target_scale", "predicted_kurtosis", "lambda", "k_response",
                                      "k_target", "k_model", "excess", "modulation", "modulation_hop_s",
                                      "frames", "freq_corr_oct", "time_corr_s", "pattern_mapping",
                                      "tf_corr", "lines", "edit")}
        it.meta["envelope"] = res.envelope
        it.meta["design"] = {"fn": res.design_fn, "psd": res.design_psd}
        it.meta["eval"] = ev
        self.note(f"synthesis {res.method}: {label}; FDS max error {ev['fds_max_error_pct']:.1f} %, "
                  f"kurtosis {ev['stats']['kurtosis']:.2f}, rms {ev['stats']['rms']:.4g}"
                  + (f"; warnings: {res.warnings}" if res.warnings else ""))
        return rid

    def reevaluate(self, rid):
        it = self.get(rid)
        ref = self.target(it.meta["target"], it.meta.get("life_h_override"))
        if ref.response_kurtosis is None:
            try:
                response_statistics(ref)
            except ValueError:
                pass
        ev = evaluate(it.x, it.fs, ref, repeats=it.meta.get("repeats", 1),
                      target_scale=it.meta.get("result", {}).get("fds_target_scale", 1.0))
        it.meta["eval"] = ev
        return ev

    def _label(self, ref_id):
        if ref_id in self.missions:
            return self.missions[ref_id].get("name", ref_id)
        return self.get(ref_id).name

    def compressibility(self, item_id, window_s=None, ranking="common", progress=None) -> dict:
        """Damage Lorenz curves and compression limits of a record (edit.compressibility)."""
        from .edit import compressibility
        it = self.get(item_id)
        if it.x is None:
            raise ValueError("compressibility needs a time history")
        p = self.params_for([it.fs])
        out = compressibility(it.x, it.fs, p, window_s=window_s, ranking=ranking, progress=progress)
        from .edit import lognormal_alpha_max
        out["name"] = it.name
        out["lognormal_alpha90"] = lognormal_alpha_max(0.9, out["kurtosis"], p.b)
        out.pop("per_fn", None)
        return out

    def sensitivity(self, rid, b_values=(), Q_values=(), progress=None):
        """FDS ratio result/target for other b and Q (target recomputed too)."""
        it = self.get(rid)
        out = []
        combos = [{"b": float(b)} for b in b_values] + [{"Q": float(q)} for q in Q_values]
        for i, ov in enumerate(combos):
            ref = self.target(it.meta["target"], it.meta.get("life_h_override"), settings_override=ov)
            p = ref.params
            Ds, _ = fds_time(it.x, it.fs, p, repeats=it.meta.get("repeats", 1))
            sc = it.meta.get("result", {}).get("fds_target_scale", 1.0)
            out.append({"b": p.b, "Q": p.Q, "fn": p.fn, "ratio": Ds / (ref.fds * sc)})
            if progress:
                progress((i + 1) / len(combos))
        return out

    # -------------------------------------------------------------- project
    def save_project(self) -> bytes:
        buf = io.BytesIO()
        arrays = {}
        items = []
        for it in self.items.values():
            d = {"id": it.id, "name": it.name, "kind": it.kind, "fs": it.fs, "meta": _project_meta(it.meta)}
            if it.x is not None:
                arrays[f"x_{it.id}"] = it.x.astype(np.float32)     # 7 significant digits are plenty
            if isinstance(it.spec, PSDSpec):
                d["spec"] = {"type": "psd", "f": list(it.spec.f), "psd": list(it.spec.psd), "name": it.spec.name}
            elif isinstance(it.spec, SineSweepSpec):
                s = it.spec
                d["spec"] = {"type": "sine", "f": list(s.f), "amplitude": list(s.amplitude), "rate": s.rate,
                             "mode": s.mode, "sweeps": s.sweeps, "name": s.name}
            items.append(d)
        arr = io.BytesIO()
        np.savez_compressed(arr, **arrays)
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("project.json", json.dumps({"version": __version__, "settings": self.settings,
                                                   "items": items, "missions": self.missions,
                                                   "log": self.log}, indent=1, default=float))
            z.writestr("arrays.npz", arr.getvalue())
        return buf.getvalue()

    @classmethod
    def load_project(cls, data: bytes) -> "Workspace":
        ws = cls()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            pj = json.loads(z.read("project.json"))
            arrays = np.load(io.BytesIO(z.read("arrays.npz")), allow_pickle=False)
        ws.settings.update(pj.get("settings", {}))
        for d in pj["items"]:
            spec = None
            if "spec" in d:
                sp = d["spec"]
                if sp["type"] == "psd":
                    spec = PSDSpec(sp["f"], sp["psd"], sp.get("name", "PSD"))
                else:
                    spec = SineSweepSpec(sp["f"], sp["amplitude"], sp["rate"], sp["mode"], sp["sweeps"],
                                         sp.get("name", "sine"))
            x = arrays[f"x_{d['id']}"].astype(float) if f"x_{d['id']}" in arrays.files else None
            ws.items[d["id"]] = Item(d["id"], d["name"], d["kind"], x, d.get("fs"), spec, d.get("meta", {}))
        ws.missions = pj.get("missions", {})
        ws.log = [tuple(e) for e in pj.get("log", [])]
        for it in ws.items.values():                     # results: recompute evaluation lazily
            it.meta.pop("eval", None)
        ws.note("project loaded")
        return ws


def _project_meta(meta):
    m = {k: v for k, v in meta.items() if k not in ("eval", "envelope", "design")}
    return json.loads(json.dumps(_jsonable(m), default=str))
