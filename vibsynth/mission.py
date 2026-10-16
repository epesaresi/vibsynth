"""Multi-situation missions: situations combined in series or in parallel.

A mission is a tree. Leaves are situations (a measured record or a
specification with the life it represents); groups combine their children
* in SERIES  : the component lives through all of them one after the other:
               FDS = sum, MRS = max, PSD = life-weighted mean, life = sum,
               moments (for RMS and kurtosis, also of every oscillator
               response) = life-weighted mixture;
* in PARALLEL: the component lives through ONE of them (alternatives, e.g.
               different customers or routes): FDS = envelope (max),
               MRS = max, PSD = envelope, life, RMS and
               kurtosis of the dominant (most damaging) child.
Groups can be nested, which covers "in series / parallel to the previous
situation / to all the previous ones" and more.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import Reference
from .fds import FDSParams, response_spectra, response_variance_operator, fds_gaussian
from .signals import trapezoid, statistics, welch_psd
from .spec import PSDSpec, SineSweepSpec

__all__ = ["Situation", "Group", "combine", "spec_target", "describe"]


@dataclass
class Situation:
    name: str
    target: Reference


@dataclass
class Group:
    kind: str                         # "series" | "parallel"
    children: list = field(default_factory=list)
    name: str = ""


def _same_grid(a: FDSParams, b: FDSParams):
    if a.fn.size != b.fn.size or not np.allclose(a.fn, b.fn) or a.Q != b.Q or a.b != b.b:
        raise ValueError("all situations must use the same damage model (fn grid, Q, b)")


def spec_target(spec, p: FDSParams, life: float, fs: float, *, seed=0, name="") -> Reference:
    """Synthesis target of a specification applied for `life` seconds."""
    if isinstance(spec, PSDSpec):
        W = response_variance_operator(p.fn, p.Q, fs, p.prefilter)
        D, M = fds_gaussian(spec.psd_on(p.fn), life, p, W=W)
        f = np.linspace(0, fs / 2, int(fs / 2 / 0.5) + 1)
        P = spec.psd_on(f)
        m2 = float(trapezoid(P, f))
        seg = spec.signal(fs, 60.0, seed=seed)
        st = {"rms": np.sqrt(m2), "std": np.sqrt(m2), "mean": 0.0, "skewness": 0.0, "kurtosis": 3.0,
              "crest_factor": float(np.max(np.abs(seg)) / np.sqrt(m2)), "peak": float(np.max(np.abs(seg)))}
        zv = np.maximum(W @ spec.psd_on(p.fn), 0.0)          # Gaussian: kurtosis 3 everywhere
        return Reference(None, fs, p, life, 1.0, D, M, f, P, st, segments=[(seg, fs, life / 60.0)],
                         kind="spec", name=name or spec.name, duration_s=life, moments=(m2, 3 * m2 * m2),
                         response_var=zv, response_kurtosis=np.full(p.fn.size, 3.0))
    if isinstance(spec, SineSweepSpec):
        one = spec.one_pass(fs)
        D1, M = spec.fds_mrs(p, fs)                     # over spec.duration
        D = D1 * life / spec.duration
        f, P = welch_psd(one, fs, resolution=0.5)
        d = one - one.mean()
        st = statistics(one)
        rs = response_spectra(one, fs, p)
        return Reference(None, fs, p, life, life / spec.pass_duration, D, M, f, P, st,
                         segments=[(one, fs, life / spec.pass_duration)], kind="spec",
                         name=name or spec.name, duration_s=life,
                         moments=(float(np.mean(d ** 2)), float(np.mean(d ** 4))),
                         response_var=rs["var"], response_kurtosis=rs["kurtosis"])
    raise TypeError("unknown specification type")


def _psd_common(children, how):
    df = min(c.psd_f[1] - c.psd_f[0] for c in children)
    fmax = max(c.psd_f[-1] for c in children)
    f = np.arange(0, fmax + df / 2, df)
    Ps = [np.interp(f, c.psd_f, c.psd, left=0.0, right=0.0) for c in children]
    if how == "max":
        return f, np.max(Ps, axis=0)
    w = np.array([c.life for c in children], float)
    return f, np.sum([wi * P for wi, P in zip(w, Ps)], axis=0) / w.sum()


def _stats_from(m2, m4, peak, crest):
    return {"rms": float(np.sqrt(m2)), "std": float(np.sqrt(m2)), "mean": 0.0, "skewness": float("nan"),
            "kurtosis": float(m4 / m2 ** 2), "crest_factor": crest, "peak": peak}


def combine(node) -> Reference:
    """Target of a mission tree (Situation / Group)."""
    if isinstance(node, Situation):
        return node.target
    kids = [combine(c) for c in node.children]
    if not kids:
        raise ValueError(f"group '{node.name}' is empty")
    if len(kids) == 1:
        return kids[0]
    for k in kids[1:]:
        _same_grid(kids[0].params, k.params)
    p = kids[0].params
    fs = max(k.fs for k in kids)
    M = np.max([k.mrs for k in kids], axis=0)
    peak = max(k.stats["peak"] for k in kids)
    if node.kind == "series":
        D = np.sum([k.fds for k in kids], axis=0)
        f, P = _psd_common(kids, "mean")
        life = float(sum(k.life for k in kids))
        w = np.array([k.life for k in kids]) / life
        m2 = float(sum(wi * k.moments[0] for wi, k in zip(w, kids)))
        m4 = float(sum(wi * k.moments[1] for wi, k in zip(w, kids)))
        segs = [s for k in kids for s in k.segments]
        st = _stats_from(m2, m4, peak, peak / np.sqrt(m2))
        notes = [f"series of {len(kids)}: " + ", ".join(k.name for k in kids)]
        zv = zk = None
        if all(k.response_var is not None and k.response_kurtosis is not None for k in kids):
            zv = np.sum([wi * k.response_var for wi, k in zip(w, kids)], axis=0)
            z4 = np.sum([wi * k.response_kurtosis * k.response_var ** 2 for wi, k in zip(w, kids)], axis=0)
            zk = z4 / np.maximum(zv ** 2, 1e-300)
    elif node.kind == "parallel":
        D = np.max([k.fds for k in kids], axis=0)
        f, P = _psd_common(kids, "max")
        dom = kids[int(np.argmax([np.sum(np.log(np.maximum(k.fds, 1e-300))) for k in kids]))]
        life = dom.life
        m2, m4 = dom.moments
        segs = dom.segments
        st = dict(dom.stats)
        notes = [f"parallel of {len(kids)} (envelope); dominant: {dom.name}"]
        zv, zk = dom.response_var, dom.response_kurtosis
    else:
        raise ValueError("group kind must be 'series' or 'parallel'")
    return Reference(None, fs, p, life, 1.0, D, M, f, P, st, notes=notes, segments=segs,
                     kind="mission", name=node.name or node.kind, duration_s=life, moments=(m2, m4),
                     response_var=zv, response_kurtosis=zk)


def describe(node, indent=0) -> str:
    pad = "  " * indent
    if isinstance(node, Situation):
        t = node.target
        return f"{pad}- {node.name}: life {t.life / 3600:.3g} h, rms {t.stats['rms']:.3g}, k {t.stats['kurtosis']:.2f}\n"
    s = f"{pad}{node.kind.upper()} {node.name}\n"
    return s + "".join(describe(c, indent + 1) for c in node.children)
