"""Figures and a plain-text summary for a synthesis result."""
from __future__ import annotations

import io

import numpy as np

from .analysis import Reference
from .fds import FDSParams, response_spectra
from .signals import window_rms

__all__ = ["summary_table", "make_figure", "response_kurtosis"]


def response_kurtosis(x, fs, fn, Q, prefilter="auto"):
    """Kurtosis of the SDOF relative-displacement response versus fn.

    Shows how much of the input non-Gaussianity reaches the component
    (Papoulis' rule, dissertation section 1.1)."""
    return response_spectra(x, fs, FDSParams(np.asarray(fn, float), Q=Q, prefilter=prefilter))["kurtosis"]


def summary_table(ref: Reference, res, ev: dict) -> list[tuple[str, str, str]]:
    s, r = ev["stats"], ref.stats
    rows = [
        ("Duration [s]", f"{ref.duration:.1f} (life {ref.life:.0f})", f"{res.duration:.1f} x {res.repeats}"),
        ("RMS", f"{r['rms']:.3g}", f"{s['rms']:.3g}"),
        ("Kurtosis", f"{r['kurtosis']:.2f}", f"{s['kurtosis']:.2f}"),
        ("Skewness", f"{r['skewness']:.2f}", f"{s['skewness']:.2f}"),
        ("Crest factor", f"{r['crest_factor']:.2f}", f"{s['crest_factor']:.2f}"),
        ("FDS error max / median" + (" (vs. reference over the same time)" if ev.get("target_scale", 1) != 1 else ""),
         "", f"{ev['fds_max_error_pct']:.1f} % / {ev['fds_median_error_pct']:.1f} %"),
        ("FDS nodes within 10 %", "", f"{100 * ev['fds_within_10pct']:.0f} %"),
        ("MRS ratio max", "", f"{np.max(ev['mrs_ratio']):.2f}"),
        ("PSD deviation in band (rms)", "", f"{ev['psd_rms_db_dev']:.1f} dB"),
    ]
    if ev.get("kz_median_abs_err") is not None:
        rows.append(("Response kurtosis k_z: median |error|", "", f"{ev['kz_median_abs_err']:.2f}"))
    return rows


def make_figure(ref: Reference, res, ev: dict, fmt="png", dpi=110):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fn = ref.params.fn
    kz_t, kz_r = ev.get("kz_target"), ev.get("k_response")
    with_kz = kz_r is not None and kz_t is not None and np.size(kz_t) == fn.size
    if with_kz:                                   # third row: response kurtosis and MRS ratio
        fig, ax = plt.subplots(3, 2, figsize=(12, 11))
    else:
        fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    c_ref, c_syn = "#1f4e79", "#c0504d"
    # time histories
    a = ax[0, 0]
    tr = np.arange(ref.x.size) / ref.fs
    ts = np.arange(res.x.size) / res.fs
    a.plot(tr, ref.x, color="0.65", lw=0.4, label="reference")
    a.plot(ts, res.x, color=c_syn, lw=0.4, alpha=0.9, label=f"{res.method} (x{res.repeats})" if res.repeats > 1 else res.method)
    a.set_xlabel("time [s]")
    a.set_ylabel("acceleration")
    a.set_title("Time histories")
    for ln in a.legend(loc="upper right", fontsize=8).get_lines():
        ln.set_linewidth(2.0)
    # PSD
    a = ax[0, 1]
    a.semilogy(ref.psd_f, ref.psd, color=c_ref, lw=1, label="reference")
    a.semilogy(ev["psd_f"], ev["psd"], color=c_syn, lw=1, label=res.method)
    a.set_xlim(0, min(ref.fs, res.fs) / 2)
    a.set_xlabel("frequency [Hz]")
    a.set_title("PSD")
    a.legend(fontsize=8)
    # FDS
    a = ax[1, 0]
    sc = ev.get("target_scale", 1.0)
    a.loglog(fn, ref.fds * sc, color=c_ref, lw=1.5,
             label="target (reference over life)" if sc == 1 else "reference over the same time")
    a.loglog(fn, ev["fds_ratio"] * ref.fds * sc, "o", ms=3, color=c_syn, label=f"{res.method} (rainflow)")
    a.set_xlabel("natural frequency [Hz]")
    a.set_title("Fatigue Damage Spectrum")
    a.legend(fontsize=8)
    b = a.twinx()
    b.semilogx(fn, 100 * (ev["fds_ratio"] - 1), color="0.5", lw=0.8)
    b.axhline(0, color="0.7", lw=0.5)
    b.set_ylabel("error [%]", color="0.4")
    lim = max(10.0, 1.1 * np.max(np.abs(100 * (ev["fds_ratio"] - 1))))
    b.set_ylim(-lim, lim)
    # level distribution over time (window RMS)
    a = ax[1, 1]
    nwr = max(int(ref.fs * 1.0), 2)
    nws = max(int(res.fs * 1.0), 2)
    rr = window_rms(ref.x, nwr)
    rs = window_rms(res.x, nws)
    bins = np.linspace(0, max(rr.max(), rs.max()) * 1.05, 40)
    a.hist(rr, bins=bins, color=c_ref, alpha=0.5, density=True, label="reference")
    a.hist(rs, bins=bins, color=c_syn, alpha=0.5, density=True, label=res.method)
    a.set_xlabel("1-s RMS")
    a.set_title("Distribution of the 1-second RMS (burst structure)")
    a.legend(fontsize=8)
    if with_kz:
        a = ax[2, 0]
        a.semilogx(fn, kz_t, color=c_ref, lw=1.5, label="target")
        a.semilogx(fn, kz_r, "o", ms=3, color=c_syn, label=res.method)
        a.axhline(3.0, color="0.7", lw=0.6, ls="--")
        a.set_xlabel("natural frequency [Hz]")
        a.set_title("Kurtosis of the oscillator responses k_z (3 = Gaussian)")
        a.legend(fontsize=8)
        a = ax[2, 1]
        a.semilogx(fn, ev["mrs_ratio"], "o-", ms=3, lw=0.8, color=c_syn)
        a.axhline(1.0, color="0.7", lw=0.6, ls="--")
        a.set_xlabel("natural frequency [Hz]")
        a.set_title("MRS ratio result / target (above 1: higher peak stresses)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi)
    plt.close(fig)
    return buf.getvalue()
