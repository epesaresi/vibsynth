"""Compression benchmark of version 2.2: how the methods behave at exponents
other than the design exponent, and the compression limit of the record.

    python tools/compression_benchmark.py [record.mat] [--alpha 4] [--b 5] [--Q 10]

Writes docs/benchmark_compression.json and two figures in docs/theory/figures
(the Lorenz curve of the record and the damage ratio as a function of b).
The default record is examples/example_road.mat. A record that is not
distributed can be given on the command line; only the numbers are written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import vibsynth as vs  # noqa: E402
from vibsynth import edit as ED  # noqa: E402
from vibsynth.fds import FDSParams, response_spectra  # noqa: E402
from vibsynth.pipeline import run_method  # noqa: E402

B_EVAL = (3.0, 5.0, 7.0, 9.0)
METHODS = (("gaussian", {}), ("kfds", {}), ("mmvsd", {"modulation": "pattern", "lines": "off"}),
           ("edit (no gain)", {"gain": False}), ("edit", {}))


def ratios(x, fs, fn, Q, D_ref_by_b):
    out = {}
    for b, D_ref in D_ref_by_b.items():
        D = response_spectra(x, fs, FDSParams(fn=fn, Q=Q, b=b), stats=False)["fds"]
        r = D / D_ref
        out[f"{b:g}"] = {"gm": float(np.exp(np.mean(np.log(r)))), "spread": float(r.max() / r.min()),
                         "ratio": r.tolist()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("record", nargs="?", default=os.path.join(HERE, "..", "examples", "example_road.mat"))
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--b", type=float, default=5.0)
    ap.add_argument("--Q", type=float, default=10.0)
    ap.add_argument("--fmin", type=float, default=5.0)
    ap.add_argument("--fmax", type=float, default=200.0)
    ap.add_argument("--ppo", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "benchmark_compression.json"))
    ap.add_argument("--figures", default=os.path.join(HERE, "..", "docs", "theory", "figures"))
    ap.add_argument("--label", default=None)
    a = ap.parse_args(argv)

    x, fs = vs.load_signal(a.record)
    label = a.label or os.path.splitext(os.path.basename(a.record))[0]
    ref = vs.analyze_reference(x, fs, fmin=a.fmin, fmax=a.fmax, Q=a.Q, b=a.b, points_per_octave=a.ppo, name=label)
    fn = ref.params.fn
    T = ref.duration
    dur = T / a.alpha
    print(f"{label}: {T:.0f} s at {fs:g} Hz, kurtosis {ref.stats['kurtosis']:.2f}; test {dur:.0f} s (alpha {a.alpha:g}), "
          f"{fn.size} oscillators, Q {a.Q:g}, design b {a.b:g}")
    D_ref = {b: response_spectra(ref.x, fs, FDSParams(fn=fn, Q=a.Q, b=b), stats=False)["fds"] for b in B_EVAL}
    comp = ED.compressibility(ref.x, fs, ref.params)
    rows = {}
    signals = {}
    for name, kw in METHODS:
        method = name.split()[0]
        t0 = time.time()
        res = run_method(ref, method, dur, seed=a.seed, **kw)
        rows[name] = {"seconds": round(time.time() - t0, 1), "kurtosis": vs.kurtosis(res.x),
                      "warnings": res.warnings, **ratios(res.x, res.fs, fn, a.Q, D_ref)}
        if method == "edit":
            e = res.info["edit"]
            rows[name]["alpha_limit"] = e["alpha_limit"]
            rows[name]["gain"] = [float(np.min(e["gain_fn"])), float(np.max(e["gain_fn"]))]
        signals[name] = res.x
        print(f"  {name:15s}" + "".join(f"  b={b:g}: {rows[name][f'{b:g}']['gm']:.2f} [{rows[name][f'{b:g}']['spread']:.1f}]"
                                         for b in B_EVAL) + f"  ({rows[name]['seconds']} s)")
    out = {"record": label, "settings": {"alpha": a.alpha, "b_design": a.b, "Q": a.Q, "fmin": a.fmin,
                                         "fmax": a.fmax, "ppo": a.ppo, "seed": a.seed, "T": T, "fs": fs},
           "reference": {"kurtosis": ref.stats["kurtosis"], "rms": ref.stats["rms"]},
           "compressibility": {"window_s": comp["window_s"], "alpha_max": comp["alpha_max"],
                               "lognormal_alpha90": ED.lognormal_alpha_max(0.9, ref.stats["kurtosis"], a.b)},
           "b_eval": list(B_EVAL), "fn": fn.tolist(), "methods": rows}
    print("compression limit (worst fn keeps 80/90/95 %): " +
          " / ".join(f"{comp['alpha_max'][k]['worst']:.2f}" for k in ("80", "90", "95")))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("written", a.out)

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(a.figures, exist_ok=True)
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for row in comp["per_fn"]:
        ax.plot(comp["q"], row, color="0.82", lw=0.6)
    ax.plot(comp["q"], comp["mean"], "k-", lw=2, label="mean over the oscillators")
    ax.plot(comp["q"], comp["worst"], "-", color="#c0504d", lw=1.6, label="worst oscillator (sets the limit)")
    ax.plot(comp["q"], comp["lognormal"], "--", color="#1f77b4", lw=1, label="lognormal closed form, same kurtosis")
    ax.plot([0, 1], [0, 1], ":", color="0.5", lw=1, label="stationary Gaussian")
    for al in (2, 4, 8):
        ax.axvline(1 / al, color="0.6", lw=0.6, ls="--")
        ax.text(1 / al, 0.02, f" alpha = {al}", rotation=90, va="bottom", ha="left", fontsize=7.5, color="0.4")
    ax.axhline(0.9, color="0.6", lw=0.6, ls="--")
    ax.set_xlabel("fraction of the time kept, q = 1/alpha")
    ax.set_ylabel("share of the damage kept, L(q)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title(f"Damage Lorenz curve of {label} (b = {a.b:g}, Q = {a.Q:g}, window {comp['window_s']:g} s)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.figures, f"lorenz_{label}.png"), dpi=150)

    bs = np.linspace(2, 10, 17)
    picks = [fn[int(round(i))] for i in np.linspace(2, fn.size - 3, 2)]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    colors = {"gaussian": "#e58b2b", "kfds": "#2ca02c", "mmvsd": "#9467bd", "edit (no gain)": "#c0504d", "edit": "#1f77b4"}
    for ax, f in zip(axes, picks):
        for name, sig in signals.items():
            r = []
            for b in bs:
                p = FDSParams(fn=np.array([f]), Q=a.Q, b=float(b))
                r.append(response_spectra(sig, fs, p, stats=False)["fds"][0] /
                         response_spectra(ref.x, fs, p, stats=False)["fds"][0])
            ax.semilogy(bs, r, "-", lw=1.5, color=colors[name], label=name)
        ax.axhline(1, color="k", lw=0.8); ax.axvline(a.b, color="0.6", ls=":", lw=1)
        ax.set_ylim(0.1, 10); ax.set_xlabel("Basquin exponent b"); ax.set_title(f"fn = {f:.1f} Hz")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("damage of the test / damage of the record"); axes[0].legend(fontsize=7.5)
    fig.suptitle(f"Damage ratio as a function of the exponent, tests designed at b = {a.b:g} (alpha = {a.alpha:g})", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.figures, f"exponent_{label}.png"), dpi=150)
    print("figures written in", a.figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
