"""Command line interface.

    python -m vibsynth analyze  record.mat --fmin 5 --fmax 200 --Q 20 --b 7
    python -m vibsynth synth    record.mat --method kfds --duration 30 --out test.mat --report test.png
    python -m vibsynth synth    record.mat --method mmvsd --duration 60 --modulation pattern --out test.mat
    python -m vibsynth synth    record.mat --method edit --duration 75 --out test.mat
    python -m vibsynth analyze  record.mat --compressibility
    python -m vibsynth gui
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import __version__
from .analysis import analyze_reference, evaluate, response_statistics
from .io import load_signal, save_result
from .pipeline import run_method
from .report import make_figure, summary_table
from .synthesis import METHODS


def _common(p):
    p.add_argument("input", help=".mat / .csv / .txt / .npy record ([time, acc] or acc + --fs)")
    p.add_argument("--fs", type=float, help="sampling frequency if the file has no time column")
    p.add_argument("--variable", help="MAT variable name")
    p.add_argument("--column", type=int, help="column of the acceleration")
    p.add_argument("--units", default="m/s^2", choices=["m/s^2", "g"])
    p.add_argument("--fmin", type=float, default=5.0)
    p.add_argument("--fmax", type=float, help="default fs/2.5")
    p.add_argument("--scale", default="octave", choices=["octave", "linear"])
    p.add_argument("--ppo", type=int, default=12, help="points per octave")
    p.add_argument("--resolution", type=float, default=1.0, help="linear scale step [Hz]")
    p.add_argument("--Q", type=float, default=10.0)
    p.add_argument("--b", type=float, default=8.0)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--K", type=float, default=1.0)
    p.add_argument("--life-h", type=float, help="represented life in hours (default: record length)")
    p.add_argument("--prefilter", default="auto", choices=["auto", "on", "off"])


def _reference(a):
    x, fs = load_signal(a.input, fs=a.fs, variable=a.variable, column=a.column, units=a.units)
    pre = {"auto": "auto", "on": True, "off": False}[a.prefilter]
    return analyze_reference(x, fs, fmin=a.fmin, fmax=a.fmax, scale=a.scale, points_per_octave=a.ppo,
                             resolution=a.resolution, Q=a.Q, b=a.b, C=a.C, K=a.K,
                             life=a.life_h * 3600 if a.life_h else None, prefilter=pre)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vibsynth", description=f"vibsynth {__version__}: "
                                 "vibration mission synthesis (kFDS, RF, PF, PSF, MMVSD, edit, kurtosis control)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("analyze", help="statistics, FDS, MRS and PSD of a record")
    _common(pa)
    pa.add_argument("--out", help="CSV with fn, FDS, MRS")
    pa.add_argument("--compressibility", action="store_true",
                    help="also the damage Lorenz curve and the compression limits (editing alone)")
    pa.add_argument("--window", type=float, help="window of the compressibility analysis [s]")
    pa.add_argument("--lorenz", help="CSV with the Lorenz curve (fraction of time kept, mean and worst share)")
    ps = sub.add_parser("synth", help="synthesise a test profile")
    _common(ps)
    ps.add_argument("--method", default="kfds", choices=list(METHODS))
    ps.add_argument("--duration", type=float, required=True, help="test duration [s]")
    ps.add_argument("--kurtosis", type=float, help="target kurtosis (default: reference)")
    ps.add_argument("--rms", type=float, help="target RMS for RF (default: reference)")
    ps.add_argument("--pattern", choices=["reference", "bursts"], help="envelope pattern for kFDS/RF/MLV")
    ps.add_argument("--window", type=float, help="burst / block duration [s]")
    ps.add_argument("--seed", type=int)
    ps.add_argument("--iterations", type=int, default=30)
    ps.add_argument("--tol", type=float, default=0.05)
    mm = ps.add_argument_group("MMVSD", "options of --method mmvsd")
    mm.add_argument("--modulation", default="random", choices=["random", "pattern"],
                    help="random field with the statistics of the record, or its own time-frequency pattern")
    mm.add_argument("--mapping", default="blocks", choices=["blocks", "compress"],
                    help="pattern of a longer record on a shorter test")
    mm.add_argument("--block", type=float, help="block length of the blocks mapping [s]")
    mm.add_argument("--lines", default="lines", choices=["lines", "random", "off"],
                    help="sine-on-random blocks as deterministic lines, in the random part, or not detected")
    mm.add_argument("--max-excess", type=float, default=30.0, help="cap of the modulation excess 3 Var(m)")
    mm.add_argument("--freq-corr", type=float, help="burst width along log2 f [octaves] (default: from the record)")
    mm.add_argument("--time-corr", type=float, help="burst duration [s] (default: from the record)")
    mm.add_argument("--anchor", type=float, default=0.0, help="pull of the PSD towards the design (e.g. 0.01)")
    ed = ps.add_argument_group("edit", "options of --method edit (the window is --window)")
    ed.add_argument("--ranking", default="common", choices=["common", "fair"],
                    help="window ranking: largest average retention, or equalised over the oscillators")
    ed.add_argument("--margin", type=float, default=2.0, help="ring-down margin [decay times of the slowest oscillator]")
    ed.add_argument("--fade", type=float, default=0.1, help="cross-fade between kept stretches [s]")
    ed.add_argument("--no-gain", action="store_true", help="editing only: no spectral gain beyond the compression limit")
    ed.add_argument("--gain-max", type=float, default=2.0, help="cap of the spectral gain")
    ed.add_argument("--retention", type=float, default=0.9, help="damage share that defines the compression limit")
    ps.add_argument("--compress-kurtosis", action="store_true",
                    help="kFDS/RF/MLV/VSD/PM: use the kurtosis of the record edited to the test duration as target")
    ps.add_argument("--out", required=True, help=".mat, .csv or .npy")
    ps.add_argument("--report", help="figure (.png/.pdf/.svg)")
    ps.add_argument("--json", help="metrics as JSON")
    sub.add_parser("gui", help="start the browser interface").add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)

    if a.cmd == "gui":
        from .gui import serve
        serve(port=a.port)
        return 0
    ref = _reference(a)
    for n in ref.notes:
        print("note:", n)
    print(f"reference: {ref.duration:.1f} s at {ref.fs:g} Hz, rms {ref.stats['rms']:.4g}, "
          f"kurtosis {ref.stats['kurtosis']:.2f}, crest {ref.stats['crest_factor']:.2f}, "
          f"{ref.params.fn.size} natural frequencies")
    if a.cmd == "analyze":
        if a.out:
            np.savetxt(a.out, np.c_[ref.params.fn, ref.fds, ref.mrs], delimiter=",",
                       header="fn_Hz,FDS,MRS", comments="")
            print("written", a.out)
        if a.compressibility or a.lorenz:
            from .edit import compressibility
            c = compressibility(ref.x, ref.fs, ref.params, window_s=a.window)
            print(f"compression limit by editing alone (window {c['window_s']:g} s): the worst oscillator keeps")
            for k in ("80", "90", "95"):
                print(f"  {k} % of the damage up to alpha {c['alpha_max'][k]['worst']:.2f}"
                      f"  (mean over the oscillators: {c['alpha_max'][k]['mean']:.2f})")
            if a.lorenz:
                np.savetxt(a.lorenz, np.c_[c["q"], c["mean"], c["worst"], c["lognormal"]], delimiter=",",
                           header="time_fraction_kept,damage_share_mean,damage_share_worst_fn,lognormal_closed_form",
                           comments="")
                print("written", a.lorenz)
        return 0
    mmkw = {}
    if a.method == "mmvsd":
        mmkw = dict(modulation=a.modulation, mapping=a.mapping, block_s=a.block, lines=a.lines,
                    max_excess=a.max_excess, freq_corr=a.freq_corr, time_corr=a.time_corr, cl_anchor=a.anchor)
    edkw = dict(ranking=a.ranking, margin=a.margin, fade_s=a.fade, gain=not a.no_gain, gain_max=a.gain_max,
                retention=a.retention, compress_kurtosis=a.compress_kurtosis)
    res = run_method(ref, a.method, a.duration, kurtosis=a.kurtosis, rms=a.rms, pattern=a.pattern,
                     seed=a.seed, iterations=a.iterations, tol=a.tol, window=a.window, log=print, **mmkw, **edkw)
    ed = res.info.get("edit")
    if ed:
        print(f"edit: alpha requested {ed['alpha_requested']:.2f}, realised {ed['alpha_realised']:.2f}; editing alone "
              f"keeps {ed['retention']:.0%} of the damage at every fn up to alpha {ed['alpha_limit']:.2f}; "
              f"{ed['n_segments']} stretches, spectral gain x{np.min(ed['gain_fn']):.2f}..x{np.max(ed['gain_fn']):.2f}")
    if res.info.get("k_response") is not None and ref.response_kurtosis is None:
        response_statistics(ref)                   # target k_z (one pass of the oscillator bank)
    ev = evaluate(res.x, res.fs, ref, repeats=res.repeats, fds=res.info.get("fds_measured"),
                  mrs=res.info.get("mrs_measured"),
                  target_scale=res.info.get("fds_target_scale", 1.0), k_response=res.info.get("k_response"))
    for name, rv, sv in summary_table(ref, res, ev):
        print(f"  {name:30s} {rv:>18s} {sv:>22s}")
    for w in res.warnings:
        print("WARNING:", w)
    save_result(a.out, res.x, res.fs, {"method": res.method, "repeats": res.repeats})
    print("written", a.out, f"(play {res.repeats} time(s))" if res.repeats > 1 else "")
    if a.report:
        with open(a.report, "wb") as f:
            f.write(make_figure(ref, res, ev, fmt=a.report.rsplit(".", 1)[-1]))
        print("written", a.report)
    if a.json:
        out = {"method": res.method, "warnings": res.warnings,
               **{k: v for k, v in ev.items() if isinstance(v, (int, float))},
               "stats": ev["stats"], "reference_stats": ref.stats}
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
