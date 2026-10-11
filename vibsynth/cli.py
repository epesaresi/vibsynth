"""Command line interface.

    python -m vibsynth analyze  record.mat --fmin 5 --fmax 200 --Q 20 --b 7
    python -m vibsynth synth    record.mat --method kfds --duration 30 --out test.mat --report test.png
    python -m vibsynth gui
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import __version__
from .analysis import analyze_reference, evaluate
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
                                 "vibration mission synthesis (kFDS, RF, PF, PSF, kurtosis control)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("analyze", help="statistics, FDS, MRS and PSD of a record")
    _common(pa)
    pa.add_argument("--out", help="CSV with fn, FDS, MRS")
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
        return 0
    res = run_method(ref, a.method, a.duration, kurtosis=a.kurtosis, rms=a.rms, pattern=a.pattern,
                     seed=a.seed, iterations=a.iterations, tol=a.tol, window=a.window, log=print)
    ev = evaluate(res.x, res.fs, ref, repeats=res.repeats, fds=res.info.get("fds_measured"),
                  mrs=res.info.get("mrs_measured"),
                  target_scale=res.info.get("fds_target_scale", 1.0))
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
