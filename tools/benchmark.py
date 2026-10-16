"""Benchmark of the FDS methods on one record, at three acceleration levels.

    python tools/benchmark.py [record.mat] [--fmin 5 --fmax 200 --Q 20 --b 7]
                              [--methods gaussian,kfds,rf,pf,psf,mmvsd] [--out results.json]

Without a record, the synthetic example (examples/example_road.mat) is used;
it is generated first if missing. Every row reports the maximum and median
FDS error (exact rainflow FDS of the profile against the target), the
kurtosis, RMS, crest factor, PSD deviation, MRS ratio and the median error of
the kurtosis of the oscillator responses k_z.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import vibsynth as vs  # noqa: E402
from vibsynth.pipeline import run_method  # noqa: E402

METHODS = ("gaussian", "kfds", "rf", "pf", "psf", "mmvsd")


def run(path, fmin, fmax, Q, b, seed=1, methods=METHODS):
    x, fs = vs.load_signal(path)
    T_ref = x.size / fs
    cases = ((1, T_ref, f"no acceleration ({T_ref:.0f} s)"),
             (10, 2 * T_ref, f"acceleration x5 ({10 * T_ref:.0f} s life in {2 * T_ref:.0f} s)"),
             (1, 30.0, f"acceleration x{T_ref / 30:.1f} ({T_ref:.0f} s life in 30 s)"))
    rows, ref = [], None
    for life_factor, T, label in cases:
        ref = vs.analyze_reference(x, fs, fmin=fmin, fmax=fmax, Q=Q, b=b, life=life_factor * T_ref)
        print("====", label)
        for name in methods:
            t0 = time.time()
            if name == "mmvsd":
                r = run_method(ref, "mmvsd", T, seed=seed)
            elif name == "gaussian":
                r = vs.gaussian(ref.fds, T, ref.fs, ref.params, seed=seed)
            elif name == "kfds":
                r = vs.kfds(ref.fds, T, ref.fs, ref.params, ref.stats["kurtosis"], reference=ref.x, ref_fs=ref.fs,
                            seed=seed)
            elif name == "rf":
                r = vs.rf(ref.fds, T, ref.fs, ref.params, ref.stats["rms"], reference=ref.x, ref_fs=ref.fs, seed=seed)
            elif name == "pf":
                r = vs.pf(ref.fds, T, ref.fs, ref.params, ref.psd_f, ref.psd, seed=seed)
            else:
                r = vs.psf(ref.fds, T, ref.fs, ref.params, ref.psd_f, ref.psd, ref.life, seed=seed)
            dt = time.time() - t0
            ev = vs.evaluate(r.x, r.fs, ref, repeats=r.repeats, fds=r.info["fds_measured"], mrs=r.info["mrs_measured"],
                             k_response=r.info.get("k_response"))
            s = ev["stats"]
            row = dict(case=label, method=r.method, seconds=round(dt, 1),
                       fds_max=round(ev["fds_max_error_pct"], 1), fds_median=round(ev["fds_median_error_pct"], 1),
                       kurtosis=round(s["kurtosis"], 2), rms=round(s["rms"], 2), crest=round(s["crest_factor"], 1),
                       psd_db=round(ev["psd_rms_db_dev"], 1), mrs_max_ratio=round(float(np.max(ev["mrs_ratio"])), 2),
                       kz_median=None if ev.get("kz_median_abs_err") is None else round(ev["kz_median_abs_err"], 2),
                       iterations=len(r.info["fds_error_history"]), warnings=r.warnings)
            rows.append(row)
            print(row)
    st = ref.stats
    return dict(record=os.path.basename(path), settings=dict(fmin=fmin, fmax=fmax, Q=Q, b=b),
                reference=dict(rms=st["rms"], kurtosis=st["kurtosis"], crest=st["crest_factor"], duration=T_ref),
                rows=rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", nargs="?", default=os.path.join(HERE, "..", "examples", "example_road.mat"))
    ap.add_argument("--fmin", type=float, default=5.0)
    ap.add_argument("--fmax", type=float, default=200.0)
    ap.add_argument("--Q", type=float, default=20.0)
    ap.add_argument("--b", type=float, default=7.0)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="benchmark.json")
    a = ap.parse_args()
    if not os.path.exists(a.record) and a.record.endswith("example_road.mat"):
        import runpy
        runpy.run_path(os.path.join(HERE, "..", "examples", "make_example.py"), run_name="__main__")
    res = run(a.record, a.fmin, a.fmax, a.Q, a.b, a.seed, tuple(m.strip() for m in a.methods.split(",") if m.strip()))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("written", a.out)
