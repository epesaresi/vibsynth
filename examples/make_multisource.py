"""Synthetic reference whose non-Gaussianity depends on frequency.

Three independent sources, as in a vehicle with a road input, an engine and
local impacts:
  A  6-25 Hz    road band, level changes every 1-5 s (lognormal)     -> leptokurtic
  B  35-90 Hz   engine band, stationary Gaussian                     -> k = 3
  C  120-250 Hz impacts band, short bursts (0.2-0.8 s, ~8 % of time)  -> strongly leptokurtic

A frequency-independent envelope (kFDS, MLV) cannot reproduce the resulting
kurtosis spectrum of the SDOF responses; this record is the test case for it.

    python examples/make_multisource.py    # writes examples/multisource.mat
    python examples/make_multisource.py out.mat
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import vibsynth as vs  # noqa: E402

FS = 1000.0
DURATION = 300.0
SEED = 7


def band_psd(f, f1, f2, level):
    g = np.zeros_like(f)
    m = (f >= f1) & (f <= f2)
    g[m] = level
    return g


def piecewise_levels(n, fs, rng, dmin, dmax, sigma, smooth_s):
    e = np.empty(n)
    i = 0
    while i < n:
        d = int(rng.uniform(dmin, dmax) * fs)
        e[i:i + d] = np.exp(sigma * rng.normal())
        i += d
    w = max(int(smooth_s * fs), 1)
    e = np.convolve(np.pad(e, w, mode="wrap"), np.ones(w) / w, mode="same")[w:-w]
    return e / np.sqrt(np.mean(e ** 2))


def bursts(n, fs, rng, duty=0.08, dmin=0.2, dmax=0.8, floor=0.25, smooth_s=0.05):
    e = np.full(n, floor)
    covered = 0
    while covered < duty * n:
        d = int(rng.uniform(dmin, dmax) * fs)
        i = rng.integers(0, n - d)
        e[i:i + d] = rng.uniform(2.5, 4.0)
        covered += d
    w = max(int(smooth_s * fs), 1)
    e = np.convolve(np.pad(e, w, mode="wrap"), np.ones(w) / w, mode="same")[w:-w]
    return e / np.sqrt(np.mean(e ** 2))


def make(fs=FS, duration=DURATION, seed=SEED):
    rng = np.random.default_rng(seed)
    f = np.linspace(0, fs / 2, 40001)
    n = int(fs * duration)
    a = vs.psd_to_timeseries(f, band_psd(f, 6, 25, 1.0), fs, duration, rng=rng)[:n]
    b = vs.psd_to_timeseries(f, band_psd(f, 35, 90, 0.15), fs, duration, rng=rng)[:n]
    c = vs.psd_to_timeseries(f, band_psd(f, 120, 250, 0.05), fs, duration, rng=rng)[:n]
    x = (a * piecewise_levels(n, fs, rng, 1.0, 5.0, 0.6, 0.3)
         + b
         + c * bursts(n, fs, rng))
    return x, fs


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "multisource.mat")
    x, fs = make()
    vs.save_result(out, x, fs)
    s = vs.statistics(x)
    print(f"written {out}: {x.size / fs:.0f} s at {fs:.0f} Hz, RMS {s['rms']:.2f}, "
          f"kurtosis {s['kurtosis']:.2f}, crest factor {s['crest_factor']:.1f}")
