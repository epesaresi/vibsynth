"""Generate the synthetic example record used in the README and the screenshots.

The record imitates a road transport measurement: a vehicle response with a
few resonances, a road roughness that changes every few seconds (which makes
the signal non-Gaussian, kurtosis around 5 to 6) and some isolated shocks.
It is fully synthetic and reproducible (fixed seed), so it can be shared
freely. Run:

    python examples/make_example.py            # writes examples/example_road.mat
    python examples/make_example.py out.csv    # or any .mat / .csv / .npy path
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import vibsynth as vs  # noqa: E402

FS = 500.0          # Hz
DURATION = 300.0    # s
SEED = 2021


def vehicle_psd(f):
    """One-sided acceleration PSD [(m/s^2)^2/Hz]: three resonances on a sloped floor."""
    f = np.asarray(f, float)
    floor = 0.02 / (1 + (f / 80.0) ** 2)
    peaks = 0.0
    for fn, zeta, a in ((12.0, 0.08, 1.2), (45.0, 0.05, 0.6), (110.0, 0.04, 0.25)):
        r = f / fn
        peaks = peaks + a / ((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2) * (2 * zeta) ** 2
    G = floor + peaks
    G[(f < 3) | (f > 220)] = 0.0
    return G


def road_envelope(n, fs, rng):
    """Piecewise level changes (1 to 6 s long) with smooth 0.3 s transitions."""
    levels, t = [], 0.0
    while t < n / fs:
        dur = rng.uniform(1.0, 6.0)
        levels.append((t, dur, np.exp(0.38 * rng.normal())))
        t += dur
    e = np.empty(n)
    for t0, dur, lev in levels:
        i0, i1 = int(t0 * fs), min(n, int((t0 + dur) * fs))
        e[i0:i1] = lev
    w = int(0.3 * fs)
    e = np.convolve(np.pad(e, w, mode="edge"), np.ones(w) / w, mode="same")[w:-w]
    return e / np.sqrt(np.mean(e ** 2))


def shocks(n, fs, rng, count=8):
    """Isolated pothole-like decaying oscillations at the first resonance."""
    x = np.zeros(n)
    t = np.arange(int(1.5 * fs)) / fs
    for _ in range(count):
        i = rng.integers(0, n - t.size)
        f0 = rng.uniform(10.0, 14.0)
        x[i:i + t.size] += rng.uniform(8, 14) * np.exp(-2 * np.pi * 0.08 * f0 * t) * np.sin(2 * np.pi * f0 * t)
    return x


def make(fs=FS, duration=DURATION, seed=SEED):
    rng = np.random.default_rng(seed)
    f = np.linspace(0, fs / 2, 20001)
    g = vs.psd_to_timeseries(f, vehicle_psd(f), fs, duration, rng=rng)
    g *= 6.0 / g.std()                                  # about 6 m/s^2 RMS
    x = g * road_envelope(g.size, fs, rng) + shocks(g.size, fs, rng)
    return x, fs


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "example_road.mat")
    x, fs = make()
    vs.save_result(out, x, fs)
    s = vs.statistics(x)
    print(f"written {out}: {x.size / fs:.0f} s at {fs:.0f} Hz, "
          f"RMS {s['rms']:.2f} m/s^2, kurtosis {s['kurtosis']:.2f}, crest factor {s['crest_factor']:.1f}")
