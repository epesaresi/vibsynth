"""Synthetic vehicle record with sine-on-random blocks.

Random sources over the whole record:
  road band 5-22 Hz with level changes (lognormal)      -> leptokurtic
  broadband 25-300 Hz, stationary
  body resonance at 70 Hz (Q = 60), excited by random noise (narrow band, but NOT a line)
Engine lines (orders of the engine speed) only in some blocks:
    0- 45 s  idle      2nd order 26 Hz, 4th order 52 Hz (small speed wander)
   45- 60 s  run-up    26 -> 44 Hz (a sweep, not a steady line)
   60-150 s  cruise    44 Hz +- 3 % slow wander, 88 Hz
  150-200 s  coasting  no lines
  200-260 s  cruise    48 Hz +- 2 %, 96 Hz
  260-300 s  idle      26 Hz, 52 Hz

    python examples/make_sor.py            # writes examples/sor_vehicle.mat
    python examples/make_sor.py out.mat
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import vibsynth as vs  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_multisource import band_psd, piecewise_levels  # noqa: E402

FS = 1000.0
DURATION = 300.0
SEED = 11


def make(fs=FS, duration=DURATION, seed=SEED):
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs
    f = np.linspace(0, fs / 2, 40001)
    road = vs.psd_to_timeseries(f, band_psd(f, 5, 22, 1.0), fs, duration, rng=rng)[:n]
    broad = vs.psd_to_timeseries(f, band_psd(f, 25, 300, 0.02), fs, duration, rng=rng)[:n]
    w0, z = 2 * np.pi * 70.0, 1 / 120.0                     # Q = 60
    H2 = w0 ** 4 / ((w0 ** 2 - (2 * np.pi * f) ** 2) ** 2 + (2 * z * w0 * 2 * np.pi * f) ** 2)
    body = vs.psd_to_timeseries(f, 0.0008 * H2 * (f > 20), fs, duration, rng=rng)[:n]
    x = road * piecewise_levels(n, fs, rng, 1.0, 5.0, 0.6, 0.3) + broad + body

    # engine speed (2nd order frequency) and line amplitude, segment by segment
    f2 = np.zeros(n)
    amp = np.zeros(n)
    wander = np.cumsum(rng.normal(size=n)) / np.sqrt(fs)
    wander = np.convolve(wander, np.ones(int(fs)) / fs, "same")
    wander /= np.max(np.abs(wander))

    def seg(t0, t1, fa, fb, rel, a):
        m = (t >= t0) & (t < t1)
        f2[m] = np.linspace(fa, fb, m.sum()) * (1 + rel * wander[m])
        amp[m] = a

    seg(0, 45, 26, 26, 0.01, 1.5)
    seg(45, 60, 26, 44, 0.0, 1.8)
    seg(60, 150, 44, 44, 0.03, 2.0)
    seg(200, 260, 48, 48, 0.02, 2.2)
    seg(260, 300, 26, 26, 0.01, 1.5)
    k = int(0.3 * fs)                                     # 0.3 s fades at the transitions
    amp = np.convolve(np.r_[np.zeros(k), amp, np.zeros(k)], np.ones(k) / k, "same")[k:-k]
    ph = 2 * np.pi * np.cumsum(f2) / fs
    lines = amp * np.sin(ph) + 0.5 * amp * np.sin(2 * ph + 1.0)
    return x + lines, fs, lines


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "sor_vehicle.mat")
    x, fs, _ = make()
    vs.save_result(out, x, fs)
    s = vs.statistics(x)
    print(f"written {out}: {x.size / fs:.0f} s at {fs:.0f} Hz, RMS {s['rms']:.2f}, "
          f"kurtosis {s['kurtosis']:.2f}, crest factor {s['crest_factor']:.1f}")
