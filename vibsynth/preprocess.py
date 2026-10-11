"""Pre-processing of measured records."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal as sps

__all__ = ["crop", "bandpass", "detrend", "resample", "split", "scale"]


def crop(x, fs, t0=None, t1=None):
    """Keep the samples between t0 and t1 [s]."""
    i0 = 0 if t0 is None else max(int(round(t0 * fs)), 0)
    i1 = x.size if t1 is None else min(int(round(t1 * fs)), x.size)
    if i1 - i0 < 16:
        raise ValueError("the selected interval is too short")
    return np.asarray(x[i0:i1], float)


def bandpass(x, fs, f_low=None, f_high=None, order=4):
    """Zero-phase Butterworth high-, low- or band-pass filter.

    A high-pass is the usual way to respect the shaker stroke (a few Hz is
    typical); a low-pass removes content above the shaker band."""
    x = np.asarray(x, float)
    nyq = fs / 2
    if f_low and f_high:
        sos = sps.butter(order, [f_low / nyq, f_high / nyq], btype="band", output="sos")
    elif f_low:
        sos = sps.butter(order, f_low / nyq, btype="high", output="sos")
    elif f_high:
        sos = sps.butter(order, f_high / nyq, btype="low", output="sos")
    else:
        return x.copy()
    return sps.sosfiltfilt(sos, x)


def detrend(x, kind="constant"):
    return sps.detrend(np.asarray(x, float), type=kind)


def resample(x, fs, fs_new):
    """Polyphase resampling to fs_new (rational approximation of the ratio)."""
    fr = Fraction(fs_new / fs).limit_denominator(1000)
    y = sps.resample_poly(np.asarray(x, float), fr.numerator, fr.denominator)
    return y, fs * fr.numerator / fr.denominator


def split(x, fs, edges):
    """Split a record at the given times [s] (extraction of situations).
    Returns a list of pieces."""
    t = sorted(set([0.0] + [float(e) for e in edges if 0 < float(e) < x.size / fs] + [x.size / fs]))
    out = []
    for a, b in zip(t[:-1], t[1:]):
        i0, i1 = int(round(a * fs)), int(round(b * fs))
        if i1 - i0 >= 16:
            out.append(np.asarray(x[i0:i1], float))
    return out


def scale(x, factor):
    return np.asarray(x, float) * float(factor)
