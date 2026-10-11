"""Test specifications: PSD breakpoints (random) and swept sine.

Definition and signal generation, so a standard (MIL-STD-810, GAM EG 13, customer spec) can
be compared with a measured mission or used as a synthesis target.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fds import FDSParams, fds_gaussian, fds_time, response_variance_operator
from .signals import trapezoid, psd_to_timeseries

__all__ = ["PSDSpec", "SineSweepSpec", "loglog_interp"]


def loglog_interp(fb, vb, f):
    """Straight lines in log-log axes between breakpoints, zero outside."""
    fb = np.asarray(fb, float)
    vb = np.asarray(vb, float)
    f = np.asarray(f, float)
    out = np.zeros_like(f)
    m = (f >= fb[0]) & (f <= fb[-1]) & (f > 0)
    out[m] = np.exp(np.interp(np.log(f[m]), np.log(fb), np.log(np.maximum(vb, 1e-300))))
    return out


@dataclass
class PSDSpec:
    """Random vibration specification given by PSD breakpoints."""
    f: list
    psd: list                      # [(m/s^2)^2/Hz]
    name: str = "PSD specification"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        f = np.asarray(self.f, float)
        g = np.asarray(self.psd, float)
        if f.size < 2 or f.size != g.size:
            raise ValueError("frequency and PSD vectors must have the same length (>= 2)")
        if np.any(np.diff(f) <= 0) or np.any(f <= 0) or np.any(g <= 0):
            raise ValueError("frequencies must be increasing and positive, PSD values positive")

    def psd_on(self, f):
        return loglog_interp(self.f, self.psd, f)

    def rms(self):
        f = np.geomspace(self.f[0], self.f[-1], 4000)
        return float(np.sqrt(trapezoid(self.psd_on(f), f)))

    def fds_mrs(self, p: FDSParams, duration: float, fs: float | None = None):
        """Gaussian (Lalanne) FDS and MRS over `duration` seconds."""
        W = response_variance_operator(p.fn, p.Q, fs, p.prefilter if fs else False)
        return fds_gaussian(self.psd_on(p.fn), duration, p, W=W)

    def signal(self, fs, duration, seed=None):
        f = np.linspace(0, fs / 2, max(int(fs / 2 / 0.25), 64) + 1)
        return psd_to_timeseries(f, self.psd_on(f), fs, duration, rng=seed, floor=0.0)


@dataclass
class SineSweepSpec:
    """Swept sine: amplitude breakpoints (peak acceleration) between f1 and f2."""
    f: list                        # breakpoint frequencies [Hz] (first = f1, last = f2)
    amplitude: list                # peak acceleration [m/s^2] at the breakpoints
    rate: float = 1.0              # octave/min (log) or Hz/s (linear)
    mode: str = "log"
    sweeps: float = 1.0            # number of passes (up + down = 2)
    name: str = "Swept sine"
    meta: dict = field(default_factory=dict)

    @property
    def pass_duration(self) -> float:
        f1, f2 = self.f[0], self.f[-1]
        if self.mode == "log":
            return float(np.log2(f2 / f1) / self.rate * 60.0)
        return float((f2 - f1) / self.rate)

    @property
    def duration(self) -> float:
        return self.pass_duration * self.sweeps

    def one_pass(self, fs):
        f1, f2 = float(self.f[0]), float(self.f[-1])
        T = self.pass_duration
        t = np.arange(int(round(T * fs))) / fs
        if self.mode == "log":
            k = np.log(f2 / f1) / T
            finst = f1 * np.exp(k * t)
            phase = 2 * np.pi * f1 * (np.exp(k * t) - 1) / k
        else:
            k = (f2 - f1) / T
            finst = f1 + k * t
            phase = 2 * np.pi * (f1 * t + 0.5 * k * t * t)
        amp = loglog_interp(self.f, self.amplitude, finst) if len(self.f) > 2 else \
            np.exp(np.interp(np.log(finst), np.log(self.f), np.log(self.amplitude)))
        return amp * np.sin(phase)

    def signal(self, fs, max_seconds=None):
        """Up/down passes alternated (a down pass is the time-reversed up pass)."""
        one = self.one_pass(fs)
        n = int(np.ceil(self.sweeps))
        parts = [one if i % 2 == 0 else one[::-1] for i in range(n)]
        x = np.concatenate(parts)
        x = x[: int(round(self.duration * fs))]
        if max_seconds:
            x = x[: int(max_seconds * fs)]
        return x

    def fds_mrs(self, p: FDSParams, fs: float):
        """Time-domain FDS of one pass times the number of passes (exact rainflow)."""
        D, M = fds_time(self.one_pass(fs), fs, p, repeats=self.sweeps)
        return D, M
