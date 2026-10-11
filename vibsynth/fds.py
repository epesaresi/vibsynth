"""Fatigue Damage Spectrum (FDS) and Maximum Response Spectrum (MRS).

* `fds_time`      exact time-domain FDS: Smallwood SDOF filter + rainflow
* `fds_fast`      fast approximate method (Appendix F of the dissertation), self-calibrated
* `fds_gaussian`  Lalanne closed form for a Gaussian PSD
* `response_variance_operator`  the linear map PSD -> rms^2 of the relative displacement

Damage is K**b / C * sum(n_i * amp_i**b) * repeats, amp = rainflow amplitude
of the relative displacement z (half range).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps
from scipy.special import gamma

from .rainflow import rainflow_damage_sum

__all__ = [
    "FDSParams", "prefilter_active", "apply_prefilter", "prefilter_power",
    "smallwood_coefficients", "sdof_relative_displacement",
    "fds_time", "fds_fast", "fds_gaussian", "response_variance_operator",
    "lalanne_matrix", "mrs_gaussian",
]

# Pre-filter (applied with filtfilt when fs <= 10*max(fn))
_PRE_B = np.array([0.8767, 1.814, 0.920])
_PRE_A = np.array([1.0, 1.6296, 0.8111, 0.0659])


@dataclass
class FDSParams:
    """Damage-model parameters shared by every FDS computation."""
    fn: np.ndarray                     # natural frequencies [Hz]
    Q: float = 10.0                    # quality factor, zeta = 1/(2Q)
    b: float = 8.0                     # Basquin / Wohler exponent
    C: float = 1.0
    K: float = 1.0
    prefilter: str | bool = "auto"     # "auto": on when fs <= 10*max(fn)
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.fn = np.asarray(self.fn, dtype=float)

    @property
    def zeta(self) -> float:
        return 1.0 / (2.0 * self.Q)


# ----------------------------------------------------------------- pre-filter
def prefilter_active(fs: float, fn: np.ndarray, prefilter: str | bool = "auto") -> bool:
    if isinstance(prefilter, str):
        return round(fs) <= 10.0 * float(np.max(fn))
    return bool(prefilter)


def apply_prefilter(x: np.ndarray, fs: float, fn: np.ndarray, prefilter: str | bool = "auto"):
    if not prefilter_active(fs, fn, prefilter):
        return np.asarray(x, float)
    # MATLAB filtfilt uses 3*(nfilt-1) reflected samples
    return sps.filtfilt(_PRE_B, _PRE_A, np.asarray(x, float), padtype="odd", padlen=9)


def prefilter_power(f: np.ndarray, fs: float) -> np.ndarray:
    """|H(f)|**4 : effect of the filtfilt pre-filter on a PSD."""
    _, h = sps.freqz(_PRE_B, _PRE_A, worN=np.asarray(f, float), fs=fs)
    return np.abs(h) ** 4


# ----------------------------------------------------------------------- SDOF
def smallwood_coefficients(fn: float, Q: float, fs: float):
    """Relative-displacement Smallwood ramp-invariant filter."""
    dt = 1.0 / fs
    damp = 1.0 / (2.0 * Q)
    om = 2.0 * np.pi * fn
    omd = om * np.sqrt(1.0 - damp ** 2)
    cosd, sind = np.cos(omd * dt), np.sin(omd * dt)
    e1 = np.exp(-damp * om * dt)
    e2 = np.exp(-2.0 * damp * om * dt)
    a1 = -2.0 * e1 * cosd
    a2 = e2
    b1 = (2 * damp * (e1 * cosd - 1) + e1 * (om / omd * (2 * damp ** 2 - 1) * sind) + om * dt) / (om ** 3 * dt)
    b2 = (-2 * om * dt * e1 * cosd + 2 * damp * (1 - e2)
          - 2 * om / omd * (2 * damp ** 2 - 1) * e1 * sind) / (om ** 3 * dt)
    b3 = ((2 * damp + om * dt) * e2 + e1 * (om / omd * (2 * damp ** 2 - 1) * sind - 2 * damp * cosd)) / (om ** 3 * dt)
    return np.array([b1, b2, b3]), np.array([1.0, a1, a2])


def sdof_relative_displacement(x: np.ndarray, fs: float, fn: float, Q: float) -> np.ndarray:
    bb, aa = smallwood_coefficients(fn, Q, fs)
    return sps.lfilter(bb, aa, x)


# ------------------------------------------------------------ time-domain FDS
def fds_time(x: np.ndarray, fs: float, p: FDSParams, repeats: float = 1.0,
             return_mrs: bool = True, progress=None):
    """Exact time-domain FDS (rainflow) and MRS of a base acceleration x."""
    y = apply_prefilter(x, fs, p.fn, p.prefilter)
    scale = p.K ** p.b / p.C * repeats
    D = np.empty(p.fn.size)
    M = np.empty(p.fn.size)

    def one(j):
        f = p.fn[j]
        z = sdof_relative_displacement(y, fs, f, p.Q)
        return j, scale * rainflow_damage_sum(z, p.b), np.max(np.abs(z)) * (2.0 * np.pi * f) ** 2

    workers = _workers(y.size)
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as ex:
            for k, (j, d, m) in enumerate(ex.map(one, range(p.fn.size))):
                D[j], M[j] = d, m
                if progress is not None:
                    progress((k + 1) / p.fn.size)
    else:
        for j in range(p.fn.size):
            _, D[j], M[j] = one(j)
            if progress is not None:
                progress((j + 1) / p.fn.size)
    return (D, M) if return_mrs else D


def _workers(n_samples):
    """Threads for the oscillator bank (NumPy/SciPy release the GIL in the heavy parts)."""
    import os
    if n_samples < 200_000:
        return 1
    return max(1, min(8, (os.cpu_count() or 2)))


def fds_fast(x: np.ndarray, fs: float, p: FDSParams, repeats: float = 1.0,
             calibration_seconds: float = 20.0, threshold_ratio: float | None = None):
    """Appendix F fast FDS: D ~ h(fn) * K^b/C * fn * dt * sum(z[z > p*zmax]**b).

    The original code used a fixed empirical correction vector h(fn) measured
    on one reference signal. Here h(fn) is calibrated on the first
    `calibration_seconds` of the signal itself against exact rainflow, which
    removes the dependence on that particular reference and sampling rate.
    """
    y = apply_prefilter(x, fs, p.fn, p.prefilter)
    pth = threshold_ratio if threshold_ratio is not None else 400.0 ** (-1.0 / p.b)
    dt = 1.0 / fs
    ncal = int(min(y.size, max(calibration_seconds * fs, 50 * fs / max(p.fn.min(), 1e-9))))
    scale = p.K ** p.b / p.C
    D = np.empty(p.fn.size)
    M = np.empty(p.fn.size)
    for j, f in enumerate(p.fn):
        z = sdof_relative_displacement(y, fs, f, p.Q)
        zc = z[:ncal]
        zmax_c = np.max(np.abs(zc))
        approx_c = f * dt * np.sum(zc[zc > pth * zmax_c] ** p.b)
        exact_c = rainflow_damage_sum(zc, p.b)
        h = exact_c / approx_c if approx_c > 0 else 1.0
        zmax = np.max(np.abs(z))
        D[j] = h * scale * f * dt * np.sum(z[z > pth * zmax] ** p.b) * repeats
        M[j] = zmax * (2.0 * np.pi * f) ** 2
    return D, M


# ----------------------------------------------------- frequency-domain (Lalanne)
def lalanne_matrix(fn: np.ndarray, Q: float) -> np.ndarray:
    """Lalanne's matrix a_mat (vectorised).

    For a PSD G that is piecewise linear between the nodes fn (and zero
    outside), the relative-displacement variance of the oscillator fn_i is
    rms_z^2 = (a_mat @ G)_i / (8*zeta*(2*pi*fn_i)^3).
    """
    fn = np.asarray(fn, float)
    z = 1.0 / (2.0 * Q)
    a = 2.0 * np.sqrt(1.0 - z ** 2)
    h = np.outer(1.0 / fn, fn)                       # h[i, j] = fn_j / fn_i
    T1 = np.log((h ** 2 + a * h + 1) / (h ** 2 - a * h + 1))
    T2 = np.arctan((2 * h + a) / (2 * z))
    T3 = np.arctan((2 * h - a) / (2 * z))
    I0 = z / (np.pi * a) * T1 + (T2 + T3) / np.pi
    I1 = 2.0 / (np.pi * a) * (T3 - T2)
    DI0 = np.diff(I0, axis=1)
    DI1 = np.diff(I1, axis=1)
    dh = np.diff(h, axis=1)
    N = fn.size
    A = np.zeros((N, N))
    left = (DI1 - h[:, :-1] * DI0) / dh              # segment terms
    right = (DI1 - h[:, 1:] * DI0) / dh
    A[:, 1:-1] = left[:, :-1] - right[:, 1:]
    A[:, 0] = (h[:, 1] * DI0[:, 0] - DI1[:, 0]) / dh[:, 0]
    A[:, -1] = left[:, -1]
    return A


def response_variance_operator(fn: np.ndarray, Q: float, fs: float | None = None,
                               prefilter: str | bool = False) -> np.ndarray:
    """W such that rms_z^2 = W @ G for a PSD G given at the nodes fn.

    When the pre-filter is active, its power response |H|^4 is
    included, so designs are consistent with the time-domain measurement.
    """
    fn = np.asarray(fn, float)
    z = 1.0 / (2.0 * Q)
    W = lalanne_matrix(fn, Q) / (8.0 * z * (2.0 * np.pi * fn[:, None]) ** 3)
    if fs is not None and prefilter_active(fs, fn, prefilter):
        W = W * prefilter_power(fn, fs)[None, :]
    return W


def fds_gaussian(G: np.ndarray, T: float, p: FDSParams, fs: float | None = None,
                 W: np.ndarray | None = None, repeats: float = 1.0):
    """Gaussian (frequency-domain) FDS and MRS for a PSD at the nodes p.fn."""
    if W is None:
        W = response_variance_operator(p.fn, p.Q, fs, p.prefilter if fs else False)
    zr2 = np.maximum(W @ np.asarray(G, float), 0.0)
    Teff = T * repeats
    D = p.K ** p.b / p.C * p.fn * Teff * (2.0 * zr2) ** (p.b / 2.0) * gamma(1.0 + p.b / 2.0)
    M = (2 * np.pi * p.fn) ** 2 * np.sqrt(zr2) * np.sqrt(2.0 * np.log(np.maximum(p.fn * Teff, 1.0001)))
    return D, M


def mrs_gaussian(G, T, p: FDSParams, fs=None):
    return fds_gaussian(G, T, p, fs)[1]
