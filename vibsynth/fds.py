"""Fatigue Damage Spectrum (FDS) and Maximum Response Spectrum (MRS).

* `fds_time`      exact time-domain FDS: Smallwood SDOF filter + rainflow
* `response_spectra`  the same pass, also giving the MRS and the variance and
                  kurtosis of every response (one call, parallel over oscillators)
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

from .rainflow import rainflow_damage_sum, turning_points

__all__ = [
    "FDSParams", "prefilter_active", "apply_prefilter", "prefilter_power", "smallwood_h2",
    "smallwood_coefficients", "sdof_relative_displacement",
    "fds_time", "response_spectra", "fds_fast", "fds_gaussian", "response_variance_operator",
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
    # zero-phase filtering, padded with 3*(nfilt-1) odd-reflected samples
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


def smallwood_h2(fn, Q, fs, f):
    """|H_n(f)|^2 of the Smallwood ramp-invariant relative-displacement filters
    (one row per node fn, one column per frequency f; vectorised).

    H_n(z) = (b1 + b2 z^-1 + b3 z^-2) / (1 + a1 z^-1 + a2 z^-2), z = exp(j 2 pi f/fs),
    with the coefficients of smallwood_coefficients. It is the exact frequency
    response of the time-domain FDS: close to the analogue
    1/((wn^2 - w^2)^2 + (2 zeta wn w)^2) for fn << fs, about 15 % lower in
    variance at fn = fs/6 and 30 % lower at fn = fs/4.4.
    """
    dt = 1.0 / fs
    z = 1.0 / (2.0 * Q)
    om = 2.0 * np.pi * np.asarray(fn, float).reshape(-1)[:, None]
    omd = om * np.sqrt(1.0 - z * z)
    cosd, sind = np.cos(omd * dt), np.sin(omd * dt)
    e1 = np.exp(-z * om * dt)
    e2 = e1 * e1
    a1, a2 = -2.0 * e1 * cosd, e2
    kk = om / omd * (2.0 * z * z - 1.0)
    den = om ** 3 * dt
    b1 = (2.0 * z * (e1 * cosd - 1.0) + e1 * kk * sind + om * dt) / den
    b2 = (-2.0 * om * dt * e1 * cosd + 2.0 * z * (1.0 - e2) - 2.0 * kk * e1 * sind) / den
    b3 = ((2.0 * z + om * dt) * e2 + e1 * (kk * sind - 2.0 * z * cosd)) / den
    w = 2.0 * np.pi * np.asarray(f, float).reshape(-1)[None, :] * dt
    c1, s1, c2, s2 = np.cos(w), np.sin(w), np.cos(2.0 * w), np.sin(2.0 * w)
    num = (b1 + b2 * c1 + b3 * c2) ** 2 + (b2 * s1 + b3 * s2) ** 2
    return num / ((1.0 + a1 * c1 + a2 * c2) ** 2 + (a1 * s1 + a2 * s2) ** 2)


def sdof_relative_displacement(x: np.ndarray, fs: float, fn: float, Q: float) -> np.ndarray:
    bb, aa = smallwood_coefficients(fn, Q, fs)
    return sps.lfilter(bb, aa, x)


# ------------------------------------------------------------ time-domain FDS
# One pass over the oscillator bank gives the rainflow FDS, the MRS and the
# statistics of every response (variance and kurtosis of the relative
# displacement). The rainflow damage is exactly that of
# rainflow_damage_sum(z, b): the turning points are the same numbers, only
# computed in cache-sized chunks. Long signals are spread over a persistent
# pool of worker PROCESSES that read the input from shared memory (numpy's
# many small calls hold the GIL, so threads do not scale); short ones use
# threads or run sequentially.

def _turning(z, chunk=1 << 16):
    """Turning points, identical to rainflow.turning_points (same values, same
    order), computed in chunks; signals with flat steps use turning_points."""
    n = z.size
    if n < 3:
        return turning_points(z)
    parts = [z[:1]]
    prev = None                                     # sign of the last slope of the previous chunk
    for i in range(0, n - 1, chunk):
        d = np.diff(z[i:min(i + chunk + 1, n)])
        if not d.all():
            return turning_points(z)
        s = d > 0
        if prev is not None and s[0] != prev:
            parts.append(z[i:i + 1])
        idx = np.flatnonzero(s[1:] != s[:-1]) + 1
        if idx.size:
            parts.append(z[i + idx])
        prev = s[-1]
    parts.append(z[-1:])
    return np.concatenate(parts)


def _moments(z, chunk=1 << 15):
    """Variance and Pearson kurtosis from raw power sums, in cache-sized chunks
    (no BLAS call: BLAS threads inside worker processes oversubscribe the CPU)."""
    n = z.size
    s1 = s2 = s3 = s4 = 0.0
    for i in range(0, n, chunk):
        zc = z[i:i + chunk]
        z2 = zc * zc
        s1 += float(zc.sum())
        s2 += float(z2.sum())
        s3 += float((z2 * zc).sum())
        s4 += float((z2 * z2).sum())
    m1, m2r, m3r, m4r = s1 / n, s2 / n, s3 / n, s4 / n
    m2 = m2r - m1 * m1
    m4 = m4r - 4 * m1 * m3r + 6 * m1 * m1 * m2r - 3 * m1 ** 4
    return m2, (m4 / m2 ** 2 if m2 > 0 else float("nan"))


def _one_node(y, fs, f, Q, b, scale, stats):
    z = sdof_relative_displacement(y, fs, f, Q)
    d = scale * rainflow_damage_sum(_turning(z), b, is_turning=True)
    m = float(np.max(np.abs(z))) * (2.0 * np.pi * f) ** 2
    v, k = _moments(z) if stats else (float("nan"), float("nan"))
    return d, m, v, k


def _nodes_shared(shm_name, size, fs, fns, Q, b, scale, stats):
    """Worker process: some oscillators of the input held in shared memory."""
    import os
    from multiprocessing import shared_memory
    try:
        shm = shared_memory.SharedMemory(name=shm_name, track=False)      # Python >= 3.13
    except TypeError:
        shm = shared_memory.SharedMemory(name=shm_name)
        if os.name == "posix":                      # the parent owns (and unlinks) the block
            try:
                from multiprocessing import resource_tracker
                resource_tracker.unregister(shm._name, "shared_memory")
            except Exception:                       # noqa: BLE001
                pass
    try:
        y = np.ndarray((size,), dtype=np.float64, buffer=shm.buf)
        out = [_one_node(y, fs, f, Q, b, scale, stats) for f in fns]
        del y
        return out
    finally:
        shm.close()


_POOL = {"ex": None, "n": 0}


def _watch_parent():
    """Worker initializer: exit when the parent process is gone (a killed
    interface or script must not leave workers behind)."""
    import multiprocessing
    import os
    import threading
    import time
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch():
        while True:
            time.sleep(1.0)
            if not parent.is_alive():
                os._exit(0)
    threading.Thread(target=watch, daemon=True).start()


def _process_pool(n):
    """Persistent pool of `n` worker processes (spawn: same behaviour on every OS)."""
    import multiprocessing
    import threading
    from concurrent.futures import ProcessPoolExecutor
    lock = _POOL.setdefault("lock", threading.Lock())
    with lock:
        if _POOL["ex"] is None or _POOL["n"] != n:
            if _POOL["ex"] is not None:
                _POOL["ex"].shutdown(wait=False, cancel_futures=True)
            _POOL["ex"] = ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"),
                                              initializer=_watch_parent)
            _POOL["n"] = n
        return _POOL["ex"]


def _n_workers(workers=None):
    """Worker count: argument, else the VIBSYNTH_WORKERS variable, else the CPU count."""
    import os
    if workers:
        return max(1, int(workers))
    env = os.environ.get("VIBSYNTH_WORKERS")
    if env:
        return max(1, int(env))
    return os.cpu_count() or 1


def response_spectra(x: np.ndarray, fs: float, p: FDSParams, repeats: float = 1.0, *,
                     stats: bool = True, progress=None, workers: int | None = None) -> dict:
    """Exact FDS, MRS and response statistics of a base acceleration, in one pass.

    Returns {"fds", "mrs", "var", "kurtosis"} on p.fn: the rainflow FDS
    (x repeats), the Maximum Response Spectrum (pseudo-acceleration), the
    variance and the Pearson kurtosis of the relative displacement of every
    oscillator (stats=False skips the last two).
    """
    y = np.ascontiguousarray(apply_prefilter(x, fs, p.fn, p.prefilter), dtype=np.float64)
    scale = p.K ** p.b / p.C * repeats
    fn = p.fn
    n = fn.size
    out = np.empty((n, 4))
    nw = _n_workers(workers)
    done = False
    if nw > 1 and y.size * n > 3e7:
        try:
            _response_processes(y, fs, p, scale, stats, min(nw, 16), out, progress)
            done = True
        except Exception as exc:                    # noqa: BLE001
            if type(exc).__name__ == "Cancelled":   # the interface cancelled the job
                raise
    if not done:
        def one(j):
            return _one_node(y, fs, fn[j], p.Q, p.b, scale, stats)
        if y.size >= 200_000 and nw > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(min(nw, 8)) as ex:
                for k, r in enumerate(ex.map(one, range(n))):
                    out[k] = r
                    if progress is not None:
                        progress((k + 1) / n)
        else:
            for k in range(n):
                out[k] = one(k)
                if progress is not None:
                    progress((k + 1) / n)
    return {"fds": out[:, 0].copy(), "mrs": out[:, 1].copy(), "var": out[:, 2].copy(),
            "kurtosis": out[:, 3].copy()}


def _response_processes(y, fs, p, scale, stats, nw, out, progress):
    from concurrent.futures import as_completed
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(create=True, size=y.nbytes)
    futs = {}
    try:
        np.ndarray(y.shape, dtype=np.float64, buffer=shm.buf)[:] = y
        pool = _process_pool(nw)
        # interleave the nodes (the cost grows with fn) and use a few tasks per worker
        ntask = min(p.fn.size, 3 * nw)
        groups = [np.arange(i, p.fn.size, ntask) for i in range(ntask)]
        futs = {pool.submit(_nodes_shared, shm.name, y.size, fs, p.fn[g].tolist(), p.Q, p.b, scale, stats): g
                for g in groups}
        n_done = 0
        for f in as_completed(futs):
            g = futs[f]
            out[g] = np.asarray(f.result(), float)
            n_done += g.size
            if progress is not None:
                progress(n_done / p.fn.size)
    finally:
        for f in futs:
            f.cancel()
        shm.close()
        shm.unlink()


def fds_time(x: np.ndarray, fs: float, p: FDSParams, repeats: float = 1.0,
             return_mrs: bool = True, progress=None):
    """Exact time-domain FDS (rainflow) and MRS of a base acceleration x."""
    r = response_spectra(x, fs, p, repeats, stats=False, progress=progress)
    return (r["fds"], r["mrs"]) if return_mrs else r["fds"]


def fds_fast(x: np.ndarray, fs: float, p: FDSParams, repeats: float = 1.0,
             calibration_seconds: float = 20.0, threshold_ratio: float | None = None):
    """Appendix F fast FDS: D ~ h(fn) * K^b/C * fn * dt * sum(z[z > p*zmax]**b).

    In the dissertation h(fn) is a fixed empirical correction vector measured
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
