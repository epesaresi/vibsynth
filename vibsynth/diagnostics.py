"""Signal diagnostics: PDF, spectrogram, spectral kurtosis, kurtogram, SDOF response, shaker check."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy.special import gamma

from .fds import apply_prefilter, sdof_relative_displacement, FDSParams, fds_time, response_spectra

__all__ = ["spectrogram", "pdf", "running_stats", "sdof_response", "response_stats",
           "spectral_kurtosis", "kurtogram", "shaker_check", "match_pdf", "fds_sensitivity",
           "decimate_minmax"]


def decimate_minmax(t, x, n_out=4000):
    """Min/max envelope for plotting long records without losing peaks."""
    x = np.asarray(x, float)
    if x.size <= n_out:
        return np.asarray(t, float), x
    m = x.size // (n_out // 2)
    k = x.size // m
    xr = x[: k * m].reshape(k, m)
    tr = np.asarray(t, float)[: k * m].reshape(k, m)
    imin, imax = xr.argmin(1), xr.argmax(1)
    first = np.minimum(imin, imax)
    second = np.maximum(imin, imax)
    rows = np.arange(k)
    tt = np.column_stack([tr[rows, first], tr[rows, second]]).ravel()
    yy = np.column_stack([xr[rows, first], xr[rows, second]]).ravel()
    return tt, yy


def spectrogram(x, fs, nperseg=None, max_t=500, max_f=400):
    nperseg = nperseg or int(min(max(fs, 256), x.size // 8))
    f, t, S = sps.spectrogram(x, fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2,
                              scaling="density", mode="psd")
    S = 10 * np.log10(np.maximum(S, 1e-300))
    if t.size > max_t:
        k = int(np.ceil(t.size / max_t))
        S = S[:, : (t.size // k) * k].reshape(S.shape[0], -1, k).max(-1)
        t = t[: (t.size // k) * k].reshape(-1, k).mean(-1)
    if f.size > max_f:
        k = int(np.ceil(f.size / max_f))
        S = S[: (f.size // k) * k].reshape(-1, k, S.shape[1]).max(1)
        f = f[: (f.size // k) * k].reshape(-1, k).mean(-1)
    return f, t, S


def pdf(x, bins=121):
    """Empirical probability density and the Gaussian with the same mean/std."""
    x = np.asarray(x, float)
    mu, sd = x.mean(), x.std()
    lim = max(abs(x.min() - mu), abs(x.max() - mu))
    h, e = np.histogram(x, bins=bins, range=(mu - lim, mu + lim), density=True)
    c = 0.5 * (e[1:] + e[:-1])
    g = np.exp(-0.5 * ((c - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    return c, h, g


def running_stats(x, fs, window_s=1.0):
    n = max(int(round(window_s * fs)), 8)
    m = x.size // n
    xr = np.asarray(x[: m * n], float).reshape(m, n)
    d = xr - xr.mean(1, keepdims=True)
    m2 = np.mean(d ** 2, 1)
    rms = np.sqrt(np.mean(xr ** 2, 1))
    k = np.mean(d ** 4, 1) / np.maximum(m2 ** 2, 1e-300)
    t = (np.arange(m) + 0.5) * n / fs
    return t, rms, k


def sdof_response(x, fs, fn, Q, quantity="pseudo-acceleration", prefilter=False):
    """Response of one SDOF system: relative displacement or (2 pi fn)^2 * z."""
    y = apply_prefilter(x, fs, np.array([fn]), prefilter)
    z = sdof_relative_displacement(y, fs, fn, Q)
    return z if quantity.startswith("disp") else z * (2 * np.pi * fn) ** 2


def response_stats(x, fs, fn, Q, prefilter="auto", progress=None):
    """Kurtosis and std of the pseudo-acceleration response versus fn
    (kurtosis and variance of response versus natural frequency).
    One parallel pass of the oscillator bank (fds.response_spectra)."""
    fn = np.asarray(fn, float)
    r = response_spectra(x, fs, FDSParams(fn, Q=Q, prefilter=prefilter), progress=progress)
    return r["kurtosis"], np.sqrt(np.maximum(r["var"], 0.0)) * (2 * np.pi * fn) ** 2


def spectral_kurtosis(x, fs, nfft=None):
    """Welch estimate of the spectral kurtosis SK(f) = M4/M2^2 - 2."""
    nw = nfft or int(min(2 ** int(np.log2(max(fs / 2, 64))), x.size // 8))
    f, t, Z = sps.stft(x, fs, window="hann", nperseg=nw, noverlap=3 * nw // 4)
    A2 = np.abs(Z) ** 2
    M2 = A2.mean(1)
    M4 = (A2 ** 2).mean(1)
    return f, M4 / np.maximum(M2 ** 2, 1e-300) - 2


def kurtogram(x, fs, levels=None):
    """STFT-based kurtogram: spectral kurtosis versus frequency and window length."""
    levels = levels or [k for k in range(4, 13) if 2 ** k <= x.size // 8]
    f_common = np.linspace(0, fs / 2, 256)
    rows = []
    for k in levels:
        f, sk = spectral_kurtosis(x, fs, 2 ** k)
        rows.append(np.interp(f_common, f, sk))
    return f_common, np.array([2 ** k for k in levels]), np.array(rows)


def _integrate(x, fs, hp):
    """Velocity and displacement by frequency-domain integration with a high-pass."""
    n = x.size
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(n, 1 / fs)
    w = 2 * np.pi * f
    H = np.ones_like(f)
    if hp:
        H = 1 / np.sqrt(1 + (hp / np.maximum(f, 1e-12)) ** 8)   # 4th-order Butterworth magnitude
    H[0] = 0
    V = np.where(w > 0, X * H / (1j * np.maximum(w, 1e-12)), 0)
    D = np.where(w > 0, X * H / (-(np.maximum(w, 1e-12) ** 2)), 0)
    return np.fft.irfft(V, n), np.fft.irfft(D, n)


def shaker_check(x, fs, *, acc_limit=None, vel_limit=None, disp_limit=None, hp=None, default_hp=1.0):
    """Peak acceleration, velocity and displacement (stroke) against shaker limits.

    If the displacement exceeds the stroke, the lowest high-pass cut-off that
    satisfies it is suggested (this is how a stroke-limiting high-pass is
    usually chosen in practice). Without a given high-pass, the integration uses
    `default_hp` (1 Hz): below that a shaker table cannot follow anyway and
    the double integration would be dominated by the lowest FFT lines."""
    hp = hp or default_hp
    v, d = _integrate(np.asarray(x, float), fs, hp)
    out = {"peak_acc": float(np.max(np.abs(x))), "peak_vel": float(np.max(np.abs(v))),
           "peak_disp": float(np.max(np.abs(d))), "peak_to_peak_disp": float(d.max() - d.min()),
           "hp": hp}
    out["ok"] = {"acc": acc_limit is None or out["peak_acc"] <= acc_limit,
                 "vel": vel_limit is None or out["peak_vel"] <= vel_limit,
                 "disp": disp_limit is None or out["peak_to_peak_disp"] <= disp_limit}
    if disp_limit and not out["ok"]["disp"]:
        for fc in np.geomspace(max(hp or 0.1, 0.1), fs / 10, 60):
            _, d2 = _integrate(np.asarray(x, float), fs, fc)
            if d2.max() - d2.min() <= disp_limit:
                out["suggested_hp"] = float(fc)
                break
    return out


def match_pdf(x, ref):
    """Monotone amplitude mapping so that x has the PDF of ref ('adjust pdf').

    Changes the PSD and FDS slightly; re-measure afterwards."""
    x = np.asarray(x, float)
    o = np.argsort(x)
    q = (np.arange(x.size) + 0.5) / x.size
    rq = np.quantile(np.asarray(ref, float), q)
    y = np.empty_like(x)
    y[o] = rq
    return y


def fds_sensitivity(x_ref, fs_ref, x_syn, fs_syn, base: FDSParams, *, b_values=(), Q_values=(),
                    repeats_ref=1.0, repeats_syn=1.0, progress=None):
    """FDS ratio synthesised/reference for other b and Q (dissertation 4.5).

    A synthesis matched with one b is robust if the ratio stays near 1 when
    the true slope of the Wohler curve is different."""
    out = []
    combos = [(b, base.Q) for b in b_values] + [(base.b, q) for q in Q_values]
    for i, (b, Q) in enumerate(combos):
        p = FDSParams(fn=base.fn, Q=Q, b=b, C=base.C, K=base.K, prefilter=base.prefilter)
        Dr, _ = fds_time(x_ref, fs_ref, p, repeats=repeats_ref)
        Ds, _ = fds_time(x_syn, fs_syn, p, repeats=repeats_syn)
        out.append({"b": b, "Q": Q, "ratio": Ds / Dr})
        if progress:
            progress((i + 1) / len(combos))
    return out
