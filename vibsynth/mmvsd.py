"""MMVSD: Moment-Matched Variable Spectral Density synthesis.

Accelerated test profile with a prescribed Fatigue Damage Spectrum D(fn) AND a
prescribed kurtosis spectrum of the oscillator responses k_z(fn).

The test signal is a conditionally Gaussian process with a random
evolutionary (time-varying) spectral density

        G(f, t) = P(f) m(f, t),        <m(f, .)>_t = 1,

so its time-averaged PSD is exactly P(f) (the condition of the VSD method,
here in continuous form). Given the modulation field m, the response z_n(t)
of the oscillator fn is a zero-mean Gaussian process whose variance s_n(t)
varies slowly, and both targets are explicit functionals of s_n:

        k_z(fn) = 3 <s_n^2> / <s_n>^2                  (response kurtosis)
        D(fn)   = K^b/C T < d(S_0,n, S_1,n, S_2,n, S_4,n) >   (Dirlik rate)

where S_k,n(t) = F_n[ sum_i Wk_ni P_i m_i(t) ] are the local spectral moments
of the response, Wk the moment operators (k = 0: Lalanne's operator) and F_n
the first-order envelope dynamics of a lightly damped oscillator, with time
constant tau_n = Q/(2 pi fn).

The model uses the exact frequency response of the sampled (Smallwood)
oscillators of the measurement and the loss of peaks between samples in the
rainflow count (sampling_factor), so that it describes what the time-domain
FDS measures. Design: m is a Gaussian copula field (burst width along log f
and burst duration per fn, estimated on the record), or the record's own
time-frequency pattern; its depth e(f) = 3 Var(m) and the PSD P (a smooth
correction of the record's PSD shape) are solved by Levenberg-Marquardt with
exact (adjoint) Jacobians. Synthesis: independent Gaussian frames (complex
normal spectral coefficients), filtered by sqrt(m), overlap-added with frame
lengths that grow towards low frequencies; then filtered by sqrt(P). A closed
loop on the realisation (exact rainflow FDS and measured k_z) corrects what
the model neglects. Sine-on-random blocks of the record are reproduced as
deterministic lines (lines.py) and the random part is designed for the rest.
The derivations are in docs/theory/vibsynth_theory.pdf.

`mmvsd()` works on arrays; `synthesize()` takes a vibsynth Reference (record,
specification or mission) and returns a SynthesisResult like every method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_fn, polygamma

from .design import target_variance, solve_variance
from .fds import (FDSParams, prefilter_active, prefilter_power, response_spectra, sdof_relative_displacement,
                  smallwood_h2)
from .lines import TestLines, ToneModel, find_tones, frame_mapping
from .signals import interp_psd, statistics, welch_psd

__all__ = ["MMVSDResult", "mmvsd", "synthesize", "measure", "estimate_structure", "ModulationField",
           "EnvelopeModel", "MultiResolution", "ReferenceField", "moment_operators", "dirlik_rate"]


def measure(x, fs, p: FDSParams, repeats: float = 1.0):
    """Rainflow FDS and response kurtosis spectrum of x (one oscillator pass)."""
    r = response_spectra(x, fs, p, repeats)
    return r["fds"], r["kurtosis"]


# ------------------------------------------------------- reference structure
def estimate_structure(x, fs, p: FDSParams, frame_s: float = 0.5):
    """Correlation widths of the reference's local band power (log scale).

    Band powers on 1/3-octave bands centred on the nodes, frames of `frame_s`
    (hop frame_s/2). The chi-square estimation noise of each band (variance
    psi'(nu/2) in natural log) is removed from the variances before the
    correlations are normalised. Returns (freq_corr_oct, time_corr_s) as the
    standard deviations of the Gaussian smoothing kernels that give the same
    1/e widths (kernel std s -> correlation exp(-lag^2 / (4 s^2))); the time
    correlation is returned per node (array on p.fn).
    """
    x = np.asarray(x, float)
    fn = p.fn
    h = max(int(round(frame_s * fs / 2)), 8)
    L = 2 * h
    nfr = (x.size - L) // h + 1
    w = np.hanning(L)
    f = np.fft.rfftfreq(L, 1 / fs)
    lo, hi = fn * 2 ** (-1 / 6), fn * 2 ** (1 / 6)
    i0, i1 = np.searchsorted(f, lo), np.searchsorted(f, hi)
    i1 = np.maximum(i1, i0 + 1)
    nlines = i1 - i0
    B = np.empty((fn.size, nfr))
    for k0 in range(0, nfr, 256):
        ks = np.arange(k0, min(nfr, k0 + 256))
        seg = x[ks[:, None] * h + np.arange(L)[None, :]] * w
        S = np.abs(np.fft.rfft(seg, axis=1)) ** 2
        c = np.cumsum(np.c_[np.zeros(len(ks)), S], axis=1)
        B[:, ks] = (c[:, i1] - c[:, i0]).T
    B = np.log(np.maximum(B, B.max() * 1e-14))
    nu = np.maximum(2.0 * nlines / 1.5, 2.0)              # Hann: correlated lines
    noise = polygamma(1, nu / 2)
    B = B - B.mean(axis=1, keepdims=True)
    var = B.var(axis=1)
    sv = np.maximum(var - noise, 0.1 * var)
    use = sv > 0.02                                       # bands that do fluctuate
    if use.sum() < 2:
        return 1.0, np.full(fn.size, 4 * frame_s)
    Bu, svu, lf = B[use], sv[use], np.log2(fn[use])
    # time: autocorrelation of every band (lags >= 1 frame), normalised by the
    # signal variance; 1/e width per band, smoothed along log f
    F = np.fft.rfft(Bu, n=2 * nfr, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, axis=1)[:, :nfr] / nfr / svu[:, None]
    ac[:, 0] = 1.0
    below = ac < np.exp(-1)
    lag = np.where(below.any(axis=1), np.argmax(below, axis=1), nfr - 1).astype(float)
    tc_band = 0.5 * np.maximum(lag, 1.0) * h / fs
    ltc = np.convolve(np.log(np.r_[tc_band[:1], tc_band, tc_band[-1:]]), np.ones(3) / 3, "valid")
    time_corr = np.exp(np.interp(np.log2(fn), lf, ltc))
    # frequency: correlation vs octave distance (off-diagonal only)
    C = (Bu @ Bu.T) / nfr / np.sqrt(np.outer(svu, svu))
    d = np.abs(lf[:, None] - lf[None, :])
    off = d > 1e-9
    bins = np.arange(1 / 12, d.max() + 1 / 6, 1 / 6)
    idx = np.digitize(d[off], bins)
    cm = np.array([C[off][idx == i].mean() if np.any(idx == i) else np.nan
                   for i in range(1, bins.size)])
    below = np.nonzero(cm < np.exp(-1))[0]
    wd = 0.5 * (bins[below[0]] + bins[below[0] + 1]) if below.size else d.max()
    return float(np.clip(0.5 * wd, 1 / 12, 3.0)), time_corr


# ------------------------------------------------------------ modulation field
class ModulationField:
    """Latent Gaussian field -> modulation m(node, frame), row means exactly 1.

    Stratified copula: every row uses the K quantiles at (r + 1/2)/K, assigned
    by the ranks of the latent field, so the sample marginal of a row depends
    on its parameter only (no sampling noise in <m^2>, <m^(b/2)>).
    """

    def __init__(self, fn, n_frames, frame_hop_s, freq_corr_oct, time_corr_s, rng,
                 marginal="gamma"):
        self.fn = np.asarray(fn, float)
        self.K = int(n_frames)
        self.marginal = marginal
        Z = rng.standard_normal((self.fn.size, self.K))
        lf = np.log2(self.fn)
        # circular Gaussian smoothing in time; the kernel std (frames) may depend on f
        st = np.broadcast_to(np.asarray(time_corr_s, float), self.fn.shape) / frame_hop_s
        k = np.fft.rfftfreq(self.K)
        Z = np.fft.irfft(np.fft.rfft(Z, axis=1) * np.exp(-2 * (np.pi * k[None, :] * st[:, None]) ** 2),
                         self.K, axis=1)
        Z /= np.maximum(Z.std(axis=1, keepdims=True), 1e-12)
        if freq_corr_oct > 0:                              # then correlate along log2(f)
            Z = np.exp(-0.5 * ((lf[:, None] - lf[None, :]) / freq_corr_oct) ** 2) @ Z
        Z = (Z - Z.mean(axis=1, keepdims=True)) / np.maximum(Z.std(axis=1, keepdims=True), 1e-12)
        self.ranks = np.argsort(np.argsort(Z, axis=1), axis=1)
        self.q = (self.ranks + 0.5) / self.K
        self.Z = Z

    _A = np.geomspace(0.01, 3000.0, 200)                  # Gamma shape grid of the quantile table

    def _gamma_table(self):
        """log of the stratified Gamma(a, 1/a) quantiles for a grid of shapes a.

        scipy's gamma.ppf costs ~1 us per value; tabulating it once for 200
        shapes and interpolating in log(a) makes field() about 100 times faster
        (the row normalisation to mean 1 absorbs the interpolation error)."""
        if getattr(self, "_tab", None) is None:
            u = (np.arange(self.K) + 0.5) / self.K
            A = self._A[:, None]
            q = stats.gamma.ppf(u[None, :], A, scale=1.0 / A)
            self._tab = np.log(np.maximum(q, 1e-30))
            self._lA = np.log(self._A)
        return self._tab

    def field(self, excess):
        """m for a per-row design excess e = 3 Var(m) (kurtosis of m-modulated noise = 3 + e)."""
        e = np.maximum(np.asarray(excess, float), 0.0)
        M = np.ones((self.fn.size, self.K))
        on = e > 1e-9
        if not np.any(on):
            return M
        if self.marginal == "gamma":
            tab = self._gamma_table()
            la = np.clip(np.log(3.0 / e[on]), self._lA[0], self._lA[-1])
            j = np.clip(np.searchsorted(self._lA, la) - 1, 0, self._lA.size - 2)
            w = ((la - self._lA[j]) / (self._lA[j + 1] - self._lA[j]))[:, None]
            srt = np.exp((1 - w) * tab[j] + w * tab[j + 1])          # sorted values per row
            M[on] = np.take_along_axis(srt, self.ranks[on], axis=1)
        else:                                              # lognormal, Var(m) = e/3
            s2 = np.log1p(e[on] / 3.0)
            if getattr(self, "_z", None) is None:
                self._z = stats.norm.ppf(self.q)
            M[on] = np.exp(np.sqrt(s2)[:, None] * self._z[on] - 0.5 * s2[:, None])
        M[on] /= M[on].mean(axis=1, keepdims=True)
        return M


# --------------------------------------------------------------- forward model
def moment_operators(fn, Q, fs=None, prefilter=False, edge_hold=True, pts=128, weight=None,
                     digital=True):
    """Spectral-moment operators of the relative-displacement response.

    Wk[j][n, i] = integral f^k |H_n(f)|^2 phi_i(f) df,  k = (0, 1, 2, 4),

    phi_i being the hat functions on the nodes (PSD piecewise linear between
    the nodes). With edge_hold the first and last basis functions are extended
    flat over 3/Q relative bandwidth beyond the grid (below fs/2), exactly as
    the synthesis holds the edge values. With fs given, |H_n|^2 is the response
    of the digital (Smallwood) filter of the measurement (digital=True) or the
    analogue one, times the vibsynth pre-filter when active. With the analogue
    response and without edge hold and weight, W0 is Lalanne's operator. With
    weight(f) (a PSD shape prior) the basis functions are weight * phi_i, i.e.
    the PSD is weight(f) times a piecewise linear correction.
    """
    fn = np.asarray(fn, float)
    n = fn.size
    zeta = 1.0 / (2.0 * Q)
    wn = 2 * np.pi * fn
    pre = fs is not None and prefilter_active(fs, fn, prefilter)
    segs = [(fn[i], fn[i + 1], i, i + 1) for i in range(n - 1)]
    if edge_hold:
        m = 3.0 / Q
        f_hi = fn[-1] * (1 + m) if fs is None else min(fn[-1] * (1 + m), 0.999 * fs / 2)
        segs = [(fn[0] * (1 - m), fn[0], 0, None)] + segs
        if f_hi > fn[-1]:
            segs.append((fn[-1], f_hi, None, n - 1))
    u = (np.arange(pts + 1)) / pts
    wts = np.full(pts + 1, 1.0 / pts)
    wts[[0, -1]] *= 0.5                                         # trapezoid
    out = np.zeros((4, n, n))
    for f0, f1, il, ir in segs:
        f = f0 + u * (f1 - f0)
        if fs is not None and digital:
            H2 = smallwood_h2(fn, Q, fs, f)
        else:
            w = 2 * np.pi * f
            H2 = 1.0 / ((wn[:, None] ** 2 - w[None, :] ** 2) ** 2 + (2 * zeta * wn[:, None] * w[None, :]) ** 2)
        if pre:
            H2 = H2 * prefilter_power(f, fs)[None, :]
        if weight is not None:
            H2 = H2 * weight(f)[None, :]
        df = (f1 - f0) * wts
        for j, k in enumerate((0, 1, 2, 4)):
            base = H2 * (f ** k)[None, :] * df[None, :]
            if il is not None and ir is not None:
                out[j][:, il] += base @ (1 - u)
                out[j][:, ir] += base @ u
            else:                                               # flat edge hold
                out[j][:, il if il is not None else ir] += base.sum(axis=1)
    return out


def sampling_factor(b, r):
    """Damage kept by rainflow counting on samples, for a cycle frequency r*fs.

    The extremes of a sampled narrow-band response are the samples nearest to
    the true peaks, at phase offsets t1, t2 uniform on (-pi r, pi r), so a
    range 2A is counted as A (cos t1 + cos t2) and the damage (range^b) is
    multiplied by c_b(r) = E[((cos t1 + cos t2)/2)^b]: 0.98 at r = 0.04, 0.86
    at r = 0.11, 0.54 at r = 0.23 for b = 8. Tabulated on r in [0, 0.5]
    (Gauss-Legendre, 24 x 24 points) and interpolated.
    """
    tab = _SAMPLING_TABLES.get(float(b))
    if tab is None:
        x, wq = np.polynomial.legendre.leggauss(24)
        rr = np.linspace(0.0, 0.5, 201)
        t = 0.5 * (x + 1.0)                                  # (0, 1), weights wq/2
        c = np.cos(np.pi * rr[:, None] * t[None, :])         # cos(pi r t)
        avg = 0.5 * (c[:, :, None] + c[:, None, :])
        tab = np.einsum("rij,i,j->r", np.power(avg, b), 0.5 * wq, 0.5 * wq)
        tab[0] = 1.0
        _SAMPLING_TABLES[float(b)] = tab = (rr, tab)
    return np.interp(np.clip(r, 0.0, 0.5), tab[0], tab[1])


_SAMPLING_TABLES = {}


def _pow(x, e):
    """x**e with repeated squaring when 2e is an integer (b = 4 ... 12): about 5
    times faster than np.power on large arrays, same result to rounding."""
    if float(e).is_integer() and 0 < e <= 32:
        e = int(e)
        out, base = None, x
        while e:
            if e & 1:
                out = base.copy() if out is None else out * base
            e >>= 1
            if e:
                base = base * base
        return out
    if float(2 * e).is_integer() and 0 < e <= 32:
        return _pow(x, e - 0.5) * np.sqrt(x)
    return np.power(x, e)


def dirlik_rate(m0, m1, m2, m4, b, block=32768):
    """Dirlik expected sum of amplitude^b per second (amplitude = range/2).

    Narrow-band limit: sqrt(m2/m0) (2 m0)^(b/2) Gamma(1 + b/2) (Rice).
    Large arrays are processed in cache-sized blocks (about 2.4 times faster).
    """
    m0, m1, m2, m4 = (np.asarray(v, float) for v in (m0, m1, m2, m4))
    if m0.size > block and m0.shape == m1.shape == m2.shape == m4.shape:
        out = np.empty(m0.size)
        f = [v.reshape(-1) for v in (m0, m1, m2, m4)]
        for i in range(0, m0.size, block):
            sl = slice(i, i + block)
            out[sl] = _dirlik(f[0][sl], f[1][sl], f[2][sl], f[3][sl], b)
        return out.reshape(m0.shape)
    return _dirlik(m0, m1, m2, m4, b)


def _dirlik(m0, m1, m2, m4, b):
    m0 = np.maximum(m0, 1e-300)
    m2 = np.maximum(m2, 1e-300)
    m4 = np.maximum(m4, 1e-300)
    r = np.sqrt(m2 / m4)                                   # 1 / nu_p
    xm = (m1 / m0) * r
    g = np.clip(m2 / np.sqrt(m0 * m4), 1e-6, 1 - 1e-9)
    g2 = g * g
    D1 = 2.0 * (xm - g2) / (1.0 + g2)
    D1s = D1 * D1
    den = 1.0 - g - D1 + D1s
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    R = np.clip((g - xm - D1s) / den, -1 + 1e-9, 1 - 1e-9)
    D2 = den / (1.0 - R)
    D3 = 1.0 - D1 - D2
    Qd = np.maximum(1.25 * (g - D3 - D2 * R) / np.where(np.abs(D1) < 1e-12, 1e-12, D1), 1e-12)
    ea = (D1 * _pow(Qd, b) * gamma_fn(1 + b)
          + 2.0 ** (b / 2) * gamma_fn(1 + b / 2) * (D2 * _pow(np.abs(R), b) + D3))
    return _pow(m0, b / 2) / r * np.maximum(ea, 0.0)


def _rfft(X):
    from scipy import fft as sfft
    return sfft.rfft(X, axis=-1, workers=-1)


def _irfft(X, n):
    from scipy import fft as sfft
    return sfft.irfft(X, n, axis=-1, workers=-1)


class EnvelopeModel:
    """Conditional-Gaussian forward model with exact (adjoint) Jacobians.

    Local spectral moments of the response of oscillator n:
        S_k,n(t) = F_n[ sum_i Wk_ni P_i m_i(t) ],   k = 0, 1, 2, 4,
    F_n = circular first-order low-pass, tau_n = Q/(2 pi fn) (envelope law).
    Response kurtosis  k_n = 3 <S_0^2> / <S_0>^2.
    Damage  D_n = K^b/C T < d(S_0, S_1, S_2, S_4) >  with d = Dirlik rate
    (damage="dirlik") or the Rice narrow-band rate fn (2 S_0)^(b/2) Gamma(1+b/2)
    (damage="rice"), times the sampling factor c_b(nu_0/fs) of the rainflow
    count on samples (nu_0 = sqrt(S_2/S_0), see sampling_factor) when fs is given.
    """

    def __init__(self, Wk, p: FDSParams, hop_s, T, K, sub=4, damage="dirlik", fs=None):
        self.Wk, self.W = Wk, Wk[0]
        self.p, self.T, self.K, self.sub, self.damage = p, T, K, sub, damage
        self.fs = fs                       # sampled measurement: peaks between samples
        self.c_fn = np.ones(p.fn.size) if fs is None else sampling_factor(p.b, p.fn / fs)
        self.Nt = K * sub
        u = (np.arange(sub) + 0.5) / sub
        self.c2 = np.cos(0.5 * np.pi * u) ** 2
        a = np.exp(-(hop_s / sub) * 2 * np.pi * p.fn / p.Q)[:, None]
        z = np.exp(-2j * np.pi * np.arange(self.Nt // 2 + 1) / self.Nt)[None, :]
        self.H = (1 - a) / (1 - a * z)
        self.q = p.b / 2.0
        self.scale = p.K ** p.b / p.C * T
        self.rice = p.fn * gamma_fn(1 + self.q) * 2.0 ** self.q   # times S0^q
        self.Uf, self.U2 = None, None

    def set_lines(self, u):
        """Deterministic lines (sine on random) in the KURTOSIS model: u =
        response variance of every line, (J, n, Nt) on the model grid.
        Conditionally on m the response is Gaussian (variance s) plus sines
        (variances u_j, U = sum u_j), so E z^2 = s + U and
        E z^4 = 3 (s + U)^2 - 1.5 sum u_j^2 (exact). The damage of a line on
        a random response is NOT additive (the line rides on the random peaks)
        and has no reliable closed form for broad-band responses, so the damage
        model stays that of the random part and the interaction is measured."""
        if u is None or len(u) == 0:
            self.Uf, self.U2 = None, None
            return
        uf = np.maximum(self.F(np.asarray(u, float)), 0.0)
        self.Uf = uf.sum(axis=0)
        self.U2 = np.mean(np.sum(uf * uf, axis=0), axis=1)

    def up(self, M):
        Mn = np.roll(M, -1, axis=1)
        return (M[:, :, None] * self.c2 + Mn[:, :, None] * (1 - self.c2)).reshape(M.shape[0], -1)

    def F(self, X):
        return _irfft(_rfft(X) * self.H, self.Nt)

    def FT(self, X):                                       # adjoint (time-reversed kernel)
        return _irfft(_rfft(X) * np.conj(self.H), self.Nt)

    def _rate(self, S):
        if self.damage == "rice":
            d = (self.rice * self.c_fn)[:, None] * S[0] ** self.q
        else:
            d = dirlik_rate(S[0], S[1], S[2], S[3], self.p.b)
            if self.fs is not None:            # cycles at the mean crossing rate sqrt(S2/S0)
                d = d * sampling_factor(self.p.b, np.sqrt(S[2] / S[0]) / self.fs)
        return d

    def run(self, P, Mup, damage=True):
        X = P[:, None] * Mup
        if damage and self.damage == "dirlik":
            S = np.maximum(self.F(np.matmul(self.Wk, X)), 1e-300)
        else:
            S = np.maximum(self.F(self.W @ X), 1e-300)[None]
        s = S[0]
        B = s.mean(axis=1)
        if self.Uf is None:
            st, Bt = s, B
            A = np.mean(s * s, axis=1)
            Ae = A
        else:
            st = s + self.Uf
            Bt = st.mean(axis=1)
            A = np.mean(st * st, axis=1)
            Ae = A - 0.5 * self.U2
        r = {"S": S, "s": s, "st": st, "B": B, "Bt": Bt, "A": A, "Ae": Ae, "k": 3.0 * Ae / Bt ** 2}
        if damage:
            d = self._rate(S)
            r["d"] = d
            r["D"] = self.scale * d.mean(axis=1)
            DG = self.scale * self.rice * self.c_fn * B ** self.q  # Gaussian (Rice) FDS of <s>
            r["c"] = r["D"] / DG
        return r

    def _rate_grad(self, r):
        """d rate / d S_k for k = 0,1,2,4 (one-sided differences, per n and t)."""
        S = r["S"]
        if self.damage == "rice":
            return [self.q * self.rice[:, None] * S[0] ** (self.q - 1.0)]
        d0 = r["d"]
        g = []
        for k in range(4):
            h = 1e-5 * S[k]
            Sp = S.copy()
            Sp[k] += h
            g.append((self._rate(Sp) - d0) / h)
        return g

    def _jac_D(self, r, Y):
        """d D_n / d theta_i, where d(P_i m_i(t))/d theta_i = Y_i(t)."""
        g = self._rate_grad(r)
        J = 0.0
        Ws = self.Wk if self.damage == "dirlik" else self.Wk[:1]
        for k, gk in enumerate(g):
            J = J + Ws[k] * ((self.FT(gk) @ Y.T) / self.Nt)
        return self.scale * J

    def jac_logP(self, P, Mup, r):
        """d log D_n / d log P_i."""
        return self._jac_D(r, P[:, None] * Mup) / r["D"][:, None]

    def jac_logD_mod(self, P, dMup, r):
        """d log D_n / d (row parameter of m)."""
        return self._jac_D(r, P[:, None] * dMup) / r["D"][:, None]

    def jac_k(self, P, dMup, r):
        """d k_n / d (row parameter) given the row derivatives dMup (row means 0)."""
        G = self.FT(r["st"])
        dA = 2.0 * self.W * P[None, :] * ((G @ dMup.T) / self.Nt)
        return 3.0 * dA / r["Bt"][:, None] ** 2

    def jac_k_logP(self, P, Mup, r):
        """d k_n / d log P_i."""
        G = self.FT(r["st"])
        dA = 2.0 * self.W * ((G @ Mup.T) / self.Nt)
        dB = self.W * Mup.mean(axis=1)[None, :]
        A, B = r["Ae"][:, None], r["Bt"][:, None]
        return 3.0 * (dA / B ** 2 - 2.0 * A * dB / B ** 3) * P[None, :]


def _second_diff(n):
    L = np.zeros((max(n - 2, 1), n))
    for i in range(n - 2):
        L[i, i:i + 3] = (1.0, -2.0, 1.0)
    return L


def _lm(y0, resid_jac, lo=-np.inf, hi=np.inf, smooth=1e-3, iters=40, tol=1e-3, y_ref=None):
    """Levenberg-Marquardt with a curvature penalty on y - y_ref (along the nodes)."""
    L = _second_diff(y0.size)
    LtL = smooth * L.T @ L
    yr = np.zeros(y0.size) if y_ref is None else np.asarray(y_ref, float)
    y = np.clip(y0.copy(), lo, hi)
    r, J = resid_jac(y)
    cost = float(r @ r + (y - yr) @ LtL @ (y - yr))
    lam = 1e-2
    for _ in range(iters):
        if np.max(np.abs(r)) < tol:
            break
        A = J.T @ J + LtL
        g = J.T @ r + LtL @ (y - yr)
        dA = np.diag(A).copy()
        improved = False
        for _ in range(12):
            step = -np.linalg.solve(A + lam * (np.diag(dA) + 1e-9 * np.eye(y.size)), g)
            yn = np.clip(y + step, lo, hi)
            rn, Jn = resid_jac(yn)
            cn = float(rn @ rn + (yn - yr) @ LtL @ (yn - yr))
            if cn < cost:
                y, r, J, cost = yn, rn, Jn, cn
                lam = max(lam / 3, 1e-6)
                improved = True
                break
            lam *= 4
        if not improved:
            break
    return y, r


# ------------------------------------------------------------------ synthesis
def _log_interp_matrix(fn, f):
    """Linear-in-log2(f) interpolation weights nodes -> lines (edges held)."""
    lf = np.log2(fn)
    x = np.log2(np.maximum(f, fn[0] * 1e-3))
    j = np.clip(np.searchsorted(lf, x) - 1, 0, fn.size - 2)
    t = np.clip((x - lf[j]) / (lf[j + 1] - lf[j]), 0.0, 1.0)
    A = np.zeros((f.size, fn.size))
    A[np.arange(f.size), j] = 1 - t
    A[np.arange(f.size), j + 1] += t
    return A


def frame_phases(rng, K, h, kind="gaussian"):
    """Spectral coefficients of K independent frames of length 2h (complex64,
    one row per frame, h + 1 lines), E|c|^2 = 1: complex normal ("gaussian",
    Gaussian frames) or unit modulus with random phase ("multisine")."""
    c = np.empty((K, h + 1), np.complex64)
    if kind == "multisine":
        ph = rng.uniform(-np.pi, np.pi, (K, h + 1)).astype(np.float32)
        c.real, c.imag = np.cos(ph), np.sin(ph)
        return c
    c.real = rng.standard_normal((K, h + 1), dtype=np.float32)
    c.imag = rng.standard_normal((K, h + 1), dtype=np.float32)
    c *= np.float32(math.sqrt(0.5))
    return c


def _modulate(phases, M, fn, h, fs, line_weight=None, chunk=256):
    """Incoherent overlap-add of independently modulated frames.

    Frame k (length 2h, hop h) is an independent periodic Gaussian noise with
    a white spectrum (complex normal coefficients, one-sided PSD 1), filtered
    by sqrt(m_k(f)) and weighted by the sqrt-Hann window w, with
    sum_k w_k(t)^2 = 1. Given m, the signal is a sum of Gaussian frames, hence
    Gaussian at every instant, and so are the oscillator responses, as the
    model assumes, however few lines a frame puts in an oscillator band (a
    flat-amplitude multisine frame would make the response of a lightly damped
    oscillator a sum of a few sines, lighter-tailed than Gaussian, with about
    half the expected damage). The local variance spectrum is
    sum_k w_k(t)^2 m_k(f) (the model's crossfade), the time average of the PSD
    is <m> = 1 per line, and m = 1 gives stationary white Gaussian noise.
    (Re-using one noise for overlapping frames would add the frames
    coherently, (sum w_k^2 sqrt(m_k))^2 <= sum w_k^2 m_k, and bias both the
    PSD and the damage of strongly modulated bands downwards.)
    `phases` (K x h+1 coefficients) may be an int seed for a one-off realisation.
    """
    K = M.shape[1]
    L = 2 * h
    if not isinstance(phases, np.ndarray):
        phases = frame_phases(np.random.default_rng(phases), K, h)
    w = np.sin(np.pi * (np.arange(L) + 0.5) / L)          # w^2 shifted by h sums to 1
    fl = np.fft.rfftfreq(L, 1 / fs)
    A = _log_interp_matrix(fn, fl)
    lw = np.ones(fl.size) if line_weight is None else line_weight(fl)
    amp = np.sqrt(L * fs / 2.0) * lw
    amp[0] = 0.0
    amp[-1] = 0.0
    V = np.zeros((K, h))
    for k0 in range(0, K, chunk):
        ks = np.arange(k0, min(K, k0 + chunk))
        F = phases[ks].astype(complex) * (np.sqrt(np.maximum(A @ M[:, ks], 0.0)) * amp[:, None]).T
        y = np.fft.irfft(F, L, axis=1) * w
        np.add.at(V, (ks - 1) % K, y[:, :h])         # frame k peaks at t = k h (model grid)
        V[ks] += y[:, h:]
    return V.ravel()


class MultiResolution:
    """Frame lengths that grow towards low frequencies (constant-Q like).

    A frame of length L resolves the modulation only on a scale 1/L in
    frequency, while the hop must stay below the correlation time of the
    modulation. Node n needs hop_n = max(T_c(n)/3, c/(2 fn)) (frame of at least
    c cycles); the nodes are grouped into hops h0*2^j (lower frequencies never
    get shorter frames) and the groups are joined by power-complementary
    crossovers in log f, with independent noises, so the local spectrum is

        sum_j  X_j(f)  sum_k  w_jk(t)^2  m_jk(f),    sum_j X_j(f) = 1,

    where m_jk is the fine-grid field averaged over the 2^j fine frames of k.
    The model uses exactly this expression (up_eff).
    """

    def __init__(self, fn, tc, fs, T, sub=4, min_cycles=8.0, overlap_oct=1 / 6, max_groups=6,
                 frame_s=None):
        fn = np.asarray(fn, float)
        need = np.maximum(np.asarray(tc, float) / 3.0, min_cycles / (2.0 * fn))
        # frames must not get longer towards high f: a node gets at most the hop of
        # every lower node (cumulative minimum from low to high f). A long burst
        # duration at high f (allowed hop) then shortens nothing and lengthens
        # nothing below it; the cycle constraint is kept, since it decreases with f
        need = np.minimum.accumulate(need)
        if frame_s is not None:
            need = np.full(fn.size, frame_s / 2.0)
        h0 = max(int(round(need.min() * fs)), 16)
        h0 += h0 % 2
        # group (hop h0 2^g) of every node: at least min_cycles per frame (hard),
        # otherwise the longest hop not above the time resolution it needs;
        # never longer towards high f (cumulative minimum over the lower nodes)
        cyc = min_cycles / (2.0 * fn)
        g_lo = np.ceil(np.log2(np.maximum(cyc * fs / h0, 1.0)) - 1e-9).astype(int)
        g_hi = np.floor(np.log2(np.maximum(need * fs / h0, 1.0)) + 1e-9).astype(int)
        g = np.minimum.accumulate(np.maximum(g_lo, g_hi))
        g = np.clip(g, 0, max_groups - 1)
        gmax = int(g.max())
        while gmax > 0 and T * fs / (h0 * 2 ** gmax) < 16:      # at least 16 coarse frames
            gmax -= 1
        g = np.minimum(g, gmax)
        self.fn, self.fs, self.sub, self.h0 = fn, fs, sub, h0
        R = 2 ** gmax
        self.K = max(int(math.ceil(T * fs / (h0 * R))), 1) * R
        self.N = self.K * h0
        self.hop = h0 / fs
        self.levels = sorted(set(g.tolist()), reverse=True)    # coarse (low f) first
        # boundaries between consecutive groups (geometric mean of the edge nodes)
        self.bounds = []
        for a_, b_ in zip(self.levels[:-1], self.levels[1:]):
            i_hi = np.nonzero(g == a_)[0].max()
            i_lo = np.nonzero(g == b_)[0].min()
            self.bounds.append(float(np.sqrt(fn[i_hi] * fn[i_lo])))
        self.overlap = overlap_oct
        self.node_w = self.weights(fn)
        self._rows = [np.flatnonzero(w > 0) for w in self.node_w]
        self.group_of_node = g

    def weights(self, f):
        """Power weights X_j(f) of the groups (telescoping, sum = 1)."""
        f = np.maximum(np.asarray(f, float), 1e-6)
        out = np.zeros((len(self.levels), f.size))
        carry = np.ones(f.size)
        for j, fb in enumerate(self.bounds):
            t = np.clip((np.log2(f / fb)) / self.overlap + 0.5, 0.0, 1.0)
            s_up = np.sin(0.5 * np.pi * t) ** 2                # share of the finer groups
            out[j] = carry * (1.0 - s_up)
            carry = carry * s_up
        out[-1] = carry
        return out

    def _group_field(self, M, lev):
        r = 2 ** lev
        return M.reshape(M.shape[0], -1, r).mean(axis=2) if r > 1 else M

    def _up(self, Mg, sub):
        u = (np.arange(sub) + 0.5) / sub
        c2 = np.cos(0.5 * np.pi * u) ** 2
        Mn = np.roll(Mg, -1, axis=1)
        dM = (Mn - Mg)[:, :, None]                    # Mg c2 + Mn (1 - c2) = Mn - dM c2
        return (Mn[:, :, None] - dM * c2).reshape(Mg.shape[0], -1)

    def up_eff(self, M):
        out = np.zeros((M.shape[0], self.K * self.sub))
        for j, lev in enumerate(self.levels):
            rows = self._rows[j]                       # only the nodes this group feeds
            if rows.size == 0:
                continue
            Mg = self._group_field(M[rows], lev)
            out[rows] += self.node_w[j][rows, None] * self._up(Mg, self.sub * 2 ** lev)
        return out

    def init_phases(self, rng, kind="gaussian"):
        """Frozen random coefficients of every frame of every group."""
        self.kind = kind
        self.phases = [frame_phases(rng, self.K // 2 ** lev, self.h0 * 2 ** lev, kind) for lev in self.levels]

    def synthesize(self, M):
        v = np.zeros(self.N)
        for j, lev in enumerate(self.levels):
            h = self.h0 * 2 ** lev
            lw = lambda fl, j=j: np.sqrt(self.weights(fl)[j])
            v += _modulate(self.phases[j], self._group_field(M, lev), self.fn, h, self.fs, lw)
        return v

    def dominant_frames(self, d_row, node, count=2):
        """Group and frames that carry most of the model damage of one node.

        d_row is the model damage rate of the node on the fine time grid.
        Returns (group index, frame indices incl. the overlapping neighbours).
        """
        j = int(np.argmax(self.node_w[:, node]))
        r = self.sub * 2 ** self.levels[j]
        per_frame = d_row.reshape(-1, r).sum(axis=1)
        top = np.argsort(per_frame)[::-1][:count]
        Kj = self.K // 2 ** self.levels[j]
        frames = np.unique(np.r_[top, (top - 1) % Kj, (top + 1) % Kj])
        return j, frames

    def realised_frames(self, dens, node, count=2):
        """Group and frames that carry most of the damage of one node in the
        realised signal. `dens` is the damage density of its response per
        sample (|z|^b); block k of group j (samples k h .. (k+1) h) is made by
        frames k and k+1, and the response lags the input, so the frames
        k-1, k, k+1 of the `count` most damaging blocks are returned."""
        j = int(np.argmax(self.node_w[:, node]))
        h = self.h0 * 2 ** self.levels[j]
        Kj = self.K // 2 ** self.levels[j]
        per_frame = np.asarray(dens, float)[: Kj * h].reshape(Kj, h).sum(axis=1)
        top = np.argsort(per_frame)[::-1][:count]
        frames = np.unique(np.r_[top, (top - 1) % Kj, (top + 1) % Kj])
        return j, frames

    def reroll(self, j, frames, rng, f_centre=None, width_oct=0.25):
        """Draw new coefficients for some frames of group j, only on the lines
        within +-width_oct of f_centre (all lines if None). Returns the old rows."""
        old = self.phases[j][frames].copy()
        new = old.copy()
        h = self.h0 * 2 ** self.levels[j]
        fl = np.fft.rfftfreq(2 * h, 1 / self.fs)
        band = np.ones(fl.size, bool) if f_centre is None else \
            np.abs(np.log2(np.maximum(fl, 1e-9) / f_centre)) <= width_oct
        nb = int(band.sum())
        if nb:
            new[:, band] = frame_phases(rng, frames.size, nb - 1, getattr(self, "kind", "gaussian"))
        self.phases[j][frames] = new
        return old

    def describe(self):
        parts = []
        for j, lev in enumerate(self.levels):
            idx = np.nonzero(self.group_of_node == lev)[0]
            if idx.size:
                parts.append(f"{self.fn[idx[0]]:.3g}-{self.fn[idx[-1]]:.3g} Hz: frame "
                             f"{2 * self.h0 * 2 ** lev / self.fs:.3g} s")
        return "; ".join(parts)


def band_power_pattern(x, fs, mr, smooth=True):
    """log band power of a signal at every node, on the fine frame grid of `mr`.

    Each node is analysed with the frame length of its synthesis group (so the
    frequency resolution is the one the synthesis can reproduce), Hann window,
    hop = group hop, band = 1/12 octave around fn (at least 2 lines). The
    values are interpolated to t_k = k * hop0 (k = 0 .. len(x)/hop0 - 1).
    smooth: Gaussian smoothing over one group hop in time and 3 nodes in
    frequency (removes most of the chi-square scatter of single frames).
    """
    x = np.asarray(x, float)
    fn = mr.fn
    h0 = mr.h0
    K = x.size // h0
    tk = np.arange(K) * h0
    out = np.zeros((fn.size, K))
    owner = np.argmax(mr.node_w, axis=0)                 # group that synthesises each node
    for j, lev in enumerate(mr.levels):
        rows = np.flatnonzero(owner == j)
        if rows.size == 0:
            continue
        hg = h0 * 2 ** lev
        L = 2 * hg
        if x.size < L + hg:
            out[rows] = np.log(np.mean(x * x) + 1e-300)
            continue
        nfr = (x.size - L) // hg + 1
        w = np.hanning(L)
        f = np.fft.rfftfreq(L, 1 / fs)
        half = np.maximum(fn[rows] * (2 ** (1 / 24) - 1), 1.0 * fs / L)
        i0 = np.searchsorted(f, fn[rows] - half)
        i1 = np.maximum(np.searchsorted(f, fn[rows] + half), i0 + 1)
        B = np.empty((rows.size, nfr))
        for k0 in range(0, nfr, 128):
            ks = np.arange(k0, min(nfr, k0 + 128))
            seg = x[ks[:, None] * hg + np.arange(L)[None, :]] * w
            S = np.abs(np.fft.rfft(seg, axis=1)) ** 2
            c = np.concatenate([np.zeros((ks.size, 1)), np.cumsum(S, axis=1)], axis=1)
            B[:, ks] = (c[:, i1] - c[:, i0]).T
        lB = np.log(np.maximum(B, B.max() * 1e-12 + 1e-300))
        if smooth and nfr > 3:
            k = np.exp(-0.5 * np.arange(-3, 4) ** 2)
            k /= k.sum()
            lB = np.apply_along_axis(lambda v: np.convolve(np.r_[v[2::-1], v, v[:-4:-1]], k, "valid"), 1, lB)
        centres = np.arange(nfr) * hg + L / 2
        for r, row in enumerate(rows):
            out[row] = np.interp(tk, centres, lB[r])
    if smooth and fn.size >= 3:
        out = np.vstack([out[:1], (out[:-2] + 2 * out[1:-1] + out[2:]) / 4, out[-1:]]) \
            if fn.size > 2 else out
    return out


class ReferenceField:
    """Modulation field that keeps the time-frequency pattern of the reference.

    m_i(t) = B_i(t)^g_i / <B_i^g_i>, B_i = band power of the reference at node
    i on the test frame grid. The rank structure (the copula: WHEN and in
    WHICH band the bursts happen) is that of the reference; the exponent g_i
    sets the modulation depth, so the design can still meet the kurtosis and
    FDS targets (g = 1: the reference's own depth).

    Mapping of the reference time axis onto a shorter test:
      "blocks"   (default) blocks of `block_s` seconds taken evenly across the
                 whole record and concatenated: every situation of the record is
                 represented and burst durations are kept;
      "compress" the whole record squeezed into the test (power averaged over
                 each test frame): event order kept, bursts shortened;
    a test as long as the record uses it frame by frame; a longer one loops it.
    """

    _G = np.r_[0.0, np.geomspace(0.02, 4.0, 60)]

    def __init__(self, x_ref, fs, mr, K, mapping="blocks", block_s=None):
        self.fn, self.K, self.mapping = mr.fn, int(K), mapping
        lB = band_power_pattern(x_ref, fs, mr)              # reference fine grid
        Kr = lB.shape[1]
        hop = mr.hop
        idx, self.mapping = frame_mapping(Kr, self.K, mapping, round((block_s or 4.0) / hop))
        if self.mapping == "compress":
            edges = np.linspace(0, Kr, self.K + 1)
            c = np.concatenate([np.zeros((lB.shape[0], 1)), np.cumsum(np.exp(lB), axis=1)], axis=1)
            lo = np.floor(edges[:-1]).astype(int)
            hi = np.maximum(np.floor(edges[1:]).astype(int), lo + 1)
            lP = np.log((c[:, hi] - c[:, lo]) / (hi - lo))
        else:
            lP = lB[:, idx]
        self.ref_index = idx                                 # reference fine frame of every test frame
        self.lB = lP - lP.mean(axis=1, keepdims=True)
        # excess e(g) = 3 Var(m_g) for every row and exponent of the grid
        E = np.empty((self.fn.size, self._G.size))
        for k, g in enumerate(self._G):
            m = np.exp(g * self.lB)
            m /= m.mean(axis=1, keepdims=True)
            E[:, k] = 3.0 * m.var(axis=1)
        self._E = np.maximum.accumulate(E, axis=1)          # monotone in g
        self.e_ref = E[:, np.argmin(np.abs(self._G - 1.0))]  # depth of the record itself

    def gamma(self, excess):
        e = np.maximum(np.asarray(excess, float), 0.0)
        return np.array([np.interp(ei, Er, self._G) for ei, Er in zip(e, self._E)])

    def field(self, excess):
        g = self.gamma(excess)
        M = np.exp(g[:, None] * self.lB)
        return M / M.mean(axis=1, keepdims=True)


def _pattern_correlation(x, fs, mr, ref_pattern, ref_index):
    """Median over the nodes of the correlation between the log band power of
    the profile and the (mapped) reference pattern: 1 = same time-frequency
    pattern, 0 = unrelated."""
    lX = band_power_pattern(x, fs, mr)
    K = min(lX.shape[1], ref_pattern.shape[1])
    a, b = lX[:, :K], ref_pattern[:, :K]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    r = (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1) + 1e-300)
    return r


def _psd_lines(fn, P, grid, Q, prior=None):
    """PSD on the FFT lines: piecewise linear P between the nodes (edges held
    over 3/Q, as vibsynth), times the full-resolution prior shape if given."""
    m = 3.0 / Q
    fe = np.r_[fn[0] * (1 - m), fn, min(fn[-1] * (1 + m), grid[-1] * 0.999)]
    ge = np.r_[P[0], P, P[-1]]
    G = interp_psd(fe, ge, grid, floor=0.0)
    if prior is not None:
        G = G * prior(grid)
    G[0] = 0.0
    G[-1] = 0.0
    return G


@dataclass
class MMVSDResult:
    x: np.ndarray
    fs: float
    psd_nodes: np.ndarray                 # P at fn (time-averaged PSD)
    modulation: np.ndarray                # m (nodes x frames)
    frame_hop_s: float
    excess_design: np.ndarray
    model: dict
    measured: dict
    history: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    lines: dict | None = None             # deterministic lines (sine-on-random blocks)

    @property
    def duration(self):
        return self.x.size / self.fs


def _prior_function(psd_prior, fn):
    """Callable PSD shape from (f, G) or from values on the nodes (log-linear)."""
    if isinstance(psd_prior, tuple):
        f, G = (np.asarray(a, float) for a in psd_prior)
    else:
        f, G = np.asarray(fn, float), np.asarray(psd_prior, float)
    G = np.maximum(G, G.max() * 1e-9)
    lf, lg = f.copy(), np.log(G)

    def prior(x):
        return np.exp(np.interp(np.asarray(x, float), lf, lg))
    return prior


def _smooth3(v):
    return np.convolve(np.r_[v[0], v, v[-1]], np.ones(3) / 3, mode="valid")


def mmvsd(D_target, k_target, T, fs, p: FDSParams, *, frame_s=None, freq_corr_oct=1 / 3,
          time_corr_s=None, marginal="gamma", max_excess=30.0, seed=None, psd_prior=None,
          damage="dirlik", iterations=40, tol=0.10, k_tol=0.3, k_weight=0.15,
          smooth_psd=0.05, smooth_mod=0.5, cl_smooth=(1e-4, 0.05), max_step=0.3,
          max_rerolls=8, patience=10, design_tol=1e-2, cl_anchor=0.0, cl_rough=0.0, polish="coupled",
          polish_mu=0.5, gn=True, frame_noise="gaussian", digital=True, starts=1, cl_box=None,
          redraw_reset=False, redraw_tries=1, branch_steps=4, early_redraws=6, early_err=0.5,
          redraw_frames="model",
          pattern=None,
          pattern_mapping="blocks", pattern_block_s=None, tones=None, life=None, tone_mapping=None,
          log=None) -> MMVSDResult:
    """Profile of duration ~T with FDS D_target and response kurtosis k_target(fn).

    k_target       scalar or array on p.fn (kurtosis of the relative displacement)
    time_corr_s    correlation time of the modulation (burst duration), scalar or
                   array on p.fn; default 4 oscillator decay times, at least 0.5 s
    frame_s        force one frame length for all f (default: multi-resolution)
    freq_corr_oct  correlation width of the modulation along log2(f)
    psd_prior      (f, G) of the reference PSD, or G on p.fn: the test PSD is
                   prior(f) times a smooth correction on the nodes, so peaks,
                   band edges and gaps of the reference are kept at full
                   resolution. Default: PSD on the nodes, curvature penalty
                   relative to the smoothed Gaussian inversion.
    damage         "dirlik" (default) or "rice" damage rate inside the model
    smooth_psd, smooth_mod  curvature penalties (Tikhonov) on log(P/prior) and log e
                   in the model design; cl_smooth: the (weaker) ones of the closed
                   loop, where node-wise corrections must remain possible
    gn             start the closed loop with coupled Gauss-Newton steps on (P, e)
                   (False: polishing of P only; the modulation stays the design)
    polish         "coupled" (default): polishing step by damped least squares with the
                   model Jacobian (polish_mu = damping relative to its diagonal), so an
                   oscillator whose damage comes from a neighbouring band is corrected
                   there; "node": node-wise multiplicative step as vibsynth
    cl_anchor      weight (relative to the median data curvature) of the pull of log P
                   towards the design in the closed-loop Gauss-Newton step, at the
                   oscillators whose damage is produced away from the node (default 0;
                   ~0.01 limits PSD drifts there, at some cost in max FDS error)
    cl_rough       weight (relative to the median data curvature) of a curvature
                   penalty on log(P/prior) in the closed-loop steps. Default 0: the
                   node-to-node ripple of P is what lets the loop fit the damage of
                   the realisation; 0.003 to 0.03 gives a smoother PSD at the cost
                   of a larger FDS error (several % median)
    The options cl_box, redraw_reset, redraw_tries, branch_steps, redraw_frames and
    starts are advanced settings for experiments; their defaults are the tested ones.
    redraw_frames  "model" (default): the re-drawn frames are those carrying most of the
                   MODEL damage of the worst oscillator; "realised": those carrying most
                   of its damage in the realised response
    k_weight       weight of the kurtosis residual against log FDS in the closed loop
    tol, k_tol     stop when max |FDS err| <= tol and median |k_z err| <= k_tol
    pattern        (x_ref, fs_ref): keep the time-frequency pattern of this record
                   (ReferenceField) instead of a random field with its statistics;
                   pattern_mapping "blocks" or "compress" for a shorter test,
                   pattern_block_s block length (default 4 s or 5 burst durations)
    tones          ToneModel from find_tones(record): its sine-on-random blocks are
                   reproduced with the vibsynth sine-on-random algorithm (deterministic
                   lines amplified to keep their share of the life damage, fixed in
                   the closed loop; the random part is designed for the rest). The
                   residual (record minus lines) replaces the record for the pattern
                   and for the PSD prior. life = life represented by D_target
                   (default: the record duration); tone_mapping: how the record time
                   is mapped on the test for the lines (default: the pattern mapping,
                   or "compress" with the random field, so every tonal block of the
                   record appears in the test, in order)
    cl_box         the closed loop keeps P within a factor cl_box of the design (None: free)
    redraw_reset   on a re-draw, P within +-0.5 octave of the worst oscillator is reset to
                   its 3-node smoothed value (the ripple fitted to the old realisation)
    redraw_tries   candidate re-draws evaluated at every re-draw (the best one is kept)
    branch_steps   polishing steps of a re-draw branch before it is kept or discarded
    starts         number of realisations (frame noises) of the design measured before the
                   closed loop; the best one is kept as its starting point
    frame_noise    "gaussian" (default): frames of complex normal coefficients, so the
                   signal is conditionally Gaussian as the model assumes;
                   "multisine": unit-modulus coefficients (exact band power per
                   frame, lighter-tailed responses when a frame has few lines in an
                   oscillator band)
    digital        model with the exact response of the sampled measurement
                   (Smallwood filters, rainflow on samples); False: analogue oscillators
    design_tol     accuracy of the model design in log FDS (the measured closed loop
                   corrects the model by ~30 % anyway, so 1 % is enough)
    patience       stop when the best profile has not improved by 1 % (score) for this
                   many measured iterations (the best profile is returned anyway)
    max_rerolls    re-draws of the phases of the frames dominating the worst
                   oscillator, tried when the polishing stalls
    early_redraws  before the first correction, while the worst oscillator misses its
                   damage by more than early_err (relative), the frames that carry
                   its largest response cycles in the realisation are re-drawn
                   (+-1/4 octave around it) and the draw is kept if the profile
                   improves: an outlier of the noise is replaced rather than
                   compensated by a notch in P (at most early_redraws tries)
    """
    say = log or (lambda s: None)
    rng = np.random.default_rng(seed)
    fn = p.fn
    n = fn.size
    D_target = np.asarray(D_target, float)
    has_lines = tones is not None and len(tones.tracks) > 0
    k_t = np.maximum(np.broadcast_to(np.asarray(k_target, float), fn.shape).copy(), 1.0 if has_lines else 3.0)
    tau = p.Q / (2 * np.pi * fn)
    if time_corr_s is None:
        time_corr_s = np.maximum(0.5, 4 * tau)
    tc = np.broadcast_to(np.asarray(time_corr_s, float), fn.shape).copy()
    mr = MultiResolution(fn, tc, fs, T, frame_s=frame_s, sub=2)
    K, N, hop = mr.K, mr.N, mr.hop
    Tn = N / fs
    warn = []
    short = tc < 2 * tau
    if np.any(short):
        warn.append(f"time correlation shorter than 2 oscillator decay times below "
                    f"{fn[short].max():.3g} Hz: bursts are low-pass filtered by the oscillators "
                    "(Papoulis); the design compensates with a deeper modulation")
    if Tn < 20 * float(np.max(tc)):
        warn.append(f"only ~{Tn / np.max(tc):.0f} correlation times in the profile at the slowest "
                    "nodes: the response kurtosis will be uneven across fn")
    say(f"MMVSD: T={Tn:.1f} s, fine hop={hop:.3f} s, frames={K}, freq_corr={freq_corr_oct:.3f} oct, "
        f"time_corr={tc.min():.2f}-{tc.max():.2f} s, damage model {damage}")
    say(f"  frames: {mr.describe()}")

    if has_lines and isinstance(psd_prior, tuple):          # random part: PSD of the residual
        fpr = np.asarray(psd_prior[0], float)
        psd_prior = welch_psd(tones.residual, tones.fs, resolution=float(fpr[1] - fpr[0]))
    prior = None if psd_prior is None else _prior_function(psd_prior, fn)
    Wk = moment_operators(fn, p.Q, fs, p.prefilter, weight=prior, digital=digital)
    if pattern is not None:
        xr, fsr = pattern
        if has_lines:
            xr, fsr = tones.residual, tones.fs
        if abs(float(fsr) - fs) > 1e-6:
            from scipy.signal import resample_poly
            from fractions import Fraction
            fr = Fraction(float(fs) / float(fsr)).limit_denominator(1000)
            xr = resample_poly(np.asarray(xr, float), fr.numerator, fr.denominator)
        bs = pattern_block_s or max(4.0, 5.0 * float(np.max(tc)))
        mod = ReferenceField(xr, fs, mr, K, pattern_mapping, bs)
        say(f"  modulation from the reference time-frequency pattern ({mod.mapping}"
            + (f", blocks of {bs:.3g} s" if mod.mapping == "blocks" else "") + ")")
    else:
        mod = ModulationField(fn, K, hop, freq_corr_oct, tc, rng, marginal)
    em = EnvelopeModel(Wk, p, hop, Tn, K, sub=2, damage=damage, fs=fs if digital else None)
    em.up = mr.up_eff                      # multi-resolution crossfade + band weights
    z_lo, z_hi, dz = math.log(1e-3), math.log(max_excess), 0.05

    # ------------------------------------------- sine-on-random blocks (lines)
    lines, Dt, xl, D_goal, lmap = None, np.zeros(n), 0.0, D_target, None
    D_rand_goal, k_rand, u_lines = D_target, k_t, None
    if has_lines:
        T_ref = tones.duration
        if isinstance(mod, ReferenceField):
            idx, lmap = mod.ref_index, mod.mapping
        else:
            bs_l = pattern_block_s or max(4.0, 5.0 * float(np.max(tc)))
            idx, lmap = frame_mapping(max(int(T_ref / hop), 1), K, tone_mapping or "compress",
                                       round(bs_l / hop))
        lv = life or T_ref
        rng_l = np.random.default_rng(None if seed is None else [int(seed), 7919])
        lines = TestLines(tones, N, fs, hop, idx, lmap, p.b, lv, rng_l)
        if lines.tracks:
            xl = lines.x
            Dt = measure(xl, fs, p)[0]
            t_model = (np.arange(em.Nt) + 0.5) * hop / em.sub
            u_lines = lines.variance(fn, p.Q, t_model, fs, p.prefilter)
            # the random part is designed for the FDS and response kurtosis of the
            # random part of the record (record minus lines), measured, so that
            # no additivity of line and random damage is assumed
            D_res, k_res = measure(tones.residual, tones.fs, p)
            D_rand_goal = D_res * (lv / T_ref)
            k_rand = np.maximum(_smooth3(k_res), 3.0)
            D_goal = np.maximum(D_target, Dt + 0.05 * D_target)
            over = Dt > 0.95 * D_target
            if np.any(over):
                warn.append("the lines alone exceed the target damage near "
                            + ", ".join(f"{v:.0f}" for v in fn[over]) + " Hz (over-tested there, as in "
                            "vibsynth sine on random)")
            say(f"  sine-on-random blocks: {len(lines.tracks)} line(s) as deterministic sines "
                f"(vibsynth SoR), mapping {lmap}: "
                + "; ".join(f"{s_['f_min']:.1f}-{s_['f_max']:.1f} Hz x{s_['gain']:.2f}" for s_ in lines.summaries))
        for s_ in lines.lost:
            warn.append(f"line {s_['f_min']:.1f}-{s_['f_max']:.1f} Hz is not in the selected blocks of the "
                        "record: its damage is carried by the random part (use the 'compress' mapping)")

    # Unknown "P" on the nodes: the PSD itself, or (with a prior) the smooth
    # multiplicative correction of the full-resolution prior shape, so that
    # resonance peaks, band edges and gaps of the reference are kept exactly.
    P0 = solve_variance(Wk[0], target_variance(D_rand_goal, Tn, p))
    if prior is None:
        lp = np.log(np.maximum(_smooth3(_smooth3(P0)), P0.max() * 1e-8))
    else:
        lp = np.zeros(n)
        P0 = np.full(n, float(np.median(P0)))

    def fit_psd(P, Mup, D_goal):
        def rj(y):
            Py = np.exp(y)
            r = em.run(Py, Mup)
            return np.log(r["D"] / D_goal), em.jac_logP(Py, Mup, r)
        y, res = _lm(np.log(np.maximum(P, 1e-300)), rj, smooth=smooth_psd, iters=60, tol=design_tol,
                     y_ref=lp)
        return np.exp(y), float(np.max(np.abs(res)))

    def fit_kurtosis(e, P, k_goal):
        def rj(z):
            ez = np.exp(z)
            M = mod.field(ez)
            Mup = em.up(M)
            r = em.run(P, Mup, damage=False)
            dM = (mod.field(ez * math.exp(dz)) - M) / dz
            return r["k"] - k_goal, em.jac_k(P, em.up(dM), r)
        z, res = _lm(np.log(np.clip(e, 1e-3, max_excess)), rj, z_lo, z_hi, smooth=smooth_mod,
                     iters=40, tol=1e-3)
        return np.exp(z), float(np.max(np.abs(res)))

    # --------------------------------------------------------------- design
    e = np.clip(k_rand - 3.0, 0.05, max_excess)
    P = P0
    for _ in range(3):                                # weakly coupled: alternate
        e, ek = fit_kurtosis(e, P, k_rand)
        P, ed = fit_psd(P, em.up(mod.field(e)), D_rand_goal)
    M = mod.field(e)
    hist = [("model", ed, ek)]
    say(f"  model design{' (random part)' if u_lines is not None else ''}: max |log FDS err| {ed:.3g}, "
        f"max |k_z err| {ek:.3f}")
    if np.any((e >= max_excess * 0.999) & (em.run(P, em.up(M), damage=False)["k"] < k_rand - 0.1)):
        warn.append("target response kurtosis not reachable at some fn (modulation depth capped)")
    # elasticity of the measured damage to the level of the random part
    # (d log D / d log P = b/2 without lines, lower where a line carries part of
    # the damage); measured on the profile at the first iteration, it scales the
    # damage rows of the closed-loop Jacobian and the polishing steps
    eta = np.full(n, p.b / 2.0)
    y_design = np.log(P)
    # local share of the damage sensitivity: the part of d log D_n / d log P that
    # comes from the node and its two neighbours. Small where the damage of an
    # oscillator is produced elsewhere (e.g. the quasi-static response of an
    # oscillator above a strong low-frequency band): there a node-wise
    # correction of P would run away without fixing the node, so the polishing
    # step is scaled by this share and the error is fixed at its source
    _Mu = em.up(M)
    _J = np.maximum(em.jac_logP(P, _Mu, em.run(P, _Mu)), 0.0)
    _band = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) <= 1
    share = np.clip((_J * _band).sum(axis=1) / np.maximum(_J.sum(axis=1), 1e-300), 0.05, 1.0)
    anchor_w = np.clip((0.5 - share) / 0.5, 0.0, 1.0)
    if np.any(share < 0.5):
        say(f"  damage produced away from the node (local share < 0.5) at {int(np.sum(share < 0.5))} "
            f"oscillators, {fn[share < 0.5].min():.3g}-{fn[share < 0.5].max():.3g} Hz")
    if u_lines is not None:
        em.set_lines(u_lines)                         # the closed loop works on totals


    # ------------------------------------------------------------ synthesis
    mr.init_phases(rng, frame_noise)
    grid = np.fft.rfftfreq(N, 1 / fs)

    def carrier(M):
        return np.fft.rfft(mr.synthesize(M))

    def build(Vf, P):
        return np.fft.irfft(Vf * np.sqrt(_psd_lines(fn, P, grid, p.Q, prior)), N)

    # ------------------------------------------------------------ closed loop
    # The phases and the ranks of the field are frozen, so the realisation is a
    # smooth function of (P, e). Three kinds of steps, all driven by MEASURED
    # residuals rD = log(D_rainflow / D_target), rk = w_k (k_meas - k_target):
    #  "gn"     coupled Gauss-Newton step with the MODEL Jacobian (block
    #           Gauss-Seidel): the modulation depth e follows the kurtosis
    #           residual, dz = argmin |J_kz dz + rk|, and the PSD compensates the
    #           damage change the model predicts for it, dy = argmin
    #           |J_Dy dy + rD + J_Dz dz|. The model carries the coupling between
    #           oscillators (W, envelope law, spectral width, modulation of the
    #           neighbouring band), the measurement what the model neglects.
    #  "polish" node-wise multiplicative correction of P (as vibsynth).
    #  "redraw" when the worst oscillator stops improving, its error is a
    #           property of the realisation (a few rare large cycles): the phases
    #           of the frames that carry most of its damage are re-drawn in a
    #           band around it, P is re-polished for a few steps (a branch), and
    #           the branch is kept only if it beats the best profile so far.
    L = _second_diff(n)
    LtL = L.T @ L
    Vf = carrier(M)
    best, branch = None, None
    main_phase = "gn" if gn else "polish"
    early_left = max(int(early_redraws), 0)
    early_lim, early_fail = math.log(1.0 + float(early_err)), 0
    lam, w, fails, phase = 0.1, 1.0, 0, ("early" if early_left else main_phase)
    branch_from = None
    stalls, redraws, max_seen = 0, 0, np.inf
    best_track, last_gain = None, 0
    rng_roll = np.random.default_rng(rng.integers(2 ** 63))

    def evaluate(P, e, M, Vf):
        x = build(Vf, P) + xl
        rs = response_spectra(x, fs, p)
        Dm, km = rs["fds"], rs["kurtosis"]
        rD = np.log(Dm / D_goal)
        rk = k_weight * (_smooth3(km) - k_t)
        ad = np.abs(rD)
        return dict(score=0.5 * float(np.max(ad)) + 0.5 * float(np.quantile(ad, 0.9))
                    + float(np.mean(np.abs(rk))), x=x, P=P.copy(), e=e.copy(), M=M, Vf=Vf,
                    Dm=Dm, km=km, Mm=rs["mrs"], rD=rD, rk=rk, err=float(np.max(np.abs(Dm / D_goal - 1))),
                    kerr=float(np.median(np.abs(km - k_t))),
                    med=float(np.median(np.abs(Dm / D_goal - 1))))

    lbox = None if not cl_box else math.log(float(cl_box))

    def boxed(P_):
        if lbox is None:
            return P_
        return np.exp(np.clip(np.log(P_), y_design - lbox, y_design + lbox))

    first = None
    if starts > 1:                                      # several noises of the design: keep the best
        rng_ms = np.random.default_rng(rng_roll.integers(2 ** 63))
        for s_ in range(int(starts)):
            if s_:
                mr.init_phases(rng_ms, frame_noise)     # a new list: earlier ones stay intact
            c_ = evaluate(P, e, M, carrier(M))
            say(f"  start {s_ + 1}/{starts}: max FDS err {100 * c_['err']:.1f} %, median {100 * c_['med']:.1f} %")
            if first is None or c_["score"] < first["score"]:
                first, best_ph = c_, mr.phases
        mr.phases = best_ph
        Vf = first["Vf"]
    pre_eval = first
    for it in range(iterations + 1):
        cur = pre_eval if pre_eval is not None else evaluate(P, e, M, Vf)
        pre_eval = None
        if it == 0 and lines is not None and lines.tracks:
            D11 = measure(1.1 * (cur["x"] - xl) + xl, fs, p)[0]
            eta = np.clip(np.log(D11 / cur["Dm"]) / (2 * math.log(1.1)), 0.1 * p.b / 2.0, p.b / 2.0)
            say(f"  damage elasticity to the random level: min {eta.min():.2f} (b/2 = {p.b / 2:g}) "
                f"at {fn[int(np.argmin(eta))]:.3g} Hz")
        hist.append(("measured", cur["err"], cur["kerr"]))
        say(f"  closed loop {it} ({phase}): max FDS err {100 * cur['err']:.1f} %, "
            f"median |FDS err| {100 * cur['med']:.1f} %, median |k_z err| {cur['kerr']:.2f}")
        ref_state = branch if branch is not None else best
        improved = ref_state is None or cur["score"] < ref_state["score"]
        if branch is not None:                          # inside a re-draw branch
            if improved:
                branch = cur
                w = min(1.6, w * 1.25)
            else:
                w *= 0.5
            branch_left -= 1
            if branch_left <= 0:
                kept = branch["score"] < best["score"]
                if branch_from == "early":              # it must fix the outlier it was drawn for
                    kept = kept and abs(branch["rD"][early_node]) < abs(best["rD"][early_node]) \
                        and branch["err"] <= best["err"]    # without creating a worse one
                if kept:
                    best = branch
                    say("    re-draw kept")
                else:                                   # restore the frame noise
                    mr.phases[pending[0]] = pending[1]
                    say("    re-draw discarded")
                if branch_from == "early":
                    early_fail = 0 if kept else early_fail + 1
                    branch, phase = None, "early"
                else:
                    branch, phase, w, stalls, max_seen = None, "polish", 1.0, 0, best["err"]
        else:
            if improved:
                best = cur
            if phase == "gn":
                lam = max(lam / 3, 1e-4) if improved else lam * 4
                fails = 0 if improved else fails + 1
                if fails >= 2:
                    phase, w, max_seen = "polish", 1.0, best["err"]
            elif phase == "polish":
                w = min(1.6, w * 1.25) if improved else w * 0.5
                stalls = 0 if best["err"] < 0.97 * max_seen else stalls + 1
                max_seen = min(max_seen, best["err"])
                if (stalls >= 3 or w < 1 / 32) and redraws < max_rerolls and best["err"] > tol:
                    phase = "redraw"
        if best_track is None or best["score"] < 0.99 * best_track:
            best_track, last_gain = best["score"], it
        if it - last_gain >= patience:
            say(f"  no improvement for {patience} iterations: stop")
            break
        if (best["err"] <= tol and best["kerr"] <= k_tol) or it == iterations:
            break
        if phase == "polish" and w < 1 / 32 and redraws >= max_rerolls:
            break
        st_ = branch if branch is not None else best
        Pb, eb, Mb = st_["P"], st_["e"], st_["M"]
        if phase == "early":
            worst = int(np.argmax(np.abs(best["rD"])))
            # stop at the tolerance, or after two useless draws in a row (the
            # error is then not an outlier of the noise but a model error)
            if early_left <= 0 or early_fail >= 2 or abs(best["rD"][worst]) <= early_lim:
                phase, last_gain, w = main_phase, it, 1.0    # patience counts from here
        if phase == "early":                            # local re-draw of a noise outlier
            early_left -= 1
            dens = np.abs(sdof_relative_displacement(best["x"], fs, fn[worst], p.Q)) ** p.b
            j, frames = mr.realised_frames(dens, worst)
            pending, early_node = (j, mr.phases[j].copy()), worst
            mr.reroll(j, frames, rng_roll, fn[worst], 0.25)
            say(f"    early re-draw of {frames.size} frames (group {j}) around fn = {fn[worst]:.3g} Hz")
            P, e, M = Pb.copy(), eb.copy(), Mb
            Vf = carrier(M)
            branch, branch_left, branch_from = dict(best, score=np.inf), 1, "early"
        elif phase == "gn":
            Mupb = em.up(Mb)
            r = em.run(Pb, Mupb)
            dMup = em.up((mod.field(eb * math.exp(dz)) - Mb) / dz)
            Jkz = k_weight * em.jac_k(Pb, dMup, r)
            A = Jkz.T @ Jkz + cl_smooth[1] * LtL
            zb = np.log(np.maximum(eb, 1e-3))
            step_z = -np.linalg.solve(A + lam * np.diag(np.diag(A)) + 1e-9 * np.eye(n),
                                      Jkz.T @ st_["rk"] + cl_smooth[1] * LtL @ zb)
            step_z = np.clip(step_z, -0.5, 0.5)
            sc = (eta / (p.b / 2.0))[:, None]               # lines: measured share of the random part
            JDy, JDz = sc * em.jac_logP(Pb, Mupb, r), sc * em.jac_logD_mod(Pb, dMup, r)
            JtJ = JDy.T @ JDy
            dmed = float(np.median(np.diag(JtJ)))
            rough = cl_smooth[0] + cl_rough * dmed
            A = JtJ + rough * LtL
            # anchor to the design, only at nodes whose damage is produced away from
            # them (local share < 0.5): there the data hardly determine P and the
            # least-squares step would let it drift
            alpha = cl_anchor * dmed * anchor_w
            A = A + np.diag(alpha)
            g = (JDy.T @ (st_["rD"] + JDz @ step_z) + rough * LtL @ (np.log(Pb) - lp)
                 + alpha * (np.log(Pb) - y_design))
            step_y = -np.linalg.solve(A + lam * np.diag(np.diag(A)) + 1e-9 * np.eye(n), g)
            P = boxed(Pb * np.exp(np.clip(step_y, -max_step, max_step)))
            e = np.clip(np.exp(np.clip(zb + step_z, z_lo, z_hi)), 1e-3, max_excess)
            M = mod.field(e)
            Vf = carrier(M)
        elif phase == "redraw":
            redraws += 1
            worst = int(np.argmax(np.abs(best["rD"])))
            if redraw_frames == "realised":
                dens = np.abs(sdof_relative_displacement(best["x"], fs, fn[worst], p.Q)) ** p.b
                j, frames = mr.realised_frames(dens, worst)
            else:
                d = em.run(Pb, em.up(Mb))["d"][worst]
                j, frames = mr.dominant_frames(d, worst)
            P, e, M = Pb.copy(), eb.copy(), Mb
            if redraw_reset:                            # drop the ripple fitted to the old noise
                band = np.abs(np.log2(fn / fn[worst])) <= 0.5
                P[band] = np.exp(_smooth3(np.log(Pb))[band])
            old_group = mr.phases[j].copy()
            cand = None
            for _t in range(max(1, int(redraw_tries))):
                if _t:
                    mr.phases[j] = old_group.copy()     # every candidate starts from the same noise
                mr.reroll(j, frames, rng_roll, fn[worst], 0.25)
                if redraw_tries > 1:
                    c_ = evaluate(P, e, M, carrier(M))
                    if cand is None or c_["score"] < cand[0]["score"]:
                        cand = (c_, mr.phases[j].copy())
            if cand is not None:
                mr.phases[j] = cand[1]
                pre_eval = cand[0]
            pending = (j, old_group)
            say(f"    re-draw of {frames.size} frames (group {j}) around fn = {fn[worst]:.3g} Hz")
            Vf = pre_eval["Vf"] if pre_eval is not None else carrier(M)
            branch = dict(best, score=np.inf)           # filled by the next evaluation
            branch_left, branch_from = int(branch_steps), "redraw"
            phase = "polish"
            w = 1.0
        else:                                           # polish (main line or branch)
            if polish == "coupled":
                # damped least squares with the model Jacobian: the correction goes
                # to the band that produces the damage of an oscillator (for a
                # diagonal Jacobian this is the node-wise vibsynth step / (1 + mu))
                Mupb = em.up(Mb)
                Jp = (eta / (p.b / 2.0))[:, None] * em.jac_logP(Pb, Mupb, em.run(Pb, Mupb))
                A = Jp.T @ Jp
                rough = cl_rough * float(np.median(np.diag(A)))
                A = A + polish_mu * np.diag(np.diag(A)) + rough * LtL + 1e-9 * np.eye(n)
                dy = -w * np.linalg.solve(A, Jp.T @ st_["rD"] + rough * LtL @ (np.log(Pb) - lp))
            else:
                dy = -w * st_["rD"] / eta
            P = boxed(Pb * np.exp(np.clip(dy, -0.8, 0.8)))
            e, M, Vf = eb.copy(), Mb, st_["Vf"]

    x, P, e, M, Dm, km = best["x"], best["P"], best["e"], best["M"], best["Dm"], best["km"]
    drift = np.abs(_smooth3(np.log(P) - y_design)) > math.log(8.0)
    if np.any(drift):
        warn.append("the closed loop moved the PSD by more than 8x from the design at "
                    f"{fn[drift].min():.3g}-{fn[drift].max():.3g} Hz (oscillators whose damage is produced "
                    "by another band, or rare cycles): try another seed or a design anchor (cl_anchor ~0.01)")
    r = em.run(P, em.up(M))
    st = statistics(x)
    tf = None
    if isinstance(mod, ReferenceField):
        tf = _pattern_correlation(x, fs, mr, mod.lB, mod.ref_index)
        say(f"  time-frequency pattern kept: median correlation {np.median(tf):.2f} "
            f"(log band power of the profile vs the mapped reference)")
    return MMVSDResult(
        x=x, fs=fs, psd_nodes=P if prior is None else P * prior(fn), modulation=M, frame_hop_s=hop, excess_design=e,
        model={"fds": r["D"], "k_response": r["k"], "c": r["c"], "s_mean": r["B"],
               "psd_correction": P if prior is not None else None, "psd_design": np.exp(y_design)},
        measured={"fds": Dm, "k_response": km, "mrs": best["Mm"], "stats": st,
                  "fds_max_err": float(np.max(np.abs(Dm / D_target - 1))),
                  "tf_corr": tf},
        history=hist,
        params={"frames": mr.describe(), "freq_corr_oct": freq_corr_oct, "time_corr_s": tc,
                "marginal": marginal, "seed": seed, "T": Tn, "damage": damage,
                "psd_prior": psd_prior is not None,
                "pattern": None if pattern is None else mod.mapping,
                "lines": None if lines is None else lmap},
        warnings=warn,
        lines=None if lines is None else {"x": lines.x, "tracks": lines.summaries, "lost": lines.lost,
                                          "fds": Dt, "mapping": lmap, "D_goal": D_goal,
                                          "fds_random_goal": D_rand_goal, "elasticity": eta})


# ------------------------------------------------------- vibsynth method
def _smooth_nodes(k, n=3):
    if n <= 1:
        return np.asarray(k, float)
    pad = n // 2
    kp = np.r_[np.full(pad, k[0]), k, np.full(pad, k[-1])]
    return np.convolve(kp, np.ones(n) / n, mode="valid")


def _profile_score(res, D_t, k_t, k_weight=0.15):
    """The closed-loop score of a result: 0.5 max + 0.5 90th percentile of
    |ln(D/D_goal)| plus the mean weighted kurtosis error (lower is better)."""
    D_goal = res.lines["D_goal"] if res.lines is not None else D_t
    rD = np.abs(np.log(res.measured["fds"] / D_goal))
    kt = np.maximum(np.asarray(k_t, float), 1.0 if res.lines is not None else 3.0)
    rk = k_weight * np.abs(_smooth3(res.measured["k_response"]) - kt)
    return 0.5 * float(rD.max()) + 0.5 * float(np.quantile(rD, 0.9)) + float(rk.mean())


def _downsample_field(M, max_cols=400):
    step = max(1, int(np.ceil(M.shape[1] / max_cols)))
    return M[:, : (M.shape[1] // step) * step].reshape(M.shape[0], -1, step).mean(axis=2)


def synthesize(ref, duration, *, fs=None, seed=None, iterations=40, tol=0.10, modulation="random",
               mapping="blocks", block_s=None, lines="lines", max_excess=30.0, freq_corr=None,
               time_corr=None, k_smooth=3, cl_anchor=0.0, max_samples=4_000_000, fallback=False, options=None,
               log=None):
    """MMVSD profile for a vibsynth target (record, specification or mission).

    Targets: the FDS of the reference and the kurtosis of its oscillator
    responses k_z(fn) (smoothed over `k_smooth` nodes), measured on the record
    or combined over the situations of a mission.
    modulation  "random": random field with the burst width and duration of the
                record; "pattern": the record's own time-frequency pattern
                (mapping "blocks" or "compress" when the test is shorter)
    lines       "lines": sine-on-random blocks of a record as deterministic lines
                (sine on random principle), "random": detected but left to the
                random part, "off": no detection
    Tests longer than max_samples samples are synthesised once and played
    `repeats` times (the profile is periodic). `options`: further keyword
    arguments of mmvsd() (advanced settings). fallback=True: when the coupled
    closed loop ends above twice the tolerance, a node-wise closed loop is run
    from the same design and noise and the profile with the better score is
    kept (it helped on some stationary-like cases, not on the measured
    transport records, where it only doubles the time).
    """
    from .analysis import response_statistics
    from .synthesis import SynthesisResult
    say = log or (lambda s: None)
    p = ref.params
    fs = float(fs or ref.fs)
    r = int(np.ceil(duration * fs / max_samples)) if duration * fs > max_samples else 1
    T = duration / r
    warn = []
    _, kz = response_statistics(ref)
    k_t = _smooth_nodes(kz, int(k_smooth or 1))
    if ref.x is not None:
        xs, fss = ref.x, ref.fs
    else:                                            # mission or specification: longest segment
        seg = max(ref.segments, key=lambda s_: s_[0].size / s_[1])
        xs, fss = seg[0], seg[1]
    tones = None
    if lines != "off" and ref.x is not None:
        tm = find_tones(ref.x, ref.fs, p.fn[0], p.fn[-1])
        if tm.tracks:
            say(f"sine-on-random blocks: {sum(1 for b in tm.blocks if b['tones'])} of {len(tm.blocks)} "
                f"blocks; {tm.describe()}")
            if lines == "lines":
                tones = tm
            xs = tm.residual                         # the random part of the record
    elif lines != "off":
        say("sine-on-random blocks are detected on single records only")
    if freq_corr is None or time_corr is None:
        fc_a, tc_a = estimate_structure(xs, fss, p)
        fc = float(freq_corr) if freq_corr is not None else fc_a
        tc = np.full(p.fn.size, float(time_corr)) if time_corr is not None else tc_a
    else:
        fc, tc = float(freq_corr), np.full(p.fn.size, float(time_corr))
    tau = p.Q / (2 * np.pi * p.fn)
    tc = np.maximum(tc, np.maximum(2 * tau, 0.25))
    pattern = None
    if modulation == "pattern":
        if ref.x is not None:
            pattern = (ref.x, ref.fs)
        else:
            warn.append("the time-frequency pattern needs a single record as target: random field used")
    D_t = np.asarray(ref.fds, float) / r
    kw = dict(freq_corr_oct=fc, time_corr_s=tc, seed=seed, psd_prior=(ref.psd_f, ref.psd),
              iterations=int(iterations), tol=float(tol), max_excess=float(max_excess), pattern=pattern,
              pattern_mapping=mapping or "blocks", pattern_block_s=block_s, tones=tones, life=ref.life / r,
              cl_anchor=float(cl_anchor or 0.0), log=log)
    kw.update(options or {})
    res = mmvsd(D_t, k_t, T, fs, p, **kw)
    loop = "coupled"
    if fallback and res.measured["fds_max_err"] > 2 * float(tol) and kw.get("gn", True):
        # the coupled loop can stall where the damage of a few oscillators rests on a
        # few bursts; the node-wise loop (as vibsynth) started from the same design
        # and noise is slower but steadier there: keep the better of the two
        say(f"max FDS error {100 * res.measured['fds_max_err']:.0f} % after the coupled closed loop: "
            "second closed loop with node-wise steps from the design")
        res2 = mmvsd(D_t, k_t, T, fs, p, **dict(kw, gn=False, polish="node"))
        s1, s2 = _profile_score(res, D_t, k_t), _profile_score(res2, D_t, k_t)
        say(f"  score (lower is better): coupled {s1:.3f}, node-wise {s2:.3f}")
        if s2 < s1:
            res, loop = res2, "node-wise"
    hist = [h[1] for h in res.history if h[0] == "measured"]
    lines_info = None
    if res.lines is not None:
        lines_info = {"tracks": res.lines["tracks"], "lost": res.lines["lost"], "mapping": res.lines["mapping"],
                      "fds_lines": res.lines["fds"] * r}
    tf = res.measured.get("tf_corr")
    info = {"fds_measured": res.measured["fds"] * r, "mrs_measured": res.measured["mrs"],
            "fds_error_history": hist, "k_response": res.measured["k_response"], "k_target": k_t,
            "k_model": res.model["k_response"], "excess": res.excess_design,
            "modulation": _downsample_field(res.modulation), "modulation_hop_s": res.frame_hop_s,
            "frames": res.params["frames"], "freq_corr_oct": fc, "time_corr_s": tc,
            "pattern_mapping": res.params.get("pattern"), "tf_corr": None if tf is None else float(np.median(tf)),
            "lines": lines_info, "closed_loop": loop, "psd_correction": res.model["psd_correction"],
            "psd_design": res.model["psd_design"], "history": res.history}
    return SynthesisResult(res.x, fs, "MMVSD", r, p.fn, res.psd_nodes, None, info, warn + res.warnings)

