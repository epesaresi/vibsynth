"""Mission synthesis and kurtosis control algorithms of the PhD work.

FDS controlled (chapter 2 of the dissertation):
    gaussian  standard Mission Synthesis (Lalanne), stationary Gaussian
    kfds      prescribed FDS + kurtosis               (kFDS)
    rf        prescribed FDS + RMS                    (RF)
    pf        prescribed FDS + PSD shape + RMS        (PF)
    psf       prescribed FDS + PSD shape              (PSF)
Kurtosis control (chapter 1):
    mlv       prescribed PSD + kurtosis, multi-level variance (modulation)
    vsd       prescribed PSD + kurtosis, variable spectral density
    pm        prescribed PSD + kurtosis, phase manipulation

Common structure:

* kFDS, RF and MLV are written as what they are mathematically, a stationary
  Gaussian carrier g(t) modulated by a slowly varying envelope e(t):
        x(t) = e(t) * g(t).
  For a Gaussian carrier the kurtosis is exactly 3*<e^4>/<e^2>^2, the RMS is
  rms(g)*sqrt(<e^2>), and (quasi-statically) the damage is D_g*<e^b>.  The
  normalisation <e^b> = 1 therefore preserves the FDS for any envelope; this
  is the continuous form of the block weights v_j of the thesis
  (e_j = (n*v_j)^(1/b)).  One scalar (the envelope contrast) is solved with a
  bracketing root finder on the *realised* signal, so the requested kurtosis
  or RMS is met exactly, without heuristic corrections or random retries.
* Using one continuous carrier removes the waveform jumps at block borders.
  The envelope transitions are linear ramps.
* PF/PSF mix two stationary carriers with a smooth switching mask:
        x = sqrt(m) * x_B + sqrt(1 - m) * x_L,
  so the overall PSD equals (t_B G_B + t_L G_L)/T = G_ref exactly.
* Every FDS method ends with a closed-loop correction: the exact rainflow FDS
  of the realised signal is measured and the carrier spectrum is reshaped by
  (D_target/D_measured)^(1/b) (zero phase, envelope untouched).  It
  removes
  the known bias between Lalanne's Gaussian formula and rainflow counting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.optimize import brentq

from .design import invert_fds
from .fds import FDSParams, fds_gaussian, fds_time, response_variance_operator
from .signals import interp_psd, kurtosis, window_rms

__all__ = ["SynthesisResult", "gaussian", "kfds", "rf", "pf", "psf", "mlv", "vsd", "pm",
           "mrs_control", "sine_on_random", "detect_tones",
           "default_window", "METHODS"]


@dataclass
class SynthesisResult:
    x: np.ndarray
    fs: float
    method: str
    repeats: int = 1                       # play the signal this many times
    design_fn: np.ndarray | None = None
    design_psd: np.ndarray | None = None
    envelope: np.ndarray | None = None
    info: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.x.size / self.fs

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.x.size) / self.fs


# --------------------------------------------------------------------- helpers
def default_window(reference_duration: float, fs: float, tau: float = 0.0) -> float:
    """Burst / block duration rule ('threshold').

    `tau` (= Q / (pi * fn_min), decay time of the slowest oscillator) adds a
    lower bound of 2*tau: a level change must last long enough for the SDOF
    systems to respond quasi-statically, otherwise the damage of a block is
    not the stationary (Lalanne) damage the design assumes.
    """
    if fs > 4000:
        th = reference_duration / 400
    elif fs > 2000:
        th = reference_duration / 200
    elif fs > 500:
        th = reference_duration / 180
    else:
        th = reference_duration / 150
    while th * fs < 300:
        th *= 1.01
    return max(th, 2.0 * tau)


class _Carrier:
    """Stationary Gaussian carrier with frozen random phases.

    The spectrum can be reshaped (closed-loop FDS correction) while keeping
    the phases, so the time structure of the realisation is preserved.
    """

    def __init__(self, fn, psd_nodes, fs, n, rng, floor=None, psd_fine=None, q=None):
        self.fs, self.n = fs, n
        self.n_lines = n // 2
        self.df = fs / n
        self.grid = np.arange(self.n_lines + 1) * self.df
        if psd_fine is not None:
            self.G = psd_fine.copy()
        else:
            # hold the edge values over a margin of a few oscillator bandwidths
            # beyond fmin/fmax: otherwise the edge oscillators only see half
            # of their band (the rest is the floor) and the closed loop has to
            # fight a strong coupling between the first/last two nodes.
            m = 3.0 / q if q else 0.0
            fe = np.r_[fn[0] * (1 - m), fn, min(fn[-1] * (1 + m), fs / 2 * 0.999)]
            ge = np.r_[psd_nodes[0], psd_nodes, psd_nodes[-1]]
            self.G = interp_psd(fe, ge, self.grid, floor)
        self.G[0] = 0.0
        self.G[-1] = 0.0
        # level from which a line that the design left at zero (NNLS gives
        # sparse solutions) can be revived by a multiplicative correction
        self.revive = 1e-6 * float(self.G.max()) if self.G.size else 0.0
        self.phases = rng.uniform(-np.pi, np.pi, self.n_lines + 1)

    def signal(self) -> np.ndarray:
        amp = np.sqrt(2.0 * self.G * self.df)
        return np.fft.irfft(amp * np.exp(1j * self.phases) * self.n_lines, self.n)

    def reshape(self, fn, gain_nodes):
        """Multiply the PSD by gain^2 (gain interpolated on fn, edges held)."""
        g = np.interp(self.grid, fn, gain_nodes)
        band = (self.grid >= fn[0]) & (self.grid <= fn[-1]) & (g > 1.0)
        G = np.where(band, np.maximum(self.G, self.revive), self.G)
        self.G = G * g ** 2


def _expand_levels(levels: np.ndarray, nw: int, n: int, ramp: int) -> np.ndarray:
    """Piecewise-constant window levels -> per-sample envelope with linear ramps."""
    e = np.repeat(levels, nw)[:n]
    if e.size < n:
        e = np.r_[e, np.full(n - e.size, levels[-1])]
    if ramp > 1:
        k = np.ones(ramp) / ramp
        pad = ramp // 2
        ep = np.r_[np.full(pad, e[0]), e, np.full(ramp - pad - 1, e[-1])]
        e = np.convolve(ep, k, mode="valid")
    return e


def _segment_levels(segments, window_s):
    """log window-RMS of every segment and its weight (life per window)."""
    vals, wts = [], []
    for x, sfs, weight in segments:
        nw = max(int(round(window_s * sfs)), 2)
        r = window_rms(x, nw)
        vals.append(np.log(np.maximum(r, r.max() * 1e-6)))
        wts.append(np.full(r.size, float(weight)))
    return np.concatenate(vals), np.concatenate(wts)


def _scores(pattern: str, n_windows: int, rng, segments=None, window_s=None,
            burst_fraction=None) -> np.ndarray:
    """Relative log-levels u_j of the windows (envelope = exp(alpha*u)).

    pattern="reference": levels drawn from the measured (life-weighted)
    distribution of the window RMS of the reference segments, by weighted
    quantile resampling, in random order.
    pattern="bursts": 10-20 % elevated windows at random positions.
    """
    if pattern == "reference":
        if not segments:
            raise ValueError("pattern='reference' needs the reference signal(s)")
        v, wt = _segment_levels(segments, window_s)
        o = np.argsort(v)
        v, wt = v[o], wt[o]
        c = np.cumsum(wt)
        c = (c - 0.5 * wt) / c[-1]
        q = (np.arange(n_windows) + rng.uniform(0, 1, n_windows)) / n_windows
        u = rng.permutation(np.interp(q, c, v))
        return u - u.max()
    if pattern == "bursts":
        frac = burst_fraction if burst_fraction else rng.uniform(0.10, 0.20)   # 10-20 % of the windows
        npk = int(min(max(np.ceil(frac * n_windows), 1), n_windows - 1)) if n_windows > 1 else 1
        u = rng.uniform(0.0, 0.4, n_windows)
        idx = rng.choice(n_windows, npk, replace=False)
        u[idx] = rng.uniform(0.6, 1.0, npk)
        return u - 1.0
    raise ValueError("pattern must be 'bursts' or 'reference'")


def _solve_scalar(fun, target, lo=0.0, hi=1.0, hi_max=1e3, increasing=True):
    """Root of fun(a) = target on a monotone branch, expanding the bracket."""
    flo = fun(lo) - target
    if (flo >= 0) if increasing else (flo <= 0):
        return lo, False
    fhi = fun(hi) - target
    while ((fhi < 0) if increasing else (fhi > 0)) and hi < hi_max:
        lo, flo = hi, fhi
        hi *= 2.0
        fhi = fun(hi) - target
    if (fhi < 0) if increasing else (fhi > 0):
        return hi, False
    return brentq(lambda a: fun(a) - target, lo, hi, xtol=1e-6, rtol=1e-8), True


def _repeats(T, fs, max_samples):
    r = int(np.ceil(T * fs / max_samples)) if T * fs > max_samples else 1
    return r, T / r


def _snapshot(objs):
    """Copy the mutable numerical state (carrier spectra, envelope scalars)."""
    out = []
    for o in objs:
        d = o.__dict__ if hasattr(o, "__dict__") else o
        out.append({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()
                    if k in ("G", "alpha")})
    return out


def _restore(objs, snap):
    for o, sv in zip(objs, snap):
        d = o.__dict__ if hasattr(o, "__dict__") else o
        for k, v in sv.items():
            d[k] = v.copy() if isinstance(v, np.ndarray) else v


def _closed_loop(build, p, fs, D_target, iterations, tol, correct, state_objs,
                 resolve=None, use=None, log=None, measure="fds"):
    """Closed-loop FDS correction with exact rainflow measurement.

    Each step multiplies the carrier PSD at node j by gain_j^2 with
    gain_j = (D_target/D_measured)_j^(w/b): damage at oscillator fn_j scales as
    amplitude^b.  The relaxation w adapts: it grows (up to 1.6, over-
    relaxation, because coupling between neighbouring oscillators slows the
    plain fixed point) while the error decreases, and a step that makes the
    error worse is undone and retried with half the relaxation.
    """
    use = np.ones(p.fn.size, bool) if use is None else use
    hist = []
    best = None
    w = 1.0
    for it in range(iterations + 1):
        if resolve is not None:
            resolve()
        x = build()
        D, M = fds_time(x, fs, p)
        meas = M if measure == "mrs" else D
        ratio = D_target / np.maximum(meas, 1e-300)
        e = np.abs(1.0 / ratio[use] - 1.0)
        err = float(np.max(e))
        # acceptance score: one stubborn oscillator must not block the
        # progress of all the others, so the 90th percentile also counts
        score = 0.5 * err + 0.5 * float(np.quantile(e, 0.9))
        hist.append(err)
        if log:
            log(f"  closed-loop iteration {it}: max |{measure.upper()} error| = {100 * err:.1f} %")
        if best is None or score < best[6]:
            best = (err, x, D, M, _snapshot(state_objs), ratio, score)
            w = min(1.6, w * 1.25) if it > 0 else w
        else:
            _restore(state_objs, best[4])
            ratio = best[5]
            w *= 0.5
        if best[0] <= tol or it == iterations or w < 1 / 16:
            break                                   # converged, budget used, or stagnating
        expo = 1.0 if measure == "mrs" else 1.0 / p.b       # MRS ~ amplitude, FDS ~ amplitude^b
        gain = np.clip(np.where(use, ratio, 1.0) ** (w * expo), 1 / 1.5, 1.5)
        correct(gain)
    err, x, D, M, snap, _, _ = best
    _restore(state_objs, snap)
    return x, D, M, hist


# --------------------------------------------------------- FDS: Gaussian standard
def _node_corrector(car, W, p):
    """Closed-loop correction acting on the design PSD nodes of a carrier."""
    state = {"G": None}

    def correct(gain):
        car.reshape(p.fn, gain)
        state["G"] = state["G"] * gain ** 2
    return state, correct


def gaussian(D_target, T, fs, p: FDSParams, *, seed=None, correction_iterations=30,
             tol=0.05, max_samples=20_000_000, log=None) -> SynthesisResult:
    """Standard Mission Synthesis: stationary Gaussian signal with the target FDS."""
    rng = np.random.default_rng(seed)
    r, Ts = _repeats(T, fs, max_samples)
    D = np.asarray(D_target, float) / r
    W = response_variance_operator(p.fn, p.Q, fs, p.prefilter)
    inv = invert_fds(D, Ts, p, fs, W=W)
    n = 2 * int(round(fs * Ts / 2))
    car = _Carrier(p.fn, inv.psd, fs, n, rng, q=p.Q)
    st, correct = _node_corrector(car, W, p)
    st["G"] = inv.psd.copy()
    x, Dm, M, hist = _closed_loop(car.signal, p, fs, D, correction_iterations, tol, correct,
                                  [car, st], log=log)
    return SynthesisResult(x, fs, "Gaussian", r, p.fn, st["G"], None,
                           {"fds_measured": Dm * r, "mrs_measured": M, "fds_error_history": hist,
                            "inversion_error": inv.max_rel_error})


# ----------------------------------------------------------------- FDS: kFDS / RF
def _modulated(D_target, T, fs, p, mode, target, *, reference=None, ref_fs=None, segments=None,
               pattern=None, window=None, ramp_fraction=1 / 3, burst_fraction=None,
               seed=None, correction_iterations=30, tol=0.05, max_samples=20_000_000, log=None):
    rng = np.random.default_rng(seed)
    warn = []
    r, Ts = _repeats(T, fs, max_samples)
    D = np.asarray(D_target, float) / r
    W = response_variance_operator(p.fn, p.Q, fs, p.prefilter)
    inv = invert_fds(D, Ts, p, fs, W=W)
    n = 2 * int(round(fs * Ts / 2))
    car = _Carrier(p.fn, inv.psd, fs, n, rng, q=p.Q)
    st, correct = _node_corrector(car, W, p)
    st["G"] = inv.psd.copy()
    if segments is None and reference is not None:
        segments = [(np.asarray(reference, float), ref_fs, 1.0)]
    if pattern is None:
        pattern = "reference" if segments else "bursts"
    td = max(sg[0].size / sg[1] for sg in segments) if segments else Ts
    tau = p.Q / (np.pi * p.fn[0])
    w = window if window else default_window(td, fs, tau)
    nw = max(int(round(w * fs)), 2)
    n_windows = int(np.ceil(n / nw))
    u = _scores(pattern, n_windows, rng, segments, nw / fs, burst_fraction)
    ramp = max(int(round(nw * ramp_fraction)), 1)
    state = {"g": None, "alpha": 0.0, "it": 0}

    def env(alpha):
        e = _expand_levels(np.exp(alpha * u), nw, n, ramp)
        return e / np.mean(e ** p.b) ** (1.0 / p.b)          # <e^b> = 1 keeps the FDS

    def resolve(force=False):
        g = car.signal()
        state["g"] = g
        state["it"] += 1
        if state["it"] > 2 and not force:
            return                     # envelope frozen: only the spectrum is corrected
        if mode == "kurtosis":
            f = lambda a: kurtosis(env(a) * g)
            a, ok = _solve_scalar(f, target, 0.0, 1.0, increasing=True)
            if not ok and f(a) < target - 0.05 and not warn:
                warn.append(f"kurtosis {target:g} not reachable with this envelope "
                            f"pattern; reached {f(a):.2f}")
        else:                                                 # RF: prescribed RMS
            f = lambda a: float(np.sqrt(np.mean((env(a) * g) ** 2)))
            if f(0.0) <= target:
                a = 0.0
                if not warn:
                    warn.append("RF: the requested RMS is not lower than the RMS of the Gaussian "
                                "profile with the same FDS, so a Gaussian profile is returned")
            else:
                a, _ = _solve_scalar(f, target, 0.0, 1.0, increasing=False)
        state["alpha"] = a

    def build():
        return env(state["alpha"]) * state["g"]

    x, Dm, M, hist = _closed_loop(build, p, fs, D, correction_iterations, tol, correct,
                                  [car, st, state], resolve=resolve, log=log)
    # final check of the envelope constraint on the returned realisation
    off = (abs(kurtosis(x) - target) > 0.1) if mode == "kurtosis" else \
        (abs(np.sqrt(np.mean(x ** 2)) / target - 1) > 0.02 and state["alpha"] > 0)
    if off:
        # re-impose the envelope constraint, then a few more spectral corrections
        resolve(force=True)
        x, Dm, M, h2 = _closed_loop(build, p, fs, D, 12, tol, correct, [car, st, state],
                                    resolve=resolve, log=log)
        hist += h2
    state["g"] = car.signal()
    e = env(state["alpha"])
    name = "kFDS" if mode == "kurtosis" else "RF"
    return SynthesisResult(x, fs, name, r, p.fn, st["G"], e,
                           {"fds_measured": Dm * r, "mrs_measured": M, "fds_error_history": hist,
                            "alpha": state["alpha"], "window_s": nw / fs, "n_windows": n_windows,
                            "pattern": pattern, "inversion_error": inv.max_rel_error,
                            "predicted_kurtosis": float(3 * np.mean(e ** 4) / np.mean(e ** 2) ** 2)},
                           warn)


def kfds(D_target, T, fs, p, kurtosis_target, **kw) -> SynthesisResult:
    """kFDS: prescribed FDS and kurtosis (dissertation 2.2.1)."""
    if kurtosis_target <= 3.0:
        res = gaussian(D_target, T, fs, p, **{k: v for k, v in kw.items()
                                             if k in ("seed", "correction_iterations", "tol", "max_samples", "log")})
        res.warnings.append("kurtosis <= 3: platykurtic signals cannot be synthesised by kFDS; "
                            "a Gaussian signal is returned")
        return res
    return _modulated(D_target, T, fs, p, "kurtosis", kurtosis_target, **kw)


def rf(D_target, T, fs, p, rms_target, **kw) -> SynthesisResult:
    """RF: prescribed FDS and RMS for any test duration (dissertation 2.2.2)."""
    return _modulated(D_target, T, fs, p, "rms", rms_target, **kw)


# ------------------------------------------------------------------ FDS: PF / PSF
def pf(D_target, T, fs, p, ref_f, ref_psd, *, burst_fraction=1 / 20, window=None,
       seed=None, correction_iterations=30, tol=0.05, max_samples=20_000_000, log=None,
       _name="PF", _keep_psd=True) -> SynthesisResult:
    """PF: prescribed FDS, PSD shape and RMS (dissertation 2.2.3).

    High-energy bursts (PSD G_B, total time t_B) and a low-energy part
    (G_L, time t_L) with t_B G_B + t_L G_L = T G_ref.  G_B is solved so that
    the damage of BOTH parts adds up to the target.  t_B starts at T/20 and
    is halved while G_L would be negative on more than 10 % of the
    significant band, with a lower bound.
    Where a Gaussian process with the reference PSD already exceeds the
    target damage (tonal components: a sine is less damaging than Gaussian
    noise of equal power), PSD and FDS cannot both be met; those frequencies
    are reported and excluded from the closed-loop criterion.
    """
    rng = np.random.default_rng(seed)
    warn = []
    r, Ts = _repeats(T, fs, max_samples)
    D = np.asarray(D_target, float) / r
    n = 2 * int(round(fs * Ts / 2))
    W = response_variance_operator(p.fn, p.Q, fs, p.prefilter)
    grid = np.arange(n // 2 + 1) * fs / n
    Gref_fine = np.maximum(np.interp(grid, ref_f, ref_psd, left=0.0, right=0.0), 0.0)
    Gref_nodes = np.maximum(np.interp(p.fn, ref_f, ref_psd), 0.0)
    signif = Gref_fine > Gref_fine.mean() / 1e3
    tau = p.Q / (np.pi * p.fn[0])
    # at least two bursts of 3 decay times of the lowest oscillator: with a
    # single short burst the damage at b = 7 rests on a handful of cycles
    # and the closed loop sees a rugged, seed dependent landscape
    tB_min = min(Ts / 4, max(0.5, 6.0 * tau))
    tB = max(Ts * burst_fraction, tB_min)
    while True:
        tL = Ts - tB
        D_L = D / 20.0
        for _ in range(8):                     # fixed point on the low-energy damage
            DB = np.maximum(D - D_L, 0.05 * D)
            invB = invert_fds(DB, tB, p, fs, W=W)
            GL_nodes = (Gref_nodes * Ts - invB.psd * tB) / tL
            D_L, _ = fds_gaussian(np.maximum(GL_nodes, 0.0), tL, p, W=W)
        GB_fine = interp_psd(p.fn, invB.psd, grid)
        GL_fine = (Gref_fine * Ts - GB_fine * tB) / tL
        neg = (GL_fine < 0) & signif
        if neg.sum() <= 0.1 * signif.sum():
            break
        if tB / 2 < max(0.25, 0.5 * tau):
            warn.append("PF: PSD, RMS and FDS cannot all be met at this acceleration "
                        "(the low-energy part would need a negative PSD); the overall PSD will "
                        "exceed the reference. Use a longer test, PSF or kFDS.")
            break
        tB /= 2.0
        if tB < tB_min and not any("quasi-static" in w_ for w_ in warn):
            warn.append(f"PF: bursts shorter than 3 decay times of the {p.fn[0]:g} Hz oscillator "
                        "(needed to keep G_L >= 0); low-frequency damage is not quasi-static and "
                        "converges less precisely.")
    if (GL_fine < 0).any():
        pos = GL_fine[GL_fine > 0]
        GL_fine = np.where(GL_fine < 0, 1e-3 * (pos.min() if pos.size else 0.0), GL_fine)
    # the inversion can leave burst PSD lines at exactly zero; the closed loop
    # corrects multiplicatively, so give them a small fraction of the
    # reference PSD to start from (negligible energy: tB << T)
    GB_fine = np.where(signif, np.maximum(GB_fine, 1e-2 * Gref_fine), GB_fine)
    carB = _Carrier(None, None, fs, n, rng, psd_fine=GB_fine)
    carL = _Carrier(None, None, fs, n, rng, psd_fine=GL_fine)
    # switching mask: bursts of length wB scattered at random
    if window is None:
        wB = float(np.clip(tB / rng.integers(11, 21), 0.5, 6.0))
        wB = max(min(wB, 8000 / fs), min(3.0 * tau, tB))
    else:
        wB = window
    nw = max(int(round(wB * fs)), 2)
    n_windows = int(np.ceil(n / nw))
    n_bursts = int(np.clip(round(tB / (nw / fs)), 1, n_windows - 1))
    levels = np.zeros(n_windows)
    levels[rng.choice(n_windows, n_bursts, replace=False)] = 1.0
    m = np.clip(_expand_levels(levels, nw, n, max(nw // 5, 1)), 0.0, 1.0)
    tB_real = float(m.mean() * Ts)
    st = {"G": invB.psd.copy()}

    def rebalance():
        GLn = (Gref_fine * Ts - carB.G * tB_real) / max(Ts - tB_real, 1e-9)
        pos = GLn[GLn > 0]
        carL.G = np.where(GLn > 0, GLn, 1e-3 * (pos.min() if pos.size else 0.0))
        carL.G[0] = carL.G[-1] = 0.0

    rebalance()

    def build():
        return np.sqrt(m) * carB.signal() + np.sqrt(1.0 - m) * carL.signal()

    def correct(gain):
        carB.reshape(p.fn, gain)
        st["G"] = st["G"] * gain ** 2
        if mode["keep_psd"]:
            # re-balance the low-energy part: the overall PSD (and RMS) stays
            # exactly the reference one
            rebalance()
        else:
            # same gain on both parts (always monotone); the PSD deviates
            # locally by gain^2 (reported by `evaluate`)
            carL.reshape(p.fn, gain)

    mode = {"keep_psd": True}
    x, Dm, M, hist = _closed_loop(build, p, fs, D, correction_iterations, tol, correct,
                                  [carB, carL, st], log=log)
    if hist and min(hist) > 2 * tol and not _keep_psd:
        # where the low-energy part dominates the damage, re-balancing moves the
        # damage the wrong way and the loop stalls; PSF only prescribes the PSD
        # shape, so continue with the same gain on both parts
        mode["keep_psd"] = False
        if log:
            log("  switching to joint correction of both parts (PSD shape relaxed locally)")
        x, Dm, M, h2 = _closed_loop(build, p, fs, D, correction_iterations, tol, correct,
                                    [carB, carL, st], log=log)
        hist += h2
        if min(hist) > 2 * tol:
            warn.append(f"{_name}: FDS error {100 * min(hist):.0f} % after the corrections. The test is "
                        "probably too short for bursts of a few decay times of the lowest oscillator; "
                        "try another seed, a longer test or kFDS.")
    elif hist and min(hist) > 2 * tol:
        warn.append(f"PF: FDS error {100 * min(hist):.0f} % after the corrections. PSD, RMS and FDS "
                    "are not all reachable here (the low-energy part, which carries the reference "
                    "PSD, controls the damage at some frequencies). PSF or kFDS will converge.")
    # over-tested nodes whose burst PSD was driven to (almost) zero: there the
    # low-energy part alone, which carries the reference PSD, exceeds the
    # target damage (typically tonal content: a sine is less damaging than
    # Gaussian noise of the same power), so PSD and FDS cannot both be met
    over = Dm / D > 1 + 2 * tol
    weak = st["G"] < 0.05 * invB.psd
    conflict = over & weak
    if conflict.any():
        warn.append("PSD and FDS cannot both be met near "
                    + ", ".join(f"{f:.0f}" for f in p.fn[conflict]) + " Hz: the part of the signal "
                    "that carries the reference PSD already exceeds the reference damage there "
                    "(tonal content); these frequencies are over-tested")
    return SynthesisResult(x, fs, _name, r, p.fn, st["G"], m,
                           {"fds_measured": Dm * r, "mrs_measured": M, "fds_error_history": hist,
                            "t_burst_s": tB_real, "burst_window_s": nw / fs, "n_bursts": n_bursts,
                            "conflict_frequencies": p.fn[conflict],
                            "psd_burst_fine": carB.G, "psd_low_fine": carL.G, "grid": grid},
                           warn)


def psf(D_target, T, fs, p, ref_f, ref_psd, life, **kw) -> SynthesisResult:
    """PSF: prescribed FDS and PSD shape (dissertation 2.2.4).

    PF is run with the damage the reference produces in the test duration
    (target / r, r = life/T) and the signal is then amplified by r^(1/b).
    """
    rr = life / T
    res = pf(np.asarray(D_target) / rr, T, fs, p, ref_f, ref_psd, _name="PSF", _keep_psd=False, **kw)
    res.x = res.x * rr ** (1.0 / p.b)
    res.info["fds_measured"] = res.info["fds_measured"] * rr
    res.info["mrs_measured"] = res.info["mrs_measured"] * rr ** (1.0 / p.b)
    res.info["reduction_factor"] = rr
    return res


# ------------------------------------------------------ kurtosis control (no FDS)
def mlv(ref_f, ref_psd, T, fs, kurtosis_target, *, reference=None, ref_fs=None, segments=None, pattern=None,
        window=None, ramp_fraction=1 / 3, burst_fraction=None, seed=None, log=None) -> SynthesisResult:
    """Multi-level variance: modulated Gaussian with the reference PSD and kurtosis."""
    rng = np.random.default_rng(seed)
    n = 2 * int(round(fs * T / 2))
    grid = np.arange(n // 2 + 1) * fs / n
    car = _Carrier(None, None, fs, n, rng,
                   psd_fine=np.maximum(np.interp(grid, ref_f, ref_psd, left=0, right=0), 0))
    g = car.signal()
    if segments is None and reference is not None:
        segments = [(np.asarray(reference, float), ref_fs, 1.0)]
    if pattern is None:
        pattern = "reference" if segments else "bursts"
    td = max(sg[0].size / sg[1] for sg in segments) if segments else T
    w = window if window else default_window(td, fs)
    nw = max(int(round(w * fs)), 2)
    n_windows = int(np.ceil(n / nw))
    u = _scores(pattern, n_windows, rng, segments, nw / fs, burst_fraction)
    ramp = max(int(round(nw * ramp_fraction)), 1)

    def env(a):
        e = _expand_levels(np.exp(a * u), nw, n, ramp)
        return e / np.sqrt(np.mean(e ** 2))                  # keeps PSD and RMS

    a, ok = _solve_scalar(lambda a: kurtosis(env(a) * g), kurtosis_target)
    e = env(a)
    x = e * g
    warn = [] if ok or kurtosis_target <= 3 else [f"kurtosis {kurtosis_target:g} not reached ({kurtosis(x):.2f})"]
    return SynthesisResult(x, fs, "MLV", 1, None, None, e,
                           {"alpha": a, "window_s": nw / fs, "pattern": pattern}, warn)


def vsd(ref_f, ref_psd, T, fs, kurtosis_target, *, n_bands=6, fmin=None, fmax=None, window=None,
        ramp_fraction=1 / 3, burst_fraction=None, seed=None, log=None) -> SynthesisResult:
    """Variable spectral density: the PSD changes in SHAPE over time.

    The band is split in `n_bands` log-spaced sub-bands, each one a Gaussian
    carrier with its own independent burst envelope (<e^2> = 1).  The time
    average PSD is the reference PSD; at a given instant some sub-bands
    dominate, which reproduces the resonance effect discussed in 3.1.2.
    """
    rng = np.random.default_rng(seed)
    n = 2 * int(round(fs * T / 2))
    grid = np.arange(n // 2 + 1) * fs / n
    G = np.maximum(np.interp(grid, ref_f, ref_psd, left=0, right=0), 0)
    lo = fmin or max(grid[1], ref_f[ref_f > 0][0])
    hi = fmax or grid[-1]
    edges = np.geomspace(lo, hi, n_bands + 1)
    edges[0], edges[-1] = 0.0, grid[-1] + 1
    w = window if window else default_window(T, fs)
    nw = max(int(round(w * fs)), 2)
    n_windows = int(np.ceil(n / nw))
    ramp = max(int(round(nw * ramp_fraction)), 1)
    carriers, scores = [], []
    for k in range(n_bands):
        mask = (grid >= edges[k]) & (grid < edges[k + 1])
        if not np.any(G[mask] > 0):
            continue
        c = _Carrier(None, None, fs, n, rng, psd_fine=G * mask)
        carriers.append(c.signal())
        scores.append(_scores("bursts", n_windows, rng, burst_fraction=burst_fraction))

    def build(a):
        x = np.zeros(n)
        for gk, uk in zip(carriers, scores):
            e = _expand_levels(np.exp(a * uk), nw, n, ramp)
            x += gk * e / np.sqrt(np.mean(e ** 2))
        return x

    a, ok = _solve_scalar(lambda a: kurtosis(build(a)), kurtosis_target)
    x = build(a)
    warn = [] if ok or kurtosis_target <= 3 else [f"kurtosis {kurtosis_target:g} not reached ({kurtosis(x):.2f})"]
    return SynthesisResult(x, fs, "VSD", 1, None, None, None,
                           {"alpha": a, "n_bands": len(carriers), "band_edges": edges}, warn)


def pm(ref_f, ref_psd, T, fs, kurtosis_target, *, band=None, block=None, peak_fraction=None,
       seed=None, log=None) -> SynthesisResult:
    """Phase manipulation: Gaussian amplitudes, partially aligned phases.

    In a fraction of the blocks the phases of the harmonics in `band` are
    pulled towards -2*pi*f*t0 (constructive interference at a random t0);
    the degree of alignment lam in [0, 1] is solved for the kurtosis.
    Blocks are joined with a short equal-power cross-fade.
    """
    rng = np.random.default_rng(seed)
    blk = block if block else default_window(T, fs)
    nb = 2 * int(round(fs * blk / 2))
    n_blocks = max(int(np.ceil(T * fs / nb)), 1)
    frac = peak_fraction if peak_fraction else rng.uniform(0.1, 0.2)
    peaks = np.zeros(n_blocks, bool)
    peaks[rng.choice(n_blocks, max(1, int(np.ceil(frac * n_blocks))), replace=False)] = True
    nl = nb // 2
    grid = np.arange(nl + 1) * fs / nb
    G = np.maximum(np.interp(grid, ref_f, ref_psd, left=0, right=0), 0)
    G[0] = G[-1] = 0
    amp = np.sqrt(2 * G * fs / nb)
    lo, hi = band if band else (0.0, fs / 2)
    sel = (grid >= lo) & (grid <= hi)
    psi = rng.uniform(-np.pi, np.pi, (n_blocks, nl + 1))
    t0 = rng.uniform(0.2, 0.8, n_blocks) * blk
    ov = max(int(0.02 * nb), 4)
    th = (np.arange(ov) + 0.5) / ov * np.pi / 2

    def build(lam):
        out = np.zeros(n_blocks * nb - (n_blocks - 1) * ov)
        pos = 0
        for j in range(n_blocks):
            ph = psi[j].copy()
            if peaks[j]:
                ph[sel] = -2 * np.pi * grid[sel] * t0[j] + (1 - lam) * psi[j, sel]
            xb = np.fft.irfft(amp * np.exp(1j * ph) * nl, nb)
            if j == 0:
                out[:nb] = xb
                pos = nb
            else:
                s = pos - ov
                out[s:pos] = out[s:pos] * np.cos(th) + xb[:ov] * np.sin(th)
                out[pos:pos + nb - ov] = xb[ov:]
                pos += nb - ov
        return out[: int(round(T * fs))]

    lam, ok = _solve_scalar(lambda l: kurtosis(build(l)), kurtosis_target, 0.0, 1.0, hi_max=1.0)
    x = build(lam)
    warn = [] if ok or kurtosis_target <= 3 else [
        f"kurtosis {kurtosis_target:g} not reachable by phase alignment alone ({kurtosis(x):.2f}); "
        "increase peak_fraction or use MLV / VSD"]
    return SynthesisResult(x, fs, "PM", 1, None, None, None,
                           {"lambda": lam, "block_s": nb / fs, "n_peak_blocks": int(peaks.sum())}, warn)


# ------------------------------------------------------------------ MRS control
def mrs_control(M_target, T, fs, p, *, seed=None, correction_iterations=20, tol=0.05,
                max_samples=20_000_000, log=None) -> SynthesisResult:
    """Gaussian profile whose Maximum Response Spectrum matches a target.

    The Gaussian MRS is (2 pi fn)^2 * rms_z * sqrt(2 ln(fn T)) (Rice), so the
    target fixes rms_z^2 = W G, a linear positive problem as for the FDS.
    The measured (time-domain) MRS is then corrected in closed loop.
    """
    from .design import solve_variance
    rng = np.random.default_rng(seed)
    r, Ts = _repeats(T, fs, max_samples)
    W = response_variance_operator(p.fn, p.Q, fs, p.prefilter)
    zr = np.asarray(M_target, float) / ((2 * np.pi * p.fn) ** 2 *
                                        np.sqrt(2 * np.log(np.maximum(p.fn * Ts, 1.0001))))
    G = solve_variance(W, zr ** 2)
    n = 2 * int(round(fs * Ts / 2))
    car = _Carrier(p.fn, G, fs, n, rng, q=p.Q)
    st, correct = _node_corrector(car, W, p)
    st["G"] = G.copy()
    x, Dm, M, hist = _closed_loop(car.signal, p, fs, np.asarray(M_target, float), correction_iterations,
                                  tol, correct, [car, st], log=log, measure="mrs")
    return SynthesisResult(x, fs, "MRS", r, p.fn, st["G"], None,
                           {"fds_measured": Dm * r, "mrs_measured": M, "mrs_error_history": hist,
                            "fds_error_history": hist})


# ------------------------------------------------------------------ sine on random
def detect_tones(f, psd, fmin, fmax, *, prominence_db=10.0, max_tones=8):
    """Tonal lines of a PSD: peaks standing `prominence_db` above a median baseline.

    Returns [(frequency, peak amplitude)] with A = sqrt(2 * power in the peak).
    """
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks
    f = np.asarray(f, float)
    P = np.maximum(np.asarray(psd, float), 1e-300)
    df = f[1] - f[0]
    width = max(int(round(5.0 / df)), 9) | 1
    base = median_filter(P, size=width, mode="nearest")
    db = 10 * np.log10(P / base)
    band = (f >= fmin) & (f <= fmax)
    pk, prop = find_peaks(np.where(band, db, -99), height=prominence_db, distance=max(int(2 / df), 1))
    pk = pk[np.argsort(-prop["peak_heights"])][:max_tones]
    tones = []
    for i in sorted(pk):
        lo, hi = i, i
        while lo > 0 and db[lo - 1] > 3:
            lo -= 1
        while hi < P.size - 1 and db[hi + 1] > 3:
            hi += 1
        power = float(np.sum(P[lo:hi + 1] - base[lo:hi + 1]) * df)
        if power > 0:
            tones.append((float(f[i]), float(np.sqrt(2 * power))))
    return tones


def sine_on_random(D_target, T, fs, p, tones, life, *, seed=None, drift=0.0, correction_iterations=30,
                   tol=0.05, max_samples=20_000_000, log=None) -> SynthesisResult:
    """Sine-on-random: deterministic tones plus a Gaussian random part.

    The tones (frequency, peak amplitude in the field) are amplified by
    (life/T)^(1/b), so they keep their share of the damage in the shortened
    test; the random part is designed for the remaining FDS and corrected in
    closed loop (the tones are kept fixed).  `drift` (e.g. 0.01) lets each
    tone wander slowly by that relative amount, as rotating machinery does.
    The tones are detected in the reference PSD or given explicitly.
    """
    rng = np.random.default_rng(seed)
    warn = []
    r, Ts = _repeats(T, fs, max_samples)
    D = np.asarray(D_target, float) / r
    n = 2 * int(round(fs * Ts / 2))
    t = np.arange(n) / fs
    amp_scale = (life / T) ** (1.0 / p.b)
    xs = np.zeros(n)
    for f0, a0 in tones:
        if drift:
            wander = np.cumsum(rng.normal(size=n)) / np.sqrt(n)
            wander = drift * wander / max(np.max(np.abs(wander)), 1e-12)
            phase = 2 * np.pi * np.cumsum(f0 * (1 + wander)) / fs
        else:
            phase = 2 * np.pi * f0 * t
        xs += a0 * amp_scale * np.sin(phase + rng.uniform(0, 2 * np.pi))
    Ds, _ = fds_time(xs, fs, p)
    over = Ds > 0.95 * D
    if over.any():
        warn.append("the tones alone exceed the target damage near "
                    + ", ".join(f"{f:.0f}" for f in p.fn[over]) + " Hz")
    W = response_variance_operator(p.fn, p.Q, fs, p.prefilter)
    inv = invert_fds(np.maximum(D - Ds, 0.05 * D), Ts, p, fs, W=W)
    car = _Carrier(p.fn, inv.psd, fs, n, rng, q=p.Q)
    st, correct = _node_corrector(car, W, p)
    st["G"] = inv.psd.copy()
    x, Dm, M, hist = _closed_loop(lambda: xs + car.signal(), p, fs, D, correction_iterations, tol,
                                  correct, [car, st], use=~over, log=log)
    return SynthesisResult(x, fs, "SoR", r, p.fn, st["G"], None,
                           {"fds_measured": Dm * r, "mrs_measured": M, "fds_error_history": hist,
                            "tones": [(f0, a0 * amp_scale) for f0, a0 in tones],
                            "tone_amplification": amp_scale, "fds_tones": Ds * r}, warn)


METHODS = {
    "gaussian": "Standard Mission Synthesis (Gaussian, prescribed FDS)",
    "kfds": "kFDS: prescribed FDS + kurtosis",
    "rf": "RF: prescribed FDS + RMS",
    "pf": "PF: prescribed FDS + PSD shape + RMS",
    "psf": "PSF: prescribed FDS + PSD shape",
    "mlv": "Kurtosis control, multi-level variance (PSD + kurtosis)",
    "vsd": "Kurtosis control, variable spectral density (PSD + kurtosis)",
    "pm": "Kurtosis control, phase manipulation (PSD + kurtosis)",
    "mrs": "MRS control: Gaussian with prescribed Maximum Response Spectrum",
    "sor": "Sine on random: tonal lines + random part, prescribed FDS",
}
