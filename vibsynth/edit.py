"""Damage-based editing of a record (method ``edit``) and the compression limit.

The idea in one paragraph
-------------------------
The damage of an oscillator is  D_b = sum n_i a_i^b  over the rainflow cycles
of its response: the b-th moment of the cycle-amplitude histogram. A test that
is right for *every* exponent b must keep the large cycles and may only drop
the small ones. On a time history there is exactly one operation that does
this: keep the damaging stretches of the measured record, unchanged, and
discard the quiet ones. How far this can go is measured by the damage Lorenz
curve L(q), the share of the damage carried by the most damaging fraction q
of the time, computed per natural frequency. The compression that keeps a
share `retention` of the damage at every fn is

    alpha_max = 1 / q      with    L_worst(q) = retention.

Up to alpha_max the test is exact for every b (except for the small cycles,
which matter at small b). Beyond it the record has no more quiet time to
drop, and the remaining compression can only come from raising the
amplitude; `edit` does that band by band (a zero-phase gain g(f) iterated
until every fn is back on the target FDS), so that the amplitude works only
where the editing could not keep the damage, and the b-dependence of the
test is confined to those bands.

Steps of the method
-------------------
1. responses z_k(t) of the oscillator family to the (pre-filtered) record;
2. window damage matrix W[k, j]: share of the damage of oscillator k that
   falls in window j (rainflow of each window of the response);
3. ranking of the windows, "common" (sum of the shares over k) or "fair"
   (greedy max-min: each step adds the window that most raises the worst
   served oscillator);
4. Lorenz curves (mean over fn and worst fn) and alpha_max for 80, 90 and
   95 % retention;
5. assembly: windows in ranking order, each extended by a ring-down margin
   (two decay times Q/(pi fn_min) of the slowest oscillator) so that the
   response a burst excites stays with the burst, merged when overlapping,
   until the time budget T_test is used; pieces joined in original order
   with raised-cosine cross-fades;
6. spectral gain (optional): g(f) interpolated in log f between the fn,
   D ~ g^b at each fn, adaptive relaxation, clipped to [1/g_max, g_max];
7. exact FDS, MRS and response kurtosis of the result (one pass).

Reference: Section 12 and 13 of the theory manual.
"""
from __future__ import annotations

import numpy as np

from .fds import FDSParams, apply_prefilter, sdof_relative_displacement, response_spectra
from .rainflow import rainflow_damage_sum
from .signals import kurtosis
from .synthesis import SynthesisResult

__all__ = ["window_damage", "rank_windows", "lorenz_curves", "alpha_for_retention",
           "compressibility", "assemble", "spectral_gain", "edit", "compressed_kurtosis",
           "default_window", "lognormal_lorenz", "lognormal_alpha_max"]


# ----------------------------------------------------------------------------- helpers
def default_window(p: FDSParams) -> float:
    """Window length: at least 2 s and three decay times of the slowest oscillator."""
    tau_d = p.Q / (np.pi * float(np.min(p.fn)))
    return float(np.round(max(2.0, 3.0 * tau_d), 1))


def window_damage(x, fs, p: FDSParams, window_s: float, progress=None):
    """Damage share of every window, per natural frequency.

    Returns (W, n_w): W[k, j] is the share of the damage of oscillator k
    (rainflow, exponent p.b) that falls in window j of n_w samples; every row
    sums to 1. The scale K^b/C cancels in the shares.
    """
    x = np.asarray(x, float)
    y = apply_prefilter(x, fs, p.fn, p.prefilter)
    n_w = int(round(window_s * fs))
    n_win = y.size // n_w
    if n_win < 2:
        raise ValueError("the record holds fewer than two windows: shorten the window")
    W = np.empty((p.fn.size, n_win))
    for k, f in enumerate(p.fn):
        z = sdof_relative_displacement(y, fs, f, p.Q)
        per = np.array([rainflow_damage_sum(z[j * n_w:(j + 1) * n_w], p.b) for j in range(n_win)])
        tot = per.sum()
        W[k] = per / tot if tot > 0 else np.full(n_win, 1.0 / n_win)
        if progress is not None:
            progress((k + 1) / p.fn.size)
    return W, n_w


def rank_windows(W: np.ndarray, ranking: str = "common") -> np.ndarray:
    """Priority order of the windows (most valuable first)."""
    n_fn, n_win = W.shape
    if ranking == "common":
        return np.argsort(W.sum(axis=0), kind="stable")[::-1]
    if ranking != "fair":
        raise ValueError("ranking must be 'common' or 'fair'")
    order, left = [], np.arange(n_win)
    r = np.zeros(n_fn)
    for _ in range(n_win):
        gain = np.min(r[:, None] + W[:, left], axis=0)          # worst fn after adding each candidate
        best = int(np.argmax(gain + 1e-9 * W[:, left].sum(axis=0)))
        k = left[best]
        order.append(k)
        r += W[:, k]
        left = np.delete(left, best)
    return np.array(order)


def lorenz_curves(W: np.ndarray, order: np.ndarray):
    """Damage Lorenz curves: q (time fraction kept), mean over fn, worst fn, per fn."""
    cum = np.cumsum(W[:, order], axis=1)
    q = np.arange(1, W.shape[1] + 1) / W.shape[1]
    return q, cum.mean(axis=0), cum.min(axis=0), cum


def alpha_for_retention(q, L, retention: float) -> float:
    """Largest compression 1/q at which the curve L still keeps `retention`."""
    L = np.asarray(L, float)
    if L[-1] < retention:
        return 1.0
    i = int(np.searchsorted(L, retention))
    return float(1.0 / q[min(i, len(q) - 1)])


def lognormal_lorenz(q, kurtosis_value: float, b: float):
    """Closed form of the Lorenz curve for a lognormal envelope (manual, Section 12.4):
    L(q) = Phi(Phi^-1(q) + b tau / 2), tau = sqrt(ln(kappa / 3))."""
    from scipy.stats import norm
    tau = np.sqrt(np.log(max(float(kurtosis_value), 3.0 + 1e-9) / 3.0))
    return norm.cdf(norm.ppf(np.clip(q, 1e-9, 1 - 1e-9)) + b * tau / 2.0)


def lognormal_alpha_max(retention: float, kurtosis_value: float, b: float) -> float:
    from scipy.stats import norm
    tau = np.sqrt(np.log(max(float(kurtosis_value), 3.0 + 1e-9) / 3.0))
    return float(1.0 / norm.cdf(norm.ppf(retention) - b * tau / 2.0))


def compressibility(x, fs, p: FDSParams, window_s: float | None = None, ranking: str = "common",
                    progress=None) -> dict:
    """Damage Lorenz curves and compression limits of a record (diagnostic).

    Returns q, mean and worst curves, the per-fn curves, alpha_max for 80, 90
    and 95 % retention (mean and worst fn), the window used, and the
    lognormal closed form for the record's kurtosis as a comparison.
    """
    window_s = window_s or default_window(p)
    W, n_w = window_damage(x, fs, p, window_s, progress)
    order = rank_windows(W, ranking)
    q, Lm, Lw, Lfn = lorenz_curves(W, order)
    am = {str(int(r * 100)): {"mean": alpha_for_retention(q, Lm, r), "worst": alpha_for_retention(q, Lw, r)}
          for r in (0.8, 0.9, 0.95)}
    k = kurtosis(np.asarray(x, float))
    return {"q": q, "mean": Lm, "worst": Lw, "per_fn": Lfn, "fn": p.fn, "alpha_max": am,
            "window_s": window_s, "ranking": ranking, "kurtosis": k,
            "lognormal": lognormal_lorenz(q, k, p.b), "n_windows": int(W.shape[1])}


def assemble(x, fs, order, n_w, n_margin, n_target, fade_s):
    """Keep windows in priority order, each extended by the ring-down margin,
    merged when overlapping, until the time budget is used; join the pieces in
    original time order with raised-cosine cross-fades.

    Returns (signal, segments) with segments as [start, stop) sample indices.
    """
    x = np.asarray(x, float)
    n_f = max(2, int(round(fade_s * fs)))
    segs, total = [], 0
    for k in order:
        new = [k * n_w, min(x.size, (k + 1) * n_w + n_margin)]
        merged = []
        for s in segs:
            if s[1] >= new[0] and s[0] <= new[1]:
                new = [min(s[0], new[0]), max(s[1], new[1])]
            else:
                merged.append(s)
        merged.append(new)
        tot = sum(e - s0 for s0, e in merged)
        budget = n_target + n_f * (len(merged) - 1)         # the cross-fades overlap n_f samples per joint
        if tot > budget and segs:
            break
        segs, total = merged, tot
        if total >= budget:
            break
    if total < n_target:                                      # fill with the next windows that still fit
        kept = {k for s0, e in segs for k in range(s0 // n_w, int(np.ceil(e / n_w)))}
        for k in order:
            if k in kept:
                continue
            if total + n_w > n_target + n_f * len(segs):
                break
            segs.append([k * n_w, min(x.size, (k + 1) * n_w)])
            kept.add(k)
            total += n_w
        segs.sort()
        merged = []
        for s in segs:
            if merged and s[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], s[1])
            else:
                merged.append(list(s))
        segs = merged
    segs.sort()
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(n_f) / n_f))
    pieces = [x[s0:e].copy() for s0, e in segs]
    y = pieces[0]
    for pc in pieces[1:]:
        if y.size < n_f or pc.size < n_f:
            y = np.concatenate([y, pc])
            continue
        tail = y[-n_f:] * (1 - ramp) + pc[:n_f] * ramp
        y = np.concatenate([y[:-n_f], tail, pc[n_f:]])
    return y, [[int(s0), int(e)] for s0, e in segs]


def spectral_gain(y, fs, p: FDSParams, D_target, repeats=1, iterations=10, gain_max=2.0, relax=0.8,
                  tol=0.03, step_cap=0.15, log=None):
    """Zero-phase frequency-dependent gain that brings the FDS of y to D_target.

    D ~ g^b at each fn, so the log-gain step is -relax log(D/D_target)/b,
    capped per step; an fn whose error changes sign gets its relaxation
    halved. Two oscillators driven by the same input band (for instance both
    above the last strong peak of the record) cannot be set independently by
    a smooth gain and end in a compromise. Returns (signal, gain_fn, history).
    """
    fn = p.fn
    gain_fn = np.ones(fn.size)
    F = np.fft.rfftfreq(y.size, 1 / fs)
    Y = np.fft.rfft(y)
    out = y.copy()
    relax_k = np.full(fn.size, relax)
    prev = None
    hist = []
    for it in range(1, iterations + 1):
        D = response_spectra(out, fs, p, repeats=repeats, stats=False)["fds"]
        lr = np.log(np.maximum(D, 1e-300) / np.maximum(D_target, 1e-300))
        err = float(np.max(np.abs(lr)))
        hist.append(float(np.max(np.abs(np.exp(lr) - 1))))
        if log:
            log(f"  spectral gain, step {it}: max |log ratio| {err:.3f}, gain "
                f"x{gain_fn.min():.2f}..x{gain_fn.max():.2f}")
        if err < tol:
            break
        if prev is not None:
            relax_k = np.where(np.sign(lr) != np.sign(prev), relax_k * 0.5, relax_k)
        prev = lr
        dlog = np.clip(-relax_k * lr / p.b, -step_cap, step_cap)
        gain_fn = np.clip(gain_fn * np.exp(dlog), 1.0 / gain_max, gain_max)
        g = np.interp(np.log(np.maximum(F, 1e-3)), np.log(fn), gain_fn, left=gain_fn[0], right=gain_fn[-1])
        out = np.fft.irfft(Y * g, n=y.size)
    return out, gain_fn, hist


# ------------------------------------------------------------------------------ method
def edit(ref, duration, *, ranking="common", window_s=None, margin=2.0, fade_s=0.1, gain=True,
         gain_max=2.0, retention=0.9, iterations=10, tol=0.03, log=None, **_ignored) -> SynthesisResult:
    """Method ``edit``: keep the damaging stretches of the target record.

    ref        Reference of a record (or of a result); missions and
               specifications have no time history to edit
    duration   test duration [s]; alpha = record duration / duration
    ranking    "common" (largest average retention) or "fair" (equalises
               the retention over the oscillators)
    window_s   window length [s]; default: max(2 s, 3 decay times of the
               slowest oscillator)
    margin     ring-down margin added to every kept window, in decay times
               Q/(pi fn_min) of the slowest oscillator
    gain       raise the bands whose damage the editing could not keep
    gain_max   cap of the spectral gain (and of 1/gain)
    retention  retention level that defines the compression limit reported
    """
    if ref.x is None:
        raise ValueError("the edit method needs a record (or a result) as target: missions and "
                         "specifications have no time history to edit")
    x = np.asarray(ref.x, float)
    x = x - x.mean()
    fs = float(ref.fs)
    p = ref.params
    T_rec = x.size / fs
    duration = float(duration)
    window_s = float(window_s) if window_s else default_window(p)
    tau_d = p.Q / (np.pi * float(np.min(p.fn)))
    n_margin = int(round(margin * tau_d * fs))
    warnings = []

    if log:
        log(f"edit: {p.fn.size} oscillators, window {window_s:g} s, ring-down margin {n_margin / fs:.2f} s, "
            f"ranking {ranking}")
    W, n_w = window_damage(x, fs, p, window_s)
    order = rank_windows(W, ranking)
    q, Lm, Lw, Lfn = lorenz_curves(W, order)
    alpha_max = {str(int(r * 100)): {"mean": alpha_for_retention(q, Lm, r), "worst": alpha_for_retention(q, Lw, r)}
                 for r in (0.8, 0.9, 0.95)}
    alpha_limit = alpha_for_retention(q, Lw, retention)
    alpha_req = T_rec / duration

    if duration >= T_rec:                                    # nothing to drop
        y, segs = x.copy(), [[0, int(x.size)]]
        if duration > T_rec * 1.001:
            warnings.append(f"the requested duration ({duration:g} s) exceeds the record ({T_rec:.1f} s): "
                            "the record is returned unchanged")
    else:
        y, segs = assemble(x, fs, order, n_w, n_margin, int(round(duration * fs)), fade_s)
    kept_idx = sorted({k for s0, e in segs for k in range(s0 // n_w, min(W.shape[1], int(np.ceil(e / n_w))))})
    kept_share = float(W[:, kept_idx].sum(axis=1).mean())
    kept_min = float(W[:, kept_idx].sum(axis=1).min())
    k_edit = kurtosis(y)
    if log:
        log(f"  kept {len(segs)} stretches, {y.size / fs:.1f} s of {T_rec:.1f} s (alpha {T_rec * fs / y.size:.2f}); "
            f"windows hold {100 * kept_share:.0f} % of the damage (worst fn {100 * kept_min:.0f} %); "
            f"editing alone keeps {retention:.0%} at every fn up to alpha {alpha_limit:.2f}")
    if alpha_req > alpha_limit * 1.02:
        warnings.append(f"requested compression {alpha_req:.2f} exceeds the editing limit {alpha_limit:.2f} "
                        f"({retention:.0%} of the damage at every fn): "
                        + ("the spectral gain raises the bands that lost damage; the test is exact at b = "
                           f"{p.b:g} and b-dependent in those bands" if gain else
                           "part of the damage is missing (spectral gain off)"))

    # life: as for the other methods, a life longer than the record is covered by
    # playing the test `repeats` times (Section 5.5); the gain only makes up for
    # the damage the editing dropped
    repeats = max(1, int(round(ref.repeats))) if ref.repeats else 1
    hist = []
    gain_fn = np.ones(p.fn.size)
    if gain:
        y, gain_fn, hist = spectral_gain(y, fs, p, ref.fds / repeats, repeats=1, iterations=iterations,
                                         gain_max=gain_max, tol=tol, log=log)
        if np.max(gain_fn) >= gain_max * 0.999 or np.min(gain_fn) <= 1.001 / gain_max:
            warnings.append(f"the spectral gain reached its cap x{gain_max:g} at some fn: the target FDS "
                            "could not be met there (band edge or too strong a compression)")
    final = response_spectra(y, fs, p, repeats=repeats, stats=True)

    info = {
        "fds_measured": final["fds"], "mrs_measured": final["mrs"], "k_response": final["kurtosis"],
        "fds_error_history": hist, "fds_target_scale": 1.0,
        "edit": {
            "alpha_requested": alpha_req, "alpha_realised": T_rec * fs / y.size, "alpha_limit": alpha_limit,
            "retention": retention, "alpha_max": alpha_max,
            "lorenz_q": q, "lorenz_mean": Lm, "lorenz_worst": Lw,
            "lorenz_lognormal": lognormal_lorenz(q, ref.stats.get("kurtosis", kurtosis(x)), p.b),
            "segments_s": [[s0 / fs, e / fs] for s0, e in segs], "n_segments": len(segs),
            "window_s": window_s, "margin_s": n_margin / fs, "ranking": ranking,
            "kept_share": kept_share, "kept_share_min": kept_min,
            "kurtosis_edited": k_edit, "kurtosis_record": float(ref.stats.get("kurtosis", kurtosis(x))),
            "gain_fn": gain_fn, "gain_on": bool(gain), "gain_max": gain_max, "gain_iterations": len(hist),
            "repeats": repeats,
        },
    }
    return SynthesisResult(y, fs, "edit", repeats=repeats, info=info, warnings=warnings)


def compressed_kurtosis(ref, duration, *, ranking="common", window_s=None, margin=2.0, fade_s=0.1) -> float:
    """Kurtosis of the target record edited to `duration` (no gain): the kurtosis
    target that a compressed test should carry (manual, Section 14)."""
    res = edit(ref, duration, ranking=ranking, window_s=window_s, margin=margin, fade_s=fade_s, gain=False)
    return float(res.info["edit"]["kurtosis_edited"])
