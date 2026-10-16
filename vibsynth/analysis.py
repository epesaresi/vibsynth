"""Reference analysis (target definition) and evaluation of a synthesised profile."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import resample_poly

from .fds import FDSParams, fds_fast, fds_gaussian, response_spectra
from .signals import natural_frequencies, statistics, welch_psd

__all__ = ["Reference", "analyze_reference", "evaluate", "response_statistics"]


@dataclass
class Reference:
    """Synthesis target: a measured record, a specification or a mission.

    `x` is the record (None for missions and PSD specifications).
    `segments` lists (signal, fs, weight) used to draw realistic burst levels:
    weight = life represented by one pass of the signal / its duration.
    """
    x: np.ndarray | None
    fs: float
    params: FDSParams
    life: float                     # represented life [s]
    repeats: float                  # life / duration of the record
    fds: np.ndarray                 # target FDS over the life
    mrs: np.ndarray
    psd_f: np.ndarray
    psd: np.ndarray
    stats: dict
    resampled: bool = False
    notes: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    kind: str = "record"
    name: str = ""
    duration_s: float | None = None
    moments: tuple | None = None           # (m2, m4) of the represented process
    response_var: np.ndarray | None = None       # variance of the relative displacement, per fn
    response_kurtosis: np.ndarray | None = None  # kurtosis of the relative displacement, per fn (k_z)

    @property
    def duration(self) -> float:
        if self.x is not None:
            return self.x.size / self.fs
        return float(self.duration_s or self.life)


def analyze_reference(x, fs, *, fmin=5.0, fmax=None, scale="octave", points_per_octave=12,
                      resolution=1.0, Q=10.0, b=8.0, C=1.0, K=1.0, life=None,
                      prefilter="auto", psd_resolution=None, progress=None, fds_method="time",
                      name="") -> Reference:
    """Target FDS / MRS / PSD of a measured record.

    * if fmax > fs/2.5 the record is up-sampled by 1.25 (enough samples per period for the rainflow)
    * life defaults to the record duration (no repetition)
    * repeats = round(life / record duration)
    * fds_method: "time" (exact rainflow), "fast" (Appendix F) or
      "frequency" (Lalanne from the Welch PSD: Gaussian assumption)
    """
    x = np.asarray(x, float).ravel()
    notes = []
    if fmax is None:
        fmax = fs / 2.5
    resampled = False
    if fmax > fs / 2.5 * (1 + 1e-9):
        x = resample_poly(x, 5, 4)
        fs = fs * 1.25
        resampled = True
        notes.append("record up-sampled by 1.25 because fmax > fs/2.5")
    if fmax > fs / 2:
        raise ValueError("fmax must be below the Nyquist frequency")
    fn = natural_frequencies(fmin, fmax, scale, points_per_octave, resolution)
    p = FDSParams(fn=fn, Q=Q, b=b, C=C, K=K, prefilter=prefilter)
    T = x.size / fs
    life = float(life) if life else T
    repeats = max(round(life / T), 1) if life >= T else life / T
    f, P = welch_psd(x, fs, resolution=psd_resolution or (resolution if scale.startswith("lin") else 0.5))
    zv = zk = None
    if fds_method == "fast":
        D, M = fds_fast(x, fs, p, repeats=repeats)
    elif fds_method == "frequency":
        D, M = fds_gaussian(np.interp(fn, f, P), T, p, fs=fs, repeats=repeats)
    else:                                   # one pass: FDS, MRS and response statistics
        rs = response_spectra(x, fs, p, repeats=repeats, progress=progress)
        D, M, zv, zk = rs["fds"], rs["mrs"], rs["var"], rs["kurtosis"]
    st = statistics(x)
    d = x - x.mean()
    return Reference(x, fs, p, life, repeats, D, M, f, P, st, resampled, notes,
                     segments=[(x, fs, life / T)], kind="record", name=name,
                     moments=(float(np.mean(d ** 2)), float(np.mean(d ** 4))),
                     response_var=zv, response_kurtosis=zk)


def response_statistics(ref: Reference, progress=None):
    """Variance and kurtosis of the oscillator responses of a target, per fn.

    Taken from the reference when its analysis already computed them (time
    domain FDS); otherwise measured on its segments (record, generated
    specification signals, mission situations) and combined with their lives:
    var = sum w_i var_i, E z^4 = sum w_i k_i var_i^2, k = E z^4 / var^2.
    The result is stored in the reference.
    """
    if ref.response_kurtosis is not None and ref.response_var is not None:
        return ref.response_var, ref.response_kurtosis
    segs = ref.segments or ([(ref.x, ref.fs, ref.life / ref.duration)] if ref.x is not None else [])
    if not segs:
        raise ValueError("the target has no time history to measure the response statistics on")
    m2 = np.zeros(ref.params.fn.size)
    m4 = np.zeros(ref.params.fn.size)
    wsum = 0.0
    for k, (sx, sfs, weight) in enumerate(segs):
        r = response_spectra(sx, sfs, ref.params, progress=None if progress is None else
                             (lambda f, k=k: progress((k + f) / len(segs))))
        w = float(weight) * sx.size / sfs                 # life represented by the segment
        m2 += w * r["var"]
        m4 += w * r["kurtosis"] * r["var"] ** 2
        wsum += w
    m2, m4 = m2 / wsum, m4 / wsum
    ref.response_var, ref.response_kurtosis = m2, m4 / np.maximum(m2 ** 2, 1e-300)
    return ref.response_var, ref.response_kurtosis


def evaluate(x, fs, ref: Reference, *, repeats=1, fds=None, mrs=None, target_scale=1.0,
             k_response=None) -> dict:
    """Compare a synthesised profile with the reference targets.

    target_scale multiplies the target FDS; kurtosis-control profiles (which
    do not prescribe damage) are compared with the reference damage over the
    same duration (target_scale = duration / life). The kurtosis of the
    oscillator responses (k_z) is compared when the target has it.
    """
    if fds is None or mrs is None:
        rs = response_spectra(x, fs, ref.params, repeats=repeats)
        fds, mrs, k_response = rs["fds"], rs["mrs"], rs["kurtosis"]
    ratio = fds / (ref.fds * target_scale)
    st = statistics(x)
    f, P = welch_psd(x, fs, resolution=ref.psd_f[1] - ref.psd_f[0])
    band = (ref.psd_f >= ref.params.fn[0]) & (ref.psd_f <= ref.params.fn[-1])
    Pi = np.interp(ref.psd_f, f, P)
    psd_db = 10 * np.log10(np.maximum(Pi[band], 1e-300) / np.maximum(ref.psd[band], 1e-300))
    return {
        "stats": st,
        "fds_ratio": ratio,
        "fds_max_error_pct": float(100 * np.max(np.abs(ratio - 1))),
        "fds_median_error_pct": float(100 * np.median(np.abs(ratio - 1))),
        "fds_within_10pct": float(np.mean(np.abs(ratio - 1) <= 0.10)),
        "mrs_ratio": mrs / ref.mrs,
        "psd_f": f, "psd": P,
        "psd_rms_db_dev": float(np.sqrt(np.mean(psd_db ** 2))),
        "duration": x.size / fs * repeats,
        "repeats": repeats,
        "target_scale": target_scale,
        **_kz_metrics(ref, k_response),
    }


def _kz_metrics(ref, k):
    """Response kurtosis of the profile against the target's (when both exist)."""
    kt = ref.response_kurtosis
    if k is None:
        return {"k_response": None, "kz_target": kt}
    k = np.asarray(k, float)
    out = {"k_response": k, "kz_target": kt}
    if kt is not None and np.size(kt) == k.size:
        e = k - kt
        out.update(kz_median_abs_err=float(np.median(np.abs(e))), kz_rms_err=float(np.sqrt(np.mean(e ** 2))))
    return out
