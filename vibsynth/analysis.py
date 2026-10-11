"""Reference analysis (target definition) and evaluation of a synthesised profile."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import resample_poly

from .fds import FDSParams, fds_fast, fds_gaussian, fds_time
from .signals import natural_frequencies, statistics, welch_psd

__all__ = ["Reference", "analyze_reference", "evaluate"]


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
    if fds_method == "fast":
        D, M = fds_fast(x, fs, p, repeats=repeats)
    elif fds_method == "frequency":
        D, M = fds_gaussian(np.interp(fn, f, P), T, p, fs=fs, repeats=repeats)
    else:
        D, M = fds_time(x, fs, p, repeats=repeats, progress=progress)
    st = statistics(x)
    d = x - x.mean()
    return Reference(x, fs, p, life, repeats, D, M, f, P, st, resampled, notes,
                     segments=[(x, fs, life / T)], kind="record", name=name,
                     moments=(float(np.mean(d ** 2)), float(np.mean(d ** 4))))


def evaluate(x, fs, ref: Reference, *, repeats=1, fds=None, mrs=None, target_scale=1.0) -> dict:
    """Compare a synthesised profile with the reference targets.

    target_scale multiplies the target FDS; kurtosis-control profiles (which
    do not prescribe damage) are compared with the reference damage over the
    same duration (target_scale = duration / life).
    """
    if fds is None or mrs is None:
        fds, mrs = fds_time(x, fs, ref.params, repeats=repeats)
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
    }
