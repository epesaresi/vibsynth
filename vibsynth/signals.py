"""Signal utilities: frequency grids, PSD estimation, PSD -> time series, statistics.

Conventions follow the PhD work (frequency grid, PSD synthesis, Pearson
kurtosis) unless stated otherwise.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

# np.trapezoid is NumPy >= 2.0; np.trapz before
trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")

__all__ = [
    "natural_frequencies",
    "welch_psd",
    "interp_psd",
    "psd_to_timeseries",
    "psd_variance",
    "statistics",
    "kurtosis",
    "window_rms",
]


def natural_frequencies(fmin: float, fmax: float, scale: str = "octave",
                        points_per_octave: int = 12, resolution: float = 1.0) -> np.ndarray:
    """SDOF natural-frequency grid (1/n octave or linear).

    scale="octave": fn = [fmin, fmin*2^(h/n) for h=1..jmax-1, fmax]
    scale="linear": fn = fmin':resolution:fmax with fmin snapped to the resolution.
    """
    if fmax <= fmin:
        raise ValueError("fmax must be larger than fmin")
    if scale.startswith("oct"):
        fmin = 0.1 if fmin == 0 else float(fmin)
        jmax = int(np.ceil(points_per_octave * np.log2(fmax / fmin)))
        inner = fmin * 2.0 ** (np.arange(1, jmax + 1) / points_per_octave)
        return np.r_[fmin, inner[:-1], float(fmax)]
    if scale.startswith("lin"):
        fmin = fmin - np.remainder(fmin, resolution)
        if fmin == 0:
            fmin = resolution
        n = int(np.floor((fmax - fmin) / resolution + 1e-9)) + 1
        return fmin + resolution * np.arange(n)
    raise ValueError("scale must be 'octave' or 'linear'")


def welch_psd(x: np.ndarray, fs: float, resolution: float | None = None,
              nfft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch PSD with MATLAB pwelch defaults (Hamming, 50 % overlap)."""
    x = np.asarray(x, dtype=float)
    if nfft is None:
        nfft = int(round(fs / (resolution or 1.0)))
    nfft = int(min(max(nfft, 8), x.size))
    f, p = sps.welch(x, fs=fs, window="hamming", nperseg=nfft, noverlap=nfft // 2,
                     nfft=nfft, detrend=False, scaling="density")
    return f, p


def interp_psd(freqs: np.ndarray, psd: np.ndarray, grid: np.ndarray,
               floor: float | None = None) -> np.ndarray:
    """Linear interpolation of a PSD defined on `freqs` onto `grid`.

    Linear (not log-log) interpolation is used on purpose: it is the
    piecewise-linear PSD model assumed by Lalanne's closed-form FDS operator,
    so the realised spectrum is exactly the one the design was computed for.
    Outside the defined band the PSD is set to `floor` (default: min/10).
    """
    freqs = np.asarray(freqs, float)
    psd = np.asarray(psd, float)
    if floor is None:
        pos = psd[psd > 0]
        floor = (pos.min() / 10.0) if pos.size else 0.0
    out = np.interp(grid, freqs, psd, left=floor, right=floor)
    return np.maximum(out, 0.0)


def psd_to_timeseries(freqs: np.ndarray, psd: np.ndarray, fs: float, duration: float,
                      rng: np.random.Generator | int | None = None,
                      phases: np.ndarray | None = None, floor: float | None = None,
                      return_spectrum: bool = False):
    """Gaussian time series with a prescribed one-sided PSD.

    x(t) = sum_k A_k cos(2*pi*k*df*t + phi_k),  A_k = sqrt(2 G(f_k) df),
    uniform random phases, df = 1/duration, N = round(fs*duration) samples.
    The DC and Nyquist lines are set to zero.
    """
    rng = np.random.default_rng(rng)
    n_lines = int(round(fs / 2.0 * duration))
    n = 2 * n_lines
    df = fs / n
    grid = np.arange(n_lines + 1) * df
    g = interp_psd(freqs, psd, grid, floor)
    amp = np.sqrt(2.0 * g * df)
    amp[0] = 0.0
    amp[-1] = 0.0
    if phases is None:
        phases = rng.uniform(-np.pi, np.pi, n_lines + 1)
    spec = amp * np.exp(1j * phases) * n_lines        # irfft scaling, see docstring
    x = np.fft.irfft(spec, n)
    if return_spectrum:
        return x, grid, g
    return x


def psd_variance(freqs: np.ndarray, psd: np.ndarray) -> float:
    return float(trapezoid(psd, freqs))


def kurtosis(x: np.ndarray) -> float:
    """Pearson kurtosis (Gaussian = 3)."""
    x = np.asarray(x, float)
    d = x - x.mean()
    m2 = np.mean(d * d)
    return float(np.mean(d ** 4) / m2 ** 2) if m2 > 0 else float("nan")


def statistics(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    d = x - x.mean()
    m2 = np.mean(d * d)
    rms = float(np.sqrt(np.mean(x * x)))
    return {
        "rms": rms,
        "std": float(np.sqrt(m2)),
        "mean": float(x.mean()),
        "skewness": float(np.mean(d ** 3) / m2 ** 1.5) if m2 > 0 else float("nan"),
        "kurtosis": float(np.mean(d ** 4) / m2 ** 2) if m2 > 0 else float("nan"),
        "crest_factor": float(np.max(np.abs(x)) / rms) if rms > 0 else float("nan"),
        "peak": float(np.max(np.abs(x))),
    }


def window_rms(x: np.ndarray, n_window: int) -> np.ndarray:
    """RMS of consecutive non-overlapping windows (last partial window dropped)."""
    x = np.asarray(x, float)
    m = x.size // n_window
    if m == 0:
        return np.array([np.sqrt(np.mean(x * x))])
    return np.sqrt(np.mean(x[: m * n_window].reshape(m, n_window) ** 2, axis=1))
