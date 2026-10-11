"""FDS -> PSD inversion (Gaussian "standard" Mission Synthesis).

Lalanne's FDS of a Gaussian process is
    D(fn) = K^b/C * fn * T * (2 * s(fn))^(b/2) * Gamma(1 + b/2),
where s = W @ G is the relative-displacement variance, LINEAR in the PSD G.
So the inversion is a non-negative linear problem W G = s_target, solved
here with a multiplicative (positivity-preserving) fixed-point iteration to a
tight tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls
from scipy.special import gamma

from .fds import FDSParams, fds_gaussian, response_variance_operator

__all__ = ["InversionResult", "target_variance", "invert_fds", "solve_variance"]


@dataclass
class InversionResult:
    fn: np.ndarray
    psd: np.ndarray              # design PSD at the nodes fn
    max_rel_error: float         # max |FDS(psd)/target - 1|
    iterations: int
    W: np.ndarray


def target_variance(D: np.ndarray, T: float, p: FDSParams) -> np.ndarray:
    """Relative-displacement variance that produces damage D in duration T."""
    D = np.maximum(np.asarray(D, float), 1e-300)
    base = D * p.C / (p.K ** p.b * p.fn * T * gamma(1.0 + p.b / 2.0))
    return 0.5 * base ** (2.0 / p.b)


def invert_fds(D: np.ndarray, T: float, p: FDSParams, fs: float | None = None,
               tol: float = 1e-4, max_iter: int = 2000, W: np.ndarray | None = None) -> InversionResult:
    """PSD (at the nodes p.fn) whose Gaussian FDS over duration T equals D."""
    if W is None:
        W = response_variance_operator(p.fn, p.Q, fs, p.prefilter if fs else False)
    s = target_variance(D, T, p)
    G = s / np.maximum(np.diag(W), 1e-300)            # diagonal (white-noise-like) start
    it = 0
    err = np.inf
    for it in range(1, max_iter + 1):
        r = W @ G
        ratio = s / np.maximum(r, 1e-300)
        err = float(np.max(np.abs(ratio - 1.0)))
        if err < tol:
            break
        G = G * ratio
    # polish with weighted NNLS when the fixed point stalls (rare)
    rel = np.abs((W @ G) / s - 1.0)
    if rel.max() > 10 * tol:
        Ws = W / s[:, None]
        G2, _ = nnls(Ws, np.ones_like(s), maxiter=50 * s.size)
        if np.max(np.abs((W @ G2) / s - 1.0)) < rel.max():
            G = G2
    Dg, _ = fds_gaussian(G, T, p, W=W)
    max_err = float(np.max(np.abs(Dg / np.maximum(D, 1e-300) - 1.0)))
    return InversionResult(p.fn.copy(), G, max_err, it, W)


def solve_variance(W: np.ndarray, s: np.ndarray, tol: float = 1e-4, max_iter: int = 2000) -> np.ndarray:
    """Non-negative G with W @ G = s (multiplicative fixed point, NNLS fallback)."""
    s = np.maximum(np.asarray(s, float), 1e-300)
    G = s / np.maximum(np.diag(W), 1e-300)
    for _ in range(max_iter):
        ratio = s / np.maximum(W @ G, 1e-300)
        if np.max(np.abs(ratio - 1.0)) < tol:
            break
        G = G * ratio
    return G
