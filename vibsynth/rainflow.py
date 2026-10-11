"""Rainflow counting.

`rainflow_damage_sum` is a vectorised (NumPy only) four-point rainflow with the
residue counted as half cycles. It returns sum(n_i * amp_i**b), amp = range/2,
which is all the FDS needs. It is equivalent to the ASTM E1049 three-point
algorithm (checked in tests against `rainflow_astm`, a plain sequential
reference implementation).
"""
from __future__ import annotations

import numpy as np

__all__ = ["turning_points", "rainflow_damage_sum", "rainflow_astm", "rainflow_cycles"]


def turning_points(y: np.ndarray) -> np.ndarray:
    """Strict local extrema (plus the end points); plateaus are collapsed."""
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return y.copy()
    d = np.diff(y)
    nz = d != 0
    if not nz.any():
        return y[[0, -1]]
    # drop flat steps, then keep points where the slope sign changes
    keep = np.r_[True, nz]
    y = y[keep]
    d = np.diff(y)
    s = np.sign(d)
    idx = np.nonzero(s[1:] != s[:-1])[0] + 1
    return np.r_[y[0], y[idx], y[-1]]


def _sequential_4point(s: np.ndarray, b: float) -> float:
    """Linear-time stack version of the 4-point method (pure Python)."""
    total = 0.0
    st = []
    for v in s.tolist():
        st.append(v)
        while len(st) >= 4:
            r2 = abs(st[-2] - st[-3])
            if r2 <= abs(st[-3] - st[-4]) and r2 <= abs(st[-1] - st[-2]):
                total += (0.5 * r2) ** b
                del st[-3:-1]
            else:
                break
    if len(st) >= 2:
        r = np.abs(np.diff(np.asarray(st)))
        total += 0.5 * float(np.sum((0.5 * r) ** b))
    return total


def rainflow_damage_sum(y: np.ndarray, b: float, *, is_turning: bool = False) -> float:
    """sum over rainflow cycles of count * (range/2)**b.

    Vectorised passes of the 4-point method remove all innermost cycles at
    once. Long monotone decays (free ringing of an oscillator after a burst)
    release only one cycle per pass, so when a pass removes few cycles the
    remaining, already much shorter sequence is finished with the linear
    stack algorithm.
    """
    s = np.asarray(y, float) if is_turning else turning_points(y)
    total = 0.0
    while s.size >= 4:
        r = np.abs(np.diff(s))
        c = (r[1:-1] <= r[:-2]) & (r[1:-1] <= r[2:])
        if not c.any():
            break
        c[1:] &= ~c[:-1]          # no two overlapping pairs in the same pass
        i = np.nonzero(c)[0] + 1  # pair (s[i], s[i+1]) is a closed cycle
        if i.size < max(16, 0.002 * s.size):
            return total + _sequential_4point(s, b)
        total += float(np.sum((0.5 * r[i]) ** b))
        keep = np.ones(s.size, bool)
        keep[i] = False
        keep[i + 1] = False
        s = s[keep]
    if s.size >= 2:
        total += 0.5 * float(np.sum((0.5 * np.abs(np.diff(s))) ** b))
    return total


def rainflow_cycles(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Amplitudes and counts (1 or 0.5) of all rainflow cycles."""
    s = turning_points(y)
    amps, counts = [], []
    while s.size >= 4:
        r = np.abs(np.diff(s))
        c = (r[1:-1] <= r[:-2]) & (r[1:-1] <= r[2:])
        if not c.any():
            break
        c[1:] &= ~c[:-1]
        i = np.nonzero(c)[0] + 1
        amps.append(0.5 * r[i])
        counts.append(np.ones(i.size))
        keep = np.ones(s.size, bool)
        keep[i] = False
        keep[i + 1] = False
        s = s[keep]
    if s.size >= 2:
        a = 0.5 * np.abs(np.diff(s))
        amps.append(a)
        counts.append(np.full(a.size, 0.5))
    if not amps:
        return np.zeros(0), np.zeros(0)
    a = np.concatenate(amps)
    n = np.concatenate(counts)
    m = a > 0
    return a[m], n[m]


def rainflow_astm(y: np.ndarray) -> np.ndarray:
    """Plain sequential ASTM E1049 rainflow (slow, for validation).

    Returns an (m, 2) array [amplitude, count].
    """
    y = np.asarray(y, float)
    # turning points
    n = y.size
    a = [y[0]]
    slope1 = y[1] - y[0]
    for i in range(1, n - 1):
        slope2 = y[i + 1] - y[i]
        if slope1 * slope2 <= 0:
            a.append(y[i])
        slope1 = slope2
    a.append(y[-1])
    a = np.array(a)
    msa = a.size
    # three-point rainflow
    out = []
    j = 1          # index of the last stored point
    pr = 2
    for _ in range(2, msa):
        j += 1
        a[j] = a[pr]
        pr += 1
        while j >= 2 and abs(a[j - 1] - a[j - 2]) <= abs(a[j] - a[j - 1]):
            ampl = abs((a[j - 1] - a[j - 2]) / 2.0)
            if j == 2:
                a[0] = a[1]
                a[1] = a[2]
                j = 1
                if ampl > 0:
                    out.append((ampl, 0.5))
                break
            a[j - 2] = a[j]
            j -= 2
            if ampl > 0:
                out.append((ampl, 1.0))
    for k in range(0, j):
        ampl = abs(a[k] - a[k + 1]) / 2.0
        if ampl > 0:
            out.append((ampl, 0.5))
    return np.array(out) if out else np.zeros((0, 2))
