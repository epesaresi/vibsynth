"""Loading measured records and saving synthesised profiles."""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["load_signal", "save_result", "list_mat_variables"]


def _from_matrix(a: np.ndarray, fs: float | None, column: int | None):
    a = np.asarray(a, float)
    if a.ndim == 1:
        if not fs:
            raise ValueError("a single column needs the sampling frequency (fs)")
        return a, float(fs)
    if a.shape[0] < a.shape[1]:
        a = a.T
    if a.shape[1] == 1:
        if not fs:
            raise ValueError("a single column needs the sampling frequency (fs)")
        return a[:, 0], float(fs)
    t = a[:, 0]
    dt = np.diff(t)
    if np.all(dt > 0) and np.std(dt) < 1e-3 * np.mean(dt):
        fs_t = 1.0 / np.mean(dt)
        col = column if column is not None else 1
        return a[:, col], float(fs or fs_t)
    col = column if column is not None else 0
    if not fs:
        raise ValueError("no time column found: give fs")
    return a[:, col], float(fs)


def list_mat_variables(path) -> list[str]:
    from scipy.io import whosmat
    try:
        return [w[0] for w in whosmat(str(path))]
    except Exception:
        try:
            import h5py  # MATLAB v7.3
            with h5py.File(path, "r") as h:
                return [k for k in h.keys() if not k.startswith("#")]
        except ImportError as e:
            raise ValueError("MAT v7.3 files need the h5py package (pip install h5py)") from e


def load_signal(path, fs: float | None = None, variable: str | None = None,
                column: int | None = None, units: str = "m/s^2"):
    """Return (x, fs). Accepts .mat (v5/v7, v7.3 with h5py), .csv, .txt, .npy.

    A two-column matrix [time, acceleration] (variable 'Realdata' or any name) gives
    fs from the time vector. `units='g'` converts with 9.81.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".mat":
        try:
            from scipy.io import loadmat
            d = loadmat(str(path))
            keys = [k for k in d if not k.startswith("__")]
            get = lambda k: d[k]
        except NotImplementedError:
            import h5py
            h = h5py.File(path, "r")
            keys = [k for k in h.keys() if not k.startswith("#")]
            get = lambda k: np.array(h[k]).T
        if fs is None:
            for k in keys:
                if k.lower() in ("fs", "fsamp", "samplerate", "sampling_frequency"):
                    fs = float(np.ravel(get(k))[0])
        cand = [variable] if variable else [k for k in keys if np.asarray(get(k)).size > 10]
        if not cand:
            raise ValueError("no numeric variable found in the MAT file")
        a = np.asarray(get(cand[0]), float).squeeze()
        x, fs = _from_matrix(a, fs, column)
    elif ext in (".csv", ".txt", ".dat"):
        try:
            a = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
        except ValueError:
            a = np.loadtxt(path, delimiter=",", skiprows=1)
        x, fs = _from_matrix(a, fs, column)
    elif ext == ".npy":
        x, fs = _from_matrix(np.load(path, allow_pickle=False), fs, column)
    else:
        raise ValueError(f"unsupported file type {ext}")
    if units.lower() == "g":
        x = x * 9.81
    return np.asarray(x, float), float(fs)


def save_result(path, x, fs, info: dict | None = None):
    """Save a profile as .mat ([t x] matrix 'Realdata'), .csv or .npy."""
    path = Path(path)
    t = np.arange(len(x)) / fs
    ext = path.suffix.lower()
    if ext == ".mat":
        from scipy.io import savemat
        out = {"Realdata": np.c_[t, x], "fs": fs}
        for k, v in (info or {}).items():
            if isinstance(v, (int, float, str)) or (isinstance(v, np.ndarray) and v.dtype != object):
                out[k[:31]] = v
        savemat(str(path), out, do_compression=True)
    elif ext == ".csv":
        np.savetxt(path, np.c_[t, x], delimiter=",", header="time_s,acceleration", comments="")
    elif ext == ".npy":
        np.save(path, np.c_[t, x])
    else:
        raise ValueError("use .mat, .csv or .npy")
    return path
