"""Export of time histories and curves (PSD, FDS, MRS, statistics...)."""
from __future__ import annotations

import io
import json
import wave
import zipfile

import numpy as np

__all__ = ["curves_bytes", "series_bytes", "FORMATS_CURVES", "FORMATS_SERIES"]

FORMATS_CURVES = ("csv", "mat", "xlsx", "json")
FORMATS_SERIES = ("mat", "csv", "npy", "txt", "wav")


def _columns(curves: dict):
    """curves: {column name: 1-D array}; pads to a common length with NaN."""
    n = max(len(np.atleast_1d(v)) for v in curves.values())
    cols = {}
    for k, v in curves.items():
        a = np.full(n, np.nan)
        v = np.atleast_1d(np.asarray(v, float))
        a[: v.size] = v
        cols[k] = a
    return cols


def curves_bytes(curves: dict, fmt: str, meta: dict | None = None, sheet="data") -> bytes:
    """Serialise named columns (e.g. f, PSD_ref, PSD_syn) to csv / mat / xlsx / json."""
    cols = _columns(curves)
    names = list(cols)
    if fmt == "csv":
        buf = io.StringIO()
        if meta:
            for k, v in meta.items():
                buf.write(f"# {k}: {v}\n")
        buf.write(",".join(names) + "\n")
        np.savetxt(buf, np.column_stack([cols[k] for k in names]), delimiter=",", fmt="%.10g")
        return buf.getvalue().encode()
    if fmt == "json":
        return json.dumps({"meta": meta or {}, **{k: [None if np.isnan(x) else float(x) for x in v]
                                                  for k, v in cols.items()}}).encode()
    if fmt == "mat":
        from scipy.io import savemat
        buf = io.BytesIO()
        d = {_matname(k): v for k, v in cols.items()}
        for k, v in (meta or {}).items():
            if isinstance(v, (int, float, str)):
                d[_matname("meta_" + k)] = v
        savemat(buf, d, do_compression=True)
        return buf.getvalue()
    if fmt == "xlsx":
        try:
            from openpyxl import Workbook
        except ImportError as e:
            raise ValueError("xlsx export needs openpyxl (pip install openpyxl); use csv") from e
        wb = Workbook()
        ws = wb.active
        ws.title = sheet[:31]
        ws.append(names)
        for row in np.column_stack([cols[k] for k in names]):
            ws.append([None if np.isnan(v) else float(v) for v in row])
        if meta:
            wm = wb.create_sheet("info")
            for k, v in meta.items():
                wm.append([k, str(v)])
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()
    raise ValueError(f"format {fmt} not supported for curves")


def _matname(k):
    s = "".join(ch if ch.isalnum() else "_" for ch in k)
    if not s[0].isalpha():
        s = "v_" + s
    return s[:31]


def series_bytes(x, fs, fmt: str, meta: dict | None = None) -> bytes:
    """Time history as mat ([t x] matrix 'Realdata'), csv, npy, txt or wav."""
    x = np.asarray(x, float)
    t = np.arange(x.size) / fs
    if fmt == "mat":
        from scipy.io import savemat
        buf = io.BytesIO()
        d = {"Realdata": np.c_[t, x], "fs": fs}
        for k, v in (meta or {}).items():
            if isinstance(v, (int, float, str)):
                d[_matname(k)] = v
        savemat(buf, d, do_compression=True)
        return buf.getvalue()
    if fmt in ("csv", "txt"):
        buf = io.StringIO()
        delim = "," if fmt == "csv" else "\t"
        buf.write(delim.join(["time_s", "acceleration"]) + "\n")
        np.savetxt(buf, np.c_[t, x], delimiter=delim, fmt="%.9g")
        return buf.getvalue().encode()
    if fmt == "npy":
        buf = io.BytesIO()
        np.save(buf, np.c_[t, x])
        return buf.getvalue()
    if fmt == "wav":
        # 32-bit float would be ideal; 16-bit PCM is the most portable.  The
        # scale factor (full scale = peak) is written in the file name by the caller.
        peak = float(np.max(np.abs(x))) or 1.0
        pcm = np.int16(np.clip(x / peak, -1, 1) * 32767)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(round(fs)))
            w.writeframes(pcm.tobytes())
        return buf.getvalue()
    raise ValueError(f"format {fmt} not supported for time histories")


def zip_bytes(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()
