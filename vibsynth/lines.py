"""Sine-on-random blocks of a record: detection, tracking and extraction of
tonal lines, and their deterministic re-synthesis on a test time axis.

Blocks where tonal lines (rotating machinery: engine orders, pumps, fans)
stand above a random background are reproduced with the sine-on-random
principle of `synthesis.sine_on_random`, block by block:

* `find_tones`  the record is cut into blocks; in every block the line
  detector `synthesis.detect_tones` is applied to the block PSD, and a peak is
  accepted as a line only if it is phase-coherent over the block (a narrow
  resonance excited by random noise is not) and persists over at least two
  blocks. Lines of consecutive blocks form tracks (frequency and amplitude
  versus time), which are demodulated along their own phase and subtracted
  from the record: the residual is the random part.
* `TestLines`   the tracks mapped on the test time axis, as deterministic
  sines amplified so that each keeps its share of the life damage
  ((life/T)^(1/b) for a stationary line, as in sine_on_random).

Used by the MMVSD method (`mmvsd.py`), which keeps the lines fixed in its
closed loop and designs the random part for the rest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .fds import prefilter_active, prefilter_power, smallwood_h2
from .synthesis import detect_tones

__all__ = ["ToneTrack", "ToneModel", "find_tones", "TestLines", "frame_mapping", "reference_time", "sdof_h2"]


@dataclass
class ToneTrack:
    t: np.ndarray            # envelope time grid of the reference [s]
    f: np.ndarray            # instantaneous frequency on t [Hz]
    a: np.ndarray            # peak amplitude on t (0 where the line is absent)
    blocks: list             # (t0, t1) of the blocks where the line was detected

    def summary(self):
        on = self.a > 0.05 * max(float(self.a.max()), 1e-300)
        fo = self.f[on] if np.any(on) else self.f
        dt = self.t[1] - self.t[0] if self.t.size > 1 else 0.0
        return {"f_min": float(fo.min()), "f_max": float(fo.max()), "f_mean": float(np.median(fo)),
                "amplitude": float(np.sqrt(np.mean(self.a[on] ** 2))) if np.any(on) else 0.0,
                "time_s": float(on.sum() * dt), "blocks": [list(map(float, b)) for b in self.blocks]}


@dataclass
class ToneModel:
    tracks: list
    fs: float
    duration: float          # reference duration [s]
    residual: np.ndarray     # reference minus the extracted lines
    lines: np.ndarray        # the extracted lines (reference)
    blocks: list             # per block: dict(t0, t1, tones=[(f, A, coherence)], rejected=[(f, A, coherence)])
    settings: dict = field(default_factory=dict)

    @property
    def tonal_time(self):
        return float(sum(b["t1"] - b["t0"] for b in self.blocks if b["tones"]))

    def describe(self):
        if not self.tracks:
            return "no sine-on-random blocks"
        return "; ".join(f"{s['f_min']:.1f}-{s['f_max']:.1f} Hz, A {s['amplitude']:.3g}, "
                         f"{s['time_s']:.0f} s" for s in (tr.summary() for tr in self.tracks))


def _stft_power(x, fs, nper, hop, chunk=256):
    w = np.hanning(nper)
    nfr = (x.size - nper) // hop + 1
    scale = 2.0 / (fs * np.sum(w * w))
    S = np.empty((nfr, nper // 2 + 1), np.float32)
    for k0 in range(0, nfr, chunk):
        ks = np.arange(k0, min(nfr, k0 + chunk))
        seg = x[ks[:, None] * hop + np.arange(nper)[None, :]] * w
        S[ks] = (np.abs(np.fft.rfft(seg, axis=1)) ** 2 * scale).astype(np.float32)
    return S


def _line_coherence(x, fs, S, f, tc, fr, f0, t0, t1, link_rel, resolution, bw=1.0):
    """Phase coherence of a candidate line over one block, in [0, 1].

    The frequency is followed frame by frame (peak of each short frame near f0,
    parabolic interpolation), the block is demodulated along that phase and
    low-pass filtered to +-bw Hz; the result c(t) is a constant phasor for a
    sine and a random phasor for narrow-band noise. Coherence =
    max over a residual frequency offset |delta| <= resolution of
    |sum c e^(-i 2 pi delta t)|^2 / (N sum |c|^2): ~1 for a sine (minus the
    share of the background in +-bw), ~1/(B T) for noise of bandwidth B.
    """
    from scipy.signal import butter, sosfiltfilt, resample_poly
    df = f[1] - f[0]
    wl = max(int(math.ceil(link_rel * f0 / df)), 3)
    i0 = int(round(f0 / df))
    lo, hi = max(i0 - wl, 1), min(i0 + wl + 1, f.size - 1)
    fk = []
    for k in fr:
        i = int(np.argmax(S[k, lo:hi])) + lo
        lp = np.log(np.maximum(S[k, i - 1:i + 2].astype(float), 1e-300))
        den = lp[0] - 2 * lp[1] + lp[2]
        fk.append((i + (0.5 * (lp[0] - lp[2]) / den if den < 0 else 0.0)) * df)
    i_0, i_1 = int(t0 * fs), min(int(t1 * fs), x.size)
    tt = np.arange(i_0, i_1) / fs
    # smooth (linear) frequency trend over the block: follows the drift of a
    # machine line but not the random wander of the peak of narrow-band noise
    fk = np.asarray(fk)
    if fr.size >= 3:
        cfit = np.polyfit(tc[fr] - tc[fr].mean(), fk, 1)
        fi = np.polyval(cfit, tt - tc[fr].mean())
    else:
        fi = np.full(tt.size, float(np.mean(fk)))
    y = x[i_0:i_1] * np.exp(-2j * np.pi * np.cumsum(fi) / fs)
    q = max(1, int(fs // (8 * bw)))
    yd = (resample_poly(y.real, 1, q) + 1j * resample_poly(y.imag, 1, q)) if q > 1 else y
    fe = fs / q
    if yd.size < 16:
        return 0.0
    sos = butter(4, bw, fs=fe, output="sos")
    c = sosfiltfilt(sos, yd.real) + 1j * sosfiltfilt(sos, yd.imag)
    m = c.size // 10                                  # drop the filter edges
    c = c[m:c.size - m]
    nz = 1 << int(math.ceil(math.log2(max(c.size * 16, 256))))
    C = np.abs(np.fft.fft(c, nz)) ** 2
    fq = np.fft.fftfreq(nz, 1 / fe)
    coh = float(C[np.abs(fq) <= 0.25 * resolution].max() / (c.size * np.sum(np.abs(c) ** 2) + 1e-300))
    ac = np.abs(c)
    return coh, float(ac.std() / max(ac.mean(), 1e-300))


def find_tones(x, fs, fmin, fmax, *, block_s=5.0, resolution=0.5, prominence_db=10.0, max_tones=8,
               min_coherence=0.75, weak_coherence=0.5, min_blocks=2, link_rel=0.05, env_bw=None,
               min_amp_rel=0.0):
    """Sine-on-random blocks of a record and the time-domain extraction of their lines.

    The record is cut into blocks of `block_s` seconds; in every block the
    vibsynth line detector (detect_tones) is applied to the block PSD
    (`resolution` Hz, Hann, 75 % overlap) and a line is accepted only if it is
    phase-coherent over the block (_line_coherence >= min_coherence: a sine
    gives ~1, a narrow-band random resonance ~1/(bandwidth x block_s)). Lines
    of consecutive blocks within `link_rel` in frequency form a track, kept if
    it is strongly coherent in at least `min_blocks` blocks (machine lines
    persist, a random resonance passing the test in one block is dropped);
    between those blocks a track may continue through weakly coherent blocks
    (>= weak_coherence: hysteresis, for lines with a wandering speed). Each
    track is followed frame by frame (peak search, parabolic interpolation), demodulated along its own phase and low-pass filtered
    (`env_bw`, default resolution / 2) to get its complex envelope; the lines
    are then subtracted from the record (residual = random part).
    """
    from scipy.signal import butter, sosfiltfilt, resample_poly
    x = np.asarray(x, float)
    x = x - x.mean()
    n = x.size
    T = n / fs
    nper = int(round(fs / resolution))
    hop = max(nper // 4, 1)
    settings = dict(block_s=block_s, resolution=resolution, prominence_db=prominence_db,
                    min_coherence=min_coherence, weak_coherence=weak_coherence, min_blocks=min_blocks,
                    max_tones=max_tones)
    empty = ToneModel([], fs, T, x.copy(), np.zeros(n), [], settings)
    if n < nper + 3 * hop:
        return empty
    S = _stft_power(x, fs, nper, hop)
    f = np.fft.rfftfreq(nper, 1 / fs)
    df = f[1] - f[0]
    tc = (np.arange(S.shape[0]) * hop + nper / 2) / fs
    nb = max(1, int(round(T / block_s)))
    edges = np.linspace(0.0, T, nb + 1)
    blocks, dets = [], []
    for b in range(nb):
        fr = np.flatnonzero((tc >= edges[b]) & (tc < edges[b + 1]))
        info = {"t0": float(edges[b]), "t1": float(edges[b + 1]), "tones": [], "rejected": []}
        blocks.append(info)
        if fr.size < 3:
            continue
        Pb = S[fr].mean(axis=0).astype(float)
        for f0, a0 in detect_tones(f, Pb, fmin, fmax, prominence_db=prominence_db, max_tones=max_tones):
            coh, _ = _line_coherence(x, fs, S, f, tc, fr, f0, edges[b], edges[b + 1], link_rel, resolution)
            if a0 < min_amp_rel * np.sqrt(np.mean(x * x)):
                continue
            if coh >= weak_coherence:
                i0 = int(round(f0 / df))
                lp = np.log(np.maximum(Pb[i0 - 1:i0 + 2], 1e-300)) if 1 <= i0 < f.size - 1 else None
                if lp is not None and lp.size == 3:
                    den = lp[0] - 2 * lp[1] + lp[2]
                    f0 = (i0 + (0.5 * (lp[0] - lp[2]) / den if den < 0 else 0.0)) * df
                dets.append((b, float(f0), float(a0), float(coh)))
            else:
                info["rejected"].append((float(f0), float(a0), coh))
    # ------------------------------------------------------------- tracks
    tracks = []                                   # [ [ (block, f, A), ... ], ... ]
    for b in range(nb):
        here = sorted([d for d in dets if d[0] == b], key=lambda d: -d[2])
        used = set()
        for d in here:
            best_j, best_d = None, np.inf
            for j, tr in enumerate(tracks):
                if j in used or tr[-1][0] < b - 2:
                    continue
                dist = abs(d[1] - tr[-1][1])
                if dist <= max(link_rel * tr[-1][1], 3 * df) and dist < best_d:
                    best_j, best_d = j, dist
            if best_j is None:
                tracks.append([d])
                used.add(len(tracks) - 1)
            else:
                tracks[best_j].append(d)
                used.add(best_j)
    # hysteresis: a track is a line if at least min_blocks of its blocks are
    # strongly coherent; between them it may pass through weakly coherent blocks
    # (a machine line with a wandering speed or a low SNR)
    keep = [sum(d[3] >= min_coherence for d in tr) >= min_blocks for tr in tracks]
    for tr, kp in zip(tracks, keep):
        for b, f0, a0, coh in tr:
            blocks[b]["tones" if kp else "rejected"].append((f0, a0, coh))
    tracks = [[d[:3] for d in tr] for tr, kp in zip(tracks, keep) if kp]
    if not tracks:
        empty.blocks = blocks
        return empty
    # ------------------------------------------------- extraction per track
    q = max(1, int(fs // (16 * max(resolution, 0.05))))        # envelope rate ~ 16 / resolution
    fe = fs / q
    te = np.arange(int(math.ceil(n / q))) * q / fs
    t_full = np.arange(n) / fs
    bw = env_bw or 0.5 * resolution
    sos = butter(4, bw, fs=fe, output="sos")
    lines = np.zeros(n)
    out = []
    taper_s = min(0.5, 0.25 * block_s)
    for tr in tracks:
        bl = sorted({d[0] for d in tr})
        # merge consecutive detected blocks into active intervals
        ivals = []                                 # gaps of up to 2 blocks are bridged
        for b in bl:
            if ivals and edges[b] - ivals[-1][1] <= 2 * (edges[1] - edges[0]) + 1e-9:
                ivals[-1][1] = edges[b + 1]
            else:
                ivals.append([edges[b], edges[b + 1]])
        fb = {d[0]: d[1] for d in tr}
        # frame-wise frequency inside the detected blocks
        fk_t, fk_f = [], []
        bl_all = [b for b in range(nb) if any(t0 - 1e-9 <= edges[b] < t1 - 1e-9 for t0, t1 in ivals)]
        for b in bl_all:
            fr = np.flatnonzero((tc >= edges[b]) & (tc < edges[b + 1]))
            c0 = fb[b] if b in fb else fb[min(fb, key=lambda k: abs(k - b))]
            w = max(link_rel * c0, 3 * df)
            lo, hi = max(int((c0 - w) / df), 1), min(int(math.ceil((c0 + w) / df)) + 1, f.size - 1)
            for k in fr:
                seg = S[k, lo:hi].astype(float)
                i = int(np.argmax(seg)) + lo
                lp = np.log(np.maximum(S[k, i - 1:i + 2].astype(float), 1e-300))
                den = lp[0] - 2 * lp[1] + lp[2]
                fk_t.append(tc[k])
                fk_f.append((i + (0.5 * (lp[0] - lp[2]) / den if den < 0 else 0.0)) * df)
        fk_t, fk_f = np.asarray(fk_t), np.asarray(fk_f)
        if fk_f.size >= 3:                          # median of 3 against spurious peaks
            fk_f = np.r_[fk_f[0], np.median(np.c_[fk_f[:-2], fk_f[1:-1], fk_f[2:]], axis=1), fk_f[-1]]
        f_e = np.interp(te, fk_t, fk_f)
        a_e = np.zeros(te.size)
        for t0, t1 in ivals:
            i0, i1 = int(t0 * fs), min(int(math.ceil(t1 * fs)), n)
            tt = t_full[i0:i1]
            fi = np.interp(tt, fk_t, fk_f)
            ph = 2 * np.pi * np.cumsum(fi) / fs
            y = x[i0:i1] * np.exp(-1j * ph)
            yd = resample_poly(y.real, 1, q) + 1j * resample_poly(y.imag, 1, q) if q > 1 else y
            if yd.size > 3 * 4 * 2:
                cd = sosfiltfilt(sos, yd.real) + 1j * sosfiltfilt(sos, yd.imag)
            else:
                cd = np.full(yd.size, yd.mean())
            td = tt[0] + np.arange(cd.size) * q / fs
            c = np.interp(tt, td, cd.real) + 1j * np.interp(tt, td, cd.imag)
            ramp = np.minimum(1.0, np.minimum(tt - tt[0], tt[-1] - tt) / taper_s + 1e-12)
            ramp = np.sin(0.5 * np.pi * np.clip(ramp, 0, 1)) ** 2
            lines[i0:i1] += 2.0 * np.real(c * np.exp(1j * ph)) * ramp
            sel = (te >= t0) & (te < t1)
            a_e[sel] = 2.0 * np.abs(np.interp(te[sel], td, cd.real) + 1j * np.interp(te[sel], td, cd.imag)) \
                * np.interp(te[sel], tt, ramp)
        out.append(ToneTrack(te, f_e, a_e, [(float(a), float(b)) for a, b in ivals]))
    return ToneModel(out, fs, T, x - lines, lines, blocks, settings)


def frame_mapping(Kr, K, mapping="blocks", block_frames=48):
    """Reference frame of every test frame (shared by the pattern field and the lines)."""
    if K >= Kr:
        return np.arange(K) % Kr, ("loop" if K > 1.01 * Kr else "identity")
    if mapping == "compress":
        edges = np.linspace(0, Kr, K + 1)
        lo = np.floor(edges[:-1]).astype(int)
        hi = np.maximum(np.floor(edges[1:]).astype(int), lo + 1)
        return (lo + hi) // 2, "compress"
    Lb = max(4, min(int(block_frames), K))
    nsel = int(np.ceil(K / Lb))
    starts = np.round(np.linspace(0, Kr - Lb, nsel)).astype(int) if nsel > 1 else np.array([(Kr - Lb) // 2])
    return np.concatenate([np.arange(s0, s0 + Lb) for s0 in starts])[:K], "blocks"


def reference_time(t, hop, idx, mapping, T_ref, T_test):
    """Reference time of the test times t for a frame mapping."""
    if mapping == "compress":
        return t * (T_ref / T_test)
    k = np.minimum((t / hop).astype(int), idx.size - 1)
    return idx[k] * hop + (t - k * hop)


def sdof_h2(fn, Q, f, fs=None, prefilter=False):
    """|H_n(f)|^2 of the relative displacement: the digital (Smallwood) response
    of the measurement when fs is given, with the vibsynth pre-filter if
    active; the analogue oscillator otherwise."""
    if fs is not None:
        H2 = smallwood_h2(fn, Q, fs, f)
    else:
        wn = 2 * np.pi * np.asarray(fn, float)[:, None]
        w = 2 * np.pi * np.asarray(f, float)[None, :]
        z = 1.0 / (2.0 * Q)
        H2 = 1.0 / ((wn ** 2 - w ** 2) ** 2 + (2 * z * wn * w) ** 2)
    if fs is not None and prefilter_active(fs, np.asarray(fn, float), prefilter):
        H2 = H2 * prefilter_power(np.asarray(f, float).ravel(), fs)[None, :]
    return H2


class TestLines:
    """The deterministic lines of the test: reference tracks mapped on the test
    time axis, amplified per track so that each keeps its share of the life
    damage:  g_j = [ (life/T_ref) int_ref a_j^b dt / int_test a_j^b dt ]^(1/b)
    (= (life/T)^(1/b) for a stationary line, as vibsynth sine_on_random)."""

    def __init__(self, model: ToneModel, N, fs, hop, idx, mapping, b, life, rng, smooth_s=0.1):
        self.N, self.fs = N, fs
        T = N / fs
        t = np.arange(N) / fs
        tau = reference_time(t, hop, idx, mapping, model.duration, T)
        self.x = np.zeros(N)
        self.tracks, self.gains, self.lost, self.summaries = [], [], [], []
        L = max(1, int(smooth_s * fs))
        kern = np.ones(L) / L
        for tr in model.tracks:
            a = np.interp(tau, tr.t, tr.a)
            if L > 1:
                a = np.convolve(np.r_[np.full(L, a[0]), a, np.full(L, a[-1])], kern, "same")[L:-L]
            fi = np.interp(tau, tr.t, tr.f)
            dt_r = tr.t[1] - tr.t[0]
            i_ref = float(np.sum(tr.a ** b) * dt_r)
            i_test = float(np.sum(a ** b) / fs)
            summ = tr.summary()
            if i_test <= 1e-12 * max(i_ref, 1e-300) or i_ref <= 0:
                self.lost.append(summ)
                continue
            g = (life / model.duration * i_ref / i_test) ** (1.0 / b)
            ph = 2 * np.pi * np.cumsum(fi) / fs + rng.uniform(0, 2 * np.pi)
            self.x += g * a * np.sin(ph)
            self.tracks.append((fi, g * a))
            self.gains.append(g)
            self.summaries.append(dict(summ, gain=float(g)))

    def variance(self, fn, Q, t_model, fs=None, prefilter=False):
        """Response variance of every line at every node on the model grid: (J, n, Nt)."""
        t = np.arange(self.N) / self.fs
        out = []
        for fi, a in self.tracks:
            fm = np.interp(t_model, t, fi)
            am = np.interp(t_model, t, a)
            fu = np.unique(np.round(fm, 2))
            H2u = sdof_h2(fn, Q, fu, fs, prefilter)
            H2 = H2u[:, np.searchsorted(fu, np.round(fm, 2))]
            out.append(0.5 * am[None, :] ** 2 * H2)
        return np.array(out)
