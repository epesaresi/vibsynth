"""One call from a target (record, specification or mission) to a test profile."""
from __future__ import annotations

from . import synthesis as S
from .analysis import Reference

__all__ = ["run_method"]


def run_method(ref: Reference, method: str, duration: float, *, kurtosis=None, rms=None,
               pattern=None, seed=None, iterations=30, tol=0.05, window=None, fs=None,
               tones=None, prominence_db=10.0, drift=0.0, log=None, modulation="random",
               mapping="blocks", block_s=None, lines="lines", max_excess=30.0, freq_corr=None,
               time_corr=None, cl_anchor=0.0, compress_kurtosis=False, ranking="common",
               margin=2.0, fade_s=0.1, gain=True, gain_max=2.0, retention=0.9):
    """Synthesise `duration` seconds that reproduce the target.

    kurtosis / rms default to the target values. For the kurtosis-control
    methods (mlv, vsd, pm) the FDS is not imposed; their FDS is compared with
    the target over the same duration. `fs` defaults to the target's rate.
    MMVSD prescribes the FDS and the kurtosis of the oscillator responses of
    the target; its own options are modulation, mapping, block_s, lines,
    max_excess, freq_corr, time_corr and cl_anchor (see mmvsd.synthesize).
    `edit` keeps the damaging stretches of the target record (edit.edit);
    its options are ranking, window, margin, fade_s, gain, gain_max and
    retention. compress_kurtosis=True replaces the kurtosis target of the
    kurtosis methods by the kurtosis of the record edited to `duration`
    (manual, Section 14), which is what a compressed test should carry.
    """
    m = method.lower()
    fs = fs or ref.fs
    if m == "edit":
        from .edit import edit as edit_record
        return edit_record(ref, duration, ranking=ranking, window_s=window, margin=margin, fade_s=fade_s,
                           gain=gain, gain_max=gain_max, retention=retention, iterations=max(iterations, 1),
                           tol=tol, log=log)
    k = kurtosis if kurtosis else ref.stats["kurtosis"]
    if compress_kurtosis and not kurtosis and ref.x is not None and m in ("kfds", "rf", "mlv", "vsd", "pm"):
        from .edit import compressed_kurtosis
        k = compressed_kurtosis(ref, duration, ranking=ranking, window_s=window, margin=margin, fade_s=fade_s)
        if log:
            log(f"kurtosis target compressed to {k:.2f} (record edited to {duration:g} s; "
                f"record kurtosis {ref.stats['kurtosis']:.2f})")
    segs = ref.segments or None
    common = dict(seed=seed, log=log)
    fdskw = dict(correction_iterations=iterations, tol=tol, **common)
    if m == "gaussian":
        return S.gaussian(ref.fds, duration, fs, ref.params, **fdskw)
    if m == "kfds":
        return S.kfds(ref.fds, duration, fs, ref.params, k, segments=segs, pattern=pattern,
                      window=window, **fdskw)
    if m == "rf":
        return S.rf(ref.fds, duration, fs, ref.params, rms if rms else ref.stats["rms"],
                    segments=segs, pattern=pattern, window=window, **fdskw)
    if m == "pf":
        return S.pf(ref.fds, duration, fs, ref.params, ref.psd_f, ref.psd, window=window, **fdskw)
    if m == "psf":
        return S.psf(ref.fds, duration, fs, ref.params, ref.psd_f, ref.psd, ref.life,
                     window=window, **fdskw)
    if m == "mrs":
        return S.mrs_control(ref.mrs, duration, fs, ref.params, **fdskw)
    if m == "sor":
        if tones is None:
            tones = S.detect_tones(ref.psd_f, ref.psd, ref.params.fn[0], ref.params.fn[-1],
                                   prominence_db=prominence_db)
            if log:
                log("detected tones: " + (", ".join(f"{f:.1f} Hz ({a:.3g})" for f, a in tones) or "none"))
        return S.sine_on_random(ref.fds, duration, fs, ref.params, tones, ref.life, drift=drift, **fdskw)
    if m == "mmvsd":
        from .mmvsd import synthesize as mmvsd_synthesize
        return mmvsd_synthesize(ref, duration, fs=fs, seed=seed, iterations=iterations, tol=tol,
                                modulation=modulation or "random", mapping=mapping or "blocks",
                                block_s=block_s, lines=lines or "lines", max_excess=max_excess or 30.0,
                                freq_corr=freq_corr, time_corr=time_corr, cl_anchor=cl_anchor or 0.0, log=log)
    if m == "mlv":
        res = S.mlv(ref.psd_f, ref.psd, duration, fs, k, segments=segs, pattern=pattern,
                    window=window, **common)
    elif m == "vsd":
        res = S.vsd(ref.psd_f, ref.psd, duration, fs, k, fmin=ref.params.fn[0],
                    fmax=ref.params.fn[-1], window=window, **common)
    elif m == "pm":
        res = S.pm(ref.psd_f, ref.psd, duration, fs, k, block=window, **common)
    else:
        raise ValueError(f"unknown method {method!r}; choose from {', '.join(S.METHODS)}")
    res.info["fds_target_scale"] = duration * res.repeats / ref.life
    return res
