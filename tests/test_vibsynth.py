"""Tests for vibsynth. Run with `python -m pytest tests` or `python tests/test_vibsynth.py`."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import vibsynth as vs  # noqa: E402
from vibsynth.signals import trapezoid  # noqa: E402
from vibsynth.fds import response_variance_operator  # noqa: E402
from vibsynth.rainflow import rainflow_astm, rainflow_cycles, rainflow_damage_sum  # noqa: E402

FS = 1000.0


def _reference(seed=0, T=240.0):
    """Synthetic 'measured' record: band-limited Gaussian with bursts (k ~ 6)."""
    rng = np.random.default_rng(seed)
    f = np.linspace(0, FS / 2, 2001)
    G = np.where((f > 8) & (f < 250), 1.0 / (1 + ((f - 60) / 40) ** 2), 0.0)
    g = vs.psd_to_timeseries(f, G, FS, T, rng=rng)
    nw = int(2 * FS)
    lev = np.exp(0.45 * rng.normal(size=int(np.ceil(g.size / nw))))
    e = np.repeat(lev, nw)[: g.size]
    e = np.convolve(np.r_[np.full(300, e[0]), e, np.full(299, e[-1])], np.ones(600) / 600, "valid")
    return g * e / np.sqrt(np.mean(e ** 2))


def _ref_analysis():
    x = _reference()
    return vs.analyze_reference(x, FS, fmin=10, fmax=250, Q=10, b=6, points_per_octave=6)


# ----------------------------------------------------------------- core blocks
def test_rainflow_equals_reference_implementation():
    rng = np.random.default_rng(1)
    for _ in range(6):
        y = np.cumsum(rng.normal(size=2500)) + 3 * rng.normal(size=2500)
        ref = rainflow_astm(y)
        for b in (3.0, 6.5, 9.0):
            exact = np.sum(ref[:, 1] * ref[:, 0] ** b)
            assert abs(rainflow_damage_sum(y, b) / exact - 1) < 1e-10
        a, n = rainflow_cycles(y)
        assert abs(np.sum(n * a ** 4) / np.sum(ref[:, 1] * ref[:, 0] ** 4) - 1) < 1e-10
    # bursts with free decays (the slow case of the vectorised passes)
    t = np.arange(20000) / 1000.0
    y = np.sin(2 * np.pi * 37 * t) * np.exp(-((t % 5) * 3)) * (1 + rng.normal(size=t.size) * 0.01)
    ref = rainflow_astm(y)
    assert abs(rainflow_damage_sum(y, 7.0) / np.sum(ref[:, 1] * ref[:, 0] ** 7) - 1) < 1e-10


def test_lalanne_operator_matches_integration():
    fn = vs.natural_frequencies(5, 200, "octave", 12)
    Q = 15.0
    W = response_variance_operator(fn, Q)
    G = np.exp(-((fn - 70) / 40) ** 2) + 0.05
    f = np.linspace(fn[0], fn[-1], 200001)
    Gf = np.interp(f, fn, G)
    num = np.array([trapezoid(Gf / ((2 * np.pi) ** 4 * ((fi ** 2 - f ** 2) ** 2 + (f * fi / Q) ** 2)), f)
                    for fi in fn])
    assert np.max(np.abs(W @ G / num - 1)) < 1e-4


def test_psd_to_timeseries_variance():
    f = np.linspace(0, 500, 1001)
    G = np.where((f > 20) & (f < 200), 2.0, 0.0)
    x = vs.psd_to_timeseries(f, G, FS, 100.0, rng=3, floor=0.0)
    assert x.size == 100000
    assert abs(np.var(x) / trapezoid(G, f) - 1) < 0.01


def test_inversion_is_exact():
    p = vs.FDSParams(fn=vs.natural_frequencies(5, 300, "octave", 12), Q=10, b=8, prefilter=False)
    G = 1.0 / (1 + (p.fn / 80) ** 2)
    D, _ = vs.fds_gaussian(G, 100.0, p)
    inv = vs.invert_fds(D, 100.0, p)
    assert inv.max_rel_error < 1e-3
    assert np.max(np.abs(inv.psd / G - 1)) < 5e-3


def test_natural_frequency_grid_convention():
    fn = vs.natural_frequencies(5, 200, "octave", 12)
    assert fn[0] == 5 and fn[-1] == 200 and fn.size == 65
    fl = vs.natural_frequencies(5.3, 50, "linear", resolution=0.5)
    assert fl[0] == 5.0 and abs(fl[1] - 5.5) < 1e-12


# ------------------------------------------------------------ FDS synthesis
def test_gaussian_mission_synthesis():
    ref = _ref_analysis()
    r = vs.gaussian(ref.fds, 60.0, ref.fs, ref.params, seed=1)
    ev = vs.evaluate(r.x, r.fs, ref, fds=r.info["fds_measured"], mrs=r.info["mrs_measured"])
    assert ev["fds_max_error_pct"] < 5.5
    assert abs(ev["stats"]["kurtosis"] - 3) < 0.15


def test_kfds_meets_fds_and_kurtosis():
    ref = _ref_analysis()
    k = ref.stats["kurtosis"]
    r = vs.kfds(ref.fds, 60.0, ref.fs, ref.params, k, reference=ref.x, ref_fs=ref.fs, seed=2)
    ev = vs.evaluate(r.x, r.fs, ref, fds=r.info["fds_measured"], mrs=r.info["mrs_measured"])
    assert ev["fds_max_error_pct"] < 6
    assert abs(ev["stats"]["kurtosis"] - k) < 0.12
    # leptokurtic signal with the same damage has a lower RMS than the Gaussian one
    rg = vs.gaussian(ref.fds, 60.0, ref.fs, ref.params, seed=1)
    assert ev["stats"]["rms"] < 0.95 * np.sqrt(np.mean(rg.x ** 2))


def test_rf_meets_fds_and_rms():
    ref = _ref_analysis()
    r = vs.rf(ref.fds, ref.duration, ref.fs, ref.params, ref.stats["rms"], reference=ref.x,
              ref_fs=ref.fs, seed=3)
    ev = vs.evaluate(r.x, r.fs, ref, fds=r.info["fds_measured"], mrs=r.info["mrs_measured"])
    assert ev["fds_max_error_pct"] < 6
    assert abs(ev["stats"]["rms"] / ref.stats["rms"] - 1) < 0.03


def test_pf_meets_fds_and_psd():
    ref = _ref_analysis()
    r = vs.pf(ref.fds, ref.duration, ref.fs, ref.params, ref.psd_f, ref.psd, seed=4)
    ev = vs.evaluate(r.x, r.fs, ref, fds=r.info["fds_measured"], mrs=r.info["mrs_measured"])
    assert ev["fds_max_error_pct"] < 6
    assert ev["psd_rms_db_dev"] < 1.5


# ----------------------------------------------------------- kurtosis control
def test_kurtosis_control_methods():
    ref = _ref_analysis()
    for fun, kw in ((vs.mlv, dict(reference=ref.x, ref_fs=ref.fs)), (vs.vsd, {}), (vs.pm, {})):
        r = fun(ref.psd_f, ref.psd, 120.0, ref.fs, 5.0, seed=5, **kw)
        ev = vs.evaluate(r.x, r.fs, ref)
        assert abs(ev["stats"]["kurtosis"] - 5.0) < 0.05, r.method
        assert ev["psd_rms_db_dev"] < 1.2, r.method


# ---------------------------------------------------------------------- I/O
def test_io_roundtrip():
    x = np.random.default_rng(0).normal(size=5000)
    with tempfile.TemporaryDirectory() as d:
        for ext in (".mat", ".csv", ".npy"):
            p = os.path.join(d, "sig" + ext)
            vs.save_result(p, x, 2048.0)
            y, fs = vs.load_signal(p)
            assert abs(fs - 2048.0) < 1e-6 and np.allclose(x, y)


if __name__ == "__main__":
    failures = 0
    for name, fun in list(globals().items()):
        if name.startswith("test_") and callable(fun):
            try:
                fun()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
