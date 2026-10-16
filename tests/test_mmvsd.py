"""Tests of the 2.1 features: response engine, MMVSD, sine-on-random blocks,
response kurtosis of missions, workspace, web API and command line.
Run with `python -m pytest tests` or `python tests/test_mmvsd.py`."""
import json
import os
import sys
import tempfile
import threading
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import vibsynth as vs  # noqa: E402
from vibsynth import lines as ln, mmvsd as mm  # noqa: E402
from vibsynth.fds import (FDSParams, _turning, apply_prefilter, response_spectra,  # noqa: E402
                          sdof_relative_displacement, smallwood_coefficients)
from vibsynth.pipeline import run_method  # noqa: E402
from vibsynth.rainflow import rainflow_damage_sum, turning_points  # noqa: E402

EXAMPLE = os.path.join(HERE, "..", "examples", "example_road.mat")


# ------------------------------------------------------------ response engine
def test_package_names():
    import importlib
    assert importlib.import_module("vibsynth.mmvsd") is mm and vs.mmvsd is mm   # the module, not shadowed
    assert vs.mmvsd_profile is mm.mmvsd and "mmvsd" in vs.METHODS


def test_turning_points_chunked_identical():
    rng = np.random.default_rng(0)
    z = np.cumsum(rng.normal(size=5000)) + rng.normal(size=5000)
    for chunk in (3, 7, 64, 1 << 16):
        assert np.array_equal(_turning(z, chunk), turning_points(z))
    flat = np.repeat(rng.normal(size=300), 3)                 # flat steps: reference path
    assert np.array_equal(_turning(flat, 16), turning_points(flat))


def test_response_spectra_exact():
    x, fs = vs.load_signal(EXAMPLE)
    x = x[: int(60 * fs)]
    p = FDSParams(fn=vs.natural_frequencies(5, 150, "octave", 6), Q=10, b=8)
    r = response_spectra(x, fs, p, repeats=2.0)
    y = apply_prefilter(x, fs, p.fn, p.prefilter)
    for j in (0, 7, p.fn.size - 1):
        z = sdof_relative_displacement(y, fs, p.fn[j], p.Q)
        assert r["fds"][j] == 2.0 * rainflow_damage_sum(z, p.b)
        assert r["mrs"][j] == np.max(np.abs(z)) * (2 * np.pi * p.fn[j]) ** 2
        assert abs(r["var"][j] / z.var() - 1) < 1e-10
        assert abs(r["kurtosis"][j] / vs.kurtosis(z) - 1) < 1e-10
    D, M = vs.fds_time(x, fs, p, repeats=2.0)
    assert np.array_equal(D, r["fds"]) and np.array_equal(M, r["mrs"])


def test_response_spectra_worker_processes():
    rng = np.random.default_rng(1)
    fs = 2000.0
    x = rng.normal(size=400_000)
    p = FDSParams(fn=vs.natural_frequencies(10, 600, "octave", 12), Q=10, b=6)   # 400k x 72 > 3e7
    a = response_spectra(x, fs, p, workers=2)                 # worker processes, shared memory
    b = response_spectra(x, fs, p, workers=1)                 # sequential
    for k in ("fds", "mrs", "var", "kurtosis"):
        assert np.allclose(a[k], b[k], rtol=1e-12, atol=0)


def test_mission_response_kurtosis_mixture():
    fs = 1000.0
    rng = np.random.default_rng(2)
    p = dict(fmin=10, fmax=250, points_per_octave=6, Q=10, b=6)
    xa = rng.normal(size=int(60 * fs))
    xb = rng.normal(size=int(60 * fs)) * np.repeat(np.exp(0.8 * rng.normal(size=60)), int(fs))
    a = vs.analyze_reference(xa, fs, life=3600, **p)
    b = vs.analyze_reference(xb, fs, life=7200, **p)
    m = vs.combine(vs.Group("series", [vs.Situation("a", a), vs.Situation("b", b)]))
    w = np.array([1.0, 2.0]) / 3.0
    var = w[0] * a.response_var + w[1] * b.response_var
    m4 = w[0] * a.response_kurtosis * a.response_var ** 2 + w[1] * b.response_kurtosis * b.response_var ** 2
    assert np.allclose(m.response_var, var) and np.allclose(m.response_kurtosis, m4 / var ** 2)
    assert np.all(b.response_kurtosis > a.response_kurtosis)


# ------------------------------------------------------------ MMVSD model
def _setup(K=96, hop=0.25, fs=500.0, digital=True):
    p = vs.FDSParams(fn=vs.natural_frequencies(5, 120, "octave", 6), Q=10, b=8)
    W = vs.response_variance_operator(p.fn, p.Q, fs, p.prefilter)
    Wk = mm.moment_operators(p.fn, p.Q, fs, p.prefilter, digital=digital)
    em = mm.EnvelopeModel(Wk, p, hop, K * hop, K, fs=fs if digital else None)
    mod = mm.ModulationField(p.fn, K, hop, 0.5, 1.0, np.random.default_rng(0))
    P = np.exp(-0.5 * np.log(p.fn / 30) ** 2) + 0.05
    return p, W, em, mod, P


def test_row_means_and_moments():
    p, W, em, mod, P = _setup()
    e = np.linspace(0.5, 6, p.fn.size)
    M = mod.field(e)
    assert np.allclose(M.mean(axis=1), 1.0)
    assert np.all(M >= 0)
    s = np.sort(mod.field(np.full(p.fn.size, 3.0)), axis=1)   # stratified copula: same sorted rows
    assert np.allclose(s, s[0])


def test_mean_variance_independent_of_modulation():
    p, W, em, mod, P = _setup()
    r1 = em.run(P, em.up(np.ones((p.fn.size, mod.K))))
    r2 = em.run(P, em.up(mod.field(np.full(p.fn.size, 4.0))))
    assert np.allclose(r1["B"], em.W @ P, rtol=1e-10)
    assert np.allclose(r2["B"], em.W @ P, rtol=1e-8)
    assert np.allclose(r1["k"], 3.0)
    assert np.all(r2["c"] > r1["c"])                          # modulation raises the damage factor


def test_moment_operator_matches_lalanne():
    p, W, em, mod, P = _setup()
    W0 = mm.moment_operators(p.fn, p.Q, 500.0, p.prefilter, edge_hold=False, digital=False)[0]
    assert np.max(np.abs(W0 - W)) / np.max(np.abs(W)) < 1e-3


def test_smallwood_response_and_digital_variance():
    import scipy.signal as sps
    fs = 500.0
    f = np.linspace(0.5, 249.0, 700)
    fn = np.array([5.0, 40.0, 113.0, 150.0])
    H2 = mm.smallwood_h2(fn, 10.0, fs, f)
    for i, x in enumerate(fn):
        b, a = smallwood_coefficients(x, 10.0, fs)
        assert np.allclose(H2[i], np.abs(sps.freqz(b, a, worN=f, fs=fs)[1]) ** 2, rtol=1e-9)
    # variance of the sampled responses to a flat band = W0 (digital) @ 1, within 1 %
    p = vs.FDSParams(fn=vs.natural_frequencies(10, 160, "octave", 6), Q=10, b=8)
    W0 = mm.moment_operators(p.fn, p.Q, fs, p.prefilter)[0]
    n = int(400 * fs)
    grid = np.fft.rfftfreq(n, 1 / fs)
    G = mm._psd_lines(p.fn, np.ones(p.fn.size), grid, p.Q)
    rng = np.random.default_rng(3)
    x = np.fft.irfft(np.sqrt(G * n * fs / 2) * np.exp(1j * rng.uniform(-np.pi, np.pi, grid.size)), n)
    v = response_spectra(x, fs, p)["var"]
    assert np.max(np.abs(v / (W0 @ np.ones(p.fn.size)) - 1)) < 0.03
    Wa = mm.moment_operators(p.fn, p.Q, fs, p.prefilter, digital=False)[0]
    assert v[-1] / (Wa @ np.ones(p.fn.size))[-1] < 0.8          # the analogue model is 30 % off at fs/3


def test_sampling_factor():
    r = np.array([0.0, 0.04, 0.226, 0.5])
    c = mm.sampling_factor(8.0, r)
    assert c[0] == 1.0 and np.all(np.diff(c) < 0)
    th = np.pi * 0.226
    t = np.random.default_rng(0).uniform(-th, th, (2, 400_000))
    assert abs(c[2] - np.mean(((np.cos(t[0]) + np.cos(t[1])) / 2) ** 8)) < 0.005


def test_dirlik_narrowband_limit():
    from scipy.special import gamma
    m0, f0 = 2.0, 50.0                                        # a pure tone spectrum
    m1, m2, m4 = m0 * f0, m0 * f0 ** 2, m0 * f0 ** 4 * (1 + 1e-9)
    rice = f0 * (2 * m0) ** 4 * gamma(5)
    assert abs(mm.dirlik_rate(m0, m1, m2, m4, 8.0) / rice - 1) < 0.02


def test_jacobians_against_finite_differences():
    p, W, em, mod, P = _setup()
    e = np.full(p.fn.size, 2.0)
    M = mod.field(e)
    Mup = em.up(M)
    r = em.run(P, Mup)
    J = em.jac_logP(P, Mup, r)
    Jk = em.jac_k_logP(P, Mup, r)
    d = 1e-6
    for i in (2, 8, 15):                  # adjoint Jacobians = finite differences of the model
        P2 = P.copy()
        P2[i] *= np.exp(d)
        r2 = em.run(P2, Mup)
        assert np.max(np.abs((np.log(r2["D"]) - np.log(r["D"])) / d - J[:, i])) < 2e-3 * np.abs(J).max()
        assert np.max(np.abs((r2["k"] - r["k"]) / d - Jk[:, i])) < 1e-4
    i = 8
    dz = 1e-5
    dM = (mod.field(e * np.exp(dz)) - M) / dz
    e2 = e.copy()
    e2[i] *= np.exp(dz)
    r3 = em.run(P, em.up(mod.field(e2)))
    Jkz = em.jac_k(P, em.up(dM), r)
    JDz = em.jac_logD_mod(P, em.up(dM), r)
    assert np.max(np.abs((r3["k"] - r["k"]) / dz - Jkz[:, i])) < 1e-3
    assert np.max(np.abs((np.log(r3["D"]) - np.log(r["D"])) / dz - JDz[:, i])) < 1e-3


def test_frame_redraw_is_band_limited():
    fs = 1000.0
    fn = vs.natural_frequencies(5, 300, "octave", 6)
    mr = mm.MultiResolution(fn, np.full(fn.size, 1.0), fs, 30.0)
    mr.init_phases(np.random.default_rng(0))
    ph = mr.phases[0].copy()
    old = mr.reroll(0, np.array([2, 3]), np.random.default_rng(1), 50.0, 0.25)
    assert np.array_equal(old, ph[[2, 3]])
    fl = np.fft.rfftfreq(2 * mr.h0 * 2 ** mr.levels[0], 1 / fs)
    changed = np.any(mr.phases[0][[2, 3]] != ph[[2, 3]], axis=0)
    assert changed.any() and np.all(np.abs(np.log2(fl[changed] / 50.0)) <= 0.25)
    assert np.array_equal(np.delete(mr.phases[0], [2, 3], 0), np.delete(ph, [2, 3], 0))


def test_realised_frames_find_the_burst():
    """The frames re-drawn early are those around the block where the response
    of the worst oscillator has its largest cycles."""
    fs = 1000.0
    fn = vs.natural_frequencies(5, 300, "octave", 6)
    mr = mm.MultiResolution(fn, np.full(fn.size, 1.0), fs, 30.0)
    node = int(np.argmin(np.abs(fn - 50.0)))
    j = int(np.argmax(mr.node_w[:, node]))
    h = mr.h0 * 2 ** mr.levels[j]
    dens = np.zeros(mr.N)
    dens[7 * h + h // 2] = 1.0                      # one large cycle in block 7 of that group
    jj, frames = mr.realised_frames(dens, node, count=1)
    assert jj == j and set(frames.tolist()) == {6, 7, 8}


def test_unit_modulation_gives_stationary_gaussian_noise():
    fs, h, K = 1000.0, 128, 400
    fn = vs.natural_frequencies(5, 300, "octave", 6)
    for kind in ("gaussian", "multisine"):
        c = mm.frame_phases(np.random.default_rng(1), K, h, kind)
        assert abs(np.mean(np.abs(c) ** 2) - 1) < 0.01
        v = mm._modulate(c, np.ones((fn.size, K)), fn, h, fs)
        assert abs(v.var() / (fs / 2) - 1) < 0.03
        seg = v.reshape(20, -1).var(axis=1) / (fs / 2)        # no level variation over time
        assert np.all(np.abs(seg - 1) < 0.12)
        assert abs(vs.kurtosis(v) - 3) < 0.1


def test_gaussian_frames_keep_oscillator_responses_gaussian():
    """Few lines per oscillator band and frame: Gaussian frames keep k_z = 3
    (unit-modulus multisine frames give lighter-tailed responses)."""
    fs = 500.0
    p = vs.FDSParams(fn=vs.natural_frequencies(20, 120, "octave", 3), Q=30, b=8)
    mr = mm.MultiResolution(p.fn, np.full(p.fn.size, 0.6), fs, 120.0, sub=2)
    k = {}
    for kind in ("gaussian", "multisine"):
        mr.init_phases(np.random.default_rng(5), kind)
        k[kind] = response_spectra(mr.synthesize(np.ones((p.fn.size, mr.K))), fs, p)["kurtosis"]
    assert abs(np.mean(k["gaussian"]) - 3) < 0.12
    assert np.mean(k["multisine"]) < np.mean(k["gaussian"]) - 0.1


def test_reference_pattern_field():
    from scipy.signal import butter, sosfilt
    fs, T = 1000.0, 60.0
    fn = vs.natural_frequencies(5, 300, "octave", 6)
    rng = np.random.default_rng(3)
    n = int(T * fs)
    t = np.arange(n) / fs
    # reference: band 20-60 Hz bursts every 7 s, band 100-250 Hz bursts every 11 s
    a = sosfilt(butter(4, [20, 60], "band", fs=fs, output="sos"), rng.standard_normal(n))
    b = sosfilt(butter(4, [100, 250], "band", fs=fs, output="sos"), rng.standard_normal(n))
    x = a * (0.2 + (np.sin(2 * np.pi * t / 7) > 0.7)) + b * (0.2 + (np.sin(2 * np.pi * t / 11) > 0.7))
    mr = mm.MultiResolution(fn, np.full(fn.size, 1.0), fs, T)
    rf = mm.ReferenceField(x, fs, mr, mr.K)
    assert rf.mapping == "identity"
    e = np.full(fn.size, 2.0)
    M = rf.field(e)
    assert np.allclose(M.mean(axis=1), 1.0) and np.all(M > 0)
    assert np.allclose(3 * M.var(axis=1), np.minimum(e, rf._E[:, -1]), rtol=0.05)
    assert np.all(np.diff(rf._E, axis=1) >= 0)
    i = np.argmin(np.abs(fn - 40))                            # the field follows the reference bursts
    assert np.corrcoef(np.log(M[i]), rf.lB[i])[0, 1] > 0.99
    mr2 = mm.MultiResolution(fn, np.full(fn.size, 1.0), fs, 20.0)
    rf2 = mm.ReferenceField(x, fs, mr2, mr2.K, "blocks", 4.0)
    assert rf2.mapping == "blocks" and rf2.ref_index.size == mr2.K
    assert rf2.ref_index.max() < rf.lB.shape[1]


# ------------------------------------------------------ sine-on-random blocks
def _sor_record(fs=500.0, T=60.0, seed=4):
    from scipy.signal import bilinear, butter, lfilter, sosfiltfilt
    rng = np.random.default_rng(seed)
    n = int(T * fs)
    t = np.arange(n) / fs
    x = sosfiltfilt(butter(4, [10, 200], "band", fs=fs, output="sos"), rng.standard_normal(n))
    w0, z = 2 * np.pi * 120, 1 / 120                          # narrow random resonance (Q 60): NOT a line
    bz, az = bilinear([w0 ** 2], [1, 2 * z * w0, w0 ** 2], fs)
    x = x + 0.25 * lfilter(bz, az, rng.standard_normal(n))
    on = (t >= 10) & (t < 40)                                 # line at 60 Hz only in 10-40 s
    f_inst = 60.0 * (1 + 0.003 * np.sin(2 * np.pi * t / 20))  # slow speed wander
    line = np.where(on, 1.5 * np.sin(2 * np.pi * np.cumsum(f_inst) / fs), 0.0)
    return x + line, fs, line


def test_find_tones_blocks_and_extraction():
    x, fs, line = _sor_record()
    tm = vs.find_tones(x, fs, 5, 200)
    assert len(tm.tracks) == 1
    s = tm.tracks[0].summary()
    assert abs(s["f_mean"] - 60) < 0.5 and abs(s["amplitude"] / 1.5 - 1) < 0.1
    tonal = [b for b in tm.blocks if b["tones"]]
    assert all(10 - 1e-9 <= b["t0"] and b["t1"] <= 40 + 1e-9 for b in tonal) and len(tonal) >= 5
    assert not any(abs(tn[0] - 120) < 3 for b in tm.blocks for tn in b["tones"])   # resonance rejected
    m = (np.arange(x.size) / fs > 11) & (np.arange(x.size) / fs < 39)
    assert np.sqrt(np.mean((tm.lines - line)[m] ** 2)) < 0.15 * 1.5 / np.sqrt(2)


def test_lines_gain_and_kurtosis_model():
    x, fs, line = _sor_record()
    tm = vs.find_tones(x, fs, 5, 200)
    p = vs.FDSParams(fn=vs.natural_frequencies(5, 200, "octave", 6), Q=10, b=8)
    mr = mm.MultiResolution(p.fn, np.full(p.fn.size, 1.0), fs, 30.0)
    Kr = int(tm.duration / mr.hop)
    idx, mp = ln.frame_mapping(Kr, mr.K, "compress")
    lines = ln.TestLines(tm, mr.N, fs, mr.hop, idx, mp, p.b, 4 * tm.duration, np.random.default_rng(0))
    # compressed by 2, life = 4 record durations -> stationary-line gain (life / T)^(1/b)
    assert abs(lines.gains[0] / (4 * tm.duration / (mr.N / fs)) ** (1 / p.b) - 1) < 0.05
    # exact kurtosis of Gaussian + sine: 3 - 1.5 u^2 / (s + u)^2 for constant s, u
    Wk = mm.moment_operators(p.fn, p.Q, fs)
    em = mm.EnvelopeModel(Wk, p, mr.hop, mr.N / fs, mr.K, sub=2)
    em.set_lines(np.full((1, p.fn.size, em.Nt), 2.0))
    r = em.run(np.full(p.fn.size, 1.0), em.up(np.ones((p.fn.size, mr.K))), damage=False)
    s_ = r["s"].mean(axis=1)
    assert np.allclose(r["k"], 3 - 1.5 * 4.0 / (s_ + 2.0) ** 2, rtol=1e-6)


# ------------------------------------------------------------ end to end
def _road(T=120.0, fmax=100.0):
    x, fs = vs.load_signal(EXAMPLE)
    return vs.analyze_reference(x[: int(T * fs)], fs, fmin=5, fmax=fmax, Q=10, b=8, points_per_octave=6)


def test_mmvsd_profile_short():
    ref = _road()
    _, k_ref = mm.measure(ref.x, ref.fs, ref.params)
    r = mm.mmvsd(ref.fds, k_ref, 40.0, ref.fs, ref.params, seed=0, iterations=10,
                 psd_prior=(ref.psd_f, ref.psd))
    assert np.median(np.abs(r.measured["fds"] / ref.fds - 1)) < 0.1
    assert np.median(np.abs(r.measured["k_response"] - k_ref)) < 1.0


def test_mmvsd_method_and_workspace():
    ref = _road()
    res = run_method(ref, "mmvsd", 30.0, seed=1, iterations=12)
    assert res.method == "MMVSD" and res.repeats == 1 and abs(res.duration - 30.0) < 1.0
    for k in ("fds_measured", "mrs_measured", "k_response", "k_target", "modulation", "frames"):
        assert res.info.get(k) is not None, k
    ev = vs.evaluate(res.x, res.fs, ref, fds=res.info["fds_measured"], mrs=res.info["mrs_measured"],
                     k_response=res.info["k_response"])
    assert ev["fds_median_error_pct"] < 10 and ev["kz_median_abs_err"] < 1.0
    # workspace: record -> MMVSD result with k_z metrics, project round trip
    ws = vs.Workspace()
    ws.settings.update(fmin=5, fmax=100, ppo=6, Q=10, b=8)
    rid = ws.add_record(ref.x, ref.fs, "road")
    sid = ws.synthesize(rid, "mmvsd", 20.0, seed=2, iterations=8, modulation="pattern", mapping="compress")
    it = ws.get(sid)
    assert it.kind == "result" and it.meta["result"]["pattern_mapping"] == "compress"
    assert it.meta["eval"].get("kz_median_abs_err") is not None
    ws2 = vs.Workspace.load_project(ws.save_project())
    assert np.allclose(ws2.get(sid).x, it.x)


def test_web_api_mmvsd():
    from http.server import ThreadingHTTPServer
    from vibsynth import server as SV
    old_app, SV.APP = SV.APP, SV.App()          # private state, restored below for the other tests
    ref = _road(T=60.0)
    rid = SV.APP.ws.add_record(ref.x, ref.fs, "road")
    SV.APP.ws.settings.update(fmin=5, fmax=100, ppo=6)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), SV.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    try:
        assert "mmvsd" in call("/api/state")["methods"]
        call("/api/synth", {"target": rid, "method": "mmvsd", "duration": 20, "seed": 1, "iterations": 6,
                            "modulation": "random", "lines": "off", "max_excess": 20})
        while call("/api/job")["running"]:
            threading.Event().wait(0.2)
        out = call("/api/job/result")
        assert out["error"] is None, out
        v = call("/api/result/" + out["result"]["ids"][0])
        assert v["method"] == "MMVSD" and v["mmvsd"]["z"] and len(v["kz"]["result"]) == len(v["fn"])
        assert v["summary"]["kz_med"] is not None
        assert SV.APP.ws.get(out["result"]["ids"][0]).meta["options"]["max_excess"] == 20.0
    finally:
        srv.shutdown()
        SV.APP = old_app


def test_cli_mmvsd_smoke():
    from vibsynth.cli import main
    d = tempfile.mkdtemp()
    x, fs = vs.load_signal(EXAMPLE)
    rec = os.path.join(d, "rec.mat")
    vs.save_result(rec, x[: int(60 * fs)], fs)
    out, js = os.path.join(d, "out.mat"), os.path.join(d, "m.json")
    assert main(["synth", rec, "--method", "mmvsd", "--duration", "20", "--fmax", "100", "--ppo", "6",
                 "--iterations", "5", "--seed", "1", "--lines", "off", "--out", out, "--json", js]) == 0
    y, fsy = vs.load_signal(out)
    assert abs(y.size / fsy - 20) < 1.0
    m = json.load(open(js))
    assert m["method"] == "MMVSD" and m["kz_median_abs_err"] < 2.0


if __name__ == "__main__":
    import time
    for name, fun in sorted(globals().items()):
        if name.startswith("test_") and callable(fun):
            t0 = time.time()
            fun()
            print(f"PASS {name} ({time.time() - t0:.1f} s)")
