"""Tests of the 2.2 features: compression limit (damage Lorenz curve), the
edit method, the compressed kurtosis target, workspace, web API and command
line. Run with `python -m pytest tests` or `python tests/test_edit.py`."""
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
from vibsynth import edit as ED  # noqa: E402
from vibsynth.fds import response_spectra  # noqa: E402
from vibsynth.pipeline import run_method  # noqa: E402

EXAMPLE = os.path.join(HERE, "..", "examples", "example_road.mat")


def _ref(T=120.0, fmax=100.0, b=5.0, Q=10.0, ppo=3):
    x, fs = vs.load_signal(EXAMPLE)
    return vs.analyze_reference(x[: int(T * fs)], fs, fmin=5, fmax=fmax, Q=Q, b=b, points_per_octave=ppo)


def _bursty(T=120.0, fs=500.0, seed=3):
    """Gaussian carrier with a slow lognormal envelope: a record whose damage
    sits in a fraction of the time (kurtosis about 6)."""
    from scipy import signal as sps
    rng = np.random.default_rng(seed)
    n = int(T * fs)
    g = rng.standard_normal(n)
    b, a = sps.butter(4, [5 / (fs / 2), 80 / (fs / 2)], "band")
    g = sps.lfilter(b, a, g)
    w = sps.lfilter(*sps.butter(2, 0.1 / (fs / 2)), rng.standard_normal(n))
    w = (w - w.mean()) / w.std()
    s = np.sqrt(np.exp(-0.8 ** 2 / 2 + 0.8 * w))
    return s * g / (s * g).std() * 5.0, fs


# ------------------------------------------------------------- Lorenz curve
def test_lorenz_curves_and_limits():
    x, fs = _bursty()
    p = vs.FDSParams(fn=vs.natural_frequencies(5, 100, "octave", 3), Q=10, b=5)
    c = ED.compressibility(x, fs, p, window_s=2.0)
    q, Lm, Lw = c["q"], c["mean"], c["worst"]
    assert np.all(np.diff(q) > 0) and abs(q[-1] - 1) < 1e-12
    assert np.all(np.diff(Lm) >= -1e-12) and np.all(np.diff(Lw) >= -1e-12)   # cumulative shares
    assert np.all(Lw <= Lm + 1e-12) and abs(Lm[-1] - 1) < 1e-9 and abs(Lw[-1] - 1) < 1e-9
    assert np.all(Lm >= q - 1e-9)                       # the ranking can only beat a uniform spread
    am = c["alpha_max"]
    assert am["95"]["worst"] <= am["90"]["worst"] <= am["80"]["worst"]
    assert am["90"]["worst"] <= am["90"]["mean"]
    assert am["90"]["worst"] > 1.3                      # a bursty record is compressible
    assert np.all((c["lognormal"] >= 0) & (c["lognormal"] <= 1))
    assert c["n_windows"] == int(120 / 2)


def test_stationary_gaussian_is_not_compressible():
    rng = np.random.default_rng(0)
    from scipy import signal as sps
    fs = 500.0
    x = sps.lfilter(*sps.butter(4, [5 / 250, 80 / 250], "band"), rng.standard_normal(int(120 * fs)))
    p = vs.FDSParams(fn=vs.natural_frequencies(5, 100, "octave", 3), Q=10, b=5)
    c = ED.compressibility(x, fs, p, window_s=2.0)
    # the curve stays close to the diagonal: keeping half of the time keeps well under 90 % of the damage
    i = int(np.searchsorted(c["q"], 0.5))
    assert c["mean"][i] < 0.85 and c["alpha_max"]["90"]["worst"] < 1.6


def test_fair_ranking_serves_the_worst_oscillator():
    # oscillators 0 and 2 are damaged in windows 0 and 1, oscillator 1 in windows 2 to 6
    W = np.zeros((3, 7))
    W[0, :2] = 0.5
    W[2, :2] = 0.5
    W[1, 2:7] = 0.2
    common = ED.rank_windows(W, "common")
    fair = ED.rank_windows(W, "fair")
    assert sorted(common) == list(range(7)) and sorted(fair) == list(range(7))
    _, _, worst_c, _ = ED.lorenz_curves(W, common)
    _, _, worst_f, _ = ED.lorenz_curves(W, fair)
    assert worst_c[1] == 0.0 and worst_f[1] > 0.0          # the common ranking leaves oscillator 1 empty
    assert np.all(worst_f >= worst_c - 1e-12)


def test_lognormal_closed_form():
    q = np.linspace(0.05, 0.95, 19)
    L6 = ED.lognormal_lorenz(q, 6.0, 5.0)
    L4 = ED.lognormal_lorenz(q, 4.0, 5.0)
    assert np.all(np.diff(L6) > 0) and np.all(L6 > q) and np.all(L6 > L4)      # burstier: higher curve
    a = ED.lognormal_alpha_max(0.9, 6.0, 5.0)
    assert 4.0 < a < 5.5                                                         # manual, Table 12.1: 4.7
    assert abs(ED.lognormal_lorenz(np.array([1 / a]), 6.0, 5.0)[0] - 0.9) < 1e-6


# ------------------------------------------------------------------ method
def test_edit_keeps_the_damage_and_reports_the_limit():
    ref = _ref()
    res = run_method(ref, "edit", 30.0)
    e = res.info["edit"]
    assert res.method == "edit" and res.repeats == 1
    assert abs(res.x.size / res.fs - 30.0) < 1.5
    assert abs(e["alpha_requested"] - 4.0) < 1e-9 and 1.0 < e["alpha_limit"] < 4.0
    assert all(0 <= s0 < s1 <= ref.x.size / ref.fs + 1e-9 for s0, s1 in e["segments_s"])
    assert e["kept_share"] > 0.7 and e["kept_share_min"] <= e["kept_share"]
    ratio = res.info["fds_measured"] / ref.fds
    assert np.max(np.abs(ratio - 1)) < 0.12                 # closed loop on the gain
    assert np.all(e["gain_fn"] >= 1 / 2.0) and np.all(e["gain_fn"] <= 2.0)
    # the measured FDS stored in the result is the exact one
    assert np.allclose(response_spectra(res.x, res.fs, ref.params, stats=False)["fds"], res.info["fds_measured"])


def test_edit_without_gain_drops_only_damage():
    ref = _ref()
    res = run_method(ref, "edit", 30.0, gain=False)
    e = res.info["edit"]
    assert np.all(e["gain_fn"] == 1) and e["gain_iterations"] == 0
    ratio = res.info["fds_measured"] / ref.fds
    assert np.all(ratio < 1.05) and np.exp(np.mean(np.log(ratio))) > 0.6
    # a shorter edit keeps less
    res2 = run_method(ref, "edit", 15.0, gain=False)
    assert res2.info["edit"]["kept_share"] <= e["kept_share"] + 1e-9


def test_edit_is_robust_to_the_exponent():
    """The design is for b = 5; at b = 9 the edited test (no gain) still
    carries most of the damage at every fn, while a Gaussian test designed at
    b = 5 is off by a factor that changes from one fn to the next."""
    ref = _ref()
    ed = run_method(ref, "edit", 30.0, gain=False)
    ga = run_method(ref, "gaussian", 30.0, seed=1, iterations=8)
    p9 = vs.FDSParams(fn=ref.params.fn, Q=ref.params.Q, b=9.0)
    D9 = response_spectra(ref.x, ref.fs, p9, stats=False)["fds"]
    r_ed = response_spectra(ed.x, ed.fs, p9, stats=False)["fds"] / D9
    r_ga = response_spectra(ga.x, ga.fs, p9, stats=False)["fds"] / D9
    assert np.exp(np.mean(np.log(r_ed))) > 0.75
    spread_ed, spread_ga = np.max(r_ed) / np.min(r_ed), np.max(r_ga) / np.min(r_ga)
    assert spread_ed < 2.5 and spread_ed < spread_ga            # the Gaussian tail is wrong fn by fn


def test_edit_longer_than_record_returns_the_record():
    ref = _ref(T=60.0)
    res = run_method(ref, "edit", 90.0, gain=False)
    assert res.x.size == ref.x.size and np.allclose(res.x, ref.x - ref.x.mean())
    assert any("unchanged" in w for w in res.warnings)


def test_edit_needs_a_record():
    spec = vs.PSDSpec([5, 20, 80, 100], [0.05, 0.3, 0.3, 0.05], "spec")
    from vibsynth.mission import spec_target
    p = vs.FDSParams(fn=vs.natural_frequencies(5, 100, "octave", 3), Q=10, b=5)
    ref = spec_target(spec, p, 600.0, 500.0, name="spec")
    try:
        run_method(ref, "edit", 30.0)
    except ValueError as exc:
        assert "record" in str(exc)
    else:
        raise AssertionError("edit accepted a target without time history")


def test_compressed_kurtosis_target():
    x, fs = _bursty()
    ref = vs.analyze_reference(x, fs, fmin=5, fmax=100, Q=10, b=5, points_per_octave=3)
    k_c = ED.compressed_kurtosis(ref, 30.0)
    assert 3.0 < k_c < ref.stats["kurtosis"]                 # dropping quiet time lowers the kurtosis
    res = run_method(ref, "kfds", 30.0, seed=1, iterations=6, compress_kurtosis=True)
    assert abs(res.info.get("k_target", vs.kurtosis(res.x)) - k_c) < 0.6 or abs(vs.kurtosis(res.x) - k_c) < 0.6


def test_workspace_edit_and_compressibility():
    ws = vs.Workspace()
    ws.settings.update(fmin=5, fmax=100, ppo=3, Q=10, b=5)
    x, fs = vs.load_signal(EXAMPLE)
    rid = ws.add_record(x[: int(120 * fs)], fs, "road")
    out = ws.synthesize(rid, "edit", 30.0, ranking="fair", retention=0.8)
    it = ws.get(out)
    e = it.meta["result"]["edit"]
    assert it.kind == "result" and it.meta["method"] == "edit"
    assert e["ranking"] == "fair" and e["retention"] == 0.8 and len(e["lorenz_q"]) == len(e["lorenz_mean"])
    assert it.meta["eval"]["fds_max_error_pct"] < 15
    c = ws.compressibility(rid)
    assert set(c) >= {"q", "mean", "worst", "alpha_max", "window_s", "lognormal_alpha90", "name"}
    # the project round trip keeps the result and its edit metadata
    ws2 = vs.Workspace.load_project(ws.save_project())
    assert ws2.get(out).meta["result"]["edit"]["n_segments"] == e["n_segments"]


def test_web_api_edit():
    from http.server import ThreadingHTTPServer
    from vibsynth import server as SV
    old_app, SV.APP = SV.APP, SV.App()
    x, fs = vs.load_signal(EXAMPLE)
    rid = SV.APP.ws.add_record(x[: int(90 * fs)], fs, "road")
    SV.APP.ws.settings.update(fmin=5, fmax=100, ppo=3, Q=10, b=5)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), SV.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    def wait():
        while call("/api/job")["running"]:
            threading.Event().wait(0.2)
        out = call("/api/job/result")
        assert out["error"] is None, out
        return out["result"]
    try:
        assert "edit" in call("/api/state")["methods"]
        call("/api/compressibility", {"ids": [rid], "window_s": 2})
        c = wait()[rid]
        assert c["alpha_max"]["90"]["worst"] >= 1 and len(c["q"]) == len(c["worst"])
        call("/api/synth", {"target": rid, "method": "edit", "duration": 30, "ranking": "common", "gain": "true",
                            "gain_max": 2, "retention": 0.9})
        res_id = wait()["ids"][0]
        v = call("/api/result/" + res_id)
        assert v["method"] == "edit" and v["edit"]["n_segments"] >= 1 and len(v["edit"]["gain_fn"]) == len(v["fn"])
        assert any("Compression" in r[0] for r in v["table"])
    finally:
        srv.shutdown()
        SV.APP = old_app


def test_cli_edit_and_compressibility():
    from vibsynth.cli import main
    d = tempfile.mkdtemp()
    x, fs = vs.load_signal(EXAMPLE)
    rec = os.path.join(d, "rec.mat")
    vs.save_result(rec, x[: int(90 * fs)], fs)
    out, js, lz = os.path.join(d, "out.mat"), os.path.join(d, "m.json"), os.path.join(d, "lorenz.csv")
    assert main(["analyze", rec, "--fmax", "100", "--ppo", "3", "--b", "5", "--compressibility", "--lorenz", lz]) == 0
    L = np.loadtxt(lz, delimiter=",", skiprows=1)
    assert L.shape[1] == 4 and np.all(np.diff(L[:, 0]) > 0)
    assert main(["synth", rec, "--method", "edit", "--duration", "30", "--fmax", "100", "--ppo", "3", "--b", "5",
                 "--out", out, "--json", js]) == 0
    y, fsy = vs.load_signal(out)
    assert abs(y.size / fsy - 30) < 1.5
    m = json.load(open(js))
    assert m["method"] == "edit" and m["fds_max_error_pct"] < 15


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
