"""Tests of the 2.0 features: specifications, missions, pre-processing, export,
diagnostics, MRS control, sine on random, workspace and web API."""
import io
import json
import os
import sys
import threading
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import vibsynth as vs  # noqa: E402
from vibsynth.signals import trapezoid  # noqa: E402
from vibsynth import diagnostics as DG, export as EX, preprocess as PP  # noqa: E402
from vibsynth.mission import Group, Situation, combine, spec_target  # noqa: E402
from vibsynth.synthesis import detect_tones, mrs_control, sine_on_random  # noqa: E402

FS = 1000.0


def _sig(seed=0, T=120.0, k_env=0.4):
    rng = np.random.default_rng(seed)
    f = np.linspace(0, FS / 2, 2001)
    G = np.where((f > 8) & (f < 250), 1.0 / (1 + ((f - 60) / 40) ** 2), 0.0)
    g = vs.psd_to_timeseries(f, G, FS, T, rng=rng)
    e = np.repeat(np.exp(k_env * rng.normal(size=int(T / 2) + 1)), int(2 * FS))[: g.size]
    return g * e / np.sqrt(np.mean(e ** 2))


def _p():
    return vs.FDSParams(fn=vs.natural_frequencies(10, 250, "octave", 6), Q=10, b=6)


def test_psd_spec_rms_and_fds():
    s = vs.PSDSpec([10, 100], [1.0, 1.0])
    assert abs(s.rms() - np.sqrt(90)) < 0.05
    p = _p()
    D1, _ = s.fds_mrs(p, 100.0)
    D2, _ = s.fds_mrs(p, 200.0)
    assert np.allclose(D2, 2 * D1)
    x = s.signal(FS, 200.0, seed=1)
    assert abs(x.std() / np.sqrt(90) - 1) < 0.03


def test_sine_sweep():
    s = vs.SineSweepSpec([10, 160], [5, 5], rate=1.0, mode="log", sweeps=2)
    assert abs(s.pass_duration - 240.0) < 1e-9 and abs(s.duration - 480.0) < 1e-9
    x = s.signal(2000.0)
    assert abs(x.size - 480 * 2000) <= 1 and abs(np.max(np.abs(x)) - 5) < 0.01
    lin = vs.SineSweepSpec([10, 110], [1, 1], rate=2.0, mode="linear")
    assert abs(lin.pass_duration - 50.0) < 1e-9


def test_mission_rules():
    p = _p()
    a = vs.analyze_reference(_sig(1), FS, fmin=10, fmax=250, points_per_octave=6, Q=10, b=6, life=3600)
    b = vs.analyze_reference(_sig(2, k_env=0.8), FS, fmin=10, fmax=250, points_per_octave=6, Q=10, b=6,
                             life=7200)
    ser = combine(Group("series", [Situation("a", a), Situation("b", b)]))
    assert np.allclose(ser.fds, a.fds + b.fds) and ser.life == 3600 + 7200
    m2 = (3600 * a.moments[0] + 7200 * b.moments[0]) / 10800
    assert abs(ser.stats["rms"] - np.sqrt(m2)) < 1e-9
    par = combine(Group("parallel", [Situation("a", a), Situation("b", b)]))
    assert np.allclose(par.fds, np.maximum(a.fds, b.fds))
    assert np.allclose(par.mrs, np.maximum(a.mrs, b.mrs))
    sp = spec_target(vs.PSDSpec([10, 250], [0.5, 0.5]), a.params, 1800.0, FS)
    nested = combine(Group("series", [Situation("a", a), Group("parallel", [Situation("b", b), Situation("s", sp)])]))
    assert np.allclose(nested.fds, a.fds + np.maximum(b.fds, sp.fds))


def test_kfds_on_mission_target():
    a = vs.analyze_reference(_sig(3), FS, fmin=10, fmax=250, points_per_octave=6, Q=10, b=6, life=3600)
    b = vs.analyze_reference(_sig(4, k_env=0.7), FS, fmin=10, fmax=250, points_per_octave=6, Q=10, b=6,
                             life=1800)
    m = combine(Group("series", [Situation("a", a), Situation("b", b)]))
    r = vs.kfds(m.fds, 120.0, FS, m.params, m.stats["kurtosis"], segments=m.segments, seed=1)
    ev = vs.evaluate(r.x, r.fs, m, fds=r.info["fds_measured"], mrs=r.info["mrs_measured"])
    assert ev["fds_max_error_pct"] < 6 and abs(ev["stats"]["kurtosis"] - m.stats["kurtosis"]) < 0.12


def test_preprocess():
    x = _sig(5, T=20)
    assert PP.crop(x, FS, 2, 5).size == 3000
    y, fs2 = PP.resample(x, FS, 400.0)
    assert fs2 == 400.0 and abs(y.size - 8000) <= 1
    parts = PP.split(x, FS, [5, 12.5])
    assert [p.size for p in parts] == [5000, 7500, 7500]
    h = PP.bandpass(x, FS, 50, None)
    f, P = vs.welch_psd(h, FS, 1.0)
    f0, P0 = vs.welch_psd(x, FS, 1.0)
    assert P[(f > 10) & (f < 20)].mean() < 1e-2 * P0[(f0 > 10) & (f0 < 20)].mean()


def test_export_formats():
    cur = {"f": np.arange(5.0), "psd": np.arange(5.0) ** 2}
    for fmt in ("csv", "mat", "json", "xlsx"):
        try:
            b = EX.curves_bytes(cur, fmt, {"name": "t"})
        except ValueError:
            assert fmt == "xlsx"          # openpyxl not installed
            continue
        assert len(b) > 20
    from scipy.io import loadmat
    d = loadmat(io.BytesIO(EX.curves_bytes(cur, "mat")))
    assert np.allclose(d["psd"].ravel(), cur["psd"])
    x = np.sin(np.arange(1000) / 10)
    import wave
    w = wave.open(io.BytesIO(EX.series_bytes(x, 1000, "wav")))
    assert w.getframerate() == 1000 and w.getnframes() == 1000


def test_diagnostics():
    x = _sig(6, T=60)
    c, h, g = DG.pdf(x)
    assert abs(trapezoid(h, c) - 1) < 0.02
    t = np.arange(int(100 * FS)) / FS
    s = 10 * np.sin(2 * np.pi * 5 * t)                       # 10 m/s^2 at 5 Hz
    r = DG.shaker_check(s, FS, disp_limit=0.001, hp=0.5)
    d_exact = 2 * 10 / (2 * np.pi * 5) ** 2                    # peak-to-peak displacement
    assert abs(r["peak_to_peak_disp"] / d_exact - 1) < 0.02 and not r["ok"]["disp"]
    f, sk = DG.spectral_kurtosis(np.random.default_rng(0).normal(size=200000), FS, 256)
    assert abs(np.median(sk)) < 0.1
    y = DG.match_pdf(np.random.default_rng(1).normal(size=20000), x)
    assert abs(vs.kurtosis(y) / vs.kurtosis(x) - 1) < 0.05


def test_mrs_control_and_sor():
    ref = vs.analyze_reference(_sig(7), FS, fmin=10, fmax=250, points_per_octave=6, Q=10, b=6)
    r = mrs_control(ref.mrs, 60.0, FS, ref.params, seed=1)
    ratio = r.info["mrs_measured"] / ref.mrs
    assert np.max(np.abs(ratio - 1)) < 0.08
    t = np.arange(int(120 * FS)) / FS
    y = _sig(8) + 3 * np.sin(2 * np.pi * 47.3 * t)
    f, P = vs.welch_psd(y, FS, 0.25)
    tones = detect_tones(f, P, 10, 250)
    assert len(tones) == 1 and abs(tones[0][0] - 47.3) < 0.3 and abs(tones[0][1] - 3) < 0.3
    ref2 = vs.analyze_reference(y, FS, fmin=10, fmax=250, points_per_octave=6, Q=10, b=6)
    s = sine_on_random(ref2.fds, 60.0, FS, ref2.params, tones, ref2.life, seed=2)
    ev = vs.evaluate(s.x, s.fs, ref2, fds=s.info["fds_measured"], mrs=s.info["mrs_measured"])
    assert ev["fds_max_error_pct"] < 8


def test_workspace_project_roundtrip():
    ws = vs.Workspace()
    ws.settings.update(fmin=10, fmax=250, ppo=6, Q=10, b=6)
    rid = ws.add_record(_sig(9, T=60), FS, "rec")
    a, b = ws.split(rid, [30])
    sid = ws.add_spec(vs.PSDSpec([10, 250], [0.3, 0.3], "spec"))
    mid = ws.set_mission({"name": "M", "root": {"type": "group", "kind": "series", "children": [
        {"type": "situation", "item": a, "life_h": 2}, {"type": "group", "kind": "parallel", "children": [
            {"type": "situation", "item": b, "life_h": 1}, {"type": "situation", "item": sid, "life_h": 1}]}]}})
    T = ws.target(mid)
    assert abs(T.life - 3 * 3600) < 1e-6
    res = ws.synthesize(mid, "gaussian", 60.0, seed=1)
    assert ws.get(res).meta["eval"]["fds_max_error_pct"] < 6
    ws2 = vs.Workspace.load_project(ws.save_project())
    assert set(ws2.items) == set(ws.items) and mid in ws2.missions
    assert np.allclose(ws2.target(mid).fds, T.fds, rtol=1e-4)
    assert "synthesis" in ws.logfile()


def test_web_api_smoke():
    from http.server import ThreadingHTTPServer
    from vibsynth import server as SV
    srv = ThreadingHTTPServer(("127.0.0.1", 0), SV.Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        st = json.load(urllib.request.urlopen(base + "/api/state"))
        assert "methods" in st and "kfds" in st["methods"]
        html = urllib.request.urlopen(base + "/").read()
        assert b"vibsynth" in html
        js = urllib.request.urlopen(base + "/static/chart.js").read()
        assert b"VChart" in js
        buf = io.BytesIO()
        np.save(buf, np.c_[np.arange(5000) / FS, _sig(10, T=5)])
        up = json.load(urllib.request.urlopen(urllib.request.Request(
            base + "/api/upload?name=x.npy", data=buf.getvalue(), method="POST")))
        imp = json.load(urllib.request.urlopen(urllib.request.Request(
            base + "/api/import", data=json.dumps({"upload_id": up["upload_id"]}).encode(), method="POST",
            headers={"Content-Type": "application/json"})))
        iid = imp["ids"][0]
        ser = json.load(urllib.request.urlopen(base + f"/api/item/{iid}/series?n=500"))
        assert len(ser["t"]) <= 500 and abs(imp["state"]["items"][0]["fs"] - FS) < 1e-6
    finally:
        srv.shutdown()


def test_example_record_generator():
    import runpy
    mod = runpy.run_path(os.path.join(os.path.dirname(__file__), "..", "examples", "make_example.py"))
    x, fs = mod["make"](duration=60.0)
    s = vs.statistics(x)
    assert fs == 500.0 and x.size == 30000
    assert 4.0 < s["kurtosis"] < 9.0 and 4.0 < s["rms"] < 8.0


if __name__ == "__main__":
    fails = 0
    for name, fun in list(globals().items()):
        if name.startswith("test_") and callable(fun):
            try:
                fun()
                print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                fails += 1
                import traceback
                print("FAIL", name, repr(exc))
                traceback.print_exc(limit=2)
    sys.exit(1 if fails else 0)
