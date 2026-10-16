# Changelog

## 2.1.0 (2026-10-16)

New method and analyses; the profiles of the 2.0 methods are unchanged
(identical samples for the same inputs and seed).

* **MMVSD** (`mmvsd`, Moment-Matched Variable Spectral Density): prescribes the
  Fatigue Damage Spectrum and the kurtosis of the oscillator responses
  k_z(fn) with a time-varying PSD G(f, t) = P(f) m(f, t). Random modulation field
  with the burst width and duration of the record, or the record's own
  time-frequency pattern (`blocks` or `compress` mapping for shorter tests).
  Model with the exact response of the sampled oscillator, the loss of peaks
  between samples in the rainflow count and Dirlik's damage; design with exact
  (adjoint) Jacobians; closed loop on the realisation with early re-draws of
  noise outliers, coupled Gauss-Newton steps, polishing and local re-draws.
* **Sine-on-random blocks**: tonal lines present only in some blocks of a record
  are detected (block PSD plus a phase coherence test), tracked, extracted and
  reproduced as deterministic sines that keep their share of the life damage;
  MMVSD designs the random part for the residual.
* **Kurtosis of the oscillator responses** (k_z): computed with the FDS for
  records, combined over missions (life-weighted fourth moments), reported for
  every result (interface, command line report and JSON, benchmark).
* Interface: MMVSD options, k_z and modulation charts in the result view, k_z
  error column in the result table, k_z curves in the analysis tab, best-so-far
  line in the convergence chart.
* Command line: `--method mmvsd` with `--modulation`, `--mapping`, `--block`,
  `--lines`, `--max-excess`, `--freq-corr`, `--time-corr`, `--anchor`; the report
  figure shows k_z and the MRS ratio when available.
* Faster exact FDS: one pass returns FDS, MRS, variance and kurtosis of every
  response; long records use worker processes (all cores, `VIBSYNTH_WORKERS`
  to limit them), with results identical to the sequential computation. The
  workers exit on their own when the interface or script that started them is
  killed.
* Theory manual `docs/theory/vibsynth_theory.pdf`: every concept derived step by
  step, from the PSD to the MMVSD Jacobians.
* Example generators `examples/make_multisource.py` and `examples/make_sor.py`;
  `tools/kz_benchmark.py`; MMVSD and k_z in `tools/benchmark.py`.
* 23 new tests (engine exactness, worker processes, mission k_z, MMVSD model,
  Jacobians, frames, lines, workspace, web API, command line); CI runs an MMVSD
  smoke test.

## 2.0.0 (2026-10-11)

First public release.

* Browser interface (local, offline): records, analysis, specifications,
  missions, synthesis and project tabs; interactive charts with zoom, pan,
  log axes, read-out, PNG and CSV export.
* Multi-situation missions: nested series and parallel groups of records and
  specifications, with the damage share of each situation.
* Specifications: PSD breakpoints and swept sine, usable as targets, mission
  situations or time histories.
* Pre-processing: crop, filters, mean removal, resampling, gain, split.
* Export of PSD, FDS, MRS and statistics (CSV, XLSX, MAT, JSON) and of time
  histories (MAT, CSV, TXT, NPY, WAV).
* New methods: MRS control and sine on random.
* Diagnostics: spectrogram, PDF, running RMS and kurtosis, SDOF response,
  spectral kurtosis, kurtogram, shaker stroke check, PDF matching, sensitivity
  to b and Q.
* Project files (.vsproj) and log.
* PF/PSF: at least two bursts of three decay times of the lowest oscillator,
  and burst PSD lines left at zero by the inversion can now be corrected by
  the closed loop.
* Synthetic example record (`examples/make_example.py`) and benchmark tool.

## 1.0.0

* Python implementation of the algorithms of the PhD work: Gaussian
  mission synthesis, kFDS, RF, PF, PSF, MLV, VSD and PM.
* Envelope times carrier formulation, exact FDS to PSD inversion, closed-loop
  FDS correction with exact rainflow counting, vectorised rainflow.
* Command line and Python API, tests against reference implementations of the
  building blocks.
