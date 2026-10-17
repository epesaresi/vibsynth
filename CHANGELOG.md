# Changelog

## 2.2.0 (2026-10-17)

The compression limit of a record and the `edit` method; the profiles of the
2.0 and 2.1 methods are unchanged (identical samples for the same inputs and
seed).

* **Compression limit** (`vibsynth.edit.compressibility`, `Workspace.compressibility`,
  `analyze --compressibility`, analysis tab): damage Lorenz curves per oscillator
  (share of the damage in the most damaging fraction of the time), mean and
  worst curves, the largest compression that keeps 80 / 90 / 95 % of the damage
  at every oscillator by editing alone, and the closed form for a lognormal
  envelope.
* **`edit` method** (`--method edit`, interface, `Workspace.synthesize(..., "edit")`):
  windows of the record ranked by damage share (common or fair ranking), kept
  with a ring-down margin of the slowest oscillator and joined with
  cross-fades up to the test duration; beyond the compression limit a zero-phase
  frequency-dependent gain brings every oscillator back on the target FDS
  (adaptive relaxation, cap). The result carries the Lorenz curves, the limit,
  the stretches kept, the gain per oscillator and the kurtosis of the edited
  record; the interface shows them, the command line prints them.
* **Compressed kurtosis target** (`--compress-kurtosis`, interface checkbox) for
  kFDS, RF, MLV, VSD and PM: the kurtosis of the record edited to the test
  duration replaces the record's kurtosis as target.
* Theory manual: Part IV (Sections 12 to 15): the damage as a moment of the
  cycle histogram, why no scaling is exact for every exponent, the Lorenz curve
  and the limit, the lognormal closed form, the edit method and its spectral
  gain, the kurtosis under compression, the non-stationarity matrix and the
  response kurtosis spectrum; settings and diagnostics sections updated. The
  numbering of Sections 1 to 11 is unchanged.
* `tools/compression_benchmark.py` and `docs/benchmark_compression.json`: damage
  ratio at b = 3, 5, 7, 9 for tests designed at b = 5, and the compression limit.
* 13 new tests (Lorenz curves, rankings, closed form, method, exponent
  robustness, compressed kurtosis, workspace, project round trip, web API,
  command line).

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
