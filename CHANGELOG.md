# Changelog

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
