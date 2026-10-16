"""vibsynth: vibration Mission Synthesis and kurtosis control (E. Pesaresi, PhD 2021).

Python implementation of the mission synthesis and kurtosis control algorithms
of the PhD work, extended with MMVSD (prescribed FDS and kurtosis spectrum of
the oscillator responses) and the detection of sine-on-random blocks.
"""
__version__ = "2.1.0"

from .signals import natural_frequencies, welch_psd, psd_to_timeseries, statistics, kurtosis
from .fds import FDSParams, fds_time, fds_fast, fds_gaussian, response_spectra, response_variance_operator
from .design import invert_fds
from .analysis import analyze_reference, evaluate, response_statistics, Reference
from .synthesis import (SynthesisResult, gaussian, kfds, rf, pf, psf, mlv, vsd, pm, METHODS)
from .io import load_signal, save_result
from .spec import PSDSpec, SineSweepSpec
from .mission import Situation, Group, combine
from .workspace import Workspace
from .lines import find_tones
from .mmvsd import mmvsd as mmvsd_profile, MMVSDResult
from . import preprocess, diagnostics, export
