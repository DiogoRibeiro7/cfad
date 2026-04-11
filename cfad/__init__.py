"""
cfad — Characteristic Function Anomaly Detector
================================================

Detect structural breaks in financial time series using contour integrals
of the empirical characteristic function in the complex plane.

Quick start
-----------
>>> from cfad import detect
>>> report = detect(returns, window=60, h=4.0)
>>> print(report.summary())

Core modules
------------
cfad.api            : high-level entry points (detect, compare_models)
cfad.empirical_cf   : rolling ECF estimation
cfad.contour        : contour integration engine
cfad.detection      : RollingDetector, AnomalyReport
cfad.models         : parametric CF models (Gaussian, NIG, CGMY, Lévy-stable)
cfad._ext           : Cython C extensions (compiled separately)

References
----------
Ribeiro, D. (2025). Residues as detectors: contour integral anomaly scoring
  for financial time series. Journal of Open Source Software (submitted).

Cont, R. & Tankov, P. (2004). Financial Modelling with Jump Processes.
  Chapman & Hall/CRC.

Epps, T. W. & Pulley, L. B. (1983). A test for normality based on the
  empirical characteristic function. Biometrika, 70(3), 723-726.
"""

from cfad.api import compare_models, detect
from cfad.detection import AnomalyReport, RollingDetector, StreamDetector
from cfad.empirical_cf import ecf_at, rolling_ecf
from cfad.gof import aic_table, cf_distance, epps_pulley_test, rolling_gof
from cfad.models.cgmy import CGMYCF
from cfad.models.gaussian import GaussianCF
from cfad.models.levy_stable import LevyStableCF
from cfad.models.nig import NIGCF
from cfad.residue_score import normalise_scores, rolling_pvalue, threshold_by_fpr
from cfad.utils import load_spy_sample, plot_scores, simulate_levy_returns

__version__ = "0.1.0"
__author__ = "Diogo Ribeiro"
__email__ = "dfr@esmad.ipp.pt"
__all__ = [
    "detect",
    "compare_models",
    "RollingDetector",
    "StreamDetector",
    "AnomalyReport",
    "ecf_at",
    "rolling_ecf",
    "cf_distance",
    "aic_table",
    "rolling_gof",
    "epps_pulley_test",
    "GaussianCF",
    "NIGCF",
    "CGMYCF",
    "LevyStableCF",
    "normalise_scores",
    "rolling_pvalue",
    "threshold_by_fpr",
    "load_spy_sample",
    "plot_scores",
    "simulate_levy_returns",
]
