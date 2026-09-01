"""cfad — Characteristic Function Anomaly Detector.

CFAD detects changes in financial-return distributional shape using rolling
empirical characteristic functions (ECFs).  Each window is compared with the
Gaussian characteristic function fitted to the same mean and variance; the
resulting real-frequency discrepancy score is monitored with Page-CUSUM.

The package also provides parametric characteristic-function models,
goodness-of-fit tools, backtesting, bootstrap diagnostics, and market utilities.
"""

from cfad.api import compare_models, detect
from cfad.backtest import BacktestResult, WalkForwardBacktest
from cfad.bootstrap import bootstrap_scores, score_stability
from cfad.contour import contour_integral, gaussian_ecf_distance_scores
from cfad.detection import AnomalyReport, RollingDetector, StreamDetector
from cfad.empirical_cf import ecf_at, rolling_ecf
from cfad.gof import aic_table, cf_distance, epps_pulley_test, rolling_gof
from cfad.models.cgmy import CGMYCF
from cfad.models.gaussian import GaussianCF
from cfad.models.levy_stable import LevyStableCF
from cfad.models.nig import NIGCF
from cfad.residue_score import normalise_scores, rolling_pvalue, threshold_by_fpr
from cfad.sensitivity import (
    frequency_sensitivity,
    recommend_params,
    window_sensitivity,
)
from cfad.utils import load_spy_sample, plot_scores, simulate_levy_returns

__version__ = "0.2.1"
__author__ = "Diogo Ribeiro"

__all__ = [
    "detect",
    "compare_models",
    "WalkForwardBacktest",
    "BacktestResult",
    "bootstrap_scores",
    "score_stability",
    "RollingDetector",
    "StreamDetector",
    "AnomalyReport",
    "ecf_at",
    "rolling_ecf",
    "gaussian_ecf_distance_scores",
    "contour_integral",
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
    "window_sensitivity",
    "frequency_sensitivity",
    "recommend_params",
    "load_spy_sample",
    "plot_scores",
    "simulate_levy_returns",
]
