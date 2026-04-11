"""Tests for utility helpers in cfad.utils."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure
from scipy.stats import shapiro

from cfad.detection import AnomalyReport
from cfad.utils import plot_scores, simulate_levy_returns


@pytest.mark.parametrize("n", [1, 7, 128, 512])
def test_simulate_levy_shape(n: int):
    returns = simulate_levy_returns(n=n, seed=123)
    assert returns.shape == (n,)


def test_simulate_levy_gaussian_limit():
    returns = simulate_levy_returns(n=200, alpha=2.0, beta=0.0, scale=0.01, seed=7)
    _, pvalue = shapiro(returns)
    assert pvalue > 0.01


def test_plot_scores_returns_figure():
    n_scores = 20
    report = AnomalyReport(
        scores=np.linspace(0.0, 2.0, n_scores),
        cusum_pos=np.linspace(0.0, 3.0, n_scores),
        cusum_neg=np.zeros(n_scores, dtype=np.float64),
        alarm_indices=np.array([5, 12], dtype=np.int64),
        window_end_indices=np.arange(40, 40 + n_scores, dtype=np.int64),
        dates=None,
        mu0=0.0,
        sigma0=1.0,
        threshold=1.25,
    )
    returns = np.random.default_rng(0).normal(0.0, 0.01, 80).astype(np.float64)
    fig = plot_scores(report, returns=returns)
    assert isinstance(fig, Figure)

