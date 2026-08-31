"""Regression tests anchored to the corrected CFAD numerical contract."""

from __future__ import annotations

import numpy as np

from cfad import detect
from cfad.empirical_cf import ecf_at
from cfad.models.gaussian import GaussianCF


def test_regression_ecf_at():
    rng = np.random.default_rng(seed=0)
    returns = rng.normal(0, 0.01, 200)
    xi = np.array([-1.0, 0.0, 1.0])
    result = ecf_at(returns, xi)
    expected = np.array(
        [
            0.9999537960307572 - 0.00015265113368525076j,
            1.0 + 0.0j,
            0.9999537960307572 + 0.00015265113368525076j,
        ],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(result, expected, atol=1e-6, rtol=0.0)


def test_regression_gaussian_fit():
    rng = np.random.default_rng(seed=0)
    returns = rng.normal(0.001, 0.02, 1000)
    model = GaussianCF().fit(returns)
    expected_mu = 3.943446474026183e-05
    expected_sigma = 0.019544834108522945
    assert np.isclose(model.mu, expected_mu, atol=1e-6, rtol=0.0)
    assert np.isclose(model.sigma, expected_sigma, atol=1e-6, rtol=0.0)


def test_regression_detect_n_windows():
    rng = np.random.default_rng(seed=0)
    returns = rng.normal(0, 0.01, 500)
    report = detect(returns, window=60, step=5)
    assert len(report.scores) == 89


def test_regression_detect_mu0_matches_independent_score_reconstruction():
    """Calibration mean must match the corrected Gaussian-reference statistic."""
    rng = np.random.default_rng(seed=0)
    returns = rng.normal(0, 0.01, 500)
    window = 60
    step = 5
    xi = np.linspace(-10.0, 10.0, 128)

    manual_scores: list[float] = []
    for start in range(0, len(returns) - window + 1, step):
        sample = returns[start : start + window]
        empirical = ecf_at(sample, xi)
        mean = float(np.mean(sample))
        std = float(np.std(sample, ddof=1))
        gaussian = np.exp(1j * mean * xi - 0.5 * std**2 * xi**2)
        squared = np.abs(empirical - gaussian) ** 2
        manual_scores.append(
            float(np.sqrt(np.trapezoid(squared, xi) / (xi[-1] - xi[0])))
        )

    n_cal = max(10, int(0.3 * len(manual_scores)))
    expected_mu0 = float(np.mean(manual_scores[:n_cal]))

    report = detect(returns, window=window, step=step)
    assert report.mu0 == np.testing.assert_allclose(
        [report.mu0],
        [expected_mu0],
        atol=1e-12,
        rtol=1e-10,
    )
