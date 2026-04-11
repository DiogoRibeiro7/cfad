"""Regression tests anchored to fixed numerical outputs."""

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
    g = GaussianCF().fit(returns)
    expected_mu = 3.943446474026183e-05
    expected_sigma = 0.019544834108522945
    assert np.isclose(g.mu, expected_mu, atol=1e-6, rtol=0.0)
    assert np.isclose(g.sigma, expected_sigma, atol=1e-6, rtol=0.0)


def test_regression_detect_n_windows():
    rng = np.random.default_rng(seed=0)
    returns = rng.normal(0, 0.01, 500)
    report = detect(returns, window=60, step=5)
    expected_n_windows = 89
    assert len(report.scores) == expected_n_windows


def test_regression_detect_mu0():
    rng = np.random.default_rng(seed=0)
    returns = rng.normal(0, 0.01, 500)
    report = detect(returns, window=60, step=5)
    expected_mu0 = 5.886403170212475
    assert np.isclose(report.mu0, expected_mu0, atol=1e-8, rtol=0.0)
