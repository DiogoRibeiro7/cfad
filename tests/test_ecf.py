"""Tests for empirical characteristic function estimation."""
import numpy as np
import pytest
from cfad.empirical_cf import ecf_at, rolling_ecf


def test_ecf_at_gaussian():
    """ECF of N(0,1) samples should approximate exp(-xi^2/2)."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 1, 5000)
    xi = np.linspace(-3, 3, 64)
    ecf = ecf_at(returns, xi)
    true_cf = np.exp(-0.5 * xi**2)
    # L2 error should be small for n=5000
    l2 = np.mean(np.abs(ecf - true_cf) ** 2)
    assert l2 < 0.01, f"L2 error too large: {l2}"


def test_ecf_normalization():
    """phi(0) = 1 always."""
    rng = np.random.default_rng(0)
    returns = rng.standard_t(3, 200)
    xi = np.array([0.0])
    ecf = ecf_at(returns, xi)
    assert abs(ecf[0] - 1.0) < 1e-12


def test_rolling_ecf_shape():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 1, 300)
    xi = np.linspace(-5, 5, 32)
    ecf_mat, end_idx = rolling_ecf(returns, xi, window=60, step=5)
    expected_windows = (300 - 60) // 5 + 1
    assert ecf_mat.shape == (expected_windows, 32)
    assert end_idx.shape == (expected_windows,)
    assert end_idx[-1] <= 300


def test_rolling_ecf_consistency():
    """Rolling ECF window should match single-window ECF."""
    rng = np.random.default_rng(2)
    returns = rng.normal(0, 0.01, 120)
    xi = np.linspace(-5, 5, 16)
    ecf_mat, _ = rolling_ecf(returns, xi, window=60, step=60)
    ecf_w0 = ecf_at(returns[:60], xi)
    np.testing.assert_allclose(ecf_mat[0], ecf_w0, atol=1e-12)
