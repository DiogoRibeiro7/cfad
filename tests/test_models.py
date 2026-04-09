"""Tests for parametric CF models."""
import numpy as np
import pytest
from cfad.models.gaussian import GaussianCF
from cfad.models.nig import NIGCF


def test_gaussian_cf_normalization():
    g = GaussianCF(mu=0.0, sigma=0.02)
    xi = np.array([0.0])
    assert abs(g.cf(xi)[0] - 1.0) < 1e-12


def test_gaussian_is_analytic():
    assert GaussianCF.is_analytic is True


def test_nig_is_not_analytic():
    assert NIGCF.is_analytic is False


def test_gaussian_fit():
    rng = np.random.default_rng(42)
    data = rng.normal(0.001, 0.02, 2000)
    g = GaussianCF().fit(data)
    assert abs(g.mu - 0.001) < 0.005
    assert abs(g.sigma - 0.02) < 0.003


def test_nig_cf_normalization():
    n = NIGCF(alpha=10.0, beta=0.0, delta=0.1, mu=0.0)
    xi = np.array([0.0])
    assert abs(n.cf(xi)[0] - 1.0) < 1e-12


def test_gaussian_entire_contour():
    """Entire function: contour integral should be near zero."""
    from cfad.contour import contour_integral
    g = GaussianCF(mu=0.0, sigma=0.02)
    re, im = contour_integral(
        lambda z: np.exp(1j * g.mu * z - 0.5 * g.sigma**2 * z**2),
        xi_min=-5, xi_max=5, height=0.3, n_pts=512
    )
    assert abs(re) < 1e-6 and abs(im) < 1e-6, f"Gaussian residue non-zero: {re}, {im}"
