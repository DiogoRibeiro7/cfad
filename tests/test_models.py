"""Tests for parametric CF models."""

import numpy as np
import pytest

from cfad.contour import contour_integral
from cfad.models.cgmy import CGMYCF
from cfad.models.gaussian import GaussianCF
from cfad.models.levy_stable import LevyStableCF
from cfad.models.nig import NIGCF
from cfad.residue_score import normalise_scores, rolling_pvalue, threshold_by_fpr
from cfad.utils import simulate_levy_returns


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
    """Entire Gaussian CF integrates to zero around a closed contour."""
    g = GaussianCF(mu=0.0, sigma=0.02)
    value = contour_integral(
        lambda z: np.exp(1j * g.mu * z - 0.5 * g.sigma**2 * z**2),
        xi_min=-5,
        xi_max=5,
        height=0.3,
        n_pts=64,
    )
    assert abs(value) < 1e-10


def test_cgmy_cf_normalization():
    c = CGMYCF(C=1.0, G=5.0, M=10.0, Y=0.5)
    xi = np.array([0.0])
    assert abs(c.cf(xi)[0] - 1.0) < 1e-12


def test_cgmy_fit():
    rng = np.random.default_rng(123)
    data = rng.standard_t(5, size=1500) * 0.01
    c = CGMYCF()
    c.fit(data)
    assert 0 < c.C
    assert 0 < c.G
    assert 0 < c.M
    assert 0 < c.Y < 2


def test_levy_stable_cf_normalization():
    model = LevyStableCF(alpha=1.7, beta=0.0, c=0.01, mu=0.0)
    xi = np.array([0.0])
    assert abs(model.cf(xi)[0] - 1.0) < 1e-12


def test_levy_stable_is_not_analytic():
    assert LevyStableCF.is_analytic is False


def test_levy_stable_fit():
    rng = np.random.default_rng(123)
    data = rng.standard_t(4, size=1500) * 0.01
    model = LevyStableCF()
    model.fit(data)
    assert 0 < model.alpha <= 2
    assert -1 <= model.beta <= 1
    assert model.c > 0


def test_cgmy_invalid_Y():
    with pytest.raises(ValueError):
        CGMYCF(Y=2.0)


def test_normalise_scores_methods():
    scores = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    z = normalise_scores(scores, method="zscore")
    assert abs(np.mean(z)) < 1e-12
    assert abs(np.std(z, ddof=1) - 1.0) < 1e-12
    scaled = normalise_scores(scores, method="minmax")
    assert np.allclose(scaled, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    robust = normalise_scores(scores, method="mad")
    assert np.allclose(robust, [-1.5, -0.5, 0.5, 1.5])


def test_threshold_by_fpr():
    scores = np.arange(100, dtype=np.float64)
    threshold = threshold_by_fpr(scores, fpr=0.05)
    assert threshold == np.quantile(scores, 0.95, method="linear")


def test_rolling_pvalue_normal():
    scores = np.concatenate([np.zeros(10), np.ones(10)])
    pvalues = rolling_pvalue(scores, window=5, dist="normal")
    assert np.isnan(pvalues[:5]).all()
    assert np.all((pvalues[5:] >= 0.0) & (pvalues[5:] <= 1.0))


def test_simulate_levy_returns_shape():
    draws = simulate_levy_returns(200, alpha=1.7, beta=0.5, scale=0.01)
    assert draws.shape == (200,)
    assert draws.dtype == np.float64
