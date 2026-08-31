"""Tests for characteristic-function geometry and ECF shape scoring."""

from __future__ import annotations

import numpy as np
import pytest

from cfad.contour import (
    contour_integral,
    gaussian_ecf_distance_scores,
    rectangular_contour,
)
from cfad.empirical_cf import ecf_at


def test_rectangular_contour_is_closed_by_integrator() -> None:
    """Integral of an entire polynomial around the contour should be near zero."""
    value = contour_integral(
        lambda z: z**3 + 2.0 * z + 1.0,
        xi_min=-2.0,
        xi_max=3.0,
        height=1.0,
        n_pts=1024,
    )
    assert abs(value) < 1e-10


def test_rectangular_contour_validates_inputs() -> None:
    """Invalid geometric parameters should fail early."""
    with pytest.raises(ValueError, match="xi_max"):
        rectangular_contour(1.0, 1.0, 0.5)
    with pytest.raises(ValueError, match="height"):
        rectangular_contour(-1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="n_pts"):
        rectangular_contour(-1.0, 1.0, 0.5, n_pts=1)


def test_gaussian_reference_has_zero_distance() -> None:
    """An exact Gaussian CF must score zero against its own parameters."""
    xi = np.linspace(-10.0, 10.0, 257, dtype=np.float64)
    means = np.asarray([0.02, -0.01], dtype=np.float64)
    stds = np.asarray([0.8, 1.3], dtype=np.float64)
    ecf = np.exp(
        1j * means[:, np.newaxis] * xi[np.newaxis, :]
        - 0.5 * stds[:, np.newaxis] ** 2 * xi[np.newaxis, :] ** 2
    )

    scores = gaussian_ecf_distance_scores(ecf, xi, means, stds)
    assert np.allclose(scores, 0.0, atol=1e-14, rtol=0.0)


def test_heavy_tailed_sample_scores_above_gaussian_sample() -> None:
    """With fixed seeds, a t sample should depart more from fitted Gaussian shape."""
    rng = np.random.default_rng(2026)
    n = 5000
    gaussian_sample = rng.normal(0.0, 1.0, n).astype(np.float64)
    heavy_sample = rng.standard_t(df=3.0, size=n).astype(np.float64)
    heavy_sample /= float(np.std(heavy_sample, ddof=1))

    xi = np.linspace(-6.0, 6.0, 257, dtype=np.float64)
    samples = (gaussian_sample, heavy_sample)
    ecf = np.vstack([ecf_at(sample, xi) for sample in samples])
    means = np.asarray([float(np.mean(sample)) for sample in samples])
    stds = np.asarray([float(np.std(sample, ddof=1)) for sample in samples])

    scores = gaussian_ecf_distance_scores(ecf, xi, means, stds)
    assert scores[1] > 3.0 * scores[0]
