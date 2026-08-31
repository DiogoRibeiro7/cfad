"""Gaussian characteristic-function model.

For real frequency ``xi`` the characteristic function is

``phi(xi) = exp(i * mu * xi - 0.5 * sigma**2 * xi**2)``.

The parametric Gaussian CF extends to an entire function of complex frequency.
CFAD uses it as the fitted location/scale reference for the rolling empirical-CF
shape score.  The detector does not infer complex singularities from the
finite-sample ECF itself, which is also entire.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import CFModel


class GaussianCF(CFModel):
    """Normal-distribution characteristic-function model."""

    is_analytic = True

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma

    def cf(self, xi: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Evaluate the characteristic function on the supplied frequency grid."""
        return np.exp(1j * self.mu * xi - 0.5 * self.sigma**2 * xi**2)

    def log_cf(self, xi: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Evaluate the log characteristic function."""
        return 1j * self.mu * xi - 0.5 * self.sigma**2 * xi**2

    def fit(self, returns: NDArray[np.float64]) -> "GaussianCF":
        """Fit mean and sample standard deviation by their usual estimators."""
        values = np.asarray(returns, dtype=np.float64)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("returns must be one-dimensional with at least 2 values")
        self.mu = float(np.mean(values))
        self.sigma = float(np.std(values, ddof=1))
        return self

    def __repr__(self) -> str:
        return f"GaussianCF(mu={self.mu:.6f}, sigma={self.sigma:.6f})"
