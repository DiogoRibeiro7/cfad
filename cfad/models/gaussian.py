"""
Gaussian model — entire characteristic function, no singularities.

phi(xi) = exp(i*mu*xi - 0.5*sigma^2*xi^2)

This is the baseline / null model. Its CF is entire (analytic
everywhere in C), so the contour integral returns zero by Cauchy's
theorem regardless of contour shape. Any non-zero residue score
in the detector signals departure from this null.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .base import CFModel


class GaussianCF(CFModel):
    """Normal distribution characteristic function model."""

    is_analytic = True

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma

    def cf(self, xi: NDArray[np.float64]) -> NDArray[np.complex128]:
        return np.exp(1j * self.mu * xi - 0.5 * self.sigma**2 * xi**2)

    def log_cf(self, xi: NDArray[np.float64]) -> NDArray[np.complex128]:
        return 1j * self.mu * xi - 0.5 * self.sigma**2 * xi**2

    def fit(self, returns: NDArray[np.float64]) -> "GaussianCF":
        self.mu = float(np.mean(returns))
        self.sigma = float(np.std(returns, ddof=1))
        return self

    def __repr__(self) -> str:
        return f"GaussianCF(mu={self.mu:.6f}, sigma={self.sigma:.6f})"
