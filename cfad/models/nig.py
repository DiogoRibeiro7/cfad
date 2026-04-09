"""
Normal Inverse Gaussian (NIG) model.

phi(xi) = exp(i*mu*xi + delta*(sqrt(alpha^2 - beta^2) - sqrt(alpha^2 - (beta + i*xi)^2)))

NIG has a BRANCH CUT in the complex plane starting at
xi = i*(alpha - |beta|), so it is NOT entire. The contour
integral returns a non-zero residue when the contour encloses
the branch point. This is the 'vortex' the detector is tuned
to distinguish from the Gaussian null.

Parameters
----------
alpha : float  -- tail heaviness (>0)
beta  : float  -- skewness (-alpha < beta < alpha)
delta : float  -- scale (>0)
mu    : float  -- location
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from .base import CFModel


class NIGCF(CFModel):
    """NIG characteristic function model (non-analytic)."""

    is_analytic = False

    def __init__(
        self,
        alpha: float = 10.0,
        beta: float = 0.0,
        delta: float = 0.1,
        mu: float = 0.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.mu = mu

    def cf(self, xi: NDArray[np.float64]) -> NDArray[np.complex128]:
        a, b, d, m = self.alpha, self.beta, self.delta, self.mu
        gamma = np.sqrt(a**2 - b**2 + 0j)
        psi = np.sqrt(a**2 - (b + 1j * xi) ** 2 + 0j)
        return np.exp(1j * m * xi + d * (gamma - psi))

    def log_cf(self, xi: NDArray[np.float64]) -> NDArray[np.complex128]:
        a, b, d, m = self.alpha, self.beta, self.delta, self.mu
        gamma = np.sqrt(a**2 - b**2 + 0j)
        psi = np.sqrt(a**2 - (b + 1j * xi) ** 2 + 0j)
        return 1j * m * xi + d * (gamma - psi)

    def fit(self, returns: NDArray[np.float64]) -> "NIGCF":
        """MLE via scipy minimize (L-BFGS-B)."""
        from cfad.empirical_cf import ecf_at

        xi = np.linspace(-15, 15, 256)
        ecf = ecf_at(returns, xi)

        def neg_ecf_distance(params):
            a, b, d, m = params
            if a <= 0 or d <= 0 or abs(b) >= a:
                return 1e10
            self.alpha, self.beta, self.delta, self.mu = a, b, d, m
            phi = self.cf(xi)
            return float(np.sum(np.abs(ecf - phi) ** 2))

        x0 = [self.alpha, self.beta, self.delta, np.mean(returns)]
        res = minimize(neg_ecf_distance, x0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-6})
        self.alpha, self.beta, self.delta, self.mu = res.x
        return self

    def __repr__(self) -> str:
        return (f"NIGCF(alpha={self.alpha:.4f}, beta={self.beta:.4f}, "
                f"delta={self.delta:.4f}, mu={self.mu:.6f})")
