"""
Lévy-stable characteristic function (Zolotarev parameterisation S1).

For alpha != 1:
  log phi(xi) = -|c*xi|^alpha * (1 - i*beta*sign(xi)*tan(pi*alpha/2)) + i*mu*xi

Parameters
----------
alpha : float  -- stability index (0 < alpha <= 2)
beta  : float  -- skewness (-1 <= beta <= 1)
c     : float  -- scale (>0)
mu    : float  -- location

When alpha=2: Gaussian (entire CF).
When alpha < 2: power-law tails → branch cut → is_analytic = False.

Reference: Zolotarev (1986), One-Dimensional Stable Distributions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .base import CFModel


class LevyStableCF(CFModel):
    """Lévy-stable characteristic function model (non-analytic for alpha<2)."""

    is_analytic = False

    def __init__(
        self,
        alpha: float = 1.7,
        beta: float = 0.0,
        c: float = 0.01,
        mu: float = 0.0,
    ):
        if not (0 < alpha <= 2):
            raise ValueError(f"alpha must be in (0, 2]; got {alpha}")
        if not (-1 <= beta <= 1):
            raise ValueError(f"beta must be in [-1, 1]; got {beta}")
        if c <= 0:
            raise ValueError("c must be positive")
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.mu = mu

    def log_cf(self, xi: NDArray[np.float64]) -> NDArray[np.complex128]:
        a, b, c, m = self.alpha, self.beta, self.c, self.mu
        xi = np.asarray(xi, dtype=np.float64)
        cxi = c * xi
        abs_cxi = np.abs(cxi)

        if abs(a - 1.0) < 1e-8:
            # Special case alpha = 1
            return (
                -abs_cxi
                + 1j * b * (2 / np.pi) * cxi * np.log(abs_cxi + 1e-300)
                + 1j * m * xi
            )
        else:
            tan_factor = np.tan(np.pi * a / 2)
            skew = -1j * b * np.sign(xi) * tan_factor * (abs_cxi**a - abs_cxi)
            return -(abs_cxi**a) + skew + 1j * m * xi

    def cf(self, xi: NDArray[np.float64]) -> NDArray[np.complex128]:
        return np.exp(self.log_cf(xi))

    def fit(self, returns: NDArray[np.float64]) -> "LevyStableCF":
        """Maximum likelihood estimation via scipy.stats.levy_stable.fit."""
        from scipy.stats import levy_stable

        returns_arr = np.asarray(returns, dtype=np.float64)
        alpha, beta, loc, scale = levy_stable.fit(returns_arr)
        if not (0 < alpha <= 2):
            raise ValueError(f"fitted alpha must be in (0, 2]; got {alpha}")
        if not (-1 <= beta <= 1):
            raise ValueError(f"fitted beta must be in [-1, 1]; got {beta}")
        if scale <= 0:
            raise ValueError(f"fitted scale must be positive; got {scale}")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.c = float(scale)
        self.mu = float(loc)
        return self

    def __repr__(self) -> str:
        return (
            f"LevyStableCF(alpha={self.alpha:.4f}, beta={self.beta:.4f}, "
            f"c={self.c:.6f}, mu={self.mu:.6f})"
        )
        return (
            f"LevyStableCF(alpha={self.alpha:.4f}, beta={self.beta:.4f}, "
            f"c={self.c:.6f}, mu={self.mu:.6f})"
        )
