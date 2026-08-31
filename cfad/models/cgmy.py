"""
CGMY characteristic-function model.

phi(xi) = exp(C * Gamma(-Y) * [(M - i*xi)^Y - M^Y + (G + i*xi)^Y - G^Y])

Parameters
----------
C : float
    Overall activity level (> 0).
G : float
    Left-tail decay (> 0).
M : float
    Right-tail decay (> 0).
Y : float
    Jump-activity index (0 < Y < 2).

The apparent pole at Y=1 is removable because the bracketed term vanishes at
Y=1. ``log_cf`` evaluates the analytic limiting expression there rather than
forming ``Gamma(-1) * 0`` numerically.

The CF has branch cuts in the complex plane, so ``is_analytic = False``.

Reference: Carr, Geman, Madan & Yor (2002), J. Business.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func

from .base import CFModel


class CGMYCF(CFModel):
    """CGMY characteristic-function model (non-analytic)."""

    is_analytic = False

    def __init__(
        self,
        C: float = 1.0,
        G: float = 5.0,
        M: float = 10.0,
        Y: float = 0.5,
    ):
        if Y >= 2 or Y <= 0:
            raise ValueError(f"Y must satisfy 0 < Y < 2; got {Y}")
        if C <= 0 or G <= 0 or M <= 0:
            raise ValueError("C, G, M must be positive")
        self.C = C
        self.G = G
        self.M = M
        self.Y = Y

    def log_cf(self, xi: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Evaluate the CGMY log characteristic function.

        At ``Y=1`` the standard expression has the removable singularity
        ``Gamma(-1) * 0``. Taking the limit in ``Y`` gives

        ``C * [(M-i*xi) log(M-i*xi) - M log(M)
             + (G+i*xi) log(G+i*xi) - G log(G)]``.
        """
        C, G, M, Y = self.C, self.G, self.M, self.Y
        xi_arr = np.asarray(xi, dtype=np.float64)

        if np.isclose(Y, 1.0, atol=1e-12, rtol=0.0):
            right = M - 1j * xi_arr
            left = G + 1j * xi_arr
            limit = (
                right * np.log(right)
                - M * np.log(M)
                + left * np.log(left)
                - G * np.log(G)
            )
            return np.asarray(C * limit, dtype=np.complex128)

        gam = gamma_func(-Y)
        term1 = (M - 1j * xi_arr) ** Y - M**Y
        term2 = (G + 1j * xi_arr) ** Y - G**Y
        return np.asarray(C * gam * (term1 + term2), dtype=np.complex128)

    def cf(self, xi: NDArray[np.float64]) -> NDArray[np.complex128]:
        return np.exp(self.log_cf(xi))

    def fit(self, returns: NDArray[np.float64]) -> "CGMYCF":
        """ECF minimum-distance estimation via Nelder-Mead."""
        from cfad.empirical_cf import ecf_at

        xi = np.linspace(-10, 10, 128)
        ecf = ecf_at(returns, xi)

        def objective(params):
            C_, G_, M_, Y_ = params
            if C_ <= 0 or G_ <= 0 or M_ <= 0 or Y_ <= 0 or Y_ >= 2:
                return 1e10
            try:
                self.C, self.G, self.M, self.Y = C_, G_, M_, Y_
                phi = self.cf(xi)
                if not np.all(np.isfinite(phi)):
                    return 1e10
                return float(np.sum(np.abs(ecf - phi) ** 2))
            except Exception:
                return 1e10

        x0 = [self.C, self.G, self.M, self.Y]
        res = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 8000, "xatol": 1e-5},
        )
        self.C, self.G, self.M, self.Y = res.x
        return self

    def __repr__(self) -> str:
        return (
            f"CGMYCF(C={self.C:.4f}, G={self.G:.4f}, "
            f"M={self.M:.4f}, Y={self.Y:.4f})"
        )
