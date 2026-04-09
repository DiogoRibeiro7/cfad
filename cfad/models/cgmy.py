"""
CGMY / Kou model characteristic function.

phi(xi) = exp(C * Gamma(-Y) * [(M - i*xi)^Y - M^Y + (G + i*xi)^Y - G^Y])

Parameters
----------
C : float  -- overall activity level (>0)
G : float  -- left tail decay (>0)
M : float  -- right tail decay (>0)
Y : float  -- jump activity index (0 < Y < 2)

When Y → 0: compound Poisson. When Y → 1: Variance Gamma.
When Y → 2: Brownian motion limit.

The CF has branch cuts in the complex plane → is_analytic = False.

Reference: Carr, Geman, Madan & Yor (2002), J. Business.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma as gamma_func
from scipy.optimize import minimize
from .base import CFModel


class CGMYCF(CFModel):
    """CGMY characteristic function model (non-analytic)."""

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
        C, G, M, Y = self.C, self.G, self.M, self.Y
        gam = gamma_func(-Y)
        # branch cut: (M - i*xi)^Y and (G + i*xi)^Y  — use complex power
        term1 = (M - 1j * xi) ** Y - M**Y
        term2 = (G + 1j * xi) ** Y - G**Y
        return C * gam * (term1 + term2)

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
                return float(np.sum(np.abs(ecf - phi) ** 2))
            except Exception:
                return 1e10

        x0 = [self.C, self.G, self.M, self.Y]
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 8000, "xatol": 1e-5})
        self.C, self.G, self.M, self.Y = res.x
        return self

    def __repr__(self) -> str:
        return (f"CGMYCF(C={self.C:.4f}, G={self.G:.4f}, "
                f"M={self.M:.4f}, Y={self.Y:.4f})")
