"""
Abstract base class for parametric characteristic function models.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class CFModel(ABC):
    """
    Abstract base for all parametric CF models.

    Subclasses must implement:
      - cf(xi)         : complex-valued characteristic function
      - log_cf(xi)     : log of cf (numerically safer for large |xi|)
      - fit(returns)   : MLE or GMM parameter estimation
      - is_analytic    : class-level bool flag

    The `is_analytic` flag is the structural marker used by the
    detector: entire functions (Gaussian) have no residue for any
    contour; non-analytic models (NIG, CGMY, Lévy-stable) may.
    """

    is_analytic: bool = True   # override in subclasses

    @abstractmethod
    def cf(self, xi: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Evaluate phi(xi) = E[exp(i xi X)]."""

    @abstractmethod
    def log_cf(self, xi: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Evaluate log phi(xi). Default: log(cf(xi))."""

    @abstractmethod
    def fit(self, returns: NDArray[np.float64]) -> "CFModel":
        """Fit model parameters to return data. Returns self."""

    def pdf_from_cf(
        self,
        x_grid: NDArray[np.float64],
        xi_max: float = 50.0,
        n_xi: int = 2048,
    ) -> NDArray[np.float64]:
        """
        Recover the PDF via numerical Fourier inversion of the CF.
        Uses scipy.fft for speed; result is real-valued.
        """
        from scipy.fft import ifft, fftfreq
        dxi = 2 * xi_max / n_xi
        xi = np.linspace(-xi_max, xi_max, n_xi)
        phi = self.cf(xi)
        pdf_raw = np.real(ifft(phi)) * dxi * n_xi / (2 * np.pi)
        # interpolate to x_grid
        from scipy.interpolate import interp1d
        x_raw = fftfreq(n_xi, d=dxi / (2 * np.pi))
        order = np.argsort(x_raw)
        f = interp1d(x_raw[order], pdf_raw[order], bounds_error=False, fill_value=0.0)
        return f(x_grid)

    def aic(self, returns: NDArray[np.float64]) -> float:
        """Akaike Information Criterion (lower = better)."""
        n = len(returns)
        xi = np.linspace(-20, 20, 512)
        log_phi = self.log_cf(xi)
        # ECF-based negative log-likelihood proxy via L2 distance
        from cfad.empirical_cf import ecf_at
        ecf = ecf_at(returns, xi)
        residual = np.abs(ecf - np.exp(log_phi)) ** 2
        nll = float(np.sum(residual))
        k = len(self.__dict__)
        return 2 * k + 2 * nll
