"""
Empirical characteristic function (ECF) estimation.

Pure-Python fallback + thin wrapper around the Cython hot path.
Falls back gracefully if C extension is not compiled.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

try:
    from cfad._ext.rolling_ecf import rolling_ecf as _rolling_ecf_c
    from cfad._ext.rolling_ecf import ecf_at as _ecf_at_c
    _HAS_C_EXT = True
except ImportError:
    _HAS_C_EXT = False


def ecf_at(
    returns: NDArray[np.float64],
    xi: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """
    Empirical CF at frequency grid xi from a single sample of returns.

    phi_n(xi) = (1/n) sum_j exp(i xi r_j)

    Uses Cython extension when available (typically 20-50x faster
    than the NumPy broadcasting version for large n).
    """
    returns = np.asarray(returns, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    if _HAS_C_EXT:
        return _ecf_at_c(returns, xi)
    # Pure NumPy fallback: broadcast (n, m)
    return np.mean(np.exp(1j * np.outer(returns, xi)), axis=0)


def rolling_ecf(
    returns: NDArray[np.float64],
    xi: NDArray[np.float64],
    window: int,
    step: int = 1,
) -> tuple[NDArray[np.complex128], NDArray[np.int64]]:
    """
    Sliding-window ECF.

    Returns
    -------
    ecf_mat : complex ndarray of shape (n_windows, m)
    end_indices : int ndarray of shape (n_windows,)
    """
    returns = np.asarray(returns, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    if _HAS_C_EXT:
        return _rolling_ecf_c(returns, xi, window, step)

    # Pure NumPy fallback (slower)
    T = len(returns)
    n_windows = (T - window) // step + 1
    m = len(xi)
    ecf_mat = np.zeros((n_windows, m), dtype=np.complex128)
    end_idx = np.zeros(n_windows, dtype=np.int64)
    for w in range(n_windows):
        s, e = w * step, w * step + window
        ecf_mat[w] = np.mean(np.exp(1j * np.outer(returns[s:e], xi)), axis=0)
        end_idx[w] = e
    return ecf_mat, end_idx


def ecf_covariance(
    returns: NDArray[np.float64],
    xi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Asymptotic covariance matrix of the ECF (real part only).
    Used for goodness-of-fit test statistics.

    Sigma(xi_j, xi_k) = Re[phi(xi_j - xi_k)] - Re[phi(xi_j)] Re[phi(xi_k)]
                        + Im[phi(xi_j - xi_k)] Im... (full expression)

    Reference: Epps & Pulley (1983), Biometrika.
    """
    n = len(returns)
    m = len(xi)
    phi = ecf_at(returns, xi)
    # outer differences
    xi_diff = xi[:, None] - xi[None, :]
    phi_diff = ecf_at(returns, xi_diff.ravel()).reshape(m, m)
    cov = np.real(phi_diff) - np.outer(np.real(phi), np.real(phi))
    return cov / n
