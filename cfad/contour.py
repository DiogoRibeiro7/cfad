"""Characteristic-function geometry and anomaly scoring utilities.

The empirical characteristic function (ECF) of a finite sample is a finite sum
of exponential functions and is therefore entire in the complex frequency
variable.  A closed-contour integral of the exact ECF cannot reveal branch cuts
or poles of the population characteristic function.

CFAD therefore uses a real-frequency discrepancy score for detection: each
rolling ECF is compared with the Gaussian characteristic function fitted to the
same window.  The resulting normalized L2 distance measures distributional
shape departure beyond the window mean and variance.
"""

from __future__ import annotations

from collections.abc import Callable
import warnings

import numpy as np
from numpy.typing import NDArray

ContourFunc = Callable[[NDArray[np.complex128]], NDArray[np.complex128]]


def rectangular_contour(
    xi_min: float,
    xi_max: float,
    height: float,
    n_pts: int = 128,
) -> NDArray[np.complex128]:
    """Build a counter-clockwise rectangular contour in the complex plane."""
    if xi_max <= xi_min:
        raise ValueError("xi_max must be greater than xi_min")
    if height <= 0.0:
        raise ValueError("height must be positive")
    if n_pts < 2:
        raise ValueError("n_pts must be at least 2")

    bottom = np.linspace(xi_min - 1j * height, xi_max - 1j * height, n_pts)
    right = np.linspace(xi_max - 1j * height, xi_max + 1j * height, n_pts)
    top = np.linspace(xi_max + 1j * height, xi_min + 1j * height, n_pts)
    left = np.linspace(xi_min + 1j * height, xi_min - 1j * height, n_pts)
    return np.concatenate([bottom, right, top, left]).astype(np.complex128)


def contour_integral(
    phi_func: ContourFunc,
    xi_min: float = -10.0,
    xi_max: float = 10.0,
    height: float = 0.5,
    n_pts: int = 256,
) -> complex:
    """Numerically integrate a complex-valued function around a closed contour.

    This helper is appropriate for parametric characteristic functions that are
    actually evaluated at complex frequencies.  It is not used as the empirical
    anomaly score.
    """
    path = rectangular_contour(xi_min, xi_max, height, n_pts)
    values = np.asarray(phi_func(path), dtype=np.complex128)
    if values.shape != path.shape:
        raise ValueError("phi_func must return one value per contour point")

    closed_path = np.concatenate([path, path[:1]])
    closed_values = np.concatenate([values, values[:1]])
    return complex(np.trapezoid(closed_values, closed_path))


def gaussian_ecf_distance_scores(
    ecf_windows: NDArray[np.complex128],
    xi_grid: NDArray[np.float64],
    means: NDArray[np.float64],
    stds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute normalized L2 distance from each ECF to its fitted Gaussian CF.

    Parameters
    ----------
    ecf_windows : complex ndarray of shape (n_windows, n_xi)
        Empirical characteristic-function values for each rolling window.
    xi_grid : float ndarray of shape (n_xi,)
        Real frequency grid used for the ECF.
    means : float ndarray of shape (n_windows,)
        Sample mean for each rolling window.
    stds : float ndarray of shape (n_windows,)
        Sample standard deviation for each rolling window.

    Returns
    -------
    scores : float ndarray of shape (n_windows,)
        Square root of the frequency-averaged integrated squared distance
        between the ECF and the Gaussian CF fitted to the same window.

    Notes
    -----
    The score is

    ``sqrt(integral |phi_hat(xi) - phi_N(xi; mu_hat, sigma_hat)|^2 dxi /
    integral 1 dxi)``.

    Fitting mean and variance within each window makes the statistic primarily
    sensitive to higher-order distributional shape changes rather than simple
    location or scale shifts.
    """
    ecf_arr = np.asarray(ecf_windows, dtype=np.complex128)
    xi = np.asarray(xi_grid, dtype=np.float64)
    mu = np.asarray(means, dtype=np.float64)
    sigma = np.asarray(stds, dtype=np.float64)

    if ecf_arr.ndim != 2:
        raise ValueError("ecf_windows must be two-dimensional")
    if xi.ndim != 1 or xi.size < 2:
        raise ValueError("xi_grid must be one-dimensional with at least 2 points")
    if ecf_arr.shape[1] != xi.size:
        raise ValueError("ecf_windows and xi_grid dimensions do not match")
    if mu.shape != (ecf_arr.shape[0],) or sigma.shape != (ecf_arr.shape[0],):
        raise ValueError("means and stds must contain one value per ECF window")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        raise ValueError("means and stds must be finite")
    if np.any(sigma < 0.0):
        raise ValueError("stds must be non-negative")

    gaussian_cf = np.exp(
        1j * mu[:, np.newaxis] * xi[np.newaxis, :]
        - 0.5 * sigma[:, np.newaxis] ** 2 * xi[np.newaxis, :] ** 2
    )
    squared_distance = np.abs(ecf_arr - gaussian_cf) ** 2
    frequency_span = float(xi[-1] - xi[0])
    if frequency_span <= 0.0:
        raise ValueError("xi_grid must be strictly increasing")

    integrated = np.trapezoid(squared_distance, xi, axis=1) / frequency_span
    return np.sqrt(np.maximum(integrated, 0.0)).astype(np.float64)


def ecf_residue_scores(
    ecf_windows: NDArray[np.complex128],
    xi_grid: NDArray[np.float64],
    height: float = 0.1,
) -> NDArray[np.float64]:
    """Deprecated legacy name for the former empirical-residue proxy.

    The old function cannot be made mathematically correct from ECF values alone
    because the exact finite-sample ECF is entire.  Call
    :func:`gaussian_ecf_distance_scores` with rolling means and standard
    deviations instead.
    """
    del ecf_windows, xi_grid, height
    warnings.warn(
        "ecf_residue_scores() is deprecated: an empirical CF has zero exact "
        "closed-contour residue. Use gaussian_ecf_distance_scores() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise RuntimeError(
        "ecf_residue_scores() no longer defines the CFAD anomaly statistic"
    )
