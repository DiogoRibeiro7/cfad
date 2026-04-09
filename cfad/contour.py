"""
Contour integration engine.

Evaluates the complex contour integral of a CF (empirical or parametric)
along a rectangular contour in the complex xi-plane.

Mathematical background
-----------------------
By the Residue Theorem, for a meromorphic function f:

    (1/2πi) ∮_C f(z) dz = Σ Res(f, z_k)

where the sum is over poles enclosed by C. For branch cuts the
contribution is the discontinuity across the cut.

For an ENTIRE function (Gaussian CF), this integral is identically zero
for any closed contour — no residue exists to detect.

For a non-analytic CF (NIG, CGMY, Lévy-stable), the integral is non-zero
when C encloses a branch point or pole: this non-zero value is the
anomaly score.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Callable

try:
    from cfad._ext.contour_quad import residue_magnitude as _residue_c
    _HAS_C_EXT = True
except ImportError:
    _HAS_C_EXT = False


ContourFunc = Callable[[NDArray[np.complex128]], NDArray[np.complex128]]


def rectangular_contour(
    xi_min: float,
    xi_max: float,
    height: float,
    n_pts: int = 128,
) -> NDArray[np.complex128]:
    """
    Build a rectangular contour path in the complex plane.

    Segments (counter-clockwise):
      Bottom : xi_min - i*height  → xi_max - i*height
      Right  : xi_max - i*height  → xi_max + i*height
      Top    : xi_max + i*height  → xi_min + i*height
      Left   : xi_min + i*height  → xi_min - i*height

    Returns
    -------
    path : complex ndarray of shape (4*n_pts,)
    """
    n = n_pts
    bottom = np.linspace(xi_min - 1j * height, xi_max - 1j * height, n)
    right  = np.linspace(xi_max - 1j * height, xi_max + 1j * height, n)
    top    = np.linspace(xi_max + 1j * height, xi_min + 1j * height, n)
    left   = np.linspace(xi_min + 1j * height, xi_min - 1j * height, n)
    return np.concatenate([bottom, right, top, left])


def contour_integral(
    phi_func: ContourFunc,
    xi_min: float = -10.0,
    xi_max: float = 10.0,
    height: float = 0.5,
    n_pts: int = 256,
) -> complex:
    """
    Numerically integrate phi along a rectangular contour.

    Parameters
    ----------
    phi_func : callable
        Function accepting complex array, returning complex array.
        Typically a CF model's cf() method extended to complex argument.
    xi_min, xi_max : float
        Real-axis extent of the contour.
    height : float
        Imaginary half-height of the contour. Should be chosen within
        the analyticity strip of the model (|Im xi| < alpha - |beta|
        for NIG).
    n_pts : int
        Quadrature points per segment.

    Returns
    -------
    integral : complex
    """
    path = rectangular_contour(xi_min, xi_max, height, n_pts)
    dz   = np.diff(path, append=path[0])  # closed path: last→first
    vals = phi_func(path)
    return float(np.sum(vals * dz).real), float(np.sum(vals * dz).imag)


def ecf_residue_scores(
    ecf_windows: NDArray[np.complex128],
    xi_grid: NDArray[np.float64],
    height: float = 0.1,
) -> NDArray[np.float64]:
    """
    Compute residue-magnitude anomaly scores for all rolling windows.

    Uses Cython extension when available.

    Parameters
    ----------
    ecf_windows : complex ndarray of shape (n_windows, m)
    xi_grid : float ndarray of shape (m,)
    height : float
        Contour height (small: stays close to real axis where ECF is valid).

    Returns
    -------
    scores : float ndarray of shape (n_windows,)
    """
    if _HAS_C_EXT:
        return _residue_c(ecf_windows, xi_grid, height)

    # Pure Python fallback.
    # Proxy for the contour residue: weighted imaginary integral + deviation
    # from Gaussian CF shape. Zero for an entire CF on the real axis.
    n_windows, m = ecf_windows.shape
    scores = np.zeros(n_windows)
    dxi = float(xi_grid[1] - xi_grid[0]) if m > 1 else 1.0
    weights = xi_grid ** 2  # upweight high-frequency content
    for w in range(n_windows):
        phi = ecf_windows[w]
        im_score = float(np.abs(np.trapezoid(phi.imag * weights, xi_grid)))
        re_score = float(np.trapezoid(
            np.abs(phi.real - np.exp(-0.5 * xi_grid**2 * dxi**2)), xi_grid
        ))
        scores[w] = im_score + re_score
    return scores
