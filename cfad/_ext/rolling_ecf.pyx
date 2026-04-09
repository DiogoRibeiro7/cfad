# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Rolling empirical characteristic function (ECF) — Cython hot path.

For a window of n returns and m frequency grid points this runs in O(n*m)
with no Python overhead per sample.
"""
import numpy as np
cimport numpy as np
from libc.math cimport cos, sin

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.complex128_t CTYPE_t


def rolling_ecf(
    np.ndarray[DTYPE_t, ndim=1] returns,
    np.ndarray[DTYPE_t, ndim=1] xi_grid,
    int window,
    int step=1,
):
    """
    Compute the empirical characteristic function on a rolling window.

    Parameters
    ----------
    returns : ndarray of shape (T,)
        Sequence of log-returns.
    xi_grid : ndarray of shape (m,)
        Frequency evaluation points in R.
    window : int
        Number of observations per window.
    step : int
        Step size between windows (default 1 = fully overlapping).

    Returns
    -------
    ecf : ndarray of shape (n_windows, m), complex128
    indices : ndarray of shape (n_windows,), int
        End-index (exclusive) of each window in `returns`.
    """
    cdef int T = returns.shape[0]
    cdef int m = xi_grid.shape[0]
    cdef int n_windows = (T - window) // step + 1
    cdef int w, t, k, start, end
    cdef double xi_k, r_t, re_sum, im_sum

    ecf = np.zeros((n_windows, m), dtype=np.complex128)
    indices = np.zeros(n_windows, dtype=np.int64)

    cdef np.ndarray[CTYPE_t, ndim=2] ecf_view = ecf
    cdef np.ndarray[np.int64_t, ndim=1] idx_view = indices

    for w in range(n_windows):
        start = w * step
        end   = start + window
        idx_view[w] = end
        for k in range(m):
            xi_k = xi_grid[k]
            re_sum = 0.0
            im_sum = 0.0
            for t in range(start, end):
                r_t = returns[t]
                re_sum += cos(xi_k * r_t)
                im_sum += sin(xi_k * r_t)
            ecf_view[w, k] = (re_sum + 1j * im_sum) / window

    return ecf, indices


def ecf_at(
    np.ndarray[DTYPE_t, ndim=1] returns,
    np.ndarray[DTYPE_t, ndim=1] xi_grid,
):
    """Single-window ECF. Convenience wrapper for calibration."""
    cdef int n = returns.shape[0]
    cdef int m = xi_grid.shape[0]
    cdef int k, t
    cdef double xi_k, r_t, re_sum, im_sum

    result = np.zeros(m, dtype=np.complex128)
    cdef np.ndarray[CTYPE_t, ndim=1] res_view = result

    for k in range(m):
        xi_k = xi_grid[k]
        re_sum = 0.0
        im_sum = 0.0
        for t in range(n):
            r_t = returns[t]
            re_sum += cos(xi_k * r_t)
            im_sum += sin(xi_k * r_t)
        res_view[k] = (re_sum + 1j * im_sum) / n

    return result
