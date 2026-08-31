# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython implementation of the two-sided Page-CUSUM recursion."""

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t


def cusum(
    np.ndarray[DTYPE_t, ndim=1] scores,
    double mu0,
    double sigma0,
    double k=0.5,
    double h=5.0,
):
    """Apply two-sided CUSUM to standardized anomaly scores.

    ``z_t = (score_t - mu0) / sigma0`` is dimensionless, so the Page reference
    value ``k`` is dimensionless as well.  Earlier versions multiplied ``k`` by
    ``sigma0`` after standardization, which mixed units and made the effective
    reference value depend incorrectly on the raw score scale.
    """
    if sigma0 <= 0.0:
        raise ValueError("sigma0 must be positive")
    if k < 0.0:
        raise ValueError("k must be non-negative")
    if h <= 0.0:
        raise ValueError("h must be positive")

    cdef int n_scores = scores.shape[0]
    cdef double s_pos = 0.0
    cdef double s_neg = 0.0
    cdef double z
    cdef int t

    positive_arr = np.zeros(n_scores, dtype=np.float64)
    negative_arr = np.zeros(n_scores, dtype=np.float64)
    alarms = []

    cdef np.ndarray[DTYPE_t, ndim=1] positive = positive_arr
    cdef np.ndarray[DTYPE_t, ndim=1] negative = negative_arr

    for t in range(n_scores):
        z = (scores[t] - mu0) / sigma0
        s_pos = max(0.0, s_pos + z - k)
        s_neg = max(0.0, s_neg - z - k)
        positive[t] = s_pos
        negative[t] = s_neg
        if s_pos > h or s_neg > h:
            alarms.append(t)
            s_pos = 0.0
            s_neg = 0.0

    return positive_arr, negative_arr, np.asarray(alarms, dtype=np.int64)
