# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
CUSUM sequential change-point detection — Cython hot path.

Implements the two-sided Page-CUSUM statistic on a stream of anomaly scores.
Alerts when S_t exceeds threshold h.
"""
import numpy as np
cimport numpy as np
from libc.math cimport fabs

ctypedef np.float64_t DTYPE_t


def cusum(
    np.ndarray[DTYPE_t, ndim=1] scores,
    double mu0,
    double sigma0,
    double k=0.5,
    double h=5.0,
):
    """
    Two-sided CUSUM on a sequence of anomaly scores.

    Parameters
    ----------
    scores : float ndarray of shape (T,)
        Anomaly score time series.
    mu0 : float
        In-control mean (estimated from calm period).
    sigma0 : float
        In-control std.
    k : float
        Allowance parameter (default 0.5 = detect 1-sigma shifts).
    h : float
        Decision threshold in units of sigma0.

    Returns
    -------
    S_pos : float ndarray of shape (T,)
    S_neg : float ndarray of shape (T,)
    alarms : int ndarray  -- time indices where |S| > h
    """
    cdef int T = scores.shape[0]
    cdef double S_pos = 0.0, S_neg = 0.0
    cdef double z, slack

    s_pos_arr = np.zeros(T, dtype=np.float64)
    s_neg_arr = np.zeros(T, dtype=np.float64)
    alarm_list = []

    cdef np.ndarray[DTYPE_t, ndim=1] sp = s_pos_arr
    cdef np.ndarray[DTYPE_t, ndim=1] sn = s_neg_arr

    slack = k * sigma0

    for t in range(T):
        z = (scores[t] - mu0) / sigma0
        S_pos = max(0.0, S_pos + z - slack)
        S_neg = max(0.0, S_neg - z - slack)
        sp[t] = S_pos
        sn[t] = S_neg
        if S_pos > h or S_neg > h:
            alarm_list.append(t)
            S_pos = 0.0
            S_neg = 0.0

    return s_pos_arr, s_neg_arr, np.array(alarm_list, dtype=np.int64)
