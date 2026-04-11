"""Score normalisation and p-value utilities for residue-based anomaly detection."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


def normalise_scores(
    scores: NDArray[np.float64],
    method: Literal["zscore", "mad", "minmax"] = "zscore",
) -> NDArray[np.float64]:
    """
    Normalise a raw anomaly score series to a common scale.

    Parameters
    ----------
    scores : float ndarray of shape (T,)
    method : "zscore"  -> (scores - mean) / std
             "mad"     -> (scores - median) / (1.4826 * MAD)  [robust]
             "minmax"  -> (scores - min) / (max - min)

    Returns
    -------
    normalised : float ndarray of shape (T,)
    """
    scores_arr = np.asarray(scores, dtype=np.float64)

    if method == "zscore":
        mu = float(np.mean(scores_arr))
        sigma = float(np.std(scores_arr))
        if sigma <= 0.0:
            return np.zeros_like(scores_arr)
        return (scores_arr - mu) / sigma

    if method == "mad":
        median = float(np.median(scores_arr))
        mad = float(np.median(np.abs(scores_arr - median)))
        scale = 1.4826 * mad
        if scale <= 0.0:
            return np.zeros_like(scores_arr)
        return (scores_arr - median) / scale

    if method == "minmax":
        lo = float(np.min(scores_arr))
        hi = float(np.max(scores_arr))
        if hi <= lo:
            return np.zeros_like(scores_arr)
        return (scores_arr - lo) / (hi - lo)

    raise ValueError("method must be one of {'zscore', 'mad', 'minmax'}")


def rolling_pvalue(
    scores: NDArray[np.float64],
    window: int,
    dist: Literal["normal", "empirical"] = "empirical",
) -> NDArray[np.float64]:
    """
    Pointwise p-value of each score under the in-control distribution
    estimated from a rolling past window of length `window`.

    For each index t, fit the in-control distribution to
    scores[max(0, t-window):t], then return P(S >= scores[t]) under that fit.

    Parameters
    ----------
    scores : float ndarray of shape (T,)
    window : int
        Lookback for in-control estimation.
    dist : "normal" or "empirical"
        In-control distribution family.

    Returns
    -------
    pvalues : float ndarray of shape (T,), values in [0, 1]
              NaN for the first `window` entries (insufficient history).
    """
    if window < 1:
        raise ValueError("window must be a positive integer")
    if dist not in ("normal", "empirical"):
        raise ValueError("dist must be one of {'normal', 'empirical'}")

    scores_arr = np.asarray(scores, dtype=np.float64)
    n_scores = int(scores_arr.shape[0])
    pvalues = np.full(n_scores, np.nan, dtype=np.float64)

    for t in range(window, n_scores):
        baseline = scores_arr[max(0, t - window):t]
        x_t = float(scores_arr[t])

        if dist == "normal":
            mu = float(np.mean(baseline))
            sigma = float(np.std(baseline))
            if sigma <= 0.0:
                pval = 1.0 if x_t <= mu else 0.0
            else:
                pval = 1.0 - float(norm.cdf(x_t, loc=mu, scale=sigma))
        else:
            pval = float(np.mean(baseline >= x_t))

        pvalues[t] = float(np.clip(pval, 0.0, 1.0))

    return pvalues


def threshold_by_fpr(
    scores: NDArray[np.float64],
    calibration_scores: NDArray[np.float64],
    fpr: float = 0.01,
) -> float:
    """
    Return the score threshold that achieves a given false-positive rate
    on the calibration (in-control) score distribution.

    Parameters
    ----------
    scores : float ndarray of shape (T,)
        Full score series (unused, kept for API symmetry).
    calibration_scores : float ndarray of shape (T_cal,)
        In-control score series used to set the threshold.
    fpr : float
        Desired false-positive rate (default 0.01 = 1%).

    Returns
    -------
    threshold : float
        The (1-fpr) quantile of calibration_scores.
    """
    if not (0.0 < fpr < 1.0):
        raise ValueError("fpr must lie in (0, 1)")

    _ = np.asarray(scores, dtype=np.float64)
    calibration_arr = np.asarray(calibration_scores, dtype=np.float64)
    if calibration_arr.size == 0:
        raise ValueError("calibration_scores must not be empty")

    return float(np.quantile(calibration_arr, 1.0 - fpr))
