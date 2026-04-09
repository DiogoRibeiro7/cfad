"""Score normalisation and p-value utilities for residue-based anomaly detection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


def normalise_scores(
    scores: NDArray[np.float64], method: str = "zscore"
) -> NDArray[np.float64]:
    """Normalise anomaly scores.

    Parameters
    ----------
    scores : ndarray
        Raw anomaly scores.
    method : {'zscore', 'mad', 'minmax'}
        Normalisation method.

    Returns
    -------
    ndarray
        Normalised scores.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    method = method.lower()

    if method == "zscore":
        mu = np.mean(scores_arr)
        sigma = np.std(scores_arr, ddof=1)
        sigma = sigma if sigma > 0 else 1.0
        return (scores_arr - mu) / sigma

    if method == "mad":
        median = np.median(scores_arr)
        mad = np.median(np.abs(scores_arr - median))
        mad = mad if mad > 0 else 1.0
        return (scores_arr - median) / mad

    if method == "minmax":
        lo = np.min(scores_arr)
        hi = np.max(scores_arr)
        if hi == lo:
            return np.zeros_like(scores_arr)
        return (scores_arr - lo) / (hi - lo)

    raise ValueError(
        f"method must be one of 'zscore', 'mad', or 'minmax'; got {method}"
    )


def rolling_pvalue(
    scores: NDArray[np.float64], window: int, dist: str = "normal"
) -> NDArray[np.float64]:
    """Compute pointwise p-values using a rolling in-control distribution.

    Parameters
    ----------
    scores : ndarray
        Anomaly score series.
    window : int
        Size of the rolling calibration window.
    dist : {'normal', 'empirical'}
        Distribution used for the in-control model.

    Returns
    -------
    ndarray
        Pointwise two-sided p-values, with NaN for the first `window` values.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    if window < 1:
        raise ValueError("window must be a positive integer")

    pvalues = np.full_like(scores_arr, np.nan, dtype=np.float64)
    n = len(scores_arr)
    dist = dist.lower()

    for t in range(window, n):
        baseline = scores_arr[t - window : t]
        if dist == "normal":
            mu = np.mean(baseline)
            sigma = np.std(baseline, ddof=1)
            if sigma < 1e-12:
                pvalues[t] = 1.0
                continue
            cdf = norm.cdf(scores_arr[t], loc=mu, scale=sigma)
            pvalues[t] = 2.0 * min(cdf, 1.0 - cdf)
        elif dist == "empirical":
            sorted_baseline = np.sort(baseline)
            rank = np.searchsorted(sorted_baseline, scores_arr[t], side="right")
            p = rank / window
            pvalues[t] = 2.0 * min(p, 1.0 - p)
        else:
            raise ValueError("dist must be 'normal' or 'empirical'")

    return pvalues


def threshold_by_fpr(scores: NDArray[np.float64], fpr: float = 0.01) -> float:
    """Return the empirical score threshold at a given false positive rate.

    Parameters
    ----------
    scores : ndarray
        Anomaly score series.
    fpr : float
        Desired false positive rate (between 0 and 1).

    Returns
    -------
    float
        Score threshold above which a fraction `fpr` of in-control observations
        would be expected.
    """
    if not (0.0 < fpr < 1.0):
        raise ValueError("fpr must be between 0 and 1")

    scores_arr = np.asarray(scores, dtype=np.float64)
    if scores_arr.size == 0:
        raise ValueError("scores must not be empty")

    return float(np.quantile(scores_arr, 1.0 - fpr, method="linear"))
