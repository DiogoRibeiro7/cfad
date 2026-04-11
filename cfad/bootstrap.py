"""Bootstrap confidence intervals and stability diagnostics for CFAD scores."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from cfad.api import detect


def _align_series(series: NDArray[np.float64], target_len: int) -> NDArray[np.float64]:
    """Interpolate a one-dimensional series onto a target length."""
    values = np.asarray(series, dtype=np.float64)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float64)
    if values.size == target_len:
        return values
    if values.size == 0:
        return np.zeros(target_len, dtype=np.float64)

    x_old = np.linspace(0.0, 1.0, values.size, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
    return np.interp(x_new, x_old, values).astype(np.float64)


def block_bootstrap_resample(
    returns: NDArray[np.float64],
    block_size: int,
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Circular block bootstrap resample.

    Draw ceil(n / block_size) blocks with replacement from all possible
    circular blocks of length ``block_size`` and return exactly ``n`` values.

    Parameters
    ----------
    returns : float ndarray of shape (n,)
        Input return series.
    block_size : int
        Circular block length.
    seed : int or None, default=None
        Random seed.

    Returns
    -------
    resampled : float ndarray of shape (n,)
        Block-bootstrap resampled series.
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
    if returns_arr.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    n = int(returns_arr.size)
    if n == 0:
        raise ValueError("returns must be non-empty")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    rng = np.random.default_rng(seed)
    b = int(block_size)
    n_blocks = int(np.ceil(n / b))

    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(b, dtype=np.int64)

    blocks = []
    for s in starts:
        idx = (int(s) + offsets) % n
        blocks.append(returns_arr[idx])

    sampled = np.concatenate(blocks)
    return sampled[:n].astype(np.float64)


def bootstrap_scores(
    returns: NDArray[np.float64],
    window: int = 60,
    n_bootstrap: int = 200,
    xi_range: tuple = (-10.0, 10.0),
    n_xi: int = 128,
    height: float = 0.2,
    step: int = 5,
    confidence: float = 0.95,
    seed: int = 0,
    n_jobs: int = 1,
) -> dict[str, NDArray[np.float64]]:
    """
    Bootstrap confidence intervals for the rolling residue score series.

    For each bootstrap replicate:
      1. Resample returns with replacement (circular block bootstrap)
      2. Run the full detection pipeline
      3. Collect and align score series

    Parameters
    ----------
    returns : float ndarray of shape (n,)
        Input return series.
    window : int, default=60
        Detector lookback window.
    n_bootstrap : int, default=200
        Number of bootstrap replicates.
    xi_range : tuple of float, default=(-10.0, 10.0)
        ECF frequency range.
    n_xi : int, default=128
        Number of ECF grid points.
    height : float, default=0.2
        Contour imaginary half-height.
    step : int, default=5
        Rolling detector step.
    confidence : float, default=0.95
        Confidence level for pointwise intervals.
    seed : int, default=0
        Base random seed.
    n_jobs : int, default=1
        Parallel jobs used by joblib when available.

    Returns
    -------
    summary : dict
        Dictionary with keys ``mean``, ``lower``, ``upper``, ``std``.
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
    if returns_arr.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if returns_arr.size <= window:
        raise ValueError("returns length must exceed window")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if n_xi <= 1:
        raise ValueError("n_xi must be greater than 1")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")

    base_report = detect(
        returns_arr,
        window=window,
        xi_range=xi_range,
        n_xi=n_xi,
        height=height,
        step=step,
    )
    target_len = int(len(base_report.scores))
    if target_len == 0:
        raise ValueError("detector produced no rolling windows")

    block_size = max(1, int(window) // 4)

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, np.iinfo(np.int64).max, size=int(n_bootstrap), dtype=np.int64)

    def _one_run(local_seed: int) -> NDArray[np.float64]:
        sample = block_bootstrap_resample(
            returns_arr,
            block_size=block_size,
            seed=int(local_seed),
        )
        report = detect(
            sample,
            window=window,
            xi_range=xi_range,
            n_xi=n_xi,
            height=height,
            step=step,
        )
        return _align_series(np.asarray(report.scores, dtype=np.float64), target_len)

    runs: list[NDArray[np.float64]]
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed

            runs = Parallel(n_jobs=n_jobs)(
                delayed(_one_run)(int(s)) for s in seeds
            )
        except Exception:
            runs = [_one_run(int(s)) for s in seeds]
    else:
        runs = [_one_run(int(s)) for s in seeds]

    score_mat = np.vstack(runs)
    alpha = 1.0 - float(confidence)

    mean = np.mean(score_mat, axis=0)
    lower = np.quantile(score_mat, alpha / 2.0, axis=0)
    upper = np.quantile(score_mat, 1.0 - alpha / 2.0, axis=0)
    ddof = 1 if score_mat.shape[0] > 1 else 0
    std = np.std(score_mat, axis=0, ddof=ddof)

    return {
        "mean": np.asarray(mean, dtype=np.float64),
        "lower": np.asarray(lower, dtype=np.float64),
        "upper": np.asarray(upper, dtype=np.float64),
        "std": np.asarray(std, dtype=np.float64),
    }


def score_stability(
    returns: NDArray[np.float64],
    window: int = 60,
    n_subsamples: int = 50,
    subsample_frac: float = 0.8,
    seed: int = 0,
) -> dict[str, NDArray[np.float64] | bool]:
    """
    Subsample stability diagnostic for anomaly scores.

    For each subsample:
      1. Draw floor(n * subsample_frac) points without replacement
      2. Run ``detect()`` with the given ``window``
      3. Interpolate to a common score grid

    Parameters
    ----------
    returns : float ndarray of shape (n,)
        Input return series.
    window : int, default=60
        Detector lookback window.
    n_subsamples : int, default=50
        Number of subsamples.
    subsample_frac : float, default=0.8
        Fraction of observations in each subsample.
    seed : int, default=0
        Random seed.

    Returns
    -------
    diagnostics : dict
        Dictionary with keys ``mean``, ``cv``, and ``stable``.
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
    if returns_arr.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    n = int(returns_arr.size)
    if n <= window:
        raise ValueError("returns length must exceed window")
    if n_subsamples <= 0:
        raise ValueError("n_subsamples must be positive")
    if not (0.0 < subsample_frac <= 1.0):
        raise ValueError("subsample_frac must be in (0, 1]")

    sub_n = int(np.floor(n * subsample_frac))
    if sub_n <= window:
        raise ValueError("subsample length must exceed window")

    baseline = detect(returns_arr, window=window)
    target_len = int(len(baseline.scores))
    if target_len == 0:
        raise ValueError("detector produced no rolling windows")

    rng = np.random.default_rng(seed)
    all_scores = np.zeros((int(n_subsamples), target_len), dtype=np.float64)

    for i in range(int(n_subsamples)):
        idx = np.sort(rng.choice(n, size=sub_n, replace=False))
        subsample = returns_arr[idx]
        rep = detect(subsample, window=window)
        all_scores[i] = _align_series(np.asarray(rep.scores, dtype=np.float64), target_len)

    mean = np.mean(all_scores, axis=0)
    ddof = 1 if all_scores.shape[0] > 1 else 0
    std = np.std(all_scores, axis=0, ddof=ddof)

    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(np.abs(mean) > 1e-12, std / np.abs(mean), np.nan)

    stable = bool(np.nanmean(cv) < 0.3)
    return {
        "mean": np.asarray(mean, dtype=np.float64),
        "cv": np.asarray(cv, dtype=np.float64),
        "stable": stable,
    }


def plot_bootstrap_bands(
    result: dict,
    report: "AnomalyReport",
    savepath: Optional[str] = None,
) -> "plt.Figure":
    """
    Plot anomaly scores with bootstrap confidence bands.

    Parameters
    ----------
    result : dict
        Output dictionary from :func:`bootstrap_scores`.
    report : AnomalyReport
        Observed detector report.
    savepath : str or None, default=None
        Optional output path for the saved figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated matplotlib figure.
    """
    import matplotlib.pyplot as plt

    required = {"mean", "lower", "upper"}
    missing = required.difference(result.keys())
    if missing:
        raise KeyError(f"result is missing required keys: {sorted(missing)}")

    obs = np.asarray(report.scores, dtype=np.float64)
    mean = np.asarray(result["mean"], dtype=np.float64)
    lower = np.asarray(result["lower"], dtype=np.float64)
    upper = np.asarray(result["upper"], dtype=np.float64)

    n = min(obs.size, mean.size, lower.size, upper.size)
    if n == 0:
        raise ValueError("score arrays must be non-empty")

    obs = obs[:n]
    mean = mean[:n]
    lower = lower[:n]
    upper = upper[:n]

    if report.dates is not None and len(report.dates) > 0 and len(report.window_end_indices) >= n:
        idx = np.clip(report.window_end_indices[:n] - 1, 0, len(report.dates) - 1)
        x = report.dates[idx]
        x_label = "Date"
    else:
        x = np.arange(n, dtype=np.int64)
        x_label = "Window Index"

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, obs, color="tab:blue", linewidth=1.3, label="Observed score")
    ax.fill_between(x, lower, upper, color="tab:blue", alpha=0.2, label="Bootstrap CI")
    ax.plot(x, mean, color="black", linestyle="--", linewidth=1.2, label="Bootstrap mean")

    threshold = float(report.mu0 + 3.0 * report.sigma0)
    ax.axhline(threshold, color="tab:red", linestyle="--", linewidth=1.0, label="mu0 + 3*sigma0")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Residue Score")
    ax.set_title("Bootstrap confidence bands for anomaly scores")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    if savepath is not None:
        out = Path(savepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


__all__ = [
    "block_bootstrap_resample",
    "bootstrap_scores",
    "score_stability",
    "plot_bootstrap_bands",
]
