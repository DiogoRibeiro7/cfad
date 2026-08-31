"""Bootstrap confidence intervals and stability diagnostics for CFAD scores."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import warnings

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
    """Return one circular-block bootstrap resample of the input series."""
    returns_arr = np.asarray(returns, dtype=np.float64)
    if returns_arr.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    n = int(returns_arr.size)
    if n == 0:
        raise ValueError("returns must be non-empty")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    rng = np.random.default_rng(seed)
    block = int(block_size)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block, dtype=np.int64)
    sampled_blocks = [returns_arr[(int(start) + offsets) % n] for start in starts]
    return np.concatenate(sampled_blocks)[:n].astype(np.float64)


def bootstrap_scores(
    returns: NDArray[np.float64],
    window: int = 60,
    n_bootstrap: int = 200,
    xi_range: tuple[float, float] = (-10.0, 10.0),
    n_xi: int = 128,
    height: Optional[float] = None,
    step: int = 5,
    confidence: float = 0.95,
    seed: int = 0,
    n_jobs: int = 1,
) -> dict[str, NDArray[np.float64]]:
    """Bootstrap pointwise intervals for the rolling ECF-shape score series.

    ``height`` is retained only for source compatibility with the retired
    empirical-contour API. When explicitly supplied it is ignored; use
    ``xi_range`` to control the real-frequency domain of the current statistic.
    """
    if height is not None:
        warnings.warn(
            "height is deprecated and ignored; tune xi_range instead",
            DeprecationWarning,
            stacklevel=2,
        )

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
        step=step,
    )
    target_len = int(len(base_report.scores))
    if target_len == 0:
        raise ValueError("detector produced no rolling windows")

    block_size = max(1, int(window) // 4)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(
        0,
        np.iinfo(np.int64).max,
        size=int(n_bootstrap),
        dtype=np.int64,
    )

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
            step=step,
        )
        return _align_series(np.asarray(report.scores, dtype=np.float64), target_len)

    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed

            runs = Parallel(n_jobs=n_jobs)(delayed(_one_run)(int(seed)) for seed in seeds)
        except Exception:
            runs = [_one_run(int(seed)) for seed in seeds]
    else:
        runs = [_one_run(int(seed)) for seed in seeds]

    score_mat = np.vstack(runs)
    alpha = 1.0 - float(confidence)
    ddof = 1 if score_mat.shape[0] > 1 else 0
    return {
        "mean": np.asarray(np.mean(score_mat, axis=0), dtype=np.float64),
        "lower": np.asarray(
            np.quantile(score_mat, alpha / 2.0, axis=0),
            dtype=np.float64,
        ),
        "upper": np.asarray(
            np.quantile(score_mat, 1.0 - alpha / 2.0, axis=0),
            dtype=np.float64,
        ),
        "std": np.asarray(np.std(score_mat, axis=0, ddof=ddof), dtype=np.float64),
    }


def score_stability(
    returns: NDArray[np.float64],
    window: int = 60,
    n_subsamples: int = 50,
    subsample_frac: float = 0.8,
    seed: int = 0,
) -> dict[str, NDArray[np.float64] | bool]:
    """Estimate ECF-shape score stability under random subsampling."""
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
    for index in range(int(n_subsamples)):
        sample_idx = np.sort(rng.choice(n, size=sub_n, replace=False))
        subsample = returns_arr[sample_idx]
        report = detect(subsample, window=window)
        all_scores[index] = _align_series(
            np.asarray(report.scores, dtype=np.float64),
            target_len,
        )

    mean = np.mean(all_scores, axis=0)
    ddof = 1 if all_scores.shape[0] > 1 else 0
    std = np.std(all_scores, axis=0, ddof=ddof)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(np.abs(mean) > 1e-12, std / np.abs(mean), np.nan)

    return {
        "mean": np.asarray(mean, dtype=np.float64),
        "cv": np.asarray(cv, dtype=np.float64),
        "stable": bool(np.nanmean(cv) < 0.3),
    }


def plot_bootstrap_bands(
    result: dict,
    report: "AnomalyReport",
    savepath: Optional[str] = None,
) -> "plt.Figure":
    """Plot observed ECF-shape scores with pointwise bootstrap bands."""
    import matplotlib.pyplot as plt

    required = {"mean", "lower", "upper"}
    missing = required.difference(result.keys())
    if missing:
        raise KeyError(f"result is missing required keys: {sorted(missing)}")

    observed = np.asarray(report.scores, dtype=np.float64)
    mean = np.asarray(result["mean"], dtype=np.float64)
    lower = np.asarray(result["lower"], dtype=np.float64)
    upper = np.asarray(result["upper"], dtype=np.float64)
    n = min(observed.size, mean.size, lower.size, upper.size)
    if n == 0:
        raise ValueError("score arrays must be non-empty")

    observed = observed[:n]
    mean = mean[:n]
    lower = lower[:n]
    upper = upper[:n]

    if (
        report.dates is not None
        and len(report.dates) > 0
        and len(report.window_end_indices) >= n
    ):
        idx = np.clip(report.window_end_indices[:n] - 1, 0, len(report.dates) - 1)
        x = report.dates[idx]
        x_label = "Date"
    else:
        x = np.arange(n, dtype=np.int64)
        x_label = "Window index"

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, observed, linewidth=1.3, label="Observed score")
    ax.fill_between(x, lower, upper, alpha=0.2, label="Bootstrap CI")
    ax.plot(x, mean, linestyle="--", linewidth=1.2, label="Bootstrap mean")
    threshold = float(report.mu0 + 3.0 * report.sigma0)
    ax.axhline(threshold, linestyle="--", linewidth=1.0, label="mu0 + 3*sigma0")
    ax.set_xlabel(x_label)
    ax.set_ylabel("ECF-shape score")
    ax.set_title("Bootstrap confidence bands for CFAD scores")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    if savepath is not None:
        output = Path(savepath)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")

    return fig


__all__ = [
    "block_bootstrap_resample",
    "bootstrap_scores",
    "score_stability",
    "plot_bootstrap_bands",
]
