"""Hyperparameter sensitivity analysis utilities for CFAD."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cfad.api import detect


def window_sensitivity(
    returns: NDArray[np.float64],
    windows: Optional[list[int]] = None,
    h: float = 5.0,
    step: int = 5,
    metric: Literal["n_alarms", "mean_score", "score_std", "cusum_max"] = "mean_score",
) -> pd.DataFrame:
    """
    Run ``detect()`` for each window size and collect a summary metric.

    Parameters
    ----------
    returns : float ndarray of shape (n,)
        Return series.
    windows : list of int or None, default=None
        Window sizes to evaluate. Defaults to ``[30, 45, 60, 90, 120]``.
    h : float, default=5.0
        CUSUM threshold.
    step : int, default=5
        Rolling step.
    metric : {"n_alarms", "mean_score", "score_std", "cusum_max"}, default="mean_score"
        Summary metric stored in ``metric_value``.

    Returns
    -------
    summary : pd.DataFrame
        DataFrame with columns ``window``, ``n_windows``, ``metric_value``.
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
    if returns_arr.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if step <= 0:
        raise ValueError("step must be positive")
    if h <= 0:
        raise ValueError("h must be positive")

    if windows is None:
        windows_eval = [30, 45, 60, 90, 120]
    else:
        windows_eval = [int(w) for w in windows]

    windows_eval = sorted(set(windows_eval))
    if len(windows_eval) == 0:
        raise ValueError("windows must contain at least one value")
    if min(windows_eval) <= 1:
        raise ValueError("all windows must be > 1")

    rows: list[dict[str, float | int]] = []
    for window in windows_eval:
        report = detect(returns_arr, window=window, step=step, h=h)

        if metric == "n_alarms":
            metric_value = float(len(report.alarm_indices))
        elif metric == "mean_score":
            metric_value = float(np.mean(report.scores))
        elif metric == "score_std":
            ddof = 1 if len(report.scores) > 1 else 0
            metric_value = float(np.std(report.scores, ddof=ddof))
        elif metric == "cusum_max":
            metric_value = float(
                max(np.max(report.cusum_pos), np.max(report.cusum_neg))
            )
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        rows.append(
            {
                "window": int(window),
                "n_windows": int(len(report.scores)),
                "metric_value": metric_value,
            }
        )

    return pd.DataFrame(rows, columns=["window", "n_windows", "metric_value"])


def height_sensitivity(
    returns: NDArray[np.float64],
    heights: Optional[list[float]] = None,
    window: int = 60,
    h: float = 5.0,
    step: int = 5,
) -> pd.DataFrame:
    """
    Run ``detect()`` for each contour height and collect score diagnostics.

    Parameters
    ----------
    returns : float ndarray of shape (n,)
        Return series.
    heights : list of float or None, default=None
        Contour heights to evaluate. Defaults to
        ``[0.05, 0.1, 0.2, 0.3, 0.5]``.
    window : int, default=60
        Detector rolling window.
    h : float, default=5.0
        CUSUM threshold.
    step : int, default=5
        Rolling step.

    Returns
    -------
    summary : pd.DataFrame
        DataFrame with columns ``height``, ``mean_score``, ``score_std``,
        and ``n_alarms``.
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
    if returns_arr.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if window <= 1:
        raise ValueError("window must be > 1")
    if step <= 0:
        raise ValueError("step must be positive")

    if heights is None:
        heights_eval = [0.05, 0.1, 0.2, 0.3, 0.5]
    else:
        heights_eval = [float(v) for v in heights]

    heights_eval = sorted(set(heights_eval))
    if len(heights_eval) == 0:
        raise ValueError("heights must contain at least one value")
    if min(heights_eval) <= 0:
        raise ValueError("all heights must be positive")

    rows: list[dict[str, float | int]] = []
    for height in heights_eval:
        report = detect(
            returns_arr,
            window=window,
            step=step,
            h=h,
            height=height,
        )
        ddof = 1 if len(report.scores) > 1 else 0
        rows.append(
            {
                "height": float(height),
                "mean_score": float(np.mean(report.scores)),
                "score_std": float(np.std(report.scores, ddof=ddof)),
                "n_alarms": int(len(report.alarm_indices)),
            }
        )

    return pd.DataFrame(rows, columns=["height", "mean_score", "score_std", "n_alarms"])


def threshold_sensitivity(
    returns: NDArray[np.float64],
    h_values: Optional[list[float]] = None,
    window: int = 60,
    step: int = 5,
    calibration_frac: float = 0.3,
) -> pd.DataFrame:
    """
    Run ``detect()`` for each CUSUM threshold and collect alarm statistics.

    Parameters
    ----------
    returns : float ndarray of shape (n,)
        Return series.
    h_values : list of float or None, default=None
        Threshold values to evaluate. Defaults to ``linspace(2.0, 8.0, 13)``.
    window : int, default=60
        Detector rolling window.
    step : int, default=5
        Rolling step.
    calibration_frac : float, default=0.3
        Calibration fraction passed to detector.

    Returns
    -------
    summary : pd.DataFrame
        DataFrame with columns ``h``, ``n_alarms``, and ``alarm_rate``.
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
    if returns_arr.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if window <= 1:
        raise ValueError("window must be > 1")
    if step <= 0:
        raise ValueError("step must be positive")

    if h_values is None:
        h_eval = np.linspace(2.0, 8.0, 13, dtype=np.float64)
    else:
        h_eval = np.asarray(h_values, dtype=np.float64)

    if h_eval.size == 0:
        raise ValueError("h_values must contain at least one value")
    if np.any(h_eval <= 0.0):
        raise ValueError("all h values must be positive")

    h_eval = np.unique(np.sort(h_eval))

    rows: list[dict[str, float | int]] = []
    for h in h_eval:
        report = detect(
            returns_arr,
            window=window,
            step=step,
            calibration_frac=calibration_frac,
            h=float(h),
        )
        n_windows = max(1, len(report.scores))
        n_alarms = int(len(report.alarm_indices))
        rows.append(
            {
                "h": float(h),
                "n_alarms": n_alarms,
                "alarm_rate": float(n_alarms / n_windows),
            }
        )

    return pd.DataFrame(rows, columns=["h", "n_alarms", "alarm_rate"])


def recommend_params(
    returns: NDArray[np.float64],
    target_fpr: float = 0.02,
    verbose: bool = True,
) -> dict[str, object]:
    """
    Recommend detector hyperparameters from sensitivity sweeps.

    Parameters
    ----------
    returns : float ndarray of shape (n,)
        Return series.
    target_fpr : float, default=0.02
        Target alarm rate for selecting ``h``.
    verbose : bool, default=True
        If ``True``, print a summary of the recommendation.

    Returns
    -------
    recommendation : dict
        Dictionary with ``window``, ``height``, ``h``, and ``rationale``.
    """
    if not (0.0 <= target_fpr <= 1.0):
        raise ValueError("target_fpr must be in [0, 1]")

    w_df = window_sensitivity(returns, metric="score_std")
    w_row = w_df.loc[w_df["metric_value"].idxmin()]
    chosen_window = int(w_row["window"])

    hgt_df = height_sensitivity(returns, window=chosen_window)
    hgt_row = hgt_df.loc[hgt_df["mean_score"].idxmax()]
    chosen_height = float(hgt_row["height"])

    thr_df = threshold_sensitivity(returns, window=chosen_window)
    thr_row = thr_df.iloc[(thr_df["alarm_rate"] - target_fpr).abs().argmin()]
    chosen_h = float(thr_row["h"])

    rationale = {
        "window": {
            "criterion": "lowest score_std",
            "value": float(w_row["metric_value"]),
        },
        "height": {
            "criterion": "highest mean_score",
            "value": float(hgt_row["mean_score"]),
        },
        "h": {
            "criterion": f"alarm_rate closest to target_fpr={target_fpr:.4f}",
            "value": float(thr_row["alarm_rate"]),
        },
    }

    recommendation = {
        "window": chosen_window,
        "height": chosen_height,
        "h": chosen_h,
        "rationale": rationale,
    }

    if verbose:
        print("CFAD parameter recommendation")
        print(f"  window : {chosen_window}")
        print(f"  height : {chosen_height:.3f}")
        print(f"  h      : {chosen_h:.3f}")
        print("  rationale:")
        print(f"    - window: {rationale['window']['criterion']}")
        print(f"    - height: {rationale['height']['criterion']}")
        print(f"    - h: {rationale['h']['criterion']}")

    return recommendation


def plot_sensitivity(
    df: pd.DataFrame,
    param: str,
    metric: str,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
) -> "plt.Figure":
    """
    Plot a sensitivity curve (metric vs parameter value).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing parameter and metric columns.
    param : str
        X-axis column name.
    metric : str
        Y-axis column name.
    title : str or None, default=None
        Optional figure title.
    savepath : str or None, default=None
        Optional PNG save path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated matplotlib figure.
    """
    import matplotlib.pyplot as plt

    if param not in df.columns:
        raise KeyError(f"param column '{param}' not found")
    if metric not in df.columns:
        raise KeyError(f"metric column '{metric}' not found")

    x = np.asarray(df[param], dtype=np.float64)
    y = np.asarray(df[metric], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker="o", linewidth=1.4, color="tab:blue")
    ax.set_xlabel(param)
    ax.set_ylabel(metric)
    ax.set_title(title if title is not None else f"{metric} vs {param}")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if savepath is not None:
        out = Path(savepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


__all__ = [
    "window_sensitivity",
    "height_sensitivity",
    "threshold_sensitivity",
    "recommend_params",
    "plot_sensitivity",
]
