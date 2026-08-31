"""Hyperparameter sensitivity utilities for the CFAD ECF-shape detector."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from cfad.api import detect


def window_sensitivity(
    returns: NDArray[np.float64],
    windows: Optional[list[int]] = None,
    h: float = 5.0,
    step: int = 5,
    metric: Literal["n_alarms", "mean_score", "score_std", "cusum_max"] = "mean_score",
) -> pd.DataFrame:
    """Evaluate detector sensitivity to rolling-window length."""
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if step <= 0:
        raise ValueError("step must be positive")
    if h <= 0.0:
        raise ValueError("h must be positive")

    windows_eval = [30, 45, 60, 90, 120] if windows is None else [int(w) for w in windows]
    windows_eval = sorted(set(windows_eval))
    if not windows_eval:
        raise ValueError("windows must contain at least one value")
    if min(windows_eval) <= 1:
        raise ValueError("all windows must be greater than 1")

    rows: list[dict[str, float | int]] = []
    for window in windows_eval:
        report = detect(values, window=window, step=step, h=h)
        if metric == "n_alarms":
            metric_value = float(len(report.alarm_indices))
        elif metric == "mean_score":
            metric_value = float(np.mean(report.scores))
        elif metric == "score_std":
            metric_value = float(np.std(report.scores, ddof=1 if len(report.scores) > 1 else 0))
        elif metric == "cusum_max":
            metric_value = float(max(np.max(report.cusum_pos), np.max(report.cusum_neg)))
        else:
            raise ValueError(f"unsupported metric: {metric}")

        rows.append(
            {
                "window": int(window),
                "n_windows": int(len(report.scores)),
                "metric_value": metric_value,
            }
        )

    return pd.DataFrame(rows, columns=["window", "n_windows", "metric_value"])


def frequency_sensitivity(
    returns: NDArray[np.float64],
    xi_max_values: Optional[list[float]] = None,
    window: int = 60,
    h: float = 5.0,
    step: int = 5,
    n_xi: int = 128,
) -> pd.DataFrame:
    """Evaluate sensitivity to the symmetric real-frequency cutoff.

    The corrected detector operates entirely on the real frequency axis.  The
    relevant geometric tuning parameter is therefore the frequency range
    ``[-xi_max, xi_max]``, not a complex contour height.
    """
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if window <= 1:
        raise ValueError("window must be greater than 1")
    if step <= 0:
        raise ValueError("step must be positive")
    if h <= 0.0:
        raise ValueError("h must be positive")
    if n_xi < 4:
        raise ValueError("n_xi must be at least 4")

    cutoffs = [5.0, 10.0, 20.0, 40.0, 80.0] if xi_max_values is None else [float(v) for v in xi_max_values]
    cutoffs = sorted(set(cutoffs))
    if not cutoffs:
        raise ValueError("xi_max_values must contain at least one value")
    if min(cutoffs) <= 0.0:
        raise ValueError("all xi_max values must be positive")

    rows: list[dict[str, float | int]] = []
    for xi_max in cutoffs:
        report = detect(
            values,
            window=window,
            xi_range=(-xi_max, xi_max),
            n_xi=n_xi,
            step=step,
            h=h,
        )
        rows.append(
            {
                "xi_max": float(xi_max),
                "mean_score": float(np.mean(report.scores)),
                "score_std": float(np.std(report.scores, ddof=1 if len(report.scores) > 1 else 0)),
                "n_alarms": int(len(report.alarm_indices)),
            }
        )

    return pd.DataFrame(rows, columns=["xi_max", "mean_score", "score_std", "n_alarms"])


def threshold_sensitivity(
    returns: NDArray[np.float64],
    h_values: Optional[list[float]] = None,
    window: int = 60,
    step: int = 5,
    calibration_frac: float = 0.3,
    xi_max: float = 10.0,
) -> pd.DataFrame:
    """Evaluate alarm-rate sensitivity to the CUSUM decision threshold."""
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if window <= 1:
        raise ValueError("window must be greater than 1")
    if step <= 0:
        raise ValueError("step must be positive")
    if xi_max <= 0.0:
        raise ValueError("xi_max must be positive")

    h_eval = (
        np.linspace(2.0, 8.0, 13, dtype=np.float64)
        if h_values is None
        else np.asarray(h_values, dtype=np.float64)
    )
    if h_eval.size == 0:
        raise ValueError("h_values must contain at least one value")
    if np.any(h_eval <= 0.0):
        raise ValueError("all h values must be positive")

    rows: list[dict[str, float | int]] = []
    for threshold in np.unique(np.sort(h_eval)):
        report = detect(
            values,
            window=window,
            xi_range=(-xi_max, xi_max),
            step=step,
            calibration_frac=calibration_frac,
            h=float(threshold),
        )
        n_windows = max(1, len(report.scores))
        n_alarms = int(len(report.alarm_indices))
        rows.append(
            {
                "h": float(threshold),
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
    """Recommend a conservative detector configuration from sensitivity sweeps.

    This routine is heuristic.  It is intended for exploratory configuration,
    not as a substitute for out-of-sample calibration against an application-
    specific false-alarm objective.
    """
    if not (0.0 <= target_fpr <= 1.0):
        raise ValueError("target_fpr must be in [0, 1]")

    window_df = window_sensitivity(returns, metric="score_std")
    window_row = window_df.loc[window_df["metric_value"].idxmin()]
    chosen_window = int(window_row["window"])

    frequency_df = frequency_sensitivity(returns, window=chosen_window)
    # Prefer the least variable score among frequency ranges; this avoids the
    # previous circular rule that selected the parameter producing the largest
    # raw anomaly score on the same data.
    frequency_row = frequency_df.loc[frequency_df["score_std"].idxmin()]
    chosen_xi_max = float(frequency_row["xi_max"])

    threshold_df = threshold_sensitivity(
        returns,
        window=chosen_window,
        xi_max=chosen_xi_max,
    )
    threshold_row = threshold_df.iloc[
        (threshold_df["alarm_rate"] - target_fpr).abs().argmin()
    ]
    chosen_h = float(threshold_row["h"])

    rationale = {
        "window": {
            "criterion": "lowest score_std",
            "value": float(window_row["metric_value"]),
        },
        "xi_max": {
            "criterion": "lowest score_std",
            "value": float(frequency_row["score_std"]),
        },
        "h": {
            "criterion": f"alarm_rate closest to target_fpr={target_fpr:.4f}",
            "value": float(threshold_row["alarm_rate"]),
        },
    }

    recommendation = {
        "window": chosen_window,
        "xi_max": chosen_xi_max,
        "h": chosen_h,
        "rationale": rationale,
    }

    if verbose:
        print("CFAD parameter recommendation")
        print(f"  window : {chosen_window}")
        print(f"  xi_max : {chosen_xi_max:.3f}")
        print(f"  h      : {chosen_h:.3f}")

    return recommendation


def plot_sensitivity(
    df: pd.DataFrame,
    param: str,
    metric: str,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
) -> "plt.Figure":
    """Plot one sensitivity metric against one parameter column."""
    import matplotlib.pyplot as plt

    if param not in df.columns:
        raise KeyError(f"param column '{param}' not found")
    if metric not in df.columns:
        raise KeyError(f"metric column '{metric}' not found")

    x = np.asarray(df[param], dtype=np.float64)
    y = np.asarray(df[metric], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker="o", linewidth=1.4)
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
    "frequency_sensitivity",
    "threshold_sensitivity",
    "recommend_params",
    "plot_sensitivity",
]
