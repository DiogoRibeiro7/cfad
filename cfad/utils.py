"""Utility helpers for plotting, data loading, and synthetic return simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cfad.detection import AnomalyReport


def plot_scores(
    report: AnomalyReport,
    returns: Optional[NDArray[np.float64]] = None,
    figsize: tuple[float, float] = (12, 6),
    ax: Optional[object] = None,
) -> plt.Figure:
    """Plot returns, ECF-shape scores, CUSUM state, and alarm locations."""
    scores = np.asarray(report.scores, dtype=np.float64)
    cusum_pos = np.asarray(report.cusum_pos, dtype=np.float64)
    alarm_idx = np.asarray(report.alarm_indices, dtype=np.int64)
    win_end_idx = np.asarray(report.window_end_indices, dtype=np.int64)
    n_scores = int(scores.shape[0])
    valid_alarm_idx = alarm_idx[(alarm_idx >= 0) & (alarm_idx < n_scores)]

    if (
        report.dates is not None
        and len(report.dates) > 0
        and win_end_idx.size >= n_scores
    ):
        score_date_idx = np.clip(
            win_end_idx[:n_scores] - 1,
            0,
            len(report.dates) - 1,
        )
        x_scores = report.dates[score_date_idx]
        alarm_x_bottom = x_scores[valid_alarm_idx]
    else:
        x_scores = np.arange(n_scores)
        alarm_x_bottom = valid_alarm_idx

    returns_arr = (
        np.asarray(returns, dtype=np.float64) if returns is not None else None
    )

    if ax is None:
        if returns_arr is None:
            fig, ax_bottom = plt.subplots(1, 1, figsize=figsize)
            ax_top = None
        else:
            fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=False)
            ax_top, ax_bottom = axes
    else:
        if returns_arr is None:
            if isinstance(ax, tuple):
                raise ValueError(
                    "ax must be a single matplotlib axis when returns is not provided"
                )
            if not hasattr(ax, "plot"):
                raise ValueError("ax must be a matplotlib axis")
            ax_top = None
            ax_bottom = ax
        else:
            if not isinstance(ax, tuple) or len(ax) != 2:
                raise ValueError(
                    "ax must be a tuple of two matplotlib axes when returns is provided"
                )
            ax_top, ax_bottom = ax
            if ax_top is None:
                raise ValueError("ax[0] cannot be None when returns is provided")
        fig = ax_bottom.figure

    if returns_arr is not None:
        n_returns = int(returns_arr.shape[0])
        if report.dates is not None and len(report.dates) >= n_returns:
            x_returns = report.dates[:n_returns]
        else:
            x_returns = np.arange(n_returns)

        if win_end_idx.size > 0:
            in_range_alarms = valid_alarm_idx[valid_alarm_idx < win_end_idx.size]
            alarm_return_idx = np.clip(
                win_end_idx[in_range_alarms] - 1,
                0,
                n_returns - 1,
            )
            if report.dates is not None and len(report.dates) >= n_returns:
                alarm_x_top = report.dates[alarm_return_idx]
            else:
                alarm_x_top = alarm_return_idx
        else:
            alarm_x_top = np.array([], dtype=np.int64)

        ax_top.plot(
            x_returns,
            returns_arr,
            linewidth=1.0,
            label="Returns",
        )
        for x_alarm in alarm_x_top:
            ax_top.axvline(
                x_alarm,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )
        ax_top.set_ylabel("Returns")
        ax_top.legend(loc="upper left")
        ax_top.grid(alpha=0.25)

    ax_bottom.plot(
        x_scores,
        scores,
        linewidth=1.2,
        label="ECF-shape score",
    )
    ax_bottom.plot(
        x_scores,
        cusum_pos,
        linewidth=1.2,
        label="CUSUM+",
    )
    ax_bottom.axhline(
        float(report.threshold),
        linestyle="--",
        linewidth=1.0,
        label="CUSUM threshold",
    )
    for x_alarm in alarm_x_bottom:
        ax_bottom.axvline(
            x_alarm,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )
    ax_bottom.set_ylabel("Score / CUSUM")
    ax_bottom.set_xlabel("Date" if report.dates is not None else "Index")
    ax_bottom.legend(loc="upper left")
    ax_bottom.grid(alpha=0.25)

    fig.suptitle("CFAD ECF-shape monitoring")
    fig.tight_layout()
    return fig


def load_spy_sample(
    start: str = "2018-01-01",
    end: str = "2023-01-01",
    cache_path: str = "data/spy_sample.csv",
) -> pd.Series:
    """Load cached or downloaded SPY log returns."""
    resolved_cache_path = Path(cache_path)
    if not resolved_cache_path.is_absolute():
        resolved_cache_path = Path(__file__).resolve().parents[1] / resolved_cache_path

    if cache_path == "data/spy_sample.csv":
        legacy_path = (
            Path(__file__).resolve().parents[1] / "data" / "spy_2018_2022.csv"
        )
        if legacy_path.exists():
            resolved_cache_path = legacy_path

    if resolved_cache_path.exists():
        data = pd.read_csv(resolved_cache_path, index_col=0, parse_dates=True)
        if "log_return" in data.columns:
            log_return = data["log_return"].astype(np.float64)
        elif "Close" in data.columns:
            close = data["Close"].astype(np.float64)
            log_return = np.log(close / close.shift(1)).dropna()
            log_return.name = "log_return"
        else:
            raise ValueError(
                "Cached file must contain either 'log_return' or 'Close' column"
            )
    else:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required to download SPY data when cache is missing"
            ) from exc

        data = yf.download(
            "SPY",
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )
        if data.empty:
            raise RuntimeError("SPY download returned no data")

        close = data["Close"]
        if hasattr(close, "ndim") and close.ndim == 2:
            close = close.iloc[:, 0]
        close = close.astype(np.float64)
        log_return = np.log(close / close.shift(1)).dropna()
        log_return.name = "log_return"

        resolved_cache_path.parent.mkdir(parents=True, exist_ok=True)
        log_return.to_frame().to_csv(resolved_cache_path, index_label="Date")

    log_return.index = pd.to_datetime(log_return.index)
    log_return.name = "log_return"
    return log_return


def simulate_levy_returns(
    n: int,
    alpha: float = 1.7,
    beta: float = 0.0,
    scale: float = 0.01,
    mu: float = 0.0,
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """Simulate returns from a Lévy-stable distribution."""
    from scipy.stats import levy_stable

    if n <= 0:
        raise ValueError("n must be a positive integer")
    if not (0.0 < alpha <= 2.0):
        raise ValueError("alpha must satisfy 0 < alpha <= 2")
    if not (-1.0 <= beta <= 1.0):
        raise ValueError("beta must satisfy -1 <= beta <= 1")
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    rng = np.random.default_rng(seed)
    samples = levy_stable.rvs(
        alpha=alpha,
        beta=beta,
        loc=mu,
        scale=scale,
        size=int(n),
        random_state=rng,
    )
    return np.asarray(samples, dtype=np.float64)
