"""Utility helpers for plotting, data loading, and synthetic return simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cfad.detection import AnomalyReport


def plot_scores(
    report: AnomalyReport,
    returns: Optional[pd.Series] = None,
    ax: Optional[Sequence[plt.Axes]] = None,
):
    """Plot returns and CUSUM anomaly scores with alarm markers.

    Parameters
    ----------
    report : AnomalyReport
        Detector output containing scores and alarms.
    returns : pandas.Series, optional
        Return series to annotate with alarm markers.
    ax : tuple[Axes, Axes], optional
        Pre-created axes for the top and bottom panels.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the score plot.
    """
    if returns is not None and not isinstance(returns, pd.Series):
        returns = pd.Series(np.asarray(returns, dtype=np.float64))

    if ax is None:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    else:
        if len(ax) != 2:
            raise ValueError("ax must contain exactly two Axes objects")
        fig = ax[0].figure
        axes = ax

    ax_returns, ax_cusum = axes
    if returns is not None:
        ax_returns.plot(returns.index, returns.values, color="tab:blue", lw=1.0)
        ax_returns.set_ylabel("Returns")
        if report.alarm_indices.size > 0:
            alarm_positions = report.window_end_indices[report.alarm_indices]
            valid = alarm_positions < len(returns)
            alarm_positions = alarm_positions[valid]
            alarm_values = returns.iloc[alarm_positions]
            ax_returns.scatter(
                alarm_values.index,
                alarm_values.values,
                color="red",
                marker="x",
                s=70,
                label="Alarms",
            )
        ax_returns.legend(loc="upper left")
    else:
        ax_returns.text(
            0.5,
            0.5,
            "No return series provided",
            ha="center",
            va="center",
            transform=ax_returns.transAxes,
            color="gray",
        )
        ax_returns.set_ylabel("Returns")

    ax_returns.set_title("Return series with CFAD alarms")
    ax_returns.grid(True, alpha=0.3)

    ax_cusum.plot(report.cusum_pos, label="CUSUM +", color="tab:green")
    ax_cusum.plot(report.cusum_neg, label="CUSUM -", color="tab:orange")
    ax_cusum.axhline(report.threshold, color="red", linestyle="--", label="Threshold")
    ax_cusum.set_ylabel("CUSUM")
    ax_cusum.set_xlabel("Window index")
    ax_cusum.legend(loc="upper left")
    ax_cusum.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def load_spy_sample() -> pd.Series:
    """Load or cache SPY returns for 2018-2022.

    Returns
    -------
    pandas.Series
        Daily SPY returns.
    """
    data_path = Path(__file__).resolve().parents[1] / "data" / "spy_2018_2022.csv"
    if data_path.exists():
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required to download SPY sample data when it is not "
                "cached locally. Install it with 'pip install yfinance'."
            ) from exc

        df = yf.download(
            "SPY",
            start="2018-01-01",
            end="2022-12-31",
            progress=False,
        )
        if df.empty:
            raise RuntimeError("Failed to download SPY sample data from yfinance")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path)

    if "Close" not in df.columns:
        raise ValueError("Downloaded SPY data does not contain a 'Close' column")

    returns = df["Close"].pct_change().dropna()
    returns.index = pd.to_datetime(returns.index)
    return returns


def simulate_levy_returns(
    n: int, alpha: float = 1.7, beta: float = 0.0, scale: float = 0.01
) -> NDArray[np.float64]:
    """Simulate synthetic Lévy-stable return series.

    Parameters
    ----------
    n : int
        Number of returns to simulate.
    alpha : float
        Stability index.
    beta : float
        Skewness parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    ndarray
        Simulated returns of length ``n``.
    """
    from scipy.stats import levy_stable

    if n < 1:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng()
    draws = levy_stable.rvs(alpha, beta, loc=0.0, scale=scale, size=n, random_state=rng)
    return np.asarray(draws, dtype=np.float64)
