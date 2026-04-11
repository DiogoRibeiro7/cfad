"""Utility helpers for plotting, data loading, and synthetic return simulation."""

from __future__ import annotations


def plot_scores(
    report: "AnomalyReport",
    returns: "Optional[NDArray[np.float64]]" = None,
    figsize: tuple = (12, 6),
    ax: "Optional[tuple]" = None,
) -> "plt.Figure":
    """
    Two-panel diagnostic plot.

    Top panel (if `returns` is provided):
      - Line plot of the return series
      - Vertical red dashed lines at each alarm date / index

    Bottom panel:
      - Line plot of report.scores (raw residue scores)
      - Line plot of report.cusum_pos in orange
      - Horizontal dashed line at report.threshold (CUSUM alarm level)
      - Vertical red dashed lines at each alarm index

    Use report.dates for x-axis labels if available, otherwise integer indices.
    Title: "CFAD Anomaly Score — RollingDetector"
    Save nothing; return the Figure object.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    scores = np.asarray(report.scores, dtype=np.float64)
    cusum_pos = np.asarray(report.cusum_pos, dtype=np.float64)
    alarm_idx = np.asarray(report.alarm_indices, dtype=np.int64)
    win_end_idx = np.asarray(report.window_end_indices, dtype=np.int64)
    n_scores = int(scores.shape[0])

    valid_alarm_idx = alarm_idx[(alarm_idx >= 0) & (alarm_idx < n_scores)]

    if report.dates is not None and len(report.dates) > 0 and win_end_idx.size >= n_scores:
        score_date_idx = np.clip(win_end_idx[:n_scores] - 1, 0, len(report.dates) - 1)
        x_scores = report.dates[score_date_idx]
        alarm_x_bottom = x_scores[valid_alarm_idx]
    else:
        x_scores = np.arange(n_scores)
        alarm_x_bottom = valid_alarm_idx

    if returns is not None:
        returns_arr = np.asarray(returns, dtype=np.float64)
    else:
        returns_arr = None

    if ax is None:
        if returns_arr is None:
            fig, ax_bottom = plt.subplots(1, 1, figsize=figsize)
            ax_top = None
        else:
            fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=False)
            ax_top, ax_bottom = axes
    else:
        if returns_arr is None:
            if not isinstance(ax, tuple) or len(ax) == 0:
                raise ValueError("ax must be a non-empty tuple of matplotlib axes")
            ax_top = None
            ax_bottom = ax[-1]
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
            alarm_return_idx = np.clip(win_end_idx[in_range_alarms] - 1, 0, n_returns - 1)
            if report.dates is not None and len(report.dates) >= n_returns:
                alarm_x_top = report.dates[alarm_return_idx]
            else:
                alarm_x_top = alarm_return_idx
        else:
            alarm_x_top = np.array([], dtype=np.int64)

        ax_top.plot(x_returns, returns_arr, color="tab:blue", linewidth=1.0, label="Returns")
        for x_alarm in alarm_x_top:
            ax_top.axvline(x_alarm, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_top.set_ylabel("Returns")
        ax_top.legend(loc="upper left")
        ax_top.grid(alpha=0.25)

    ax_bottom.plot(x_scores, scores, color="tab:blue", linewidth=1.2, label="Residue Score")
    ax_bottom.plot(x_scores, cusum_pos, color="orange", linewidth=1.2, label="CUSUM+")
    ax_bottom.axhline(
        float(report.threshold),
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Threshold",
    )
    for x_alarm in alarm_x_bottom:
        ax_bottom.axvline(x_alarm, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_bottom.set_ylabel("Score")
    ax_bottom.set_xlabel("Date" if report.dates is not None else "Index")
    ax_bottom.legend(loc="upper left")
    ax_bottom.grid(alpha=0.25)

    fig.suptitle("CFAD Anomaly Score — RollingDetector")
    fig.tight_layout()
    return fig


def load_spy_sample(
    start: str = "2018-01-01",
    end: str = "2023-01-01",
    cache_path: str = "data/spy_sample.csv",
) -> "pd.Series":
    """
    Load SPY log-returns.

    If `cache_path` exists, load from CSV (column "log_return", index "Date").
    Otherwise download via yfinance, compute log-returns as
      log_return = log(Close_t / Close_{t-1}),
    save to cache_path, and return the Series.

    Returns pd.Series with DatetimeIndex, name="log_return".
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd

    resolved_cache_path = Path(cache_path)
    if not resolved_cache_path.is_absolute():
        resolved_cache_path = Path(__file__).resolve().parents[1] / resolved_cache_path

    if resolved_cache_path.exists():
        data = pd.read_csv(resolved_cache_path, index_col="Date", parse_dates=["Date"])
        if "log_return" not in data.columns:
            raise ValueError("Cached file must contain a 'log_return' column")
        log_return = data["log_return"].astype(np.float64)
    else:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required to download SPY data when cache is missing."
            ) from exc

        data = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=False)
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
    seed: "Optional[int]" = None,
) -> "NDArray[np.float64]":
    """
    Simulate n observations from a Lévy-stable distribution.

    Uses the Chambers-Mallows-Stuck method via scipy.stats.levy_stable.rvs.
    When alpha=2.0 this reduces to Gaussian(mu, sqrt(2)*scale).

    Parameters
    ----------
    n : int
        Sample size.
    alpha : float
        Stability index (0 < alpha <= 2).
    beta : float
        Skewness (-1 <= beta <= 1).
    scale : float
        Scale parameter (analogous to std for Gaussian).
    mu : float
        Location.
    seed : int | None
        Random seed.

    Returns
    -------
    returns : float ndarray of shape (n,)
    """
    import numpy as np
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
