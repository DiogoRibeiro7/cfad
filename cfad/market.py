"""Market data helpers and multi-asset detection workflows.

Typical usage
-------------
from cfad.market import load_returns, detect_portfolio
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cfad.api import detect
from cfad.detection import AnomalyReport
from cfad.residue_score import normalise_scores


def _extract_close_column(data: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract a close-price series from a yfinance download frame."""
    if data.empty:
        raise ValueError(f"Downloaded data for {ticker} is empty")

    close: pd.Series | pd.DataFrame
    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", ticker) in data.columns:
            close = data[("Close", ticker)]
        elif ("Adj Close", ticker) in data.columns:
            close = data[("Adj Close", ticker)]
        elif "Close" in data.columns.get_level_values(0):
            close = data["Close"]
        elif "Adj Close" in data.columns.get_level_values(0):
            close = data["Adj Close"]
        else:
            raise ValueError(f"No Close/Adj Close column found for {ticker}")
    else:
        if "Close" in data.columns:
            close = data["Close"]
        elif "Adj Close" in data.columns:
            close = data["Adj Close"]
        else:
            raise ValueError(f"No Close/Adj Close column found for {ticker}")

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    out = pd.Series(close, name=ticker).astype(np.float64)
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def load_returns(
    tickers: list[str],
    start: str,
    end: str,
    return_type: Literal["log", "pct"] = "log",
    cache_dir: str = "data/",
) -> pd.DataFrame:
    """
    Download and cache multi-asset returns from Yahoo Finance.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance ticker strings.
    start, end : str
        Date boundaries in ``YYYY-MM-DD`` format.
    return_type : {"log", "pct"}, default="log"
        Return definition.
    cache_dir : str, default="data/"
        Directory for one CSV cache file per ticker.

    Returns
    -------
    returns_df : pd.DataFrame
        DataFrame of shape ``(T, len(tickers))`` with DatetimeIndex and one
        column per ticker.
    """
    if len(tickers) == 0:
        raise ValueError("tickers must contain at least one symbol")
    if return_type not in {"log", "pct"}:
        raise ValueError("return_type must be 'log' or 'pct'")

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for load_returns(). Install it with `pip install yfinance`."
        ) from exc

    series_list: list[pd.Series] = []
    for ticker in tickers:
        file_name = f"{ticker}_{start}_{end}.csv"
        cache_path = cache_root / file_name

        if cache_path.exists():
            raw = pd.read_csv(cache_path, index_col=0, parse_dates=[0])
        else:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
            if raw.empty:
                raise RuntimeError(f"No data downloaded for ticker '{ticker}'")
            raw.to_csv(cache_path, index_label="Date")

        close = _extract_close_column(raw, ticker)
        if return_type == "log":
            returns = np.log(close / close.shift(1))
        else:
            returns = close.pct_change()
        returns = returns.dropna().rename(ticker)
        series_list.append(returns)

    out = pd.concat(series_list, axis=1)
    out.index = pd.to_datetime(out.index)
    out = out.sort_index().dropna(how="all")
    return out


def detect_portfolio(
    returns_df: pd.DataFrame,
    mode: Literal["univariate", "joint"] = "univariate",
    detector_kwargs: Optional[dict] = None,
) -> dict[str, AnomalyReport]:
    """
    Run anomaly detection across a portfolio return matrix.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Multi-asset return DataFrame.
    mode : {"univariate", "joint"}, default="univariate"
        Per-asset univariate detection or joint multivariate detection.
    detector_kwargs : dict or None, default=None
        Keyword arguments for ``detect()`` (univariate) or
        ``MultivariateDetector`` (joint).

    Returns
    -------
    reports : dict[str, AnomalyReport]
        Mapping from ticker to report. For joint mode: ``{"joint": report}``.
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame")
    if returns_df.shape[1] == 0:
        raise ValueError("returns_df must contain at least one column")

    kwargs = dict(detector_kwargs or {})

    if mode == "univariate":
        reports: dict[str, AnomalyReport] = {}
        for ticker in returns_df.columns:
            series = pd.to_numeric(returns_df[ticker], errors="coerce").dropna()
            if series.empty:
                raise ValueError(f"Column '{ticker}' contains no valid return values")
            series.name = str(ticker)
            reports[str(ticker)] = detect(series, **kwargs)
        return reports

    if mode == "joint":
        from cfad.multivariate import MultivariateDetector

        clean_df = returns_df.apply(pd.to_numeric, errors="coerce").dropna(how="any")
        if clean_df.empty:
            raise ValueError("returns_df has no complete rows for joint detection")
        detector = MultivariateDetector(**kwargs)
        report = detector.fit_transform(
            clean_df.to_numpy(dtype=np.float64),
            dates=pd.DatetimeIndex(clean_df.index),
        )
        return {"joint": report}

    raise ValueError("mode must be either 'univariate' or 'joint'")


def correlation_break_score(
    returns_df: pd.DataFrame,
    window: int = 60,
    step: int = 5,
) -> pd.Series:
    """
    Compute rolling correlation-instability scores.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Multi-asset return DataFrame.
    window : int, default=60
        Rolling window size.
    step : int, default=5
        Step between windows.

    Returns
    -------
    score : pd.Series
        Frobenius-distance score versus in-control correlation matrix,
        indexed by window end timestamps.
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame")
    if window <= 1:
        raise ValueError("window must be greater than 1")
    if step <= 0:
        raise ValueError("step must be positive")

    clean_df = returns_df.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    n = len(clean_df)
    if n < window:
        return pd.Series(dtype=np.float64, name="correlation_break_score")

    base_arr = clean_df.iloc[:window].to_numpy(dtype=np.float64)
    base_corr = np.corrcoef(base_arr, rowvar=False)
    base_corr = np.atleast_2d(np.asarray(base_corr, dtype=np.float64))

    n_windows = (n - window) // step + 1
    scores = np.zeros(n_windows, dtype=np.float64)
    idx = []

    for w in range(n_windows):
        start = w * step
        end = start + window
        arr = clean_df.iloc[start:end].to_numpy(dtype=np.float64)
        corr = np.corrcoef(arr, rowvar=False)
        corr = np.atleast_2d(np.asarray(corr, dtype=np.float64))
        diff = corr - base_corr
        scores[w] = float(np.linalg.norm(diff, ord="fro"))
        idx.append(clean_df.index[end - 1])

    return pd.Series(scores, index=pd.DatetimeIndex(idx), name="correlation_break_score")


def summarise_portfolio_alarms(
    portfolio_reports: dict[str, AnomalyReport],
) -> pd.DataFrame:
    """
    Aggregate alarm summaries across assets.

    Parameters
    ----------
    portfolio_reports : dict[str, AnomalyReport]
        Detector outputs keyed by ticker label.

    Returns
    -------
    summary : pd.DataFrame
        DataFrame with columns ``ticker``, ``n_alarms``, ``first_alarm_date``,
        ``last_alarm_date``, ``mean_score``, and ``max_score``.
    """
    columns = [
        "ticker",
        "n_alarms",
        "first_alarm_date",
        "last_alarm_date",
        "mean_score",
        "max_score",
    ]
    if len(portfolio_reports) == 0:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for ticker, report in portfolio_reports.items():
        n_alarms = int(len(report.alarm_indices))
        if n_alarms > 0 and report.alarm_dates is not None and len(report.alarm_dates) > 0:
            first_alarm = report.alarm_dates[0]
            last_alarm = report.alarm_dates[-1]
        else:
            first_alarm = pd.NaT
            last_alarm = pd.NaT

        scores = np.asarray(report.scores, dtype=np.float64)
        mean_score = float(np.mean(scores)) if scores.size else np.nan
        max_score = float(np.max(scores)) if scores.size else np.nan

        rows.append(
            {
                "ticker": str(ticker),
                "n_alarms": n_alarms,
                "first_alarm_date": first_alarm,
                "last_alarm_date": last_alarm,
                "mean_score": mean_score,
                "max_score": max_score,
            }
        )

    out = pd.DataFrame(rows, columns=columns)
    out = out.sort_values("n_alarms", ascending=False, kind="mergesort").reset_index(drop=True)
    return out


def plot_portfolio_heatmap(
    portfolio_reports: dict[str, AnomalyReport],
    savepath: Optional[str] = None,
) -> "plt.Figure":
    """
    Plot a heatmap of normalized anomaly scores across assets and windows.

    Parameters
    ----------
    portfolio_reports : dict[str, AnomalyReport]
        Detector outputs keyed by ticker label.
    savepath : str or None, default=None
        Optional output path for saving the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Heatmap figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if len(portfolio_reports) == 0:
        raise ValueError("portfolio_reports must contain at least one report")

    tickers = list(portfolio_reports.keys())
    n_assets = len(tickers)
    max_windows = max(int(len(report.scores)) for report in portfolio_reports.values())
    if max_windows == 0:
        raise ValueError("reports contain no score windows")

    zscores = np.full((n_assets, max_windows), np.nan, dtype=np.float64)
    alarm_mask = np.zeros((n_assets, max_windows), dtype=bool)

    for i, ticker in enumerate(tickers):
        report = portfolio_reports[ticker]
        scores = np.asarray(report.scores, dtype=np.float64)
        if scores.size == 0:
            continue

        if np.std(scores) <= 1e-12:
            z = np.zeros_like(scores)
        else:
            z = normalise_scores(scores, method="zscore")
        zscores[i, : scores.size] = z

        valid_alarm_idx = report.alarm_indices[
            (report.alarm_indices >= 0) & (report.alarm_indices < scores.size)
        ]
        alarm_mask[i, valid_alarm_idx] = True

    fig_width = max(8.0, 0.15 * max_windows)
    fig_height = max(3.5, 0.65 * n_assets)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    masked = np.ma.masked_invalid(zscores)
    im = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-3.0,
        vmax=3.0,
    )

    for i in range(n_assets):
        alarm_cols = np.where(alarm_mask[i])[0]
        for j in alarm_cols:
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="black",
                    linewidth=0.8,
                )
            )

    ax.set_yticks(np.arange(n_assets))
    ax.set_yticklabels(tickers)
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Ticker")
    ax.set_title("Portfolio Anomaly Heatmap (within-asset z-scores)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Score z-score")

    fig.tight_layout()
    if savepath is not None:
        out = Path(savepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


__all__ = [
    "load_returns",
    "detect_portfolio",
    "correlation_break_score",
    "summarise_portfolio_alarms",
    "plot_portfolio_heatmap",
]
