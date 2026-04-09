"""
Public API — single-function entry points for common use cases.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Optional, Union

from cfad.detection import RollingDetector, AnomalyReport


def detect(
    returns: Union[NDArray, pd.Series],
    window: int = 60,
    xi_range: tuple[float, float] = (-10.0, 10.0),
    n_xi: int = 128,
    height: float = 0.2,
    step: int = 1,
    calibration_frac: float = 0.3,
    h: float = 5.0,
) -> AnomalyReport:
    """
    Detect structural anomalies in a financial return series.

    Parameters
    ----------
    returns : array-like or pd.Series
        Log-returns (daily, intraday, etc.).
    window : int
        Rolling window size for ECF estimation.
    xi_range : (float, float)
        Frequency grid [xi_min, xi_max].
    n_xi : int
        Number of frequency grid points.
    height : float
        Contour imaginary half-height (keep < analyticity strip).
    step : int
        Rolling step.
    calibration_frac : float
        Fraction of data for in-control calibration.
    h : float
        CUSUM alarm threshold.

    Returns
    -------
    AnomalyReport
        Contains scores, CUSUM statistics, alarm indices and dates.

    Examples
    --------
    >>> import yfinance as yf
    >>> from cfad import detect
    >>> prices = yf.download("SPY", start="2019-01-01", end="2021-01-01")["Close"]
    >>> rets = prices.pct_change().dropna()
    >>> report = detect(rets, window=60, h=4.0)
    >>> print(report.summary())
    """
    dates = None
    if isinstance(returns, pd.Series):
        dates = returns.index if hasattr(returns.index, "to_pydatetime") else None
        returns_arr = returns.values.astype(np.float64)
    else:
        returns_arr = np.asarray(returns, dtype=np.float64)

    detector = RollingDetector(
        window=window,
        xi_min=xi_range[0],
        xi_max=xi_range[1],
        n_xi=n_xi,
        height=height,
        step=step,
        calibration_frac=calibration_frac,
        h=h,
    )
    return detector.fit_transform(returns_arr, dates=dates)


def compare_models(
    returns: NDArray[np.float64],
    xi: Optional[NDArray[np.float64]] = None,
) -> dict:
    """
    Fit Gaussian and NIG models and compare their AIC and CF distance.

    Returns a dict with keys 'gaussian', 'nig', and 'winner'.
    Useful for confirming whether non-analytic structure is present.
    """
    from cfad.models.gaussian import GaussianCF
    from cfad.models.nig import NIGCF
    from cfad.empirical_cf import ecf_at

    returns = np.asarray(returns, dtype=np.float64)
    if xi is None:
        xi = np.linspace(-15, 15, 256)

    ecf = ecf_at(returns, xi)

    g = GaussianCF().fit(returns)
    n = NIGCF().fit(returns)

    g_dist = float(np.mean(np.abs(ecf - g.cf(xi)) ** 2))
    n_dist = float(np.mean(np.abs(ecf - n.cf(xi)) ** 2))

    return {
        "gaussian": {"model": g, "ecf_l2": g_dist, "aic": g.aic(returns)},
        "nig":      {"model": n, "ecf_l2": n_dist, "aic": n.aic(returns)},
        "winner":   "nig" if n_dist < g_dist else "gaussian",
    }
