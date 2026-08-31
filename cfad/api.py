"""Public API entry points for common CFAD workflows."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from cfad.detection import AnomalyReport, RollingDetector


def detect(
    returns: Union[NDArray, pd.Series],
    window: int = 60,
    xi_range: tuple[float, float] = (-10.0, 10.0),
    n_xi: int = 128,
    step: int = 1,
    calibration_frac: float = 0.3,
    k: float = 0.5,
    h: float = 5.0,
) -> AnomalyReport:
    """Detect distributional-shape changes in a financial return series.

    Each rolling empirical characteristic function is compared with the
    Gaussian characteristic function fitted to the same window.  The resulting
    real-frequency L2 distance is monitored with a two-sided Page-CUSUM.

    Parameters
    ----------
    returns : array-like or pandas.Series
        One-dimensional return series.
    window : int, default=60
        Rolling window size for ECF estimation.
    xi_range : tuple[float, float], default=(-10, 10)
        Real-frequency grid bounds.
    n_xi : int, default=128
        Number of frequency grid points.
    step : int, default=1
        Rolling step.
    calibration_frac : float, default=0.3
        Fraction of score windows used to estimate the in-control score mean and
        standard deviation.
    k : float, default=0.5
        Dimensionless Page-CUSUM reference value on standardized scores.
    h : float, default=5.0
        CUSUM decision threshold.

    Returns
    -------
    AnomalyReport
        Scores, CUSUM statistics, alarm indices, and optional dates.
    """
    dates = None
    if isinstance(returns, pd.Series):
        dates = returns.index if hasattr(returns.index, "to_pydatetime") else None
        returns_arr = returns.to_numpy(dtype=np.float64)
    else:
        returns_arr = np.asarray(returns, dtype=np.float64)

    detector = RollingDetector(
        window=window,
        xi_min=xi_range[0],
        xi_max=xi_range[1],
        n_xi=n_xi,
        step=step,
        calibration_frac=calibration_frac,
        k=k,
        h=h,
    )
    return detector.fit_transform(returns_arr, dates=dates)


def compare_models(
    returns: NDArray[np.float64],
    xi: Optional[NDArray[np.float64]] = None,
) -> dict[str, object]:
    """Fit Gaussian and NIG models and compare real-frequency ECF distance.

    This model comparison is descriptive evidence about distributional fit.  It
    must not be interpreted as a test for branch cuts or population-CF
    singularities from a finite-sample empirical characteristic function.
    """
    from cfad.empirical_cf import ecf_at
    from cfad.models.gaussian import GaussianCF
    from cfad.models.nig import NIGCF

    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("returns must be one-dimensional with at least 2 values")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite values")

    grid = (
        np.linspace(-15.0, 15.0, 256, dtype=np.float64)
        if xi is None
        else np.asarray(xi, dtype=np.float64)
    )
    if grid.ndim != 1 or grid.size < 4:
        raise ValueError("xi must be one-dimensional with at least 4 values")

    empirical = ecf_at(values, grid)
    gaussian = GaussianCF().fit(values)
    nig = NIGCF().fit(values)

    gaussian_distance = float(np.mean(np.abs(empirical - gaussian.cf(grid)) ** 2))
    nig_distance = float(np.mean(np.abs(empirical - nig.cf(grid)) ** 2))

    return {
        "gaussian": {
            "model": gaussian,
            "ecf_l2": gaussian_distance,
            "aic": gaussian.aic(values),
        },
        "nig": {
            "model": nig,
            "ecf_l2": nig_distance,
            "aic": nig.aic(values),
        },
        "winner": "nig" if nig_distance < gaussian_distance else "gaussian",
    }
