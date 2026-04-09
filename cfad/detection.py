"""
Anomaly detection layer.

Combines rolling ECF estimation, contour scoring, and sequential
detection (CUSUM) into a unified detection pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Optional

from cfad.empirical_cf import rolling_ecf
from cfad.contour import ecf_residue_scores

try:
    from cfad._ext.cusum import cusum as _cusum_c
    _HAS_C_EXT = True
except ImportError:
    _HAS_C_EXT = False


@dataclass
class AnomalyReport:
    """Container for detector output."""
    scores: NDArray[np.float64]
    cusum_pos: NDArray[np.float64]
    cusum_neg: NDArray[np.float64]
    alarm_indices: NDArray[np.int64]
    window_end_indices: NDArray[np.int64]
    dates: Optional[pd.DatetimeIndex] = None
    mu0: float = 0.0
    sigma0: float = 1.0
    threshold: float = 5.0

    @property
    def alarm_dates(self) -> Optional[pd.DatetimeIndex]:
        if self.dates is None:
            return None
        valid = self.alarm_indices[self.alarm_indices < len(self.window_end_indices)]
        win_idx = self.window_end_indices[valid]
        valid_win = win_idx[win_idx < len(self.dates)]
        return self.dates[valid_win]

    def summary(self) -> str:
        n_alarms = len(self.alarm_indices)
        lines = [
            f"CFAD Anomaly Report",
            f"  Windows evaluated : {len(self.scores)}",
            f"  Alarms fired      : {n_alarms}",
            f"  In-control mean   : {self.mu0:.4f}",
            f"  In-control std    : {self.sigma0:.4f}",
            f"  CUSUM threshold   : {self.threshold}",
        ]
        if self.alarm_dates is not None and n_alarms:
            lines.append(f"  First alarm       : {self.alarm_dates[0].date()}")
        return "\n".join(lines)


def _cusum_python(
    scores: NDArray[np.float64],
    mu0: float,
    sigma0: float,
    k: float = 0.5,
    h: float = 5.0,
) -> tuple:
    """Pure Python CUSUM fallback."""
    T = len(scores)
    S_pos, S_neg = 0.0, 0.0
    sp = np.zeros(T)
    sn = np.zeros(T)
    alarms = []
    slack = k * sigma0
    for t in range(T):
        z = (scores[t] - mu0) / sigma0
        S_pos = max(0.0, S_pos + z - slack)
        S_neg = max(0.0, S_neg - z - slack)
        sp[t], sn[t] = S_pos, S_neg
        if S_pos > h or S_neg > h:
            alarms.append(t)
            S_pos, S_neg = 0.0, 0.0
    return sp, sn, np.array(alarms, dtype=np.int64)


class RollingDetector:
    """
    Main detector object: rolling ECF → contour score → CUSUM alarm.

    Parameters
    ----------
    window : int
        Number of returns per ECF estimation window.
    xi_min, xi_max : float
        Frequency grid bounds for ECF evaluation.
    n_xi : int
        Number of frequency grid points.
    height : float
        Contour imaginary half-height.
    step : int
        Rolling step (1 = fully overlapping, window = non-overlapping).
    calibration_frac : float
        Fraction of data used to estimate in-control parameters mu0, sigma0.
    k : float
        CUSUM allowance (default 0.5σ shift detection).
    h : float
        CUSUM alarm threshold.
    """

    def __init__(
        self,
        window: int = 60,
        xi_min: float = -10.0,
        xi_max: float = 10.0,
        n_xi: int = 128,
        height: float = 0.2,
        step: int = 1,
        calibration_frac: float = 0.3,
        k: float = 0.5,
        h: float = 5.0,
    ):
        self.window = window
        self.xi_grid = np.linspace(xi_min, xi_max, n_xi)
        self.height = height
        self.step = step
        self.calibration_frac = calibration_frac
        self.k = k
        self.h = h
        self._mu0: Optional[float] = None
        self._sigma0: Optional[float] = None

    def fit_transform(
        self,
        returns: NDArray[np.float64],
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> AnomalyReport:
        """
        Run the full detection pipeline on a return series.

        Steps
        -----
        1. Rolling ECF estimation (Cython when available)
        2. Contour residue scoring
        3. In-control calibration on first `calibration_frac` of scores
        4. CUSUM sequential detection
        """
        returns = np.asarray(returns, dtype=np.float64)
        ecf_mat, end_idx = rolling_ecf(returns, self.xi_grid, self.window, self.step)
        scores = ecf_residue_scores(ecf_mat, self.xi_grid, self.height)

        n_cal = max(10, int(self.calibration_frac * len(scores)))
        self._mu0 = float(np.mean(scores[:n_cal]))
        self._sigma0 = float(np.std(scores[:n_cal], ddof=1)) + 1e-12

        cusum_fn = _cusum_c if _HAS_C_EXT else _cusum_python
        sp, sn, alarms = cusum_fn(scores, self._mu0, self._sigma0, self.k, self.h)

        return AnomalyReport(
            scores=scores,
            cusum_pos=sp,
            cusum_neg=sn,
            alarm_indices=alarms,
            window_end_indices=end_idx,
            dates=dates,
            mu0=self._mu0,
            sigma0=self._sigma0,
            threshold=self.h,
        )
