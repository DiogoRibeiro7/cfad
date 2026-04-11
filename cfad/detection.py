"""
Anomaly detection layer.

Combines rolling ECF estimation, contour scoring, and sequential
detection (CUSUM) into a unified detection pipeline.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Optional

from cfad.empirical_cf import ecf_at, rolling_ecf
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


class StreamDetector:
    """
    Online (streaming) anomaly detector. Processes one return at a time.

    Maintains a circular buffer of the last `window` returns, recomputes
    the ECF and residue score on each update, and runs the CUSUM recursion
    incrementally. No full-series storage required after initialisation.

    Parameters
    ----------
    window : int
        Lookback window for ECF estimation.
    xi_min, xi_max : float
        Frequency grid bounds.
    n_xi : int
        Number of frequency grid points.
    height : float
        Contour imaginary half-height.
    mu0 : float or None
        In-control score mean. If None, estimated from first `warmup` updates.
    sigma0 : float or None
        In-control score std. If None, estimated from first `warmup` updates.
    warmup : int
        Number of initial observations used for in-control calibration
        when mu0/sigma0 are not provided.
    k : float
        CUSUM allowance parameter.
    h : float
        CUSUM alarm threshold.
    """

    def __init__(
        self,
        window: int,
        xi_min: float,
        xi_max: float,
        n_xi: int,
        height: float,
        mu0: Optional[float] = None,
        sigma0: Optional[float] = None,
        warmup: Optional[int] = None,
        k: float = 0.5,
        h: float = 5.0,
    ):
        if window <= 0:
            raise ValueError("window must be positive")
        if n_xi <= 1:
            raise ValueError("n_xi must be greater than 1")
        if (mu0 is None) != (sigma0 is None):
            raise ValueError("mu0 and sigma0 must be both provided or both omitted")

        self.window = int(window)
        self.xi_grid = np.linspace(xi_min, xi_max, int(n_xi))
        self.height = float(height)
        self.k = float(k)
        self.h = float(h)

        self._fixed_mu0 = None if mu0 is None else float(mu0)
        self._fixed_sigma0 = None if sigma0 is None else float(sigma0)

        if warmup is None:
            self.warmup = self.window
        else:
            self.warmup = int(warmup)
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")

        self._buffer: deque[float] = deque(maxlen=self.window)
        self._warmup_scores: list[float] = []
        self.n_obs: int = 0
        self.cusum_pos: float = 0.0
        self.cusum_neg: float = 0.0
        self.mu0: Optional[float] = None
        self.sigma0: Optional[float] = None
        self._calibrated: bool = False

        self.reset()

    @property
    def is_calibrated(self) -> bool:
        """True once warmup observations have been processed."""
        return self._calibrated

    def _make_output(
        self,
        score: float,
        alarm: bool,
    ) -> dict[str, float | bool | int]:
        return {
            "score": float(score),
            "cusum_pos": float(self.cusum_pos),
            "cusum_neg": float(self.cusum_neg),
            "alarm": bool(alarm),
            "n_obs": int(self.n_obs),
            "calibrated": bool(self._calibrated),
        }

    def _calibrate_if_ready(self) -> None:
        if self._calibrated:
            return
        if len(self._warmup_scores) < self.warmup:
            return
        warmup_arr = np.asarray(self._warmup_scores, dtype=np.float64)
        self.mu0 = float(np.mean(warmup_arr))
        if warmup_arr.size > 1:
            sigma0 = float(np.std(warmup_arr, ddof=1))
        else:
            sigma0 = 0.0
        self.sigma0 = sigma0 + 1e-12
        self._calibrated = True

    def update(self, r: float) -> dict[str, float | bool | int]:
        """
        Ingest one new return observation.

        Returns a dict with keys:
          - "score"      : float, current residue score (NaN during warmup)
          - "cusum_pos"  : float, current S+ statistic
          - "cusum_neg"  : float, current S- statistic
          - "alarm"      : bool, True if alarm fired this step
          - "n_obs"      : int, total observations ingested so far
          - "calibrated" : bool, True once warmup is complete

        During warmup (fewer than `window` observations in buffer, or fewer
        than `warmup` score observations for calibration), score is NaN and
        alarm is always False.
        """
        self.n_obs += 1
        self._buffer.append(float(r))

        if len(self._buffer) < self.window:
            return self._make_output(np.nan, alarm=False)

        buffer_arr = np.asarray(self._buffer, dtype=np.float64)
        ecf_vec = ecf_at(buffer_arr, self.xi_grid)
        score = float(
            ecf_residue_scores(
                ecf_vec[np.newaxis, :],
                self.xi_grid,
                self.height,
            )[0]
        )

        if not self._calibrated:
            self._warmup_scores.append(score)
            self._calibrate_if_ready()
            return self._make_output(np.nan, alarm=False)

        if self.mu0 is None or self.sigma0 is None:
            raise RuntimeError("Detector is calibrated but mu0/sigma0 is missing")

        slack = self.k * self.sigma0
        z = (score - self.mu0) / self.sigma0
        self.cusum_pos = max(0.0, self.cusum_pos + z - slack)
        self.cusum_neg = max(0.0, self.cusum_neg - z - slack)
        alarm = bool(self.cusum_pos > self.h or self.cusum_neg > self.h)
        if alarm:
            self.cusum_pos = 0.0
            self.cusum_neg = 0.0
        return self._make_output(score, alarm=alarm)

    def update_batch(self, returns: NDArray[np.float64]) -> list[dict[str, float | bool | int]]:
        """Convenience: call update() for each element, return list of dicts."""
        returns_arr = np.asarray(returns, dtype=np.float64)
        return [self.update(float(r)) for r in returns_arr]

    def reset(self) -> None:
        """Reset buffer, CUSUM, and calibration state."""
        self._buffer.clear()
        self._warmup_scores = []
        self.n_obs = 0
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0

        if self._fixed_mu0 is not None and self._fixed_sigma0 is not None:
            self.mu0 = self._fixed_mu0
            self.sigma0 = self._fixed_sigma0 + 1e-12
            self._calibrated = True
        else:
            self.mu0 = None
            self.sigma0 = None
            self._calibrated = False
