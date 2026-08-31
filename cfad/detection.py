"""Anomaly detection layer.

Combines rolling empirical characteristic-function (ECF) estimation, a
real-frequency Gaussian-shape discrepancy score, and sequential CUSUM detection
into a unified pipeline.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from cfad.contour import gaussian_ecf_distance_scores
from cfad.empirical_cf import ecf_at, rolling_ecf

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
        """Return dates corresponding to alarm windows when dates are available."""
        if self.dates is None:
            return None
        valid = self.alarm_indices[self.alarm_indices < len(self.window_end_indices)]
        win_idx = self.window_end_indices[valid]
        valid_win = win_idx[win_idx < len(self.dates)]
        return self.dates[valid_win]

    def summary(self) -> str:
        """Return a compact human-readable report summary."""
        n_alarms = len(self.alarm_indices)
        lines = [
            "CFAD Anomaly Report",
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
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Pure-Python two-sided Page-CUSUM fallback.

    The recursion operates on standardized scores.  Consequently ``k`` is a
    dimensionless reference value and must not be multiplied by ``sigma0``.
    """
    if sigma0 <= 0.0:
        raise ValueError("sigma0 must be positive")
    if k < 0.0:
        raise ValueError("k must be non-negative")
    if h <= 0.0:
        raise ValueError("h must be positive")

    n_scores = len(scores)
    s_pos = 0.0
    s_neg = 0.0
    positive = np.zeros(n_scores, dtype=np.float64)
    negative = np.zeros(n_scores, dtype=np.float64)
    alarms: list[int] = []

    for t, score in enumerate(np.asarray(scores, dtype=np.float64)):
        z = (float(score) - mu0) / sigma0
        s_pos = max(0.0, s_pos + z - k)
        s_neg = max(0.0, s_neg - z - k)
        positive[t] = s_pos
        negative[t] = s_neg
        if s_pos > h or s_neg > h:
            alarms.append(t)
            s_pos = 0.0
            s_neg = 0.0

    return positive, negative, np.asarray(alarms, dtype=np.int64)


def _rolling_moments(
    returns: NDArray[np.float64],
    window: int,
    step: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute sample mean and standard deviation for every detector window."""
    values = np.asarray(returns, dtype=np.float64)
    n_windows = (values.size - window) // step + 1
    means = np.empty(n_windows, dtype=np.float64)
    stds = np.empty(n_windows, dtype=np.float64)

    for idx in range(n_windows):
        start = idx * step
        sample = values[start : start + window]
        means[idx] = float(np.mean(sample))
        stds[idx] = float(np.std(sample, ddof=1)) if window > 1 else 0.0

    return means, stds


class RollingDetector:
    """Rolling ECF shape detector followed by a two-sided Page-CUSUM.

    Parameters
    ----------
    window : int
        Number of returns per ECF estimation window.
    xi_min, xi_max : float
        Real-frequency grid bounds for ECF evaluation.
    n_xi : int
        Number of frequency grid points.
    step : int
        Rolling step (1 = fully overlapping, ``window`` = non-overlapping).
    calibration_frac : float
        Fraction of score windows used to estimate the in-control score mean and
        standard deviation.
    k : float
        Dimensionless Page-CUSUM reference value on standardized scores.
    h : float
        CUSUM alarm threshold.
    """

    def __init__(
        self,
        window: int = 60,
        xi_min: float = -10.0,
        xi_max: float = 10.0,
        n_xi: int = 128,
        step: int = 1,
        calibration_frac: float = 0.3,
        k: float = 0.5,
        h: float = 5.0,
    ) -> None:
        if window <= 1:
            raise ValueError("window must be greater than 1")
        if xi_max <= xi_min:
            raise ValueError("xi_max must be greater than xi_min")
        if n_xi < 4:
            raise ValueError("n_xi must be at least 4")
        if step <= 0:
            raise ValueError("step must be positive")
        if not (0.0 < calibration_frac < 1.0):
            raise ValueError("calibration_frac must lie in (0, 1)")
        if k < 0.0:
            raise ValueError("k must be non-negative")
        if h <= 0.0:
            raise ValueError("h must be positive")

        self.window = int(window)
        self.xi_grid = np.linspace(xi_min, xi_max, int(n_xi), dtype=np.float64)
        self.step = int(step)
        self.calibration_frac = float(calibration_frac)
        self.k = float(k)
        self.h = float(h)
        self._mu0: Optional[float] = None
        self._sigma0: Optional[float] = None

    def fit_transform(
        self,
        returns: NDArray[np.float64],
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> AnomalyReport:
        """Run rolling ECF scoring, calibration, and sequential detection."""
        values = np.asarray(returns, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("returns must be one-dimensional")
        if values.size < self.window:
            raise ValueError("returns must contain at least one full window")
        if not np.all(np.isfinite(values)):
            raise ValueError("returns must contain only finite values")

        ecf_mat, end_idx = rolling_ecf(values, self.xi_grid, self.window, self.step)
        means, stds = _rolling_moments(values, self.window, self.step)
        scores = gaussian_ecf_distance_scores(ecf_mat, self.xi_grid, means, stds)

        n_cal = max(10, int(self.calibration_frac * len(scores)))
        n_cal = min(n_cal, len(scores))
        self._mu0 = float(np.mean(scores[:n_cal]))
        sigma0 = float(np.std(scores[:n_cal], ddof=1)) if n_cal > 1 else 0.0
        self._sigma0 = max(sigma0, 1e-12)

        cusum_fn = _cusum_c if _HAS_C_EXT else _cusum_python
        sp, sn, alarms = cusum_fn(
            scores,
            self._mu0,
            self._sigma0,
            self.k,
            self.h,
        )

        return AnomalyReport(
            scores=np.asarray(scores, dtype=np.float64),
            cusum_pos=np.asarray(sp, dtype=np.float64),
            cusum_neg=np.asarray(sn, dtype=np.float64),
            alarm_indices=np.asarray(alarms, dtype=np.int64),
            window_end_indices=np.asarray(end_idx, dtype=np.int64),
            dates=dates,
            mu0=self._mu0,
            sigma0=self._sigma0,
            threshold=self.h,
        )


class StreamDetector:
    """Online detector using the same ECF-shape score as ``RollingDetector``."""

    def __init__(
        self,
        window: int,
        xi_min: float,
        xi_max: float,
        n_xi: int,
        mu0: Optional[float] = None,
        sigma0: Optional[float] = None,
        warmup: Optional[int] = None,
        k: float = 0.5,
        h: float = 5.0,
    ) -> None:
        if window <= 1:
            raise ValueError("window must be greater than 1")
        if xi_max <= xi_min:
            raise ValueError("xi_max must be greater than xi_min")
        if n_xi < 4:
            raise ValueError("n_xi must be at least 4")
        if (mu0 is None) != (sigma0 is None):
            raise ValueError("mu0 and sigma0 must be both provided or both omitted")
        if sigma0 is not None and sigma0 <= 0.0:
            raise ValueError("sigma0 must be positive")
        if k < 0.0:
            raise ValueError("k must be non-negative")
        if h <= 0.0:
            raise ValueError("h must be positive")

        self.window = int(window)
        self.xi_grid = np.linspace(xi_min, xi_max, int(n_xi), dtype=np.float64)
        self.k = float(k)
        self.h = float(h)

        self._fixed_mu0 = None if mu0 is None else float(mu0)
        self._fixed_sigma0 = None if sigma0 is None else float(sigma0)
        self.warmup = self.window if warmup is None else int(warmup)
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")

        self._buffer: deque[float] = deque(maxlen=self.window)
        self._warmup_scores: list[float] = []
        self.n_obs = 0
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.mu0: Optional[float] = None
        self.sigma0: Optional[float] = None
        self._calibrated = False
        self.reset()

    @property
    def is_calibrated(self) -> bool:
        """Return whether score calibration is complete."""
        return self._calibrated

    def _make_output(
        self,
        score: float,
        alarm: bool,
    ) -> dict[str, float | bool | int]:
        """Create the public result object for one streamed observation."""
        return {
            "score": float(score),
            "cusum_pos": float(self.cusum_pos),
            "cusum_neg": float(self.cusum_neg),
            "alarm": bool(alarm),
            "n_obs": int(self.n_obs),
            "calibrated": bool(self._calibrated),
        }

    def _calibrate_if_ready(self) -> None:
        """Estimate in-control score moments once enough score windows exist."""
        if self._calibrated or len(self._warmup_scores) < self.warmup:
            return
        warmup_arr = np.asarray(self._warmup_scores, dtype=np.float64)
        self.mu0 = float(np.mean(warmup_arr))
        sigma0 = float(np.std(warmup_arr, ddof=1)) if warmup_arr.size > 1 else 0.0
        self.sigma0 = max(sigma0, 1e-12)
        self._calibrated = True

    def update(self, r: float) -> dict[str, float | bool | int]:
        """Ingest one return observation and update the sequential detector."""
        if not np.isfinite(r):
            raise ValueError("streamed returns must be finite")

        self.n_obs += 1
        self._buffer.append(float(r))
        if len(self._buffer) < self.window:
            return self._make_output(np.nan, alarm=False)

        sample = np.asarray(self._buffer, dtype=np.float64)
        ecf_vec = ecf_at(sample, self.xi_grid)
        mean = np.asarray([float(np.mean(sample))], dtype=np.float64)
        std = np.asarray([float(np.std(sample, ddof=1))], dtype=np.float64)
        score = float(
            gaussian_ecf_distance_scores(
                ecf_vec[np.newaxis, :],
                self.xi_grid,
                mean,
                std,
            )[0]
        )

        if not self._calibrated:
            self._warmup_scores.append(score)
            self._calibrate_if_ready()
            return self._make_output(np.nan, alarm=False)

        if self.mu0 is None or self.sigma0 is None:
            raise RuntimeError("detector is calibrated but mu0/sigma0 is missing")

        z = (score - self.mu0) / self.sigma0
        self.cusum_pos = max(0.0, self.cusum_pos + z - self.k)
        self.cusum_neg = max(0.0, self.cusum_neg - z - self.k)
        alarm = bool(self.cusum_pos > self.h or self.cusum_neg > self.h)
        if alarm:
            self.cusum_pos = 0.0
            self.cusum_neg = 0.0
        return self._make_output(score, alarm=alarm)

    def update_batch(
        self,
        returns: NDArray[np.float64],
    ) -> list[dict[str, float | bool | int]]:
        """Process a one-dimensional array through :meth:`update`."""
        values = np.asarray(returns, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("returns must be one-dimensional")
        return [self.update(float(value)) for value in values]

    def reset(self) -> None:
        """Reset buffer, CUSUM state, and score calibration."""
        self._buffer.clear()
        self._warmup_scores = []
        self.n_obs = 0
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0

        if self._fixed_mu0 is not None and self._fixed_sigma0 is not None:
            self.mu0 = self._fixed_mu0
            self.sigma0 = self._fixed_sigma0
            self._calibrated = True
        else:
            self.mu0 = None
            self.sigma0 = None
            self._calibrated = False
