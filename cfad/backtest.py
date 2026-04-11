"""Walk-forward backtesting utilities for CFAD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cfad.detection import AnomalyReport, RollingDetector, _cusum_python


@dataclass
class BacktestResult:
    """Container for walk-forward backtest output."""

    fold_reports: list[AnomalyReport]
    fold_dates: list[tuple]
    aggregate_scores: NDArray[np.float64]
    aggregate_alarms: NDArray[np.int64]
    n_folds: int
    window_size: int
    step: int
    _global_dates: Optional[pd.DatetimeIndex] = field(default=None, repr=False)

    def summary(self) -> str:
        """Return a concise textual summary of backtest outcomes."""
        n_windows = int(self.aggregate_scores.size)
        n_alarms = int(self.aggregate_alarms.size)
        lines = [
            "CFAD Walk-Forward Backtest",
            f"  Folds            : {self.n_folds}",
            f"  Window size      : {self.window_size}",
            f"  Step             : {self.step}",
            f"  Aggregate windows: {n_windows}",
            f"  Aggregate alarms : {n_alarms}",
        ]
        if self.fold_dates:
            first = self.fold_dates[0]
            last = self.fold_dates[-1]
            lines.append(f"  First fold       : train[{first[0]} -> {first[1]}], test[{first[2]} -> {first[3]}]")
            lines.append(f"  Last fold        : train[{last[0]} -> {last[1]}], test[{last[2]} -> {last[3]}]")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return fold-concatenated score diagnostics as a DataFrame.

        Returns
        -------
        dataframe : pd.DataFrame
            DataFrame indexed by test-period date/index with columns:
            ``score``, ``cusum_pos``, ``cusum_neg``, ``alarm``.
        """
        columns = ["score", "cusum_pos", "cusum_neg", "alarm"]
        if len(self.fold_reports) == 0:
            return pd.DataFrame(columns=columns)

        frames: list[pd.DataFrame] = []
        for i, report in enumerate(self.fold_reports):
            n = int(len(report.scores))
            if n == 0:
                continue

            alarm_mask = np.zeros(n, dtype=bool)
            valid_alarm_idx = report.alarm_indices[
                (report.alarm_indices >= 0) & (report.alarm_indices < n)
            ]
            alarm_mask[valid_alarm_idx] = True

            if report.dates is not None and len(report.dates) > 0:
                local_idx = np.clip(report.window_end_indices[:n] - 1, 0, len(report.dates) - 1)
                index = report.dates[local_idx]
            else:
                test_start = int(self.fold_dates[i][2])
                index = test_start + report.window_end_indices[:n] - 1

            frame = pd.DataFrame(
                {
                    "score": np.asarray(report.scores, dtype=np.float64),
                    "cusum_pos": np.asarray(report.cusum_pos, dtype=np.float64),
                    "cusum_neg": np.asarray(report.cusum_neg, dtype=np.float64),
                    "alarm": alarm_mask,
                },
                index=index,
            )
            frames.append(frame)

        if len(frames) == 0:
            return pd.DataFrame(columns=columns)

        out = pd.concat(frames, axis=0)
        if out.index.has_duplicates:
            raise ValueError("Backtest folds produced overlapping test indices")
        out.index.name = "time"
        return out


class WalkForwardBacktest:
    """
    Walk-forward evaluation of the cfad detector.

    Splits the return series into expanding or rolling training folds,
    calibrates the detector on each training fold, and evaluates on the
    subsequent test fold while avoiding look-ahead bias.

    Parameters
    ----------
    detector_kwargs : dict
        Keyword arguments passed to ``RollingDetector``.
    n_folds : int, default=5
        Number of test folds.
    train_frac : float, default=0.6
        Fraction of data used as initial training fold.
    expanding : bool, default=True
        If ``True`` use expanding train folds, otherwise rolling train folds.
    """

    def __init__(
        self,
        detector_kwargs: dict,
        n_folds: int = 5,
        train_frac: float = 0.6,
        expanding: bool = True,
    ):
        if n_folds <= 0:
            raise ValueError("n_folds must be positive")
        if not (0.0 < train_frac < 1.0):
            raise ValueError("train_frac must be in (0, 1)")
        if not isinstance(detector_kwargs, dict):
            raise TypeError("detector_kwargs must be a dict")

        self.detector_kwargs = dict(detector_kwargs)
        self.n_folds = int(n_folds)
        self.train_frac = float(train_frac)
        self.expanding = bool(expanding)

    def _split_folds(self, n_obs: int, window: int) -> list[tuple[int, int, int, int]]:
        """Create fold boundaries as half-open index intervals."""
        initial_train = int(np.floor(self.train_frac * n_obs))
        initial_train = max(initial_train, window)

        if initial_train >= n_obs:
            raise ValueError("Initial training fold leaves no samples for testing")

        remaining = n_obs - initial_train
        if remaining < self.n_folds:
            raise ValueError("Not enough samples in the test region for n_folds")

        base = remaining // self.n_folds
        extra = remaining % self.n_folds

        train_size_fixed = initial_train
        folds: list[tuple[int, int, int, int]] = []
        test_start = initial_train
        for i in range(self.n_folds):
            fold_len = base + (1 if i < extra else 0)
            test_end = test_start + fold_len

            if self.expanding:
                train_start = 0
                train_end = test_start
            else:
                train_end = test_start
                train_start = max(0, train_end - train_size_fixed)

            if train_end - train_start < window:
                raise ValueError("Training fold is shorter than detector window")
            if test_end - test_start < window:
                raise ValueError("Test fold is shorter than detector window")

            folds.append((train_start, train_end, test_start, test_end))
            test_start = test_end

        return folds

    @staticmethod
    def _sanitize_sigma(value: float) -> float:
        """Ensure a strictly positive finite calibration scale."""
        if not np.isfinite(value) or value <= 0.0:
            return 1e-12
        return float(value)

    def run(
        self,
        returns: NDArray[np.float64],
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> BacktestResult:
        """
        Execute the walk-forward backtest.

        For each fold:
          1. Slice training and test periods
          2. Instantiate ``RollingDetector(**self.detector_kwargs)``
          3. Call ``fit_transform(train_returns)`` to calibrate ``mu0``, ``sigma0``
          4. Create a new ``RollingDetector`` for test scoring and apply train
             calibration to CUSUM without recalibration on test
          5. Call ``fit_transform(test_returns)`` and collect ``AnomalyReport``
        """
        returns_arr = np.asarray(returns, dtype=np.float64)
        if returns_arr.ndim != 1:
            raise ValueError("returns must be one-dimensional")

        n_obs = int(returns_arr.size)
        window = int(self.detector_kwargs.get("window", 60))
        step = int(self.detector_kwargs.get("step", 1))

        if n_obs <= window:
            raise ValueError("returns length must exceed detector window")

        if dates is not None:
            dates_idx = pd.DatetimeIndex(dates)
            if len(dates_idx) != n_obs:
                raise ValueError("dates length must match returns length")
        else:
            dates_idx = None

        folds = self._split_folds(n_obs=n_obs, window=window)

        fold_reports: list[AnomalyReport] = []
        fold_dates: list[tuple] = []
        aggregate_scores: list[NDArray[np.float64]] = []
        aggregate_alarms: list[np.int64] = []

        for train_start, train_end, test_start, test_end in folds:
            train_returns = returns_arr[train_start:train_end]
            test_returns = returns_arr[test_start:test_end]

            train_dates = None if dates_idx is None else dates_idx[train_start:train_end]
            test_dates = None if dates_idx is None else dates_idx[test_start:test_end]

            train_detector = RollingDetector(**self.detector_kwargs)
            train_report = train_detector.fit_transform(train_returns, dates=train_dates)
            mu0 = float(train_report.mu0)
            sigma0 = self._sanitize_sigma(float(train_report.sigma0))

            test_kwargs = dict(self.detector_kwargs)
            test_kwargs["calibration_frac"] = 0.0
            test_detector = RollingDetector(**test_kwargs)

            # Keep fit_transform for score generation, then replace CUSUM with
            # train-calibrated parameters to avoid look-ahead on the test fold.
            raw_test_report = test_detector.fit_transform(test_returns, dates=test_dates)
            sp, sn, alarms = _cusum_python(
                np.asarray(raw_test_report.scores, dtype=np.float64),
                mu0,
                sigma0,
                test_detector.k,
                test_detector.h,
            )

            test_report = AnomalyReport(
                scores=np.asarray(raw_test_report.scores, dtype=np.float64),
                cusum_pos=np.asarray(sp, dtype=np.float64),
                cusum_neg=np.asarray(sn, dtype=np.float64),
                alarm_indices=np.asarray(alarms, dtype=np.int64),
                window_end_indices=np.asarray(raw_test_report.window_end_indices, dtype=np.int64),
                dates=test_dates,
                mu0=mu0,
                sigma0=sigma0,
                threshold=test_detector.h,
            )
            fold_reports.append(test_report)
            aggregate_scores.append(test_report.scores)

            valid_alarm_idx = test_report.alarm_indices[
                (test_report.alarm_indices >= 0)
                & (test_report.alarm_indices < len(test_report.window_end_indices))
            ]
            if valid_alarm_idx.size > 0:
                alarm_global = (
                    test_start
                    + test_report.window_end_indices[valid_alarm_idx]
                    - 1
                ).astype(np.int64)
                aggregate_alarms.extend(alarm_global.tolist())

            if dates_idx is None:
                fold_dates.append(
                    (train_start, train_end - 1, test_start, test_end - 1)
                )
            else:
                fold_dates.append(
                    (
                        dates_idx[train_start],
                        dates_idx[train_end - 1],
                        dates_idx[test_start],
                        dates_idx[test_end - 1],
                    )
                )

        aggregate_scores_arr = (
            np.concatenate(aggregate_scores).astype(np.float64)
            if aggregate_scores
            else np.zeros(0, dtype=np.float64)
        )
        aggregate_alarms_arr = np.asarray(aggregate_alarms, dtype=np.int64)

        return BacktestResult(
            fold_reports=fold_reports,
            fold_dates=fold_dates,
            aggregate_scores=aggregate_scores_arr,
            aggregate_alarms=aggregate_alarms_arr,
            n_folds=self.n_folds,
            window_size=window,
            step=step,
            _global_dates=dates_idx,
        )

    def score_alarms(
        self,
        result: BacktestResult,
        known_breaks: list,
        tolerance_windows: int = 10,
    ) -> dict[str, float | int]:
        """
        Score alarms against known structural break dates/indices.

        Parameters
        ----------
        result : BacktestResult
            Walk-forward result produced by :meth:`run`.
        known_breaks : list
            Known break points as integer indices or datetimes.
        tolerance_windows : int, default=10
            Maximum distance (in window units) for counting an alarm as a hit.

        Returns
        -------
        metrics : dict
            Dictionary with keys ``hits``, ``misses``, ``false_alarms``,
            ``precision``, ``recall``, and ``f1``.
        """
        if tolerance_windows < 0:
            raise ValueError("tolerance_windows must be non-negative")

        alarms = np.asarray(result.aggregate_alarms, dtype=np.int64)

        break_indices: list[int] = []
        for br in known_breaks:
            if isinstance(br, (int, np.integer)):
                break_indices.append(int(br))
                continue

            if result._global_dates is None:
                continue

            try:
                ts = pd.Timestamp(br)
            except Exception:
                continue

            date_values = result._global_dates.view("int64")
            target = int(ts.value)
            pos = int(np.searchsorted(date_values, target))
            if pos <= 0:
                nearest = 0
            elif pos >= len(date_values):
                nearest = len(date_values) - 1
            else:
                left = date_values[pos - 1]
                right = date_values[pos]
                nearest = pos - 1 if abs(target - left) <= abs(right - target) else pos
            break_indices.append(nearest)

        if len(break_indices) == 0:
            false_alarms = int(alarms.size)
            precision = np.nan if alarms.size == 0 else 0.0
            return {
                "hits": 0,
                "misses": 0,
                "false_alarms": false_alarms,
                "precision": precision,
                "recall": np.nan,
                "f1": np.nan,
            }

        breaks_arr = np.asarray(break_indices, dtype=np.int64)

        hits = int(
            np.sum([
                np.any(np.abs(alarms - br) <= tolerance_windows)
                for br in breaks_arr
            ])
        )
        misses = int(len(breaks_arr) - hits)

        if alarms.size == 0:
            true_alarm_count = 0
            false_alarms = 0
            precision = np.nan
        else:
            true_alarm_mask = np.asarray(
                [np.any(np.abs(breaks_arr - a) <= tolerance_windows) for a in alarms],
                dtype=bool,
            )
            true_alarm_count = int(np.sum(true_alarm_mask))
            false_alarms = int(alarms.size - true_alarm_count)
            precision = true_alarm_count / float(alarms.size)

        recall = hits / float(len(breaks_arr))

        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0.0:
            f1 = np.nan if np.isnan(precision) or np.isnan(recall) else 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

        return {
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "precision": float(precision) if not np.isnan(precision) else np.nan,
            "recall": float(recall),
            "f1": float(f1) if not np.isnan(f1) else np.nan,
        }


__all__ = ["BacktestResult", "WalkForwardBacktest"]
