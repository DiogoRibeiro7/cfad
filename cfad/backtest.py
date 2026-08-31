"""Walk-forward backtesting utilities for CFAD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cfad.detection import AnomalyReport, RollingDetector


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
        lines = [
            "CFAD Walk-Forward Backtest",
            f"  Folds            : {self.n_folds}",
            f"  Window size      : {self.window_size}",
            f"  Step             : {self.step}",
            f"  Aggregate windows: {self.aggregate_scores.size}",
            f"  Aggregate alarms : {self.aggregate_alarms.size}",
        ]
        if self.fold_dates:
            first = self.fold_dates[0]
            last = self.fold_dates[-1]
            lines.append(
                f"  First fold       : train[{first[0]} -> {first[1]}], "
                f"test[{first[2]} -> {first[3]}]"
            )
            lines.append(
                f"  Last fold        : train[{last[0]} -> {last[1]}], "
                f"test[{last[2]} -> {last[3]}]"
            )
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Return fold-concatenated score diagnostics as a DataFrame."""
        columns = ["score", "cusum_pos", "cusum_neg", "alarm"]
        frames: list[pd.DataFrame] = []

        for i, report in enumerate(self.fold_reports):
            n_scores = int(len(report.scores))
            if n_scores == 0:
                continue

            alarm_mask = np.zeros(n_scores, dtype=bool)
            valid_alarm_idx = report.alarm_indices[
                (report.alarm_indices >= 0) & (report.alarm_indices < n_scores)
            ]
            alarm_mask[valid_alarm_idx] = True

            if report.dates is not None and len(report.dates) > 0:
                local_idx = np.clip(
                    report.window_end_indices[:n_scores] - 1,
                    0,
                    len(report.dates) - 1,
                )
                index = report.dates[local_idx]
            else:
                test_start = int(self.fold_dates[i][2])
                index = test_start + report.window_end_indices[:n_scores] - 1

            frames.append(
                pd.DataFrame(
                    {
                        "score": np.asarray(report.scores, dtype=np.float64),
                        "cusum_pos": np.asarray(report.cusum_pos, dtype=np.float64),
                        "cusum_neg": np.asarray(report.cusum_neg, dtype=np.float64),
                        "alarm": alarm_mask,
                    },
                    index=index,
                )
            )

        if not frames:
            return pd.DataFrame(columns=columns)

        out = pd.concat(frames, axis=0)
        if out.index.has_duplicates:
            raise ValueError("Backtest folds produced overlapping test indices")
        out.index.name = "time"
        return out


class WalkForwardBacktest:
    """Walk-forward evaluation with train-only score calibration."""

    def __init__(
        self,
        detector_kwargs: dict,
        n_folds: int = 5,
        train_frac: float = 0.6,
        expanding: bool = True,
    ) -> None:
        if n_folds <= 0:
            raise ValueError("n_folds must be positive")
        if not (0.0 < train_frac < 1.0):
            raise ValueError("train_frac must be in (0, 1)")
        if not isinstance(detector_kwargs, dict):
            raise TypeError("detector_kwargs must be a dict")

        kwargs = dict(detector_kwargs)
        kwargs.pop("height", None)
        self.detector_kwargs = kwargs
        self.n_folds = int(n_folds)
        self.train_frac = float(train_frac)
        self.expanding = bool(expanding)

    def _split_folds(
        self,
        n_obs: int,
        window: int,
    ) -> list[tuple[int, int, int, int]]:
        """Create non-overlapping test folds as half-open index intervals."""
        initial_train = max(int(np.floor(self.train_frac * n_obs)), window)
        if initial_train >= n_obs:
            raise ValueError("Initial training fold leaves no samples for testing")

        remaining = n_obs - initial_train
        if remaining < self.n_folds:
            raise ValueError("Not enough samples in the test region for n_folds")

        base, extra = divmod(remaining, self.n_folds)
        train_size_fixed = initial_train
        folds: list[tuple[int, int, int, int]] = []
        test_start = initial_train

        for i in range(self.n_folds):
            fold_len = base + (1 if i < extra else 0)
            test_end = test_start + fold_len
            train_end = test_start
            train_start = (
                0 if self.expanding else max(0, train_end - train_size_fixed)
            )

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
        """Execute a leakage-free walk-forward evaluation.

        Each fold estimates ``mu0`` and ``sigma0`` only from training-window
        scores. Test-window ECF scores are produced by ``score_windows`` and the
        frozen training calibration is then applied via ``apply_calibration``.
        No test-fold statistic is used to fit the sequential decision rule.
        """
        values = np.asarray(returns, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("returns must be one-dimensional")
        if not np.all(np.isfinite(values)):
            raise ValueError("returns must contain only finite values")

        n_obs = int(values.size)
        window = int(self.detector_kwargs.get("window", 60))
        step = int(self.detector_kwargs.get("step", 1))
        if n_obs <= window:
            raise ValueError("returns length must exceed detector window")

        dates_idx = None if dates is None else pd.DatetimeIndex(dates)
        if dates_idx is not None and len(dates_idx) != n_obs:
            raise ValueError("dates length must match returns length")

        folds = self._split_folds(n_obs=n_obs, window=window)
        fold_reports: list[AnomalyReport] = []
        fold_dates: list[tuple] = []
        aggregate_scores: list[NDArray[np.float64]] = []
        aggregate_alarms: list[int] = []

        for train_start, train_end, test_start, test_end in folds:
            train_returns = values[train_start:train_end]
            test_returns = values[test_start:test_end]
            test_dates = None if dates_idx is None else dates_idx[test_start:test_end]

            train_detector = RollingDetector(**self.detector_kwargs)
            train_scores, _ = train_detector.score_windows(train_returns)
            n_cal = max(10, int(train_detector.calibration_frac * len(train_scores)))
            n_cal = min(n_cal, len(train_scores))
            mu0 = float(np.mean(train_scores[:n_cal]))
            sigma0_raw = (
                float(np.std(train_scores[:n_cal], ddof=1)) if n_cal > 1 else 0.0
            )
            sigma0 = self._sanitize_sigma(sigma0_raw)

            test_detector = RollingDetector(**self.detector_kwargs)
            test_scores, end_idx = test_detector.score_windows(test_returns)
            positive, negative, alarms = test_detector.apply_calibration(
                test_scores,
                mu0,
                sigma0,
            )

            report = AnomalyReport(
                scores=test_scores,
                cusum_pos=positive,
                cusum_neg=negative,
                alarm_indices=alarms,
                window_end_indices=end_idx,
                dates=test_dates,
                mu0=mu0,
                sigma0=sigma0,
                threshold=test_detector.h,
            )
            fold_reports.append(report)
            aggregate_scores.append(report.scores)

            valid_alarm_idx = alarms[(alarms >= 0) & (alarms < len(end_idx))]
            if valid_alarm_idx.size:
                global_alarm_idx = test_start + end_idx[valid_alarm_idx] - 1
                aggregate_alarms.extend(global_alarm_idx.astype(int).tolist())

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
        return BacktestResult(
            fold_reports=fold_reports,
            fold_dates=fold_dates,
            aggregate_scores=aggregate_scores_arr,
            aggregate_alarms=np.asarray(aggregate_alarms, dtype=np.int64),
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
        """Score alarms against known break dates or integer indices."""
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
                nearest = (
                    pos - 1 if abs(target - left) <= abs(right - target) else pos
                )
            break_indices.append(nearest)

        if not break_indices:
            return {
                "hits": 0,
                "misses": 0,
                "false_alarms": int(alarms.size),
                "precision": np.nan if alarms.size == 0 else 0.0,
                "recall": np.nan,
                "f1": np.nan,
            }

        breaks = np.asarray(break_indices, dtype=np.int64)
        hits = int(
            sum(np.any(np.abs(alarms - br) <= tolerance_windows) for br in breaks)
        )
        misses = int(len(breaks) - hits)

        if alarms.size == 0:
            precision = np.nan
            false_alarms = 0
        else:
            true_alarm_mask = np.asarray(
                [
                    np.any(np.abs(breaks - alarm) <= tolerance_windows)
                    for alarm in alarms
                ],
                dtype=bool,
            )
            true_alarm_count = int(np.sum(true_alarm_mask))
            false_alarms = int(alarms.size - true_alarm_count)
            precision = true_alarm_count / float(alarms.size)

        recall = hits / float(len(breaks))
        if np.isnan(precision) or (precision + recall) == 0.0:
            f1 = np.nan if np.isnan(precision) else 0.0
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
