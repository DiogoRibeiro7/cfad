from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfad import WalkForwardBacktest
from cfad.detection import RollingDetector


def _make_backtest(n_folds: int = 4) -> WalkForwardBacktest:
    return WalkForwardBacktest(
        detector_kwargs={
            "window": 30,
            "xi_min": -10.0,
            "xi_max": 10.0,
            "n_xi": 64,
            "step": 5,
            "calibration_frac": 0.3,
            "k": 0.5,
            "h": 4.0,
        },
        n_folds=n_folds,
        train_frac=0.6,
        expanding=True,
    )


def test_walkforward_n_folds():
    returns = np.random.default_rng(0).normal(0.0, 0.01, 400).astype(np.float64)
    result = _make_backtest(n_folds=4).run(returns)
    assert len(result.fold_reports) == 4
    assert result.n_folds == 4


def test_walkforward_fold_boundaries_do_not_overlap():
    returns = np.random.default_rng(1).normal(0.0, 0.01, 400).astype(np.float64)
    result = _make_backtest(n_folds=4).run(returns)

    for train_start, train_end, test_start, test_end in result.fold_dates:
        assert train_start <= train_end
        assert test_start <= test_end
        assert train_end < test_start


def test_walkforward_uses_training_calibration_only():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0, 0.01, 420).astype(np.float64)
    wf = _make_backtest(n_folds=4)
    result = wf.run(returns)

    first_train_start, first_train_end, _, _ = result.fold_dates[0]
    train_values = returns[first_train_start : first_train_end + 1]
    detector = RollingDetector(**wf.detector_kwargs)
    train_scores, _ = detector.score_windows(train_values)
    n_cal = min(
        max(10, int(detector.calibration_frac * len(train_scores))),
        len(train_scores),
    )
    expected_mu0 = float(np.mean(train_scores[:n_cal]))
    expected_sigma0 = float(np.std(train_scores[:n_cal], ddof=1))

    first_report = result.fold_reports[0]
    assert first_report.mu0 == pytest.approx(expected_mu0)
    assert first_report.sigma0 == pytest.approx(max(expected_sigma0, 1e-12))


def test_walkforward_test_scores_match_direct_score_windows():
    rng = np.random.default_rng(12)
    returns = rng.normal(0.0, 0.01, 420).astype(np.float64)
    wf = _make_backtest(n_folds=4)
    result = wf.run(returns)

    _, _, test_start, test_end = result.fold_dates[0]
    test_values = returns[test_start : test_end + 1]
    detector = RollingDetector(**wf.detector_kwargs)
    expected_scores, expected_end_idx = detector.score_windows(test_values)

    first_report = result.fold_reports[0]
    assert np.allclose(first_report.scores, expected_scores)
    assert np.array_equal(first_report.window_end_indices, expected_end_idx)


def test_walkforward_accepts_legacy_height_config():
    returns = np.random.default_rng(13).normal(0.0, 0.01, 400).astype(np.float64)
    wf = WalkForwardBacktest(
        detector_kwargs={"window": 30, "height": 0.2, "step": 5},
        n_folds=4,
        train_frac=0.6,
    )
    result = wf.run(returns)
    assert len(result.fold_reports) == 4
    assert "height" not in wf.detector_kwargs


def test_walkforward_to_dataframe_length():
    returns = np.random.default_rng(2).normal(0.0, 0.01, 420).astype(np.float64)
    dates = pd.date_range("2020-01-01", periods=420, freq="B")
    result = _make_backtest(n_folds=4).run(returns, dates=dates)

    df = result.to_dataframe()
    expected_rows = sum(len(report.scores) for report in result.fold_reports)
    assert len(df) == expected_rows
    assert list(df.columns) == ["score", "cusum_pos", "cusum_neg", "alarm"]


def test_score_alarms_perfect_detector():
    returns = np.random.default_rng(3).normal(0.0, 0.01, 400).astype(np.float64)
    wf = _make_backtest(n_folds=4)
    result = wf.run(returns)
    known_break = 300
    result.aggregate_alarms = np.array([known_break], dtype=np.int64)

    metrics = wf.score_alarms(result, known_breaks=[known_break], tolerance_windows=0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_score_alarms_no_alarms():
    returns = np.random.default_rng(4).normal(0.0, 0.01, 400).astype(np.float64)
    wf = _make_backtest(n_folds=4)
    result = wf.run(returns)
    result.aggregate_alarms = np.array([], dtype=np.int64)

    metrics = wf.score_alarms(result, known_breaks=[300], tolerance_windows=5)
    assert metrics["recall"] == pytest.approx(0.0)
    assert np.isnan(metrics["precision"]) or metrics["precision"] == pytest.approx(0.0)
