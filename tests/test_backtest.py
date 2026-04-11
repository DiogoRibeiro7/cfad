from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfad import WalkForwardBacktest


def _make_backtest(n_folds: int = 4) -> WalkForwardBacktest:
    return WalkForwardBacktest(
        detector_kwargs={
            "window": 30,
            "xi_min": -10.0,
            "xi_max": 10.0,
            "n_xi": 64,
            "height": 0.2,
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
    wf = _make_backtest(n_folds=4)
    result = wf.run(returns)
    assert len(result.fold_reports) == 4
    assert result.n_folds == 4


def test_walkforward_no_lookahead():
    returns = np.random.default_rng(1).normal(0.0, 0.01, 400).astype(np.float64)
    wf = _make_backtest(n_folds=4)
    result = wf.run(returns)

    for train_start, train_end, test_start, test_end in result.fold_dates:
        assert train_start < train_end
        assert test_start < test_end
        assert train_end < test_start


def test_walkforward_to_dataframe_length():
    returns = np.random.default_rng(2).normal(0.0, 0.01, 420).astype(np.float64)
    dates = pd.date_range("2020-01-01", periods=420, freq="B")
    wf = _make_backtest(n_folds=4)
    result = wf.run(returns, dates=dates)

    df = result.to_dataframe()
    expected_rows = int(sum(len(report.scores) for report in result.fold_reports))
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
