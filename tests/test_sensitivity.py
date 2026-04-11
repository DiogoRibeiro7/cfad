from __future__ import annotations

import numpy as np

from cfad.sensitivity import (
    height_sensitivity,
    recommend_params,
    threshold_sensitivity,
    window_sensitivity,
)


def test_window_sensitivity_shape():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, 400).astype(np.float64)
    windows = [30, 60, 90]

    df = window_sensitivity(returns, windows=windows)
    assert len(df) == len(windows)
    assert {"window", "n_windows", "metric_value"}.issubset(df.columns)


def test_window_sensitivity_monotone_windows():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.01, 420).astype(np.float64)

    df = window_sensitivity(returns, windows=[120, 45, 60, 30])
    window_values = df["window"].to_numpy(dtype=np.int64)
    assert np.all(np.diff(window_values) > 0)


def test_height_sensitivity_shape():
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0, 0.01, 380).astype(np.float64)
    heights = [0.05, 0.1, 0.2]

    df = height_sensitivity(returns, heights=heights, window=60)
    assert len(df) == len(heights)
    assert {"height", "mean_score", "score_std", "n_alarms"}.issubset(df.columns)


def test_threshold_sensitivity_alarm_rate_decreasing():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.01, 500).astype(np.float64)

    df = threshold_sensitivity(
        returns,
        h_values=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        window=60,
        step=5,
        calibration_frac=0.3,
    )
    alarm_rate = df.sort_values("h")["alarm_rate"].to_numpy(dtype=np.float64)
    assert np.all(np.diff(alarm_rate) <= 1e-12)


def test_recommend_params_keys():
    rng = np.random.default_rng(4)
    returns = rng.normal(0.0, 0.01, 450).astype(np.float64)

    out = recommend_params(returns, target_fpr=0.02, verbose=False)
    assert {"window", "height", "h", "rationale"}.issubset(out.keys())


def test_recommend_params_valid_ranges():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0, 0.01, 450).astype(np.float64)

    out = recommend_params(returns, target_fpr=0.03, verbose=False)
    assert 20 <= int(out["window"]) <= 200
    assert 0.0 < float(out["height"]) <= 1.0
    assert float(out["h"]) > 0.0
