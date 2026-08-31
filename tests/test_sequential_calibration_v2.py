"""Tests for the frozen v2 sequential-calibration benchmark helpers."""

import numpy as np

from benchmarks.sequential_calibration_v2 import (
    ScoredSeries,
    classify_changed_alarm,
    empirical_threshold,
    first_page_cusum_alarm,
    max_page_cusum_path,
)


def test_max_page_cusum_path_has_no_alarm_reset():
    z = np.array([1.0, 1.0, -2.0], dtype=np.float64)
    assert max_page_cusum_path(z, k=0.5) == 1.5


def test_first_alarm_ignores_score_calibration_block():
    scored = ScoredSeries(
        z=np.array([100.0, 100.0, 100.0, 2.0, 0.0], dtype=np.float64),
        end_indices=np.array([60, 65, 70, 75, 80], dtype=np.int64),
        n_calibration=3,
    )
    assert first_page_cusum_alarm(scored, k=0.5, h=1.0) == (3, 75)


def test_change_classification_respects_half_open_endpoints():
    assert classify_changed_alarm(None, change_point=360, horizon=120) == (
        "none",
        None,
    )
    assert classify_changed_alarm(360, change_point=360, horizon=120) == (
        "pre_change_false_alarm",
        None,
    )
    assert classify_changed_alarm(365, change_point=360, horizon=120) == (
        "detection",
        5,
    )
    assert classify_changed_alarm(485, change_point=360, horizon=120) == (
        "late_alarm",
        None,
    )


def test_empirical_threshold_uses_higher_order_statistic():
    maxima = np.arange(20, dtype=np.float64)
    assert empirical_threshold(maxima, 0.95) == 19.0
