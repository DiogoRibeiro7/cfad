"""Invariant tests for the frozen v3 score benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "empirical_ecf_v3.py"
SPEC = importlib.util.spec_from_file_location("empirical_ecf_v3", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
v3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v3)


def test_auc_is_half_for_identical_score_sets() -> None:
    values = np.array([0.1, 0.2, 0.3, 0.4])
    assert v3.auc_from_scores(values, values) == 0.5


def test_auc_is_one_for_perfect_separation() -> None:
    negative = np.array([0.0, 0.1, 0.2])
    positive = np.array([1.0, 1.1, 1.2])
    assert v3.auc_from_scores(negative, positive) == 1.0


def test_ks_zero_for_identical_samples() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert v3.ks_statistic(values, values) == 0.0


def test_standardization_removes_location_and_scale() -> None:
    base = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    shifted_scaled = 7.0 + 3.5 * base
    np.testing.assert_allclose(v3.standardize(base), v3.standardize(shifted_scaled))


def test_frozen_windows_are_exactly_five_non_overlapping_blocks() -> None:
    protocol = v3.load_protocol()
    values = np.linspace(-0.02, 0.02, 600)
    rows = v3.score_windows(values, protocol)
    empirical = [
        row
        for row in rows
        if row["method"] == "standardized_empirical_reference"
    ]
    bounds = [(row["start"], row["end"]) for row in empirical]
    assert bounds == [(300, 360), (360, 420), (420, 480), (480, 540), (540, 600)]
