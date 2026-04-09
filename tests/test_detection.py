"""Integration tests for the detection pipeline."""
import numpy as np
import pytest
from cfad import detect, RollingDetector


def make_returns_with_jump(n=500, jump_at=350, jump_size=0.15):
    rng = np.random.default_rng(99)
    returns = rng.normal(0, 0.01, n)
    returns[jump_at:jump_at+5] += jump_size
    return returns


def test_detect_returns_report():
    returns = np.random.default_rng(0).normal(0, 0.01, 300)
    report = detect(returns, window=60, step=5)
    assert len(report.scores) > 0
    assert report.mu0 >= 0


def test_detect_finds_jump():
    returns = make_returns_with_jump()
    report = detect(returns, window=60, step=1, h=3.0, calibration_frac=0.5)
    # At least one alarm should fire near the jump
    assert len(report.alarm_indices) > 0


def test_report_summary_string():
    returns = np.random.default_rng(1).normal(0, 0.01, 200)
    report = detect(returns, window=50, step=10)
    s = report.summary()
    assert "CFAD" in s
    assert "Alarms" in s
