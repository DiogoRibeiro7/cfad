"""Tests for publication-quality plotting helpers in cfad.viz."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure
from pathlib import Path
import shutil
import uuid

from cfad.detection import AnomalyReport
from cfad.viz import (
    plot_cf_families,
    plot_contour_anatomy,
    plot_detection_timeline,
    plot_score_distribution,
)


def _make_report(n_scores: int = 30) -> AnomalyReport:
    return AnomalyReport(
        scores=np.linspace(0.0, 2.0, n_scores),
        cusum_pos=np.linspace(0.0, 5.0, n_scores),
        cusum_neg=np.linspace(0.0, 2.0, n_scores),
        alarm_indices=np.array([5, 12, 20], dtype=np.int64),
        window_end_indices=np.arange(40, 40 + n_scores, dtype=np.int64),
        dates=None,
        mu0=0.9,
        sigma0=0.15,
        threshold=1.25,
    )


def test_plot_cf_families_returns_figure():
    fig = plot_cf_families()
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3


def test_plot_contour_anatomy_gaussian():
    fig = plot_contour_anatomy("gaussian")
    assert isinstance(fig, Figure)


def test_plot_contour_anatomy_nig():
    fig = plot_contour_anatomy("nig")
    assert isinstance(fig, Figure)


def test_plot_score_distribution_returns_figure():
    report = _make_report()
    fig = plot_score_distribution(report)
    assert isinstance(fig, Figure)


def test_plot_detection_timeline_four_panels():
    report = _make_report()
    returns = np.random.default_rng(0).normal(0.0, 0.01, 100).astype(np.float64)
    fig = plot_detection_timeline(report, returns=returns)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4


def test_plot_detection_timeline_with_events():
    report = _make_report()
    _ = plot_detection_timeline(report, returns=None, events={"test event": 50})


def test_savepath_creates_file():
    temp_dir = Path.cwd() / ".tmp_viz_tests" / f"cfad_viz_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        out = temp_dir / "test_cf.png"
        _ = plot_cf_families(savepath=str(out))
        assert out.exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
