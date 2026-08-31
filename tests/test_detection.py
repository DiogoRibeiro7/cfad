"""Integration tests for the detection pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from cfad import RollingDetector, StreamDetector, detect
from cfad.detection import _cusum_python
from cfad.residue_score import normalise_scores, rolling_pvalue, threshold_by_fpr


def make_returns_with_shape_shift(
    n: int = 500,
    shift_at: int = 350,
) -> np.ndarray:
    """Return Gaussian data followed by a standardized heavy-tailed regime."""
    rng = np.random.default_rng(99)
    left = rng.normal(0.0, 0.01, shift_at)
    right = rng.standard_t(df=3.0, size=n - shift_at)
    right = right / float(np.std(right, ddof=1)) * 0.01
    return np.concatenate([left, right]).astype(np.float64)


def test_detect_returns_report() -> None:
    returns = np.random.default_rng(0).normal(0.0, 0.01, 300)
    report = detect(returns, window=60, step=5)
    assert len(report.scores) > 0
    assert report.mu0 >= 0.0
    assert report.sigma0 > 0.0


def test_detect_finds_distributional_shape_shift() -> None:
    returns = make_returns_with_shape_shift()
    report = detect(returns, window=60, step=1, h=3.0, calibration_frac=0.5)
    assert len(report.alarm_indices) > 0


def test_report_summary_string() -> None:
    returns = np.random.default_rng(1).normal(0.0, 0.01, 200)
    report = detect(returns, window=50, step=10)
    summary = report.summary()
    assert "CFAD" in summary
    assert "Alarms" in summary


def test_cusum_uses_dimensionless_reference_value() -> None:
    """Standardized z=1 with k=.5 must add .5 per step, independent of sigma0."""
    mu0 = 10.0
    sigma0 = 0.01
    scores = np.asarray([mu0 + sigma0, mu0 + sigma0], dtype=np.float64)

    positive, negative, alarms = _cusum_python(
        scores,
        mu0=mu0,
        sigma0=sigma0,
        k=0.5,
        h=10.0,
    )

    assert np.allclose(positive, [0.5, 1.0])
    assert np.allclose(negative, 0.0)
    assert alarms.size == 0


def test_normalise_zscore_mean_zero() -> None:
    rng = np.random.default_rng(321)
    scores = rng.normal(loc=2.0, scale=0.7, size=2048).astype(np.float64)
    zscores = normalise_scores(scores, method="zscore")
    assert float(np.mean(zscores)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.std(zscores)) == pytest.approx(1.0, rel=1e-6)


def test_normalise_mad_robust() -> None:
    scores = np.linspace(-1.0, 1.0, 101, dtype=np.float64)
    scores[-1] = 50.0
    mad_scaled = normalise_scores(scores, method="mad")
    assert float(np.median(np.abs(mad_scaled[:-1]))) < 1.0
    assert float(mad_scaled[-1]) > 20.0


def test_rolling_pvalue_shape() -> None:
    scores = np.linspace(-2.0, 2.0, 20, dtype=np.float64)
    window = 6
    pvalues = rolling_pvalue(scores, window=window, dist="empirical")
    assert pvalues.shape == scores.shape
    assert np.all(np.isnan(pvalues[:window]))
    assert np.all((pvalues[window:] >= 0.0) & (pvalues[window:] <= 1.0))


def test_threshold_by_fpr() -> None:
    rng = np.random.default_rng(12345)
    calibration_scores = rng.normal(0.0, 1.0, 10_000).astype(np.float64)
    scores = rng.normal(0.0, 1.0, 10_000).astype(np.float64)
    threshold = threshold_by_fpr(scores, calibration_scores, fpr=0.05)
    realised_fpr = float(np.mean(calibration_scores > threshold))
    assert realised_fpr == pytest.approx(0.05, abs=0.01)


def test_stream_vs_batch_with_fixed_calibration() -> None:
    rng = np.random.default_rng(2026)
    returns = rng.normal(0.0, 0.01, 400).astype(np.float64)

    batch_detector = RollingDetector(
        window=60,
        xi_min=-10.0,
        xi_max=10.0,
        n_xi=128,
        step=1,
        calibration_frac=0.3,
        k=0.5,
        h=5.0,
    )
    report = batch_detector.fit_transform(returns)

    stream = StreamDetector(
        window=60,
        xi_min=-10.0,
        xi_max=10.0,
        n_xi=128,
        mu0=report.mu0,
        sigma0=report.sigma0,
        warmup=0,
        k=0.5,
        h=5.0,
    )
    out = stream.update_batch(returns)

    stream_scores = np.asarray([entry["score"] for entry in out], dtype=np.float64)
    assert np.all(np.isnan(stream_scores[:59]))
    assert np.allclose(
        stream_scores[59:],
        report.scores,
        atol=1e-12,
        rtol=1e-10,
    )


def test_stream_alarm_fires_after_shape_shift() -> None:
    returns = make_returns_with_shape_shift(n=450, shift_at=300)
    calibration = RollingDetector(
        window=60,
        xi_min=-10.0,
        xi_max=10.0,
        n_xi=128,
        step=1,
        calibration_frac=0.5,
        k=0.5,
        h=4.0,
    ).fit_transform(returns[:300])

    stream = StreamDetector(
        window=60,
        xi_min=-10.0,
        xi_max=10.0,
        n_xi=128,
        mu0=calibration.mu0,
        sigma0=calibration.sigma0,
        warmup=0,
        k=0.5,
        h=4.0,
    )
    out = stream.update_batch(returns)
    alarms_after_shift = [
        bool(entry["alarm"])
        for idx, entry in enumerate(out)
        if idx >= 300
    ]
    assert any(alarms_after_shift)


def test_stream_reset() -> None:
    stream = StreamDetector(
        window=40,
        xi_min=-10.0,
        xi_max=10.0,
        n_xi=64,
        mu0=None,
        sigma0=None,
        warmup=20,
        k=0.5,
        h=4.0,
    )
    _ = stream.update_batch(np.linspace(-0.01, 0.01, 120, dtype=np.float64))
    stream.reset()

    assert stream.n_obs == 0
    assert stream.is_calibrated is False
    assert stream.cusum_pos == 0.0


def test_stream_update_returns_dict_keys() -> None:
    stream = StreamDetector(
        window=20,
        xi_min=-10.0,
        xi_max=10.0,
        n_xi=32,
        mu0=0.0,
        sigma0=1.0,
        warmup=0,
        k=0.5,
        h=4.0,
    )
    out = stream.update(0.001)
    assert set(out.keys()) == {
        "score",
        "cusum_pos",
        "cusum_neg",
        "alarm",
        "n_obs",
        "calibrated",
    }
