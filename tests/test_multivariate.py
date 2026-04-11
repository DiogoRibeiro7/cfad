from __future__ import annotations

import numpy as np

from cfad.multivariate import MultivariateDetector, joint_ecf, random_directions


def test_joint_ecf_normalization():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, size=(120, 4)).astype(np.float64)
    xi_directions = np.zeros((1, 4), dtype=np.float64)
    phi = joint_ecf(returns, xi_directions)
    assert phi.shape == (1,)
    assert abs(phi[0] - 1.0) < 1e-12


def test_random_directions_unit_norm():
    dirs = random_directions(d=5, m=256, xi_max=5.0, seed=123)
    norms = np.linalg.norm(dirs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_multivariate_detector_shape():
    rng = np.random.default_rng(10)
    returns = rng.normal(0.0, 0.01, size=(300, 3)).astype(np.float64)

    detector = MultivariateDetector(
        window=60,
        m_directions=64,
        xi_max=5.0,
        step=5,
        calibration_frac=0.3,
        h=4.0,
        seed=42,
    )
    report = detector.fit_transform(returns)

    expected_n_windows = (300 - 60) // 5 + 1
    assert report.scores.shape[0] == expected_n_windows


def test_multivariate_detects_correlation_break():
    rng = np.random.default_rng(2026)
    n_pre = 240
    n_post = 120

    cov_pre = np.eye(3)
    cov_post = np.full((3, 3), 0.9, dtype=np.float64)
    np.fill_diagonal(cov_post, 1.0)

    pre = rng.multivariate_normal(np.zeros(3), cov_pre, size=n_pre) * 0.01
    post = rng.multivariate_normal(np.zeros(3), cov_post, size=n_post) * 0.01
    returns = np.vstack([pre, post]).astype(np.float64)

    detector = MultivariateDetector(
        window=60,
        m_directions=96,
        xi_max=6.0,
        step=1,
        calibration_frac=0.4,
        k=0.3,
        h=2.5,
        seed=7,
    )
    report = detector.fit_transform(returns)

    break_index = n_pre
    in_post_break = []
    for alarm_w in report.alarm_indices:
        if alarm_w < 0 or alarm_w >= len(report.window_end_indices):
            continue
        alarm_global = int(report.window_end_indices[int(alarm_w)] - 1)
        in_post_break.append(alarm_global >= break_index)

    assert any(in_post_break)
