from __future__ import annotations

import numpy as np

from cfad import detect
from cfad.bootstrap import (
    block_bootstrap_resample,
    bootstrap_scores,
    score_stability,
)


def test_block_bootstrap_length():
    returns = np.arange(200, dtype=np.float64)
    resampled = block_bootstrap_resample(returns, block_size=15, seed=0)
    assert resampled.shape == returns.shape


def test_block_bootstrap_values_from_input():
    returns = np.linspace(-1.0, 1.0, 101, dtype=np.float64)
    resampled = block_bootstrap_resample(returns, block_size=8, seed=1)
    assert np.all(np.isin(resampled, returns))


def test_bootstrap_scores_keys():
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0, 0.01, 320).astype(np.float64)

    result = bootstrap_scores(
        returns,
        window=60,
        n_bootstrap=10,
        n_xi=64,
        step=5,
        seed=3,
        n_jobs=1,
    )
    assert {"mean", "lower", "upper", "std"}.issubset(set(result.keys()))


def test_bootstrap_scores_shape():
    rng = np.random.default_rng(4)
    returns = rng.normal(0.0, 0.01, 340).astype(np.float64)

    report = detect(returns, window=60, step=5, n_xi=64)
    result = bootstrap_scores(
        returns,
        window=60,
        n_bootstrap=12,
        n_xi=64,
        step=5,
        seed=5,
        n_jobs=1,
    )

    n_ref = len(report.scores)
    n_mean = len(result["mean"])
    n_lower = len(result["lower"])
    n_upper = len(result["upper"])

    assert abs(n_mean - n_ref) <= 2
    assert abs(n_lower - n_ref) <= 2
    assert abs(n_upper - n_ref) <= 2


def test_bootstrap_lower_le_upper():
    rng = np.random.default_rng(6)
    returns = rng.normal(0.0, 0.01, 300).astype(np.float64)
    result = bootstrap_scores(
        returns,
        window=50,
        n_bootstrap=8,
        n_xi=64,
        step=5,
        seed=7,
        n_jobs=1,
    )

    assert np.all(result["lower"] <= result["upper"])


def test_score_stability_returns_stable_flag():
    rng = np.random.default_rng(8)
    returns = rng.normal(0.0, 0.01, 500).astype(np.float64)

    out = score_stability(
        returns,
        window=60,
        n_subsamples=40,
        subsample_frac=0.8,
        seed=9,
    )

    assert isinstance(out["stable"], bool)
    assert out["stable"] is True
