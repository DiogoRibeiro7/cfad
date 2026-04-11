from __future__ import annotations

import numpy as np
import pandas as pd

from cfad.market import (
    correlation_break_score,
    detect_portfolio,
    summarise_portfolio_alarms,
)


def _make_df(n: int = 320, d: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, size=(n, d)).astype(np.float64)
    cols = [f"TK{i+1}" for i in range(d)]
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(data, index=idx, columns=cols)


def test_detect_portfolio_univariate_keys():
    df = _make_df(n=320, d=3, seed=1)
    out = detect_portfolio(
        df,
        mode="univariate",
        detector_kwargs={"window": 60, "step": 5, "h": 4.0, "n_xi": 64},
    )
    assert set(out.keys()) == set(df.columns)


def test_detect_portfolio_joint_single_key():
    df = _make_df(n=320, d=3, seed=2)
    out = detect_portfolio(
        df,
        mode="joint",
        detector_kwargs={
            "window": 60,
            "step": 5,
            "h": 4.0,
            "m_directions": 32,
            "seed": 42,
        },
    )
    assert set(out.keys()) == {"joint"}


def test_correlation_break_score_shape():
    df = _make_df(n=300, d=3, seed=3)
    series = correlation_break_score(df, window=60, step=5)
    expected = (300 - 60) // 5 + 1
    assert len(series) == expected


def test_summarise_portfolio_alarms_columns():
    df = _make_df(n=320, d=3, seed=4)
    reports = detect_portfolio(
        df,
        mode="univariate",
        detector_kwargs={"window": 60, "step": 5, "h": 4.0, "n_xi": 64},
    )
    summary = summarise_portfolio_alarms(reports)

    required = {
        "ticker",
        "n_alarms",
        "first_alarm_date",
        "last_alarm_date",
        "mean_score",
        "max_score",
    }
    assert required.issubset(set(summary.columns))


def test_summarise_sorted_by_alarms():
    df = _make_df(n=360, d=3, seed=5)
    reports = detect_portfolio(
        df,
        mode="univariate",
        detector_kwargs={"window": 60, "step": 5, "h": 3.5, "n_xi": 64},
    )
    summary = summarise_portfolio_alarms(reports)
    alarms = summary["n_alarms"].to_numpy(dtype=np.int64)
    assert np.all(np.diff(alarms) <= 0)
