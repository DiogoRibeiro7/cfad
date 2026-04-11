"""Tests for ECF-based goodness-of-fit utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t

from cfad.gof import aic_table, cf_distance, epps_pulley_test
from cfad.models.gaussian import GaussianCF
from cfad.models.nig import NIGCF


def test_cf_distance_gaussian_self():
    rng = np.random.default_rng(123)
    returns = rng.normal(0.0, 0.01, 800).astype(np.float64)
    model = GaussianCF().fit(returns)
    d_l2 = cf_distance(returns, model, xi_max=10.0, n_xi=256, metric="l2")
    assert d_l2 < 0.05


def test_cf_distance_gaussian_vs_nig():
    rng = np.random.default_rng(456)
    returns = t.rvs(df=3, loc=0.0, scale=0.01, size=900, random_state=rng).astype(
        np.float64
    )
    g = GaussianCF().fit(returns)
    n = NIGCF().fit(returns)
    d_g = cf_distance(returns, g, xi_max=10.0, n_xi=256, metric="l2")
    d_n = cf_distance(returns, n, xi_max=10.0, n_xi=256, metric="l2")
    assert d_n < d_g


def test_aic_table_shape():
    rng = np.random.default_rng(999)
    returns = rng.normal(0.0, 0.01, 300).astype(np.float64)
    table = aic_table(returns)
    required = {"model", "is_analytic", "n_params", "aic", "ecf_l2", "winner"}
    assert isinstance(table, pd.DataFrame)
    assert table.shape[0] == 4
    assert required.issubset(set(table.columns))


def test_aic_table_winner():
    rng = np.random.default_rng(321)
    returns = rng.normal(0.0, 0.01, 700).astype(np.float64)
    table = aic_table(returns)
    winner_row = table.loc[table["winner"]].iloc[0]
    assert winner_row["model"] == "GaussianCF"


def test_epps_pulley_reject_nonnormal():
    rng = np.random.default_rng(42)
    returns = t.rvs(df=2, loc=0.0, scale=0.01, size=500, random_state=rng).astype(
        np.float64
    )
    model = GaussianCF().fit(returns)
    out = epps_pulley_test(returns, model, xi_max=3.0, n_xi=50, B=199)
    assert out["pvalue"] < 0.05
    assert out["reject_5pct"] is True
