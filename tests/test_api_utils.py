"""Tests for API, utils, and auxiliary coverage-critical code."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from cfad import compare_models, detect
from cfad.detection import AnomalyReport
from cfad.empirical_cf import ecf_covariance
from cfad.models.base import CFModel
from cfad.models.gaussian import GaussianCF
from cfad.models.nig import NIGCF
from cfad.residue_score import normalise_scores, rolling_pvalue, threshold_by_fpr
from cfad.utils import load_spy_sample, plot_scores, simulate_levy_returns


def test_api_detect_pandas_series():
    rng = np.random.default_rng(0)
    returns = pd.Series(
        rng.normal(0, 0.01, 120), index=pd.date_range("2021-01-01", periods=120)
    )
    report = detect(returns, window=40, step=10, h=3.0)
    assert report.dates is not None
    assert report.scores.shape[0] > 0
    assert hasattr(report, "alarm_indices")


def test_api_compare_models_returns_structure():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, 300)
    result = compare_models(returns)
    assert "gaussian" in result and "nig" in result and "winner" in result
    assert isinstance(result["gaussian"]["model"], GaussianCF)
    assert isinstance(result["nig"]["model"], NIGCF)
    assert result["winner"] in {"gaussian", "nig"}


def test_empirical_covariance_symmetry():
    rng = np.random.default_rng(2)
    returns = rng.normal(0, 0.01, 200)
    xi = np.linspace(-2, 2, 5)
    cov = ecf_covariance(returns, xi)
    assert cov.shape == (5, 5)
    assert np.allclose(cov, cov.T, atol=1e-12)


def test_cfmodel_pdf_and_aic():
    rng = np.random.default_rng(3)
    returns = rng.normal(0, 0.01, 250)
    model = GaussianCF(mu=0.0, sigma=0.01)
    x_grid = np.linspace(-0.1, 0.1, 50)
    pdf = model.pdf_from_cf(x_grid, xi_max=20.0, n_xi=512)
    assert pdf.shape == x_grid.shape
    assert np.all(pdf >= 0)
    aic = model.aic(returns)
    assert isinstance(aic, float)


def test_normalise_scores_invalid_method():
    with pytest.raises(ValueError):
        normalise_scores(np.array([1.0, 2.0]), method="unknown")


def test_rolling_pvalue_invalid_args():
    with pytest.raises(ValueError):
        rolling_pvalue(np.array([1.0]), window=0)
    with pytest.raises(ValueError):
        rolling_pvalue(np.array([1.0, 2.0, 3.0]), window=2, dist="invalid")


def test_threshold_by_fpr_errors():
    with pytest.raises(ValueError):
        threshold_by_fpr(np.array([1.0]), fpr=0.0)
    with pytest.raises(ValueError):
        threshold_by_fpr(np.array([]), fpr=0.05)


def test_plot_scores_with_returns_and_alarms():
    dates = pd.date_range("2021-01-01", periods=100)
    report = AnomalyReport(
        scores=np.linspace(0.0, 1.0, 10),
        cusum_pos=np.linspace(0.0, 1.0, 10),
        cusum_neg=np.zeros(10),
        alarm_indices=np.array([0], dtype=np.int64),
        window_end_indices=np.array([0], dtype=np.int64),
        dates=dates,
        mu0=0.0,
        sigma0=1.0,
        threshold=0.5,
    )
    returns = pd.Series(np.random.normal(0, 0.01, 100), index=dates)
    fig = plot_scores(report, returns=returns)
    assert fig is not None
    assert len(fig.axes) == 2


def test_plot_scores_invalid_axes():
    _, axs = plt.subplots(1, 1)
    with pytest.raises(ValueError):
        plot_scores(
            AnomalyReport(
                scores=np.zeros(5),
                cusum_pos=np.zeros(5),
                cusum_neg=np.zeros(5),
                alarm_indices=np.array([], dtype=np.int64),
                window_end_indices=np.array([], dtype=np.int64),
                dates=None,
                mu0=0.0,
                sigma0=1.0,
                threshold=1.0,
            ),
            returns=None,
            ax=(axs,),
        )


def test_load_spy_sample_cached():
    data_path = Path(__file__).resolve().parents[1] / "data" / "spy_2018_2022.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]}, index=pd.date_range("2020-01-01", periods=3)
    )
    df.to_csv(data_path)
    try:
        returns = load_spy_sample()
        assert len(returns) == 2
        assert pd.api.types.is_datetime64_any_dtype(returns.index.dtype)
    finally:
        data_path.unlink()


def test_simulate_levy_returns_invalid_n():
    with pytest.raises(ValueError):
        simulate_levy_returns(0)
