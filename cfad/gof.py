"""
Goodness-of-fit utilities based on empirical characteristic functions (ECF).
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cfad.empirical_cf import ecf_at
from cfad.models import CGMYCF, CFModel, GaussianCF, LevyStableCF, NIGCF


def _ecf_statistic(
    returns: NDArray[np.float64],
    model: CFModel,
    xi_max: float,
    n_xi: int,
    weighted: bool,
) -> float:
    """Internal ECF discrepancy statistic."""
    returns_arr = np.asarray(returns, dtype=np.float64)
    xi = np.linspace(-xi_max, xi_max, n_xi, dtype=np.float64)
    phi_hat = ecf_at(returns_arr, xi)
    phi_theta = model.cf(xi)
    diff_sq = np.abs(phi_hat - phi_theta) ** 2
    if weighted:
        weight = np.exp(-(xi**2))
        integrand = diff_sq * weight
    else:
        integrand = diff_sq
    return float(np.trapezoid(integrand, xi))


def _build_inverse_cdf_sampler(
    model: CFModel,
    returns: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build inverse-CDF lookup arrays for generic model sampling."""
    returns_arr = np.asarray(returns, dtype=np.float64)
    std = float(np.std(returns_arr, ddof=1))
    std = max(std, 1e-4)
    x_max = 12.0 * std
    x_grid = np.linspace(-x_max, x_max, 4096, dtype=np.float64)
    pdf = np.asarray(model.pdf_from_cf(x_grid, xi_max=60.0, n_xi=4096), dtype=np.float64)
    pdf = np.clip(pdf, 0.0, None)
    total = float(np.trapezoid(pdf, x_grid))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Unable to build sampling distribution from model CF")
    pdf = pdf / total
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    return x_grid, cdf


def _sample_from_model(
    model: CFModel,
    n: int,
    rng: np.random.Generator,
    returns_for_fallback: NDArray[np.float64],
    fallback_cache: Optional[tuple[NDArray[np.float64], NDArray[np.float64]]] = None,
) -> tuple[NDArray[np.float64], Optional[tuple[NDArray[np.float64], NDArray[np.float64]]]]:
    """Draw synthetic returns from a fitted CFModel."""
    if isinstance(model, GaussianCF):
        sample = rng.normal(model.mu, model.sigma, n)
        return np.asarray(sample, dtype=np.float64), fallback_cache

    if isinstance(model, NIGCF):
        from scipy.stats import norminvgauss

        sample = norminvgauss.rvs(
            a=model.alpha,
            b=model.beta,
            loc=model.mu,
            scale=model.delta,
            size=n,
            random_state=rng,
        )
        return np.asarray(sample, dtype=np.float64), fallback_cache

    if isinstance(model, LevyStableCF):
        from scipy.stats import levy_stable

        sample = levy_stable.rvs(
            alpha=model.alpha,
            beta=model.beta,
            loc=model.mu,
            scale=model.c,
            size=n,
            random_state=rng,
        )
        return np.asarray(sample, dtype=np.float64), fallback_cache

    if fallback_cache is None:
        fallback_cache = _build_inverse_cdf_sampler(model, returns_for_fallback)

    x_grid, cdf = fallback_cache
    u = rng.random(n)
    sample = np.interp(u, cdf, x_grid)
    return np.asarray(sample, dtype=np.float64), fallback_cache


def epps_pulley_test(
    returns: NDArray[np.float64],
    model: CFModel,
    xi_max: float = 3.0,
    n_xi: int = 50,
    B: int = 999,
) -> dict[str, float | int | str | bool]:
    """
    Epps-Pulley (1983) ECF-based goodness-of-fit test.

    Test statistic:
      T_n = n * integral_{-xi_max}^{xi_max} |phi_hat_n(xi) - phi_theta(xi)|^2 w(xi) dxi

    where w(xi) = exp(-xi^2) (Gaussian weight, standard in ECF tests).
    phi_theta is the fitted model's CF.

    Under H0 (data ~ model), T_n is asymptotically chi-squared.
    Use a simulation-based p-value: simulate B samples of size n from
    the fitted model, compute T_n for each, p-value = fraction >= observed T_n.

    Parameters
    ----------
    returns : float ndarray of shape (n,)
    model : CFModel
        Fitted CFModel instance.
    xi_max : float, default=3.0
        Frequency cutoff.
    n_xi : int, default=50
        Number of grid points.
    B : int, default=999
        Number of simulation replicates for p-value estimation.

    Returns
    -------
    result : dict
        Dictionary with keys:
        "statistic", "pvalue", "n", "model", "reject_5pct".
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
    if returns_arr.ndim != 1 or returns_arr.size < 2:
        raise ValueError("returns must be a one-dimensional array with at least 2 values")
    if xi_max <= 0:
        raise ValueError("xi_max must be positive")
    if n_xi < 4:
        raise ValueError("n_xi must be at least 4")
    if B < 1:
        raise ValueError("B must be at least 1")

    n = int(returns_arr.size)
    xi = np.linspace(-xi_max, xi_max, n_xi, dtype=np.float64)
    weight = np.exp(-(xi**2))

    # Affine normalization mirrors standard ECF testing practice and avoids
    # scale-driven degeneracy when testing heavy-tailed data against Gaussian.
    center = float(np.mean(returns_arr))
    scale = float(np.std(returns_arr, ddof=1)) + 1e-12

    def statistic(sample: NDArray[np.float64]) -> float:
        sample_arr = np.asarray(sample, dtype=np.float64)
        y = (sample_arr - center) / scale
        phi_hat = ecf_at(y, xi)
        phi_theta = np.exp(-1j * xi * center / scale) * model.cf(xi / scale)
        integrand = np.abs(phi_hat - phi_theta) ** 2 * weight
        return float(sample_arr.size * np.trapezoid(integrand, xi))

    observed = statistic(returns_arr)

    rng = np.random.default_rng(0)
    sim_stats = np.zeros(B, dtype=np.float64)
    fallback_cache: Optional[tuple[NDArray[np.float64], NDArray[np.float64]]] = None
    for b in range(B):
        sim_returns, fallback_cache = _sample_from_model(
            model=model,
            n=n,
            rng=rng,
            returns_for_fallback=returns_arr,
            fallback_cache=fallback_cache,
        )
        sim_stats[b] = statistic(sim_returns)

    pvalue = float(np.mean(sim_stats >= observed))
    return {
        "statistic": observed,
        "pvalue": pvalue,
        "n": n,
        "model": repr(model),
        "reject_5pct": bool(pvalue < 0.05),
    }


def cf_distance(
    returns: NDArray[np.float64],
    model: CFModel,
    xi_max: float = 10.0,
    n_xi: int = 256,
    metric: Literal["l2", "l1", "sup"] = "l2",
) -> float:
    """
    Distance between empirical CF and parametric CF.

    L2: sqrt( integral |phi_hat - phi_theta|^2 dxi )
    L1: integral |phi_hat - phi_theta| dxi
    Sup: max |phi_hat(xi) - phi_theta(xi)|

    All integrals via numpy.trapezoid on uniform xi grid.
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
    if returns_arr.ndim != 1 or returns_arr.size < 2:
        raise ValueError("returns must be a one-dimensional array with at least 2 values")
    if xi_max <= 0:
        raise ValueError("xi_max must be positive")
    if n_xi < 4:
        raise ValueError("n_xi must be at least 4")

    xi = np.linspace(-xi_max, xi_max, n_xi, dtype=np.float64)
    phi_hat = ecf_at(returns_arr, xi)
    phi_theta = model.cf(xi)
    diff = np.abs(phi_hat - phi_theta)

    if metric == "l2":
        return float(np.sqrt(np.trapezoid(diff**2, xi)))
    if metric == "l1":
        return float(np.trapezoid(diff, xi))
    if metric == "sup":
        return float(np.max(diff))
    raise ValueError("metric must be one of {'l2', 'l1', 'sup'}")


def aic_table(
    returns: NDArray[np.float64],
    models: Optional[list[CFModel]] = None,
) -> pd.DataFrame:
    """
    Fit all provided CF models and return an AIC comparison table.

    Default models if None: [GaussianCF(), NIGCF(), CGMYCF(), LevyStableCF()]

    Returns pd.DataFrame with columns:
      model, is_analytic, n_params, aic, ecf_l2, winner (bool)
    Sorted by aic ascending.
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
    if returns_arr.ndim != 1 or returns_arr.size < 2:
        raise ValueError("returns must be a one-dimensional array with at least 2 values")

    if models is None:
        fitted_models: list[CFModel] = [GaussianCF(), NIGCF(), CGMYCF(), LevyStableCF()]
    else:
        if len(models) == 0:
            raise ValueError("models must not be empty")
        fitted_models = models

    rows: list[dict[str, object]] = []
    for model in fitted_models:
        fitted = model.fit(returns_arr)
        rows.append(
            {
                "model": type(fitted).__name__,
                "is_analytic": bool(getattr(fitted, "is_analytic", False)),
                "n_params": int(len(fitted.__dict__)),
                "aic": float(fitted.aic(returns_arr)),
                "ecf_l2": float(
                    cf_distance(
                        returns_arr,
                        fitted,
                        xi_max=10.0,
                        n_xi=256,
                        metric="l2",
                    )
                ),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("aic", ascending=True, kind="mergesort").reset_index(drop=True)
    df["winner"] = False
    if len(df) > 0:
        df.loc[0, "winner"] = True
    return df[["model", "is_analytic", "n_params", "aic", "ecf_l2", "winner"]]


def rolling_gof(
    returns: NDArray[np.float64],
    model_class,
    window: int = 120,
    step: int = 5,
    xi_max: float = 8.0,
    n_xi: int = 64,
) -> NDArray[np.float64]:
    """
    Rolling window goodness-of-fit distance (L2) between empirical CF
    and a freshly fitted parametric model on each window.

    Used to track how well the null model fits over time — a sustained
    increase in distance signals model inadequacy (structural break).

    Returns float ndarray of shape (n_windows,).
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
    if returns_arr.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if window <= 1:
        raise ValueError("window must be greater than 1")
    if step <= 0:
        raise ValueError("step must be positive")
    if returns_arr.size < window:
        return np.zeros(0, dtype=np.float64)

    n_windows = (returns_arr.size - window) // step + 1
    out = np.zeros(n_windows, dtype=np.float64)
    for w in range(n_windows):
        start = w * step
        end = start + window
        sample = returns_arr[start:end]
        model = model_class()
        fitted = model.fit(sample)
        out[w] = cf_distance(sample, fitted, xi_max=xi_max, n_xi=n_xi, metric="l2")
    return out
