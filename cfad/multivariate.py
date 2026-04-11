"""Multivariate characteristic-function anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cfad.detection import AnomalyReport, _cusum_python


def joint_ecf(
    returns: NDArray[np.float64],
    xi_directions: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """
    Estimate the joint ECF along a set of directions.

    Parameters
    ----------
    returns : float ndarray of shape (n, d)
        Observed multivariate returns.
    xi_directions : float ndarray of shape (m, d)
        Direction vectors where the joint ECF is evaluated.

    Returns
    -------
    ecf : complex ndarray of shape (m,)
        ``phi_n(xi_k) = (1/n) sum_j exp(i * xi_k^T r_j)``.
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
    xi_arr = np.asarray(xi_directions, dtype=np.float64)

    if returns_arr.ndim != 2:
        raise ValueError("returns must be a two-dimensional array with shape (n, d)")
    if xi_arr.ndim != 2:
        raise ValueError("xi_directions must be a two-dimensional array with shape (m, d)")
    if returns_arr.shape[1] != xi_arr.shape[1]:
        raise ValueError("returns and xi_directions must have the same feature dimension")
    if returns_arr.shape[0] == 0:
        raise ValueError("returns must contain at least one observation")

    projections = returns_arr @ xi_arr.T
    return np.mean(np.exp(1j * projections), axis=0)


def random_directions(
    d: int,
    m: int,
    xi_max: float = 5.0,
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Sample unit directions on ``S^{d-1}`` via normalized Gaussian vectors.

    Parameters
    ----------
    d : int
        Feature dimension.
    m : int
        Number of direction vectors.
    xi_max : float, default=5.0
        Frequency magnitude used downstream by the detector.
    seed : int or None, default=None
        Random seed.

    Returns
    -------
    directions : float ndarray of shape (m, d)
        Unit-norm random directions.
    """
    if d <= 0:
        raise ValueError("d must be positive")
    if m <= 0:
        raise ValueError("m must be positive")
    if xi_max <= 0.0:
        raise ValueError("xi_max must be positive")

    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(int(m), int(d))).astype(np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def joint_residue_score(
    returns: NDArray[np.float64],
    m_directions: int = 64,
    xi_max: float = 5.0,
    seed: int = 0,
) -> float:
    """
    Single-window joint residue score for multivariate returns.

    Parameters
    ----------
    returns : float ndarray of shape (n, d)
        Windowed multivariate return matrix.
    m_directions : int, default=64
        Number of Monte Carlo projection directions.
    xi_max : float, default=5.0
        Frequency magnitude multiplier for random directions.
    seed : int, default=0
        Random seed controlling direction sampling.

    Returns
    -------
    score : float
        Weighted average magnitude of joint ECF values.
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
    if returns_arr.ndim != 2:
        raise ValueError("returns must be two-dimensional with shape (n, d)")
    n_obs, d = returns_arr.shape
    if n_obs < 2:
        raise ValueError("returns must contain at least two observations")

    dirs = random_directions(d=d, m=m_directions, xi_max=xi_max, seed=seed)
    xi = float(xi_max) * dirs

    phi_hat = joint_ecf(returns_arr, xi)

    mu = np.mean(returns_arr, axis=0)
    sigma = np.cov(returns_arr, rowvar=False, ddof=1)
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.ndim == 0:
        sigma = np.array([[float(sigma)]], dtype=np.float64)
    sigma = np.atleast_2d(sigma)
    sigma = sigma + 1e-12 * np.eye(d)

    mean_term = xi @ mu
    quad_term = np.einsum("md,dd,md->m", xi, sigma, xi)
    phi_gaussian = np.exp(1j * mean_term - 0.5 * quad_term)

    weights = np.clip(1.0 - np.abs(phi_gaussian), 0.0, 1.0)
    weighted_residual = np.abs(phi_hat - phi_gaussian) * weights
    return float(np.mean(weighted_residual))


class MultivariateDetector:
    """
    Rolling joint-CF anomaly detector for multivariate return series.

    Parameters
    ----------
    window : int, default=60
        Rolling window size.
    m_directions : int, default=64
        Number of random projection directions per window.
    xi_max : float, default=5.0
        Frequency magnitude multiplier.
    step : int, default=1
        Rolling step.
    calibration_frac : float, default=0.3
        Fraction of score windows for in-control calibration.
    k : float, default=0.5
        CUSUM allowance parameter.
    h : float, default=5.0
        CUSUM alarm threshold.
    seed : int, default=0
        Base random seed for direction sampling.
    """

    def __init__(
        self,
        window: int = 60,
        m_directions: int = 64,
        xi_max: float = 5.0,
        step: int = 1,
        calibration_frac: float = 0.3,
        k: float = 0.5,
        h: float = 5.0,
        seed: int = 0,
    ):
        if window <= 1:
            raise ValueError("window must be greater than 1")
        if m_directions <= 0:
            raise ValueError("m_directions must be positive")
        if xi_max <= 0:
            raise ValueError("xi_max must be positive")
        if step <= 0:
            raise ValueError("step must be positive")
        if not (0.0 <= calibration_frac <= 1.0):
            raise ValueError("calibration_frac must be in [0, 1]")

        self.window = int(window)
        self.m_directions = int(m_directions)
        self.xi_max = float(xi_max)
        self.step = int(step)
        self.calibration_frac = float(calibration_frac)
        self.k = float(k)
        self.h = float(h)
        self.seed = int(seed)

    def fit_transform(
        self,
        returns: NDArray[np.float64],
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> AnomalyReport:
        """
        Run rolling multivariate detection and return an ``AnomalyReport``.

        Parameters
        ----------
        returns : float ndarray of shape (T, d)
            Multivariate return matrix.
        dates : pd.DatetimeIndex or None, default=None
            Optional date index aligned with ``returns`` rows.

        Returns
        -------
        report : AnomalyReport
            Detector output containing scores, CUSUM paths, and alarms.
        """
        returns_arr = np.asarray(returns, dtype=np.float64)
        if returns_arr.ndim != 2:
            raise ValueError("returns must be two-dimensional with shape (T, d)")

        T, _ = returns_arr.shape
        if T < self.window:
            raise ValueError("returns length must be at least window")

        if dates is not None:
            dates_idx = pd.DatetimeIndex(dates)
            if len(dates_idx) != T:
                raise ValueError("dates length must match returns length")
        else:
            dates_idx = None

        n_windows = (T - self.window) // self.step + 1
        scores = np.zeros(n_windows, dtype=np.float64)
        end_idx = np.zeros(n_windows, dtype=np.int64)

        for w in range(n_windows):
            start = w * self.step
            end = start + self.window
            window_returns = returns_arr[start:end]
            scores[w] = joint_residue_score(
                window_returns,
                m_directions=self.m_directions,
                xi_max=self.xi_max,
                seed=self.seed + w,
            )
            end_idx[w] = end

        if n_windows == 0:
            raise ValueError("No rolling windows were generated")

        n_cal = int(np.floor(self.calibration_frac * n_windows))
        n_cal = max(1, min(n_windows, n_cal))

        mu0 = float(np.mean(scores[:n_cal]))
        if n_cal > 1:
            sigma0 = float(np.std(scores[:n_cal], ddof=1)) + 1e-12
        else:
            sigma0 = float(np.std(scores[:n_cal], ddof=0)) + 1e-12

        sp, sn, alarms = _cusum_python(scores, mu0, sigma0, self.k, self.h)

        return AnomalyReport(
            scores=scores,
            cusum_pos=np.asarray(sp, dtype=np.float64),
            cusum_neg=np.asarray(sn, dtype=np.float64),
            alarm_indices=np.asarray(alarms, dtype=np.int64),
            window_end_indices=end_idx,
            dates=dates_idx,
            mu0=mu0,
            sigma0=sigma0,
            threshold=self.h,
        )


__all__ = [
    "joint_ecf",
    "random_directions",
    "joint_residue_score",
    "MultivariateDetector",
]
