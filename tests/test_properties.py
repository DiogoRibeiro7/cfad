"""Property-based tests for CFAD core invariants."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from cfad.contour import contour_integral
from cfad.empirical_cf import ecf_at, rolling_ecf
from cfad.models.cgmy import CGMYCF


@given(
    n=st.integers(min_value=50, max_value=500),
    sigma=st.floats(
        min_value=0.001,
        max_value=0.1,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@settings(max_examples=50)
def test_ecf_normalization_always_holds(n: int, sigma: float):
    """phi_n(0) == 1 for any return series."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0, sigma, n)
    value = ecf_at(returns, np.array([0.0]))[0]
    assert abs(value - 1.0) < 1e-10


@given(
    n=st.integers(min_value=80, max_value=400),
    window=st.integers(min_value=30, max_value=80),
)
@settings(max_examples=30)
def test_rolling_ecf_window_count(n: int, window: int):
    """Number of windows matches (T - window) // step + 1."""
    assume(window < n)
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, n)
    xi = np.linspace(-3, 3, 16)
    ecf_mat, _ = rolling_ecf(returns, xi, window=window, step=1)
    assert ecf_mat.shape[0] == n - window + 1


@given(
    sigma=st.floats(
        min_value=0.001,
        max_value=0.05,
        allow_nan=False,
        allow_infinity=False,
    )
)
@settings(max_examples=20)
def test_gaussian_cf_contour_is_zero(sigma: float):
    """Gaussian CF contour integral is zero for any positive sigma."""
    value = contour_integral(
        lambda z: np.exp(-0.5 * sigma**2 * z**2),
        xi_min=-5,
        xi_max=5,
        height=0.3,
        n_pts=32,
    )
    assert abs(value) < 1e-10


@given(
    C=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    Y=st.floats(min_value=0.1, max_value=1.9, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20)
def test_cgmy_normalization(C: float, Y: float):
    """phi(0) == 1 for any valid CGMY parameters."""
    model = CGMYCF(C=C, G=5.0, M=10.0, Y=Y)
    value = model.cf(np.array([0.0]))[0]
    assert abs(value - 1.0) < 1e-10
