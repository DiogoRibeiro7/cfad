---
title: 'cfad: Characteristic Function Anomaly Detection for Financial Time Series'
tags:
  - Python
  - anomaly detection
  - characteristic function
  - contour integration
  - Lévy processes
  - change-point detection
  - financial econometrics
authors:
  - name: Diogo Ribeiro
    orcid: 0009-0001-2022-7072
    affiliation: 1
affiliations:
  - name: ESMAD – Escola Superior de Média Arte e Design, Instituto Politécnico do Porto, Portugal
    index: 1
date: 2025
bibliography: references.bib
---

# Summary

`cfad` is a Python package for structural anomaly detection in financial time
series using characteristic functions and contour integration. The package is
designed for computational scientists who want a mathematically explicit
alternative to detectors based only on moments, variance shifts, or generic
forecast errors. Instead of monitoring only what is visible in the return
series, `cfad` monitors how the *data-generating law* changes in the complex
frequency domain.

An intuitive analogy comes from oceanography. A surface observer may see a calm
water field, while a hidden subsurface vortex is still present. The vortex can
be detected by integrating circulation along a closed curve, even if local
pointwise observations look benign. `cfad` translates this idea to finance.
Returns in a rolling window are mapped to the empirical characteristic function
(ECF), then integrated along a contour in the complex $\xi$-plane. Under an
entire characteristic function, circulation is zero by complex analysis. When
the window is better explained by a model with branch cuts or poles, the
integral magnitude becomes non-zero.

The software contribution is an end-to-end detection pipeline that combines:
rolling ECF estimation, contour-based residue scoring, and sequential alarming
with CUSUM. Performance-critical steps are implemented in Cython so the method
is practical on long historical series and Monte Carlo experiments. The package
still provides pure-Python fallbacks for portability, exposing the same public
API regardless of whether extension modules are compiled. In practice, users
get a one-call interface (`detect`) for operational monitoring and a model
comparison interface (`compare_models`) for interpretation, while retaining
direct access to lower-level components for reproducible research.

# Statement of Need

Practitioners have mature software for change-point analysis, but most
financial workflows remain tied to return-domain summaries (mean, variance,
volatility proxies) or segmented time-domain likelihoods. Widely used tools
such as `ruptures`, `changepoint`, and `strucchange` are powerful for generic
distributional shifts, yet they do not explicitly encode the analytic topology
of the underlying characteristic function. In jump-process modeling, however,
that topology is economically meaningful: branch points and heavy-tail behavior
are core signals of regime change in markets with discontinuities
[@cont2004].

At the same time, libraries focused on characteristic-function modeling (for
example, implementations around QuantLib-style pricing routines or
Lévy-distribution fitting utilities) are usually parameter-estimation tools.
They fit models but do not provide an integrated anomaly score and sequential
decision rule for ongoing monitoring. In other words, model calibration and
online detection are typically disconnected.

`cfad` addresses this gap by coupling residue-based scoring with sequential
CUSUM alarming [@page1954] in a single workflow. The package estimates rolling
ECFs, computes contour-integral scores that are sensitive to non-analytic
structure, calibrates in-control behavior, and emits alarms through a unified
API. To the author's knowledge, this is the first open-source package that
operationalizes this specific combination for financial time series. The result
is a tool that is both mathematically interpretable and engineering-ready:
users can reproduce the full pipeline from data to alert without hand-assembling
heterogeneous libraries.

# Mathematics

Let $r_1, \ldots, r_T$ be a sequence of log-returns. The empirical
characteristic function (ECF) at frequency $\xi \in \mathbb{R}$ is

$$\hat{\varphi}_n(\xi) = \frac{1}{n} \sum_{j=1}^n e^{i\xi r_j}.$$

For a rolling window of size $n$ ending at time $t$, `cfad` computes
$\hat{\varphi}_n(\xi)$ at a uniform grid $\xi_1, \ldots, \xi_m$ using a
Cython-accelerated O$(nm)$ inner loop. The anomaly score for that window is

$$S_t = \left| \oint_C \hat{\varphi}_n(\xi)\, d\xi \right|$$

where $C$ is a rectangular contour with corners $\xi_{\min} \pm i\eta$,
$\xi_{\max} \pm i\eta$ in the complex $\xi$-plane. By the Residue Theorem:

- **Entire CF** (Gaussian): $S_t \equiv 0$ for any contour $C$.
- **Non-analytic CF** (NIG, CGMY, Lévy-stable): $S_t \neq 0$ when $C$
  encloses a branch point or pole.

For the Normal Inverse Gaussian family, analyticity holds only inside a strip
of the complex plane:

$$\left\{\xi \in \mathbb{C} : |\operatorname{Im}\xi| < \alpha - |\beta| \right\}.$$

Contours that stay fully inside this strip behave as if the integrand were
analytic; contours that cross branch structure produce non-zero circulation.
This gives a direct geometric interpretation of parameter-dependent sensitivity.

The key theoretical guarantee is Cauchy's theorem: if the (window-level)
characteristic function is holomorphic on and inside the chosen contour, then
the closed-path integral is exactly zero. Therefore, in the idealized
noise-free setting, the residue score vanishes for entire models and non-zero
scores are tied to singular structure. In finite samples, `cfad` evaluates a
discrete contour quadrature, so scores are not exactly zero numerically, but
they remain near a stable baseline under analytic dynamics and increase during
regimes with jump-like behavior.

The approach is connected to classical ECF goodness-of-fit ideas. The
Epps-Pulley test [@epps1983] measures discrepancy from Gaussianity in a single
sample through ECF functionals. `cfad` extends this spirit from one-shot
testing to *sequential detection*: each rolling window yields an ECF-based
score, and the resulting score stream is monitored online.

Sequential detection is then performed by applying a two-sided Page-CUSUM
[@page1954] to the score series $\{S_t\}$, with in-control parameters
$\mu_0, \sigma_0$ estimated from a calm calibration period:

$$S_t^+ = \max\!\left(0,\; S_{t-1}^+ + \frac{S_t - \mu_0}{\sigma_0} - k\right), \qquad \text{alarm if } S_t^+ > h.$$

An analogous recursion is used for negative excursions, yielding a two-sided
procedure robust to both upward and downward score drifts. The detector
therefore separates two roles cleanly: contour integration extracts structural
evidence from the frequency domain, and CUSUM transforms that evidence into a
controlled sequential decision process.

# Implementation

The package is structured in four layers:

1. **Cython extensions** (`cfad/_ext/`): three `.pyx` modules for rolling ECF
   estimation, rectangular contour quadrature in the complex plane, and CUSUM
   updates. Each uses `boundscheck=False`, `wraparound=False`, and
   `cdivision=True`, with O3 optimisation. Pure-Python fallbacks are provided
   for environments without a C compiler.

2. **Core engine** (`cfad/empirical_cf.py`, `cfad/contour.py`): thin wrappers
   around the Cython hot paths with graceful fallback.

3. **Parametric CF models** (`cfad/models/`): `GaussianCF` (entire, baseline),
   `NIGCF` [@barndorff1977], `CGMYCF` [@carr2002], `LevyStableCF`. Each
   implements `cf(xi)`, `log_cf(xi)`, and `fit(returns)` (ECF-based minimum
   distance estimation via Nelder-Mead).

4. **Detection API** (`cfad/api.py`): `detect(returns, ...)` and
   `compare_models(returns)` for one-call usage.

# Usage

```python
import yfinance as yf
from cfad import detect, compare_models

prices = yf.download("SPY", start="2019-01-01", end="2021-06-01")["Close"]
returns = prices.pct_change().dropna()

report = detect(returns, window=60, h=4.0)
print(report.summary())
# CFAD Anomaly Report
#   Windows evaluated : 588
#   Alarms fired      : 3
#   First alarm       : 2020-02-28

result = compare_models(returns.values)
print(result["winner"])   # → "nig"
```

```python
from cfad import compare_models

result = compare_models(returns.values)
print(result)  # {'gaussian': {...}, 'nig': {...}, 'winner': 'nig'}
```

# Performance Note

`cfad` is engineered so the high-frequency inner loops run in Cython while the
public API remains Pythonic. In pure NumPy/Python, rolling ECF evaluation over
a long series can incur substantial interpreter overhead, effectively scaling as
O$(T \cdot n \cdot m)$ with costly per-window array operations. The Cython
extensions keep the same asymptotic structure but execute as tight C loops with
typed memory access and no Python overhead per sample update, substantially
reducing wall-clock time in benchmarks and Monte Carlo studies. For users in
restricted environments (for example, no local C compiler), `cfad` ships
feature-equivalent pure-Python fallbacks so experiments remain reproducible.

# Acknowledgements

The author thanks the students of TSIW at ESMAD-IPP whose questions about
market microstructure motivated the pedagogical framing of this work.

# References

- [@cont2004]
- [@page1954]
- [@epps1983]
- [@barndorff1977]
- [@carr2002]
- [@inclan1994]
- [@madan1998]
- [@kuchler2013]
