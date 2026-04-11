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

`cfad` is a Python package for detecting structural breaks in financial return
series using contour integrals of the empirical characteristic function (ECF).
The package is aimed at computational scientists who want a mathematically
interpretable detector, but do not necessarily work in quantitative finance.
Rather than only tracking changes in first and second moments, `cfad` monitors
changes in the analytic structure of the data-generating law in the complex
frequency domain.

The intuition is borrowed from oceanography: a subsurface vortex can remain
invisible in local surface measurements but still be detected by circulation
integrals over a closed loop. `cfad` translates this idea into time-series
analysis. For each rolling window of returns, the package estimates the ECF and
computes a rectangular contour integral in the complex $\xi$-plane. If the
underlying characteristic function is entire, the contour integral should vanish
up to numerical error. If singular structure (for example branch-cut behavior)
is present, the score rises.

The software contribution is a complete detection stack that combines
Cython-accelerated rolling ECF estimation, contour-residue scoring, and
sequential CUSUM alarming in a unified API. The same package also includes
parametric characteristic-function models (Gaussian, NIG, CGMY, and
Lévy-stable), enabling structural interpretation through model comparison.
Companion notebooks produce reproducible empirical outputs used in the paper,
including concept-level separation, SPY event annotations, detection-delay
distributions, ROC comparisons, and window-sensitivity analyses. Pure-Python
fallbacks are provided so all functionality remains available even when C
extensions cannot be compiled.

# Statement of Need

Financial change-point workflows are well served by generic segmentation and
CUSUM-style tools, but the dominant methods still operate on returns,
volatility, or related moments. In practice this means that widely used
packages such as `ruptures`, `changepoint`, and `strucchange` are optimized for
distributional changes in observed time-domain summaries, not for the topology
of characteristic functions. For jump-process modeling, this distinction
matters: singular structures associated with heavy tails and discontinuities are
first-order modeling objects, not edge cases [@cont2004].

Conversely, characteristic-function-focused software ecosystems (for example
QuantLib-centered pricing workflows or Lévy-fitting utilities such as
`py-levy`) primarily target calibration and pricing. They typically do not
provide a direct anomaly score and sequential alarm layer for online monitoring.
As a result, practitioners often combine multiple toolchains, with ad hoc glue
code and inconsistent calibration assumptions between modeling and detection.

`cfad` addresses this gap by integrating residue-based structural scoring with
Page-CUSUM sequential detection [@page1954] in one consistent API. It maps
rolling windows to ECFs, computes contour-derived anomaly scores, calibrates
in-control behavior, and emits alarms in a single pipeline. To the author's
knowledge, this is the first open-source package that operationalizes this
specific combination for financial time series. The package therefore fills both
a scientific need (topology-aware detection) and an engineering need
(reproducible, maintainable, end-to-end implementation).

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

This strip determines where contour paths can be interpreted as sampling the
analytic regime of NIG and where branch structure may contribute to the
integral. In practical terms, contour height controls structural sensitivity.

The key guarantee is Cauchy's theorem: if a function is holomorphic on and
inside a closed contour, the contour integral is zero. Therefore, if the
window-level characteristic function is entire in the relevant domain,
$S_t = 0$ in the idealized limit. Under finite samples and discrete quadrature,
scores are not exactly zero numerically, but they remain concentrated near an
in-control baseline for analytic dynamics and increase when non-analytic
features are present.

The method is related to the ECF goodness-of-fit literature. Epps-Pulley
[@epps1983] uses ECF functionals for a single-sample test of normality; `cfad`
extends this from one-window testing to a rolling sequential detector in which
each window yields a structural score that feeds a monitoring process.

Sequential detection is then performed by applying a two-sided Page-CUSUM
[@page1954] to the score series $\{S_t\}$, with in-control parameters
$\mu_0, \sigma_0$ estimated from a calm calibration period:

$$S_t^+ = \max\!\left(0,\; S_{t-1}^+ + \frac{S_t - \mu_0}{\sigma_0} - k\right), \qquad \text{alarm if } S_t^+ > h.$$

An analogous recursion is used for negative excursions, producing a two-sided
procedure robust to asymmetric deviations. This decomposition is central to the
design: contour integration extracts topology-sensitive evidence, and CUSUM
converts that evidence into a controlled sequential decision rule with explicit
thresholding.

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

In addition, the repository includes benchmark scripts and reproducible
notebooks for empirical validation. These workflows generate the paper outputs
under `paper/figures/` from fixed seeds and explicit simulation settings,
supporting end-to-end reproducibility from raw returns to final metrics.

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

`cfad` keeps the same algorithmic structure in Python and Cython, but moves the
hot path to compiled loops. In pure Python, rolling ECF evaluation scales as
O$(T \cdot n \cdot m)$ with interpreter overhead at each sample/update step.
The Cython extensions retain O$(T \cdot n \cdot m)$ complexity while executing
as tight C loops, eliminating per-sample Python overhead and reducing wall-clock
runtime in long-series and Monte Carlo settings. The package also provides
feature-equivalent pure-Python fallbacks for environments without a working C
compiler, so portability and reproducibility are preserved.

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
