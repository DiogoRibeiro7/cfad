---
title: 'cfad: Characteristic Function Anomaly Detection for Financial Time Series'
tags:
  - Python
  - anomaly detection
  - empirical characteristic function
  - change-point detection
  - financial econometrics
  - CUSUM
authors:
  - name: Diogo Ribeiro
    orcid: 0009-0001-2022-7072
    affiliation: 1
affiliations:
  - name: Faculty of Media Arts and Design, Technical University of Porto, Portugal
    index: 1
bibliography: references.bib
---

# Summary

`cfad` is a Python package for detecting changes in the distributional shape of
financial returns with empirical characteristic functions (ECFs). For each
rolling window, the package evaluates the ECF on a real-frequency grid and
compares it with the Gaussian characteristic function fitted to that window's
sample mean and variance. The resulting normalized integrated squared distance
is monitored with a two-sided Page-CUSUM to produce sequential alarms.

The package is designed for research workflows where changes in tails,
skewness, or other higher-order distributional features may matter beyond
changes in mean or variance. It includes batch and streaming detectors,
parametric characteristic-function models (Gaussian, NIG, CGMY, and
Lévy-stable), ECF goodness-of-fit tools, walk-forward backtesting, bootstrap and
sensitivity diagnostics, visualization utilities, benchmark scripts, and
optional Cython acceleration for rolling ECF evaluation and CUSUM updates.

A central design principle is that the statistical score has one definition
regardless of whether compiled extensions are installed. Cython changes
execution speed, not the scientific estimand.

# Statement of Need

Financial change-point tools commonly monitor returns, volatility, regression
parameters, or generic distributional summaries. Those approaches are often
appropriate, but they do not directly expose how the full empirical
characteristic function changes over time. ECF methods provide a natural way to
compare distributions because they exist for every probability distribution and
encode all distributional information under standard uniqueness results.

`cfad` provides an end-to-end workflow that joins an ECF discrepancy statistic
to sequential monitoring. The package is intended to complement rather than
replace established change-point and goodness-of-fit methods. Its main software
contribution is the integration of rolling ECF estimation, model comparison,
sequential decision rules, diagnostics, and reproducible evaluation in one API.

# Statistical Method

Let $r_1,\ldots,r_n$ denote returns in one rolling window. The empirical
characteristic function is

$$
\widehat\varphi_n(\xi)
=
\frac{1}{n}\sum_{j=1}^{n}e^{i\xi r_j},
\qquad \xi\in\mathbb R.
$$

Within the same window, define

$$
\widehat\mu_n = \frac1n\sum_{j=1}^n r_j,
\qquad
\widehat\sigma_n^2
=\frac{1}{n-1}\sum_{j=1}^n(r_j-\widehat\mu_n)^2.
$$

The fitted Gaussian characteristic function is

$$
\varphi_G(\xi)
=
\exp\left(
 i\widehat\mu_n\xi
 -\frac12\widehat\sigma_n^2\xi^2
\right).
$$

CFAD's window score is the normalized real-frequency $L^2$ discrepancy

$$
D_n
=
\left[
\frac{1}{\xi_{\max}-\xi_{\min}}
\int_{\xi_{\min}}^{\xi_{\max}}
\left|
\widehat\varphi_n(\xi)-\varphi_G(\xi)
\right|^2d\xi
\right]^{1/2}.
$$

Because the Gaussian reference is fitted separately in every window, changes in
location and scale are absorbed by $\widehat\mu_n$ and
$\widehat\sigma_n$. The remaining score is therefore primarily a measure of
non-Gaussian distributional shape, including tail and skewness changes. This
interpretation should be validated for each application by simulation and
out-of-sample benchmarks rather than treated as a universal detector guarantee.

## Why the empirical score is not a contour residue

For a finite sample,

$$
\widehat\varphi_n(z)
=
\frac1n\sum_{j=1}^n e^{izr_j}
$$

is a finite sum of entire functions of $z\in\mathbb C$, hence is itself entire.
Its exact integral around any closed contour is therefore zero by Cauchy's
theorem. Consequently, a finite-sample ECF contour integral cannot identify
branch cuts or poles of the population characteristic function.

Complex contour integration remains useful in `cfad.contour` for parametric
functions that are explicitly evaluated at complex arguments, but it is not the
empirical anomaly statistic. Parametric model comparisons are similarly treated
as distributional-fit evidence rather than direct empirical singularity tests.

# Sequential Monitoring

Let $D_t$ denote the rolling score sequence and let $\mu_0,\sigma_0$ be
estimated from a prespecified in-control calibration prefix. Standardized scores
are

$$
z_t=\frac{D_t-\mu_0}{\sigma_0}.
$$

CFAD applies the two-sided Page-CUSUM [@page1954]

$$
S_t^+ = \max(0,S_{t-1}^+ + z_t-k),
$$

$$
S_t^- = \max(0,S_{t-1}^- - z_t-k),
$$

and emits an alarm when either statistic exceeds a decision threshold $h$. Since
$z_t$ is standardized, the reference value $k$ is dimensionless.

# Implementation

The package is organized into four main layers:

1. **ECF estimation** (`cfad/empirical_cf.py`) with an optional Cython rolling
   implementation.
2. **Scoring and detection** (`cfad/contour.py`, `cfad/detection.py`) with a
   single NumPy implementation of the Gaussian ECF distance and Python/Cython
   implementations of the same CUSUM recursion.
3. **Parametric models and diagnostics** (`cfad/models/`, `cfad/gof.py`) for
   Gaussian, NIG, CGMY, and Lévy-stable comparison and goodness of fit.
4. **Evaluation and operational utilities** (`cfad/backtest.py`,
   `cfad/bootstrap.py`, `cfad/sensitivity.py`, `cfad/market.py`) for temporal
   evaluation, uncertainty diagnostics, sensitivity analysis, and multi-asset
   workflows.

The repository also includes a Streamlit dashboard, Sphinx documentation,
notebooks, and benchmark scripts.

# Usage

```python
import numpy as np
from cfad import detect

rng = np.random.default_rng(42)
returns = np.concatenate([
    rng.normal(0.0, 0.01, 300),
    rng.standard_t(df=3.0, size=150) * 0.01 / np.sqrt(3.0),
])

report = detect(
    returns,
    window=60,
    xi_range=(-10.0, 10.0),
    calibration_frac=0.4,
    k=0.5,
    h=5.0,
)
print(report.summary())
```

# Validation Strategy

A useful anomaly detector must be evaluated on more than one annotated market
series. The repository's benchmark contract therefore emphasizes controlled
experiments with explicit null and alternative data-generating processes. Core
metrics include false-positive rate, power, detection delay, and sensitivity to
window length and real-frequency cutoff. Comparisons against simpler baselines
are required to establish whether ECF-based shape information adds useful signal.

Any figures produced under the earlier empirical-contour-residue definition are
provisional and must be regenerated before being used as evidence for the
corrected method.

# Current Validation Evidence

The corrected method has been subjected to two frozen validation programmes.
The v2 sequential experiment calibrated the maximum two-sided Page-CUSUM path by
Monte Carlo under a Gaussian null. CFAD's Gaussian false-alarm rate was 0.064,
but the same calibrated detector produced a 0.256 false-alarm rate under a
stationary Student-t null. First-alarm power was 0.234 for a
Gaussian-to-Student-t change and 0.208 for a Gaussian-to-skew change. The v2
publication screen therefore failed.

The v3 experiment removed the sequential layer and tested the score directly on
non-overlapping windows. It separated the raw-frequency formulation from
standardized-frequency scoring and compared a Gaussian reference with a frozen
empirical in-control ECF. Standardization corrected the legacy raw score's
near-perfect discrimination of a pure variance change: variance-shift AUC fell
from 0.999 for the raw-frequency score to approximately 0.50 for both
standardized CF scores. The empirical reference also produced strong null-law
stability between stationary Gaussian and Student-t data, with a median-score
ratio of 1.024 and KS statistic of 0.0505.

Those improvements did not translate into superior shape discrimination. The
primary empirical-reference CF score achieved AUC 0.646 for the heavy-tail
change and 0.783 for the skew change, averaging 0.715. The prespecified minimum
was 0.75 for each alternative. A simple excess-kurtosis distance averaged 0.738
across the two shape alternatives, so the primary CF score was worse by 0.023
rather than better by the required 0.05. It was also slightly worse than the
standardized-Gaussian CF score on both shape alternatives. The v3 screen failed.

These results delimit the current claim. The software implements a coherent ECF
scoring and sequential-monitoring framework, and the validation work identifies
useful properties of frequency standardization and empirical in-control
referencing. However, the present evidence does not establish a statistically
validated sequential detector or an accuracy advantage over simpler moment-based
shape summaries. The frozen negative results are retained in the repository and
are not tuned away after inspection.

# Acknowledgements

The project benefited from questions raised in teaching and research discussions
about characteristic functions, distributional change, and financial time
series.

# References

- [@page1954]
- [@epps1983]
- [@cont2004]
- [@barndorff1977]
- [@carr2002]
