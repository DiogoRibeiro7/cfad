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

`cfad` is a Python package for detecting structural anomalies in financial
time series using contour integrals of the empirical characteristic function
(ECF) in the complex frequency plane. The core idea is grounded in classical
complex analysis: by the Residue Theorem, the contour integral of an entire
(analytic) function is identically zero, while a function with poles or branch
cuts yields a non-zero residue. Standard diffusive models — Brownian motion,
geometric Brownian motion, and their GARCH variants — produce characteristic
functions that are entire. Heavy-tailed and jump-diffusion processes — Normal
Inverse Gaussian (NIG), CGMY, Lévy-stable — produce characteristic functions
with branch cuts. `cfad` exploits this contrast directly: it estimates the ECF
on a rolling window, evaluates a rectangular contour integral in the complex
$\xi$-plane, and interprets a non-zero result as evidence of non-analytic
structure — the financial analogue of a subsurface vortex invisible from the
surface but detectable by its circulation.

# Statement of Need

Change-point detection in financial time series is typically performed on
returns or volatility directly, using methods such as CUSUM on raw moments
[@page1954], GARCH-based likelihood ratio tests [@inclan1994], or
non-parametric kernel approaches. These methods are sensitive to the *size*
of moves but structurally blind to the *topology* of the generating process:
a model constrained to produce an entire characteristic function cannot detect
the signature of a regime shift into a jump-diffusion or heavy-tailed world,
because the mathematical object that would carry that signature — the branch
cut or pole — cannot exist within the model by construction.

`cfad` fills this gap by operating in the space of characteristic functions
rather than the space of returns. It is, to the author's knowledge, the first
open-source package to implement residue-based anomaly scoring for financial
time series as a general-purpose tool with a scikit-learn-compatible API,
Cython-accelerated hot paths, and a validated benchmark suite on known
structural break events (Flash Crash 2010, August 2015 correction, COVID-19
crash March 2020).

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

Sequential detection is then performed by applying a two-sided Page-CUSUM
[@page1954] to the score series $\{S_t\}$, with in-control parameters
$\mu_0, \sigma_0$ estimated from a calm calibration period:

$$S_t^+ = \max\!\left(0,\; S_{t-1}^+ + \frac{S_t - \mu_0}{\sigma_0} - k\right), \qquad \text{alarm if } S_t^+ > h.$$

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

# Acknowledgements

The author thanks the students of TSIW at ESMAD-IPP whose questions about
market dynamics motivated the pedagogical framing of this work.

# References
