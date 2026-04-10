# cfad — Characteristic Function Anomaly Detector

[![PyPI version](https://badge.fury.io/py/cfad.svg)](https://pypi.org/project/cfad/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/diogoribeiro7/cfad/actions/workflows/ci.yml/badge.svg)](https://github.com/diogoribeiro7/cfad/actions)
[![DOI](https://joss.theoj.org/papers/placeholder/badge.svg)](paper/paper.md)

> *"If your equations enforce a smooth surface, you haven't proven the vortex is missing — you've made it impossible to exist."*

**cfad** detects structural breaks in financial time series by measuring the
**residue of the empirical characteristic function (ECF)** along a contour
in the complex frequency plane.

The key idea: under diffusive (Gaussian) dynamics, the characteristic function
is *entire* — analytic everywhere — so any closed contour integral returns
exactly zero (Cauchy's theorem). Heavy-tailed and jump processes (NIG, CGMY,
Lévy-stable) introduce *branch cuts* and *poles* — non-analytic structure that
makes the contour integral non-zero. This non-zero value is the anomaly score.

---

## Installation

```bash
# Standard (pure Python, no C extensions)
pip install cfad

# With Cython C extensions (20-50× faster on large windows)
pip install cfad[speed]
# or from source:
git clone https://github.com/diogoribeiro7/cfad
cd cfad
pip install -e . --no-build-isolation
```

## Quick start

```python
import yfinance as yf
from cfad import detect

prices = yf.download("SPY", start="2018-01-01", end="2022-01-01")["Close"]
returns = prices.pct_change().dropna()

report = detect(returns, window=60, h=4.0)
print(report.summary())
# → CFAD Anomaly Report
# →   Windows evaluated : 952
# →   Alarms fired      : 3
# →   First alarm       : 2020-02-28

# Compare structural models
from cfad import compare_models
result = compare_models(returns.values)
print(result["winner"])  # → "nig" (non-analytic wins → structure present)
```

## How it works

```
Returns r_1..r_T
       │
       ▼ rolling_ecf()              [Cython: O(n·m)]
  φ̂_n(ξ) for each window
       │
       ▼ ecf_residue_scores()       [Cython: complex quadrature]
  Residue magnitude score s_t
       │
       ▼ cusum()                    [Cython: sequential]
  S_t (CUSUM statistic)
       │
       ▼ threshold h
  AnomalyReport (alarms, dates)
```

The anomaly score is the magnitude of:

$$I = \frac{1}{2\pi i} \oint_C \hat{\varphi}_n(\xi)\, d\xi$$

Under the Gaussian null: $I \equiv 0$.
Under non-analytic alternatives: $I \neq 0 \Rightarrow$ structural break.

## CF model families

| Model | `is_analytic` | CF structure | Captures |
|---|---|---|---|
| `GaussianCF` | `True` | Entire | Drift + diffusion |
| `NIGCF` | `False` | Branch cut at $i(\alpha-|\beta|)$ | Semi-heavy tails, skew |
| `CGMYCF` | `False` | Branch cut | Jump intensity + tail index |
| `LevyStableCF` | `False` | Branch cut | Power-law tails ($\alpha < 2$) |

## Utilities

`cfad` also provides convenience tools for model comparison and sample generation:

- `load_spy_sample()` — load or cache SPY daily returns for 2018–2022.
- `simulate_levy_returns(n, alpha, beta, scale)` — generate synthetic Lévy-stable returns.
- `normalise_scores(...)`, `rolling_pvalue(...)`, `threshold_by_fpr(...)` — score normalisation and thresholding utilities.

## Notebooks

The repository includes example notebooks demonstrating the package pipeline and scientific motivation:

- `notebooks/01_concept_illustration.ipynb`
- `notebooks/02_cf_families.ipynb`
- `notebooks/03_contour_detection.ipynb`
- `notebooks/04_empirical_validation.ipynb`

## Paper

This package accompanies the manuscript:

> Ribeiro, D. (2025). *Residues as detectors: contour integral anomaly
> scoring for financial time series*. Submitted to Journal of Open Source
> Software (JOSS).

## Citation

```bibtex
@article{ribeiro2025cfad,
  author  = {Ribeiro, Diogo},
  title   = {cfad: Characteristic Function Anomaly Detector},
  journal = {Journal of Open Source Software},
  year    = {2025},
  doi     = {10.xxxx/joss.xxxxx}
}
```

## Author

**Diogo Ribeiro** — ESMAD, Instituto Politécnico do Porto  
ORCID: [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)  
Email: dfr@esmad.ipp.pt

## License

MIT
