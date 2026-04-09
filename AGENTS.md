# AGENTS.md — cfad

## Project overview

`cfad` is a Python package (with Cython C extensions) for detecting structural
breaks in financial time series using contour integrals of the empirical
characteristic function. It accompanies a research paper submitted to JOSS.

## Repository layout

```
cfad/
├── cfad/                   # main package
│   ├── __init__.py         # public API exports
│   ├── empirical_cf.py     # ECF estimation (Python + Cython wrapper)
│   ├── contour.py          # contour integration engine
│   ├── residue_score.py    # score normalisation utilities
│   ├── detection.py        # RollingDetector, AnomalyReport
│   ├── api.py              # detect(), compare_models() entry points
│   ├── utils.py            # plotting, data loading helpers
│   ├── models/             # parametric CF models
│   │   ├── base.py         # CFModel abstract base class
│   │   ├── gaussian.py     # Gaussian (entire, baseline)
│   │   ├── nig.py          # Normal Inverse Gaussian (branch cut)
│   │   ├── cgmy.py         # CGMY / Kou model (branch cut)
│   │   └── levy_stable.py  # Lévy-stable (branch cut)
│   └── _ext/               # Cython extensions
│       ├── rolling_ecf.pyx  # O(n*m) rolling ECF — HOT PATH
│       ├── contour_quad.pyx # complex-plane quadrature — HOT PATH
│       └── cusum.pyx        # sequential CUSUM — HOT PATH
├── tests/                  # pytest suite
├── notebooks/              # Jupyter notebooks (concept → validation)
├── benchmarks/             # timing + ROC benchmarks
├── paper/                  # JOSS manuscript (LaTeX + paper.md)
├── data/                   # sample datasets (SPY, synthetic Lévy)
├── setup.py                # Cython build
└── pyproject.toml          # project metadata
```

## Development conventions

- **Python 3.10+**, type-annotated throughout (`from __future__ import annotations`)
- **Numpy docstrings** on all public functions
- **Cython compiler directives** on every `.pyx` file header:
  `# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True`
- All public Python modules have a **pure-Python fallback** when C extensions
  are not compiled. Guard with `try/except ImportError`.
- **No global state** — all stateful objects are `RollingDetector` instances.
- Formatting: `black` (line length 88) + `ruff` linting.
- All PRs must pass `pytest tests/ --cov=cfad --cov-fail-under=80`.

## Task instructions for coding agents

### Task A — Implement cgmy.py
File: `cfad/models/cgmy.py`
Pattern: follow `cfad/models/nig.py` exactly.

The CGMY characteristic function is:
  phi(xi) = exp(C * Gamma(-Y) * [(M - i*xi)^Y - M^Y + (G + i*xi)^Y - G^Y])
where C>0, G>0, M>0, 0<Y<2.
- Set `is_analytic = False`.
- Implement `cf(xi)`, `log_cf(xi)`, `fit(returns)` (Nelder-Mead on ECF L2).
- Add `__repr__`.
- Write unit tests in `tests/test_models.py` mirroring NIG tests.
- Guard: if Y >= 2 raise ValueError.

### Task B — Implement levy_stable.py
File: `cfad/models/levy_stable.py`
The Lévy-stable CF (Zolotarev parameterisation):
  log phi(xi) = -|c*xi|^alpha * (1 - i*beta*sign(xi)*tan(pi*alpha/2)) + i*mu*xi
for alpha ≠ 1. Set `is_analytic = False`.
- Fit via scipy.stats.levy_stable.fit (MLE).
- Implement `cf`, `log_cf`, `fit`, `__repr__`.
- Add tests confirming phi(0)=1 and is_analytic=False.

### Task C — Implement residue_score.py
File: `cfad/residue_score.py`
- `normalise_scores(scores, method="zscore"|"mad"|"minmax")` → ndarray
- `rolling_pvalue(scores, window, dist="normal"|"empirical")` → ndarray
  (pointwise p-value of score under in-control distribution)
- `threshold_by_fpr(scores, fpr=0.01)` → float (threshold at given FPR)
- All functions fully type-annotated, numpy docstrings.

### Task D — Implement utils.py
File: `cfad/utils.py`
- `plot_scores(report: AnomalyReport, returns=None, ax=None)` — matplotlib
  two-panel plot: top = return series with alarm markers, bottom = CUSUM.
- `load_spy_sample()` → pd.Series — loads `data/spy_2018_2022.csv` if present,
  otherwise downloads via yfinance and caches.
- `simulate_levy_returns(n, alpha=1.7, beta=0.0, scale=0.01)` → ndarray
  — synthetic Lévy-stable returns for testing.

### Task E — Implement benchmarks
File: `benchmarks/benchmark_detector.py`
- Use `timeit` to benchmark `detect()` for T = [500, 2000, 10000] with C ext
  and pure-Python fallback. Print a table.
File: `benchmarks/roc_benchmark.py`
- Simulate 500 normal series and 500 series with a single Lévy jump.
- Run `detect()` with varying `h` thresholds.
- Compute ROC curve + AUC. Save figure to `benchmarks/roc_curve.png`.

### Task F — Paper companion (paper/paper.md)
Write the JOSS paper.md following https://joss.readthedocs.io/en/latest/submitting.html
- Title, authors (Diogo Ribeiro, ESMAD-IPP, ORCID: 0009-0001-2022-7072)
- Summary (~250 words): what cfad does, why it matters
- Statement of need: gap in current software
- Mathematics: ECF definition, residue theorem connection, CUSUM
- Usage example (code block)
- Acknowledgements, References (BibTeX keys from paper/references.bib)

### Task G — Notebooks
For each notebook follow this structure:
1. Import cfad + dependencies
2. Theoretical motivation (markdown with LaTeX)
3. Code demonstration
4. Figure saved to `paper/figures/`

`01_concept_illustration.ipynb` — oceanography→finance analogy, residue theorem,
animate ECF for Gaussian vs NIG vs Lévy-stable.

`02_cf_families.ipynb` — compare all four CF models on SPY data, AIC table,
CF distance plot.

`03_contour_detection.ipynb` — run RollingDetector on SPY 2019-2021,
annotate Flash Crash (Feb 2020 pre-signal), COVID crash (Mar 2020).

`04_empirical_validation.ipynb` — Monte Carlo: ROC curves, detection delay
distribution, comparison with CUSUM on raw returns.

## Key scientific references (cite in code comments)

- Cont & Tankov (2004) — *Financial Modelling with Jump Processes*. CRC Press.
- Epps & Pulley (1983) — ECF normality test. *Biometrika* 70(3), 723-726.
- Küchler & Tappe (2013) — Tempered stable distributions. *Stochastic Processes*.
- Page (1954) — Continuous inspection schemes. *Biometrika* 41(1-2), 100-115.
- Madan, Carr & Chang (1998) — VG model. *European Finance Review* 2(1), 79-105.
