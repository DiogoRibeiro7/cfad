# cfad — Characteristic Function Anomaly Detector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/DiogoRibeiro7/cfad/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DiogoRibeiro7/cfad/actions/workflows/ci.yml)

**cfad** is research software for studying distributional change in financial
return series with empirical characteristic functions (ECFs). It turns rolling
ECF discrepancies into shape-sensitive anomaly scores, supports sequential
Page-CUSUM monitoring, and includes model-comparison, goodness-of-fit,
backtesting, bootstrap, sensitivity, notebook, dashboard, and benchmark
workflows for reproducible experimentation.

The project is aimed at methodological research and validation rather than
turnkey production trading signals. Its benchmarks explicitly preserve negative
results and document where the current ECF score does, and does not, outperform
simpler distributional summaries.

For each rolling window, CFAD compares the empirical characteristic function
with the Gaussian characteristic function fitted to that window's sample mean
and variance. The normalized real-frequency discrepancy is the anomaly score:

$$
D_t =
\left[
\frac{1}{\xi_{\max}-\xi_{\min}}
\int_{\xi_{\min}}^{\xi_{\max}}
\left|
\widehat\varphi_t(\xi)
-
\varphi_{\mathcal N(\widehat\mu_t,\widehat\sigma_t^2)}(\xi)
\right|^2
\,d\xi
\right]^{1/2}.
$$

Because location and scale are fitted within each window, the score is aimed at
higher-order shape changes such as tail and skewness changes. A two-sided
Page-CUSUM then converts the score sequence into sequential alarms.

> **Scientific scope.** The finite-sample ECF
> $\widehat\varphi_n(z)=n^{-1}\sum_j e^{izx_j}$ is a finite sum of entire
> functions and is itself entire. Therefore its exact closed-contour integral is
> zero. CFAD does **not** infer population-CF branch cuts or poles from an
> empirical contour residue. Complex contour integration remains available as a
> diagnostic helper for parametric characteristic functions evaluated at complex
> arguments, but it is not the empirical anomaly statistic.

---

## Status

The repository is currently a **research/development project**. The codebase
contains release scaffolding, documentation, benchmarks, notebooks, and a draft
software paper. GitHub releases are archival research-software snapshots; no
PyPI distribution is claimed unless a separate PyPI publication is explicitly
performed and verified.

### Current validation boundary

Two frozen validation programmes have now tested the corrected method rather
than the retired contour-residue interpretation.

The v2 sequential benchmark showed that Monte Carlo calibration can control the
Gaussian-null false-alarm rate, but the Gaussian-reference score was not robust
to a stable Student-t in-control law and had weak first-alarm power once false
alarms were controlled. The v3 score-level ablation then removed CUSUM and
separated frequency scaling from reference-law choice. Standardizing each window
corrected the legacy score's sensitivity to pure variance changes, and a frozen
empirical in-control ECF produced strong null-law stability. However, the
empirical-reference score achieved AUC 0.646 for a Gaussian-to-Student-t shape
change and 0.783 for a Gaussian-to-skew change, for an average of 0.715 versus
0.738 for a simple kurtosis-distance comparator.

Accordingly, the current evidence does **not** establish that CFAD is a validated
sequential detector or that its ECF score outperforms simpler moment summaries.
The negative v2 and v3 results are retained as reproducible evidence in
`benchmarks/` and are treated as design constraints rather than tuned away.

## Installation

From the default branch:

```bash
git clone https://github.com/DiogoRibeiro7/cfad
cd cfad
git switch main
python -m pip install -e ".[dev]" --no-build-isolation
```

The package includes optional Cython acceleration for rolling ECF evaluation and
CUSUM updates. The statistical score itself is implemented once in NumPy, so
installing the extensions changes performance rather than the definition of the
statistic.

## Quick start

```python
import numpy as np
from cfad import detect

rng = np.random.default_rng(42)
returns = np.concatenate(
    [
        rng.normal(0.0, 0.01, 300),
        rng.standard_t(df=3.0, size=150) * 0.01 / np.sqrt(3.0),
    ]
)

report = detect(
    returns,
    window=60,
    xi_range=(-10.0, 10.0),
    step=1,
    calibration_frac=0.4,
    k=0.5,
    h=5.0,
)
print(report.summary())
```

For market data, fetch the series explicitly with the provider of your choice,
then pass returns to `detect`. Keeping data acquisition outside the core example
makes the detector reproducible without relying on network access.

## Detection pipeline

```text
returns
  │
  ├─ rolling ECF on a real-frequency grid
  │
  ├─ fitted Gaussian CF in each window
  │
  ├─ normalized ECF L2 shape distance D_t
  │
  ├─ calibration of score mean/std on an in-control prefix
  │
  └─ two-sided Page-CUSUM → alarms
```

## Parametric CF models

CFAD also implements characteristic-function models for descriptive model
comparison and goodness-of-fit work:

| Model | Main use |
|---|---|
| `GaussianCF` | location/scale baseline |
| `NIGCF` | semi-heavy tails and skewness |
| `CGMYCF` | jump/tail-shape modelling |
| `LevyStableCF` | power-law tail modelling |

`compare_models()` compares fitted models by real-frequency ECF discrepancy and
AIC. A better-fitting non-Gaussian model is evidence of distributional fit, not
a direct empirical test for complex singularities.

## Utilities

The repository includes:

- `rolling_gof`, `cf_distance`, and `epps_pulley_test` for ECF goodness of fit;
- `WalkForwardBacktest` for temporal evaluation;
- bootstrap and score-stability diagnostics;
- `window_sensitivity`, `frequency_sensitivity`, and threshold sensitivity;
- multivariate and market-oriented helpers;
- a Streamlit dashboard under `apps/`;
- reproducible notebooks and benchmark scripts.

## Reproducibility

The main scientific validation target is not "does an alarm fire on one famous
market event?" but how the detector behaves under controlled null and
alternative data-generating processes. The benchmark layer reports false-positive
behaviour, power/discrimination under prespecified shape changes, specificity to
location/scale changes, and comparison against simpler baselines.

Frozen failed experiments are part of the evidence record. In particular,
`benchmarks/v2_failed_calibration_record.json` records the failed sequential
screen and `benchmarks/v3_failed_score_validation_record.json` records the failed
score-level screen. Those failures are not retroactively reclassified after
parameter or method changes.

## Documentation and paper

- MkDocs sources: [`docs/`](docs/)
- Draft software paper: [`paper/paper.md`](paper/paper.md)
- Reproducible notebooks: [`notebooks/`](notebooks/)
- Benchmarks: [`benchmarks/`](benchmarks/)

The manuscript is a draft companion to the software. Citation metadata should
only advertise a journal DOI after an actual accepted/published record exists.

## Author

**Diogo Ribeiro**  
Faculty of Media Arts and Design, Technical University of Porto  
ORCID: [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)

## License

MIT
