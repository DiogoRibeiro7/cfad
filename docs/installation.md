# Installation

CFAD supports Python 3.10, 3.11, and 3.12. The core package depends on NumPy,
SciPy, pandas, matplotlib, statsmodels, and joblib. Optional Cython extensions
accelerate selected rolling ECF and CUSUM routines but do not change the
definition of the statistic.

## Install From the Repository

For development or for the current default branch:

```bash
git clone https://github.com/DiogoRibeiro7/cfad
cd cfad
python -m pip install -e ".[dev]" --no-build-isolation
```

The `--no-build-isolation` flag is used consistently in this repository because
the optional Cython extension build imports build-time dependencies from the
active environment. If installation fails with `ModuleNotFoundError: Cython`,
install the build prerequisites first:

```bash
python -m pip install "setuptools>=68" wheel "numpy>=1.24" "Cython>=3.0" "packaging>=24.2"
python -m pip install -e ".[dev]" --no-build-isolation
```

## Install From PyPI

If a PyPI release has been published and verified:

```bash
python -m pip install cfad
```

GitHub release snapshots and PyPI releases are separate publication events in
this project. Check the repository release notes before assuming a given version
is available on PyPI.

## Optional Dependency Groups

| Extra | Purpose |
|---|---|
| `[dev]` | pytest, coverage, ruff, black, mypy, Cython, and Hypothesis. |
| `[docs]` | MkDocs, Material for MkDocs, mkdocstrings, and Markdown extensions. |
| `[notebooks]` | Jupyter, seaborn, and yfinance for exploratory notebooks. |

Install multiple groups when needed:

```bash
python -m pip install -e ".[dev,docs,notebooks]" --no-build-isolation
```

## Build Cython Extensions

Build the optional extensions in place when running performance-sensitive tests
or benchmarks:

```bash
python setup.py build_ext --inplace
```

If the extensions are unavailable, CFAD falls back to the NumPy/Python
implementation. Results should remain statistically equivalent; only runtime
changes.

## Verify the Install

```python
import numpy as np
from cfad import detect

returns = np.random.default_rng(0).normal(0.0, 0.01, 200)
report = detect(returns, window=50)

print(report.summary())
```

You should see a `CFAD Anomaly Report` with evaluated windows, alarms, and
calibration statistics.

## Build the Documentation

```bash
python -m pip install -e ".[docs]" --no-build-isolation
mkdocs build --strict
```

Serve the site locally while editing:

```bash
mkdocs serve
```
