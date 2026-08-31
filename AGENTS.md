# AGENTS.md — cfad

## Project purpose

`cfad` is a research-oriented Python package for detecting distributional-shape
changes in financial return series with empirical characteristic functions
(ECFs) and Page-CUSUM sequential monitoring.

The scientific contract is part of the code contract. Do not describe or
implement the detector as an empirical contour-residue detector.

## Core statistic

For each rolling window, evaluate the empirical characteristic function on a
real-frequency grid and compare it with the Gaussian CF fitted to the same
window mean and sample standard deviation:

$$
D_t =
\left[
\frac{1}{\xi_{\max}-\xi_{\min}}
\int_{\xi_{\min}}^{\xi_{\max}}
|\widehat\varphi_t(\xi)-
\varphi_{\mathcal N(\widehat\mu_t,\widehat\sigma_t^2)}(\xi)|^2 d\xi
\right]^{1/2}.
$$

Then standardize the score with in-control calibration moments and apply
Page-CUSUM:

$$
z_t=(D_t-\mu_0)/\sigma_0,
$$

$$
S_t^+=\max(0,S_{t-1}^+ + z_t-k),
\qquad
S_t^-=\max(0,S_{t-1}^- - z_t-k).
$$

`k` is dimensionless because `z_t` is standardized.

## Mathematical guardrail

For a finite sample $x_1,\ldots,x_n$,

$$
\widehat\varphi_n(z)=n^{-1}\sum_j e^{izx_j}
$$

is a finite sum of entire functions and is itself entire. Its exact integral
around any closed contour is zero. Therefore:

- never claim that a finite-sample ECF residue identifies population branch cuts
  or poles;
- never use a real-axis numerical integral and call it a contour residue;
- complex contour integration is permitted only for functions actually evaluated
  at complex arguments, such as parametric CF diagnostics;
- a compiled extension must implement exactly the same statistic as the Python
  path. Acceleration may change runtime, never the estimand.

## Repository structure

```text
cfad/
├── cfad/
│   ├── api.py              # detect(), compare_models()
│   ├── empirical_cf.py     # ECF evaluation, rolling ECF
│   ├── contour.py          # ECF shape score + parametric contour helper
│   ├── detection.py        # RollingDetector, StreamDetector, Page-CUSUM
│   ├── gof.py              # CF distance and goodness-of-fit tools
│   ├── backtest.py         # walk-forward evaluation
│   ├── sensitivity.py      # window/frequency/threshold sensitivity
│   ├── models/             # parametric CF models
│   └── _ext/               # optional rolling-ECF and CUSUM Cython paths
├── tests/
├── benchmarks/
├── notebooks/
├── docs/
├── paper/
├── apps/
├── pyproject.toml
└── setup.py
```

## Engineering conventions

- Python 3.10+.
- Public Python APIs are typed and use NumPy-style docstrings.
- Validate dimensionality, finiteness, and parameter ranges at public boundaries.
- Use `numpy.trapezoid`, not deprecated `numpy.trapz`.
- Keep Python/Cython numerical behavior covered by parity tests.
- Avoid `-march=native` and unsafe floating-point flags in distributable wheels.
- Formatting: Black, line length 88.
- Linting: Ruff.
- Type checking: mypy where practical.
- Tests: pytest + coverage.
- CI target branch is `develop` unless repository policy explicitly changes.

## Evaluation rules

A detector claim needs evidence against an explicit alternative and an explicit
null. At minimum, controlled validation should report:

1. false-positive rate under the in-control process;
2. power under prespecified distributional-shape changes;
3. detection delay;
4. sensitivity to window length and frequency cutoff;
5. comparison against simpler baselines;
6. seeds/configuration sufficient to reproduce the result.

Do not use one famous market event as proof of general detection ability. Do not
select hyperparameters on the same event and then report that event as unbiased
validation.

## Documentation rules

The README, Sphinx documentation, paper, code docstrings, archive metadata, and
actual implementation must describe the same statistic.

Do not add:

- placeholder DOIs;
- unverified publication/submission claims;
- unverified PyPI badges;
- old institutional affiliation text.

Current affiliation text:

**Faculty of Media Arts and Design, Technical University of Porto**

ORCID: `0009-0001-2022-7072`.

## Key references

- Epps & Pulley (1983), *Biometrika* — ECF-based normality testing.
- Page (1954), *Biometrika* — CUSUM sequential monitoring.
- Cont & Tankov (2004) — jump-process background and parametric CFs.

These references motivate components of the software; they do not establish the
empirical detector's performance. Performance claims require the repository's
own reproducible evidence.
