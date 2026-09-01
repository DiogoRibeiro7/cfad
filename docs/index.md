# cfad - Characteristic Function Anomaly Detector

`cfad` is research software for studying distributional change in financial
return series with empirical characteristic functions (ECFs). It is designed for
method development, reproducible validation, and diagnostic analysis of return
distributions, not for turnkey trading signals.

At a high level, CFAD computes a rolling ECF for each window of returns,
compares that ECF with a fitted Gaussian characteristic function on a
real-frequency grid, and monitors the resulting shape score with a two-sided
Page-CUSUM.

## What CFAD Is For

Use CFAD when you want to:

- study changes in distributional shape, especially tails, skewness, and
  departures from a Gaussian reference;
- run reproducible experiments around empirical characteristic functions;
- compare parametric characteristic-function models against observed returns;
- inspect score stability, threshold sensitivity, and walk-forward behavior;
- preserve validation evidence, including negative benchmark outcomes.

CFAD is a poor fit if you need:

- a production-ready market-alert service;
- a detector whose superiority over simpler summaries has already been
  established across domains;
- a branch-cut or pole test from a finite-sample empirical characteristic
  function.

!!! important "Validation status"
    The repository deliberately preserves failed confirmatory screens. Current
    evidence does not establish CFAD as a validated sequential detector or show
    that its ECF score generally outperforms simpler moment-based summaries.
    See [Validation](validation.md) before citing performance claims.

## Core Pipeline

```text
returns
  |
  |-- rolling windows
  |-- empirical characteristic function on a real-frequency grid
  |-- fitted Gaussian characteristic function per window
  |-- normalized ECF L2 shape distance
  |-- in-control calibration on an initial prefix
  `-- two-sided Page-CUSUM alarms
```

The score fits location and scale inside each rolling window. That design makes
the statistic less sensitive to pure mean and variance changes and more focused
on higher-order shape changes. It does not remove all finite-sample effects, so
operating behavior must be checked empirically.

## Main Entry Points

| Task | Use |
|---|---|
| Run the standard detector | `cfad.detect()` |
| Configure the detector directly | `cfad.detection.RollingDetector` |
| Process observations one at a time | `cfad.detection.StreamDetector` |
| Compare Gaussian and NIG fits | `cfad.compare_models()` |
| Run leakage-aware temporal evaluation | `cfad.backtest.WalkForwardBacktest` |
| Compute ECF goodness-of-fit diagnostics | `cfad.gof` |
| Sweep windows, frequencies, and thresholds | `cfad.sensitivity` |

## Documentation Map

- [Installation](installation.md): install paths, docs dependencies, and
  verification checks.
- [Quickstart](quickstart.md): complete examples for detection, dates, model
  comparison, and walk-forward evaluation.
- [Detector Guide](guide.md): parameter meanings, output interpretation, and
  practical workflow guidance.
- [Validation](validation.md): current evidence boundary and benchmark record.
- [Mathematics](mathematics.md): score definition, CUSUM layer, and why the
  empirical-residue interpretation was retired.
- [API Reference](api.md): generated reference material with a module map.
- [Contributing](contributing.md): development, docs, testing, and evidence
  practices.
