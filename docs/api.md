# API Reference

This page combines a short API map with generated reference documentation from
the package docstrings.

## Which API Should I Use?

| Workflow | API |
|---|---|
| Standard one-shot detection | `cfad.detect` |
| Custom rolling detector configuration | `cfad.detection.RollingDetector` |
| Online updates | `cfad.detection.StreamDetector` |
| Train/test temporal evaluation | `cfad.backtest.WalkForwardBacktest` |
| ECF model comparison | `cfad.compare_models` |
| ECF goodness-of-fit tests and distances | `cfad.gof` |
| Parameter sweeps | `cfad.sensitivity` |
| Plotting diagnostics | `cfad.viz` |

## Return Objects

`detect()` and `RollingDetector.fit_transform()` return an `AnomalyReport`.
The most important fields are `scores`, `cusum_pos`, `cusum_neg`,
`alarm_indices`, `window_end_indices`, `mu0`, `sigma0`, and `threshold`.

`WalkForwardBacktest.run()` returns a `BacktestResult`, which can be summarized
with `summary()` or converted to a fold-concatenated DataFrame with
`to_dataframe()`.

## High-Level API

::: cfad.api
    options:
      members: true

## Detection Objects

::: cfad.detection.RollingDetector
    options:
      members: true

::: cfad.detection.StreamDetector
    options:
      members: true

::: cfad.detection.AnomalyReport
    options:
      members: true

## Backtesting

::: cfad.backtest.WalkForwardBacktest
    options:
      members: true

::: cfad.backtest.BacktestResult
    options:
      members: true

## ECF and Scoring Functions

::: cfad.empirical_cf.ecf_at

::: cfad.empirical_cf.rolling_ecf

::: cfad.contour.gaussian_ecf_distance_scores

::: cfad.contour.ecf_residue_scores

::: cfad.contour.rectangular_contour

::: cfad.residue_score.normalise_scores

::: cfad.residue_score.rolling_pvalue

::: cfad.residue_score.threshold_by_fpr

## Goodness of Fit

::: cfad.gof.cf_distance

::: cfad.gof.epps_pulley_test

::: cfad.gof.aic_table

::: cfad.gof.rolling_gof

## Sensitivity

::: cfad.sensitivity.window_sensitivity

::: cfad.sensitivity.frequency_sensitivity

::: cfad.sensitivity.threshold_sensitivity

::: cfad.sensitivity.recommend_params

## Model Classes

::: cfad.models.gaussian.GaussianCF
    options:
      members: true

::: cfad.models.nig.NIGCF
    options:
      members: true

::: cfad.models.cgmy.CGMYCF
    options:
      members: true

::: cfad.models.levy_stable.LevyStableCF
    options:
      members: true
