# API Reference

## High-Level API

::: cfad.api
    options:
      members: true

## Detection Objects

::: cfad.detection.RollingDetector
    options:
      members: true

::: cfad.detection.AnomalyReport
    options:
      members: true

## ECF and Scoring Functions

::: cfad.empirical_cf.ecf_at

::: cfad.empirical_cf.rolling_ecf

::: cfad.contour.ecf_residue_scores

::: cfad.contour.rectangular_contour

::: cfad.residue_score.normalise_scores

::: cfad.residue_score.rolling_pvalue

::: cfad.residue_score.threshold_by_fpr

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
