# Changelog

All notable changes to cfad are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Replaced the empirical contour-residue proxy with a real-frequency ECF shape
  score: normalized L2 distance from each window ECF to the Gaussian CF fitted
  to that window's mean and sample standard deviation.
- Corrected the scientific interpretation: a finite-sample empirical
  characteristic function is entire, so its exact closed-contour integral is
  zero and cannot identify population-CF branch cuts or poles.
- Corrected Page-CUSUM units so the reference value `k` is dimensionless after
  score standardization.
- Unified batch and streaming scoring so optional Cython extensions change
  performance rather than the definition of the statistic.
- Replaced contour-height sensitivity with real-frequency cutoff sensitivity.
- Updated README, paper, citation, archive metadata, and affiliation to match the
  corrected method and current project status.
- Reattached CI to the repository's actual development branch, `develop`.

### Removed
- Removed the obsolete Cython `contour_quad` extension, which used a real-axis
  proxy inconsistent with the Python score.
- Removed placeholder/unverified PyPI and JOSS publication claims from the
  README.

### Fixed
- Fixed alarm-date indexing to use the final observation of each half-open
  rolling window rather than the exclusive endpoint.
- Removed CPU-specific and unsafe floating-point build flags from distributable
  Cython extensions.

## [0.1.0] — 2025-01-01

### Added
- Rolling empirical characteristic function estimation with optional Cython
  acceleration.
- Sequential anomaly detection with Page-CUSUM.
- Parametric CF models: GaussianCF, NIGCF, CGMYCF, LevyStableCF.
- RollingDetector and StreamDetector classes.
- AnomalyReport dataclass with summary and alarm-date helpers.
- ECF goodness-of-fit tests and model comparison.
- Walk-forward backtesting.
- Sensitivity and bootstrap diagnostics.
- Publication-oriented visualization utilities.
- Multi-asset market helpers.
- CLI entry point.
- Sphinx documentation and Read the Docs configuration.
- Wheel, publish, and security workflow scaffolding.

> **Historical note:** the original 0.1.0 development code described its score
> as an empirical contour residue. That interpretation is superseded by the
> correction documented under **Unreleased** above.
