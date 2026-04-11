# Changelog

All notable changes to cfad will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2025-01-01

### Added
- Rolling empirical characteristic function estimation (Cython-accelerated)
- Rectangular contour integration in the complex ξ-plane
- Residue-based anomaly scoring
- Page-CUSUM sequential detection
- Parametric CF models: GaussianCF, NIGCF, CGMYCF, LevyStableCF
- RollingDetector and StreamDetector classes
- AnomalyReport dataclass with summary(), alarm_dates
- cfad.simulate: synthetic return generators
- cfad.gof: ECF goodness-of-fit tests and AIC comparison
- cfad.backtest: walk-forward backtesting
- cfad.sensitivity: hyperparameter calibration
- cfad.bootstrap: bootstrap confidence intervals
- cfad.viz: publication-quality figures
- cfad.market: multi-asset detection
- CLI entry point: cfad detect / cfad compare
- Sphinx documentation with Read the Docs configuration
- CI/CD: GitHub Actions matrix (Python 3.10/3.11/3.12, Ubuntu/macOS)
- Wheel builds via cibuildwheel
