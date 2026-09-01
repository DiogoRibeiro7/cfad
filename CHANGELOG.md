# Changelog

All notable changes to cfad are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.1] — 2026-08-31

### Fixed
- Repaired the GitHub release build after the public `v0.2.0` tag failed during
  macOS wheel construction because an old cibuildwheel build environment paired
  modern setuptools with an incompatible `packaging` version.
- Added `packaging>=24.2` to the isolated build requirements and upgraded the
  release wheel builder from cibuildwheel 2.19.2 to 4.2.0.
- Disabled matrix fail-fast for release wheels so one platform cannot hide build
  results from the remaining operating systems.
- Aligned CI, README installation guidance, badges, and project URLs with the
  repository's current default branch, `main`.
- Changed PyPI publishing from an automatic post-build workflow to an explicit
  manual workflow requiring a successful wheel-build run ID. GitHub release
  creation no longer implies PyPI publication.

### Release history
- The `v0.2.0` annotated tag remains immutable as the first release attempt.
  Its sdist built successfully, but the wheel workflow failed on macOS and no
  GitHub Release was published from that tag.
- The PyPI follow-on workflow for `v0.2.0` was skipped; no package was published
  to PyPI.
- Version 0.2.1 contains release-engineering fixes only. The statistical method,
  frozen validation results, benchmarks, and scientific interpretation are
  unchanged from 0.2.0.

## [0.2.0] — 2026-08-31

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
- Updated README, software paper, citation metadata, Zenodo metadata, and
  affiliation to match the corrected method and current project status.

### Added
- Frozen v2 sequential-calibration validation with explicit false-alarm,
  first-alarm power, and detection-delay rules.
- Frozen v3 empirical-ECF score validation separating frequency standardization,
  reference-law choice, null robustness, and location/scale specificity.
- Comparative finite-window validation framework covering targeted moments,
  empirical-reference ECF, energy distance, Gaussian-kernel MMD, and
  Wasserstein-1 across registered shape alternatives and null laws.
- Machine-readable evidence records preserving failed confirmatory screens and
  workflow/artifact provenance rather than tuning negative results away.
- Separate methodological-paper scaffold for omnibus-versus-targeted finite-window
  distributional change detection.

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
- Corrected CGMY normalization at the removable `Y = 1` singularity and enforced
  exact normalization at zero frequency.
- Fixed leakage in walk-forward calibration so thresholds are estimated from
  in-control training information only.

### Validation status
- CFAD's frozen v2 and v3 programmes do not establish a statistically validated
  sequential detector or superiority over simpler moment-based shape summaries.
- The comparative finite-window study found strong average performance for
  energy distance, Gaussian MMD, and Wasserstein-1, but no registered omnibus
  method passed the prespecified worst-case window-60 AUC criterion.
- The current comparative benchmark's per-method computational timing column is
  known to be invalid because the combined score call was timed once and divided
  equally across methods. Statistical scores, AUCs, robustness summaries, and
  confirmatory decisions are unaffected; timing will be repaired separately.

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
> correction documented in version 0.2.0.
