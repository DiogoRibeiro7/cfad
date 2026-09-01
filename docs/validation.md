# Validation and Evidence Status

CFAD is a research project with an explicit evidence boundary. The repository
keeps failed validation records because those results constrain future claims
and prevent method changes from rewriting the historical interpretation.

## Current Bottom Line

The implementation is tested and reproducible, but the current evidence does
not establish CFAD as a validated sequential anomaly detector for financial
returns. In the frozen validation programs, the corrected ECF score showed useful
properties in some settings, but it did not satisfy the registered performance
screens and did not generally outperform simpler moment summaries.

That status is intentional documentation, not a temporary README caveat.

## Frozen Validation Results

| Evidence record | What it tested | Outcome |
|---|---|---|
| `benchmarks/v2_failed_calibration_record.json` | Sequential Page-CUSUM calibration and first-alarm behavior. | Failed the publication screen. Gaussian-null calibration was not robust to a stable Student-t in-control law, and shape-change power was weak once false alarms were controlled. |
| `benchmarks/v3_failed_score_validation_record.json` | Score-level ablation without CUSUM, separating frequency standardization and reference-law choice. | Failed the score-validation screen. Standardization fixed variance sensitivity, but the primary empirical-reference score underperformed the registered criteria and a kurtosis comparator on average. |
| `benchmarks/validation_study_primary_result_record.json` | Comparative finite-window validation of omnibus and targeted distributional-change scores. | No registered omnibus method passed all confirmatory criteria at window size 60. The lighter-tail transition was the shared bottleneck. |

## What Can Be Claimed

It is reasonable to claim that CFAD provides:

- a reproducible implementation of rolling ECF discrepancy scoring;
- a sequential Page-CUSUM monitoring layer over standardized scores;
- characteristic-function model comparison and goodness-of-fit utilities;
- benchmark scripts and immutable evidence records for method evaluation;
- optional Cython acceleration for selected computations.

It is not currently supported to claim that CFAD:

- is production validated for financial anomaly detection;
- reliably detects market crises or regime changes out of sample;
- outperforms simpler moment-based shape summaries in general;
- extracts empirical branch cuts, poles, or residues from finite-sample ECFs.

## Reproducing Evidence

The benchmark scripts live in `benchmarks/` and the methodological paper assets
live in `paper/`. The exact runtime requirements vary by benchmark, but the
basic development setup is:

```bash
python -m pip install -e ".[dev]" --no-build-isolation
python setup.py build_ext --inplace
```

Useful entry points include:

```bash
python benchmarks/sequential_calibration_v2.py
python benchmarks/empirical_ecf_v3.py
python benchmarks/validation_study_benchmark.py
```

Run times can be substantial. Treat machine-readable records in `benchmarks/`
as the frozen evidence layer, and avoid overwriting them after inspection.

## Citation Guidance

When citing this repository, cite the software and describe the validation state
plainly. Do not cite negative screens as positive detector validation. If a
future method revision changes the statistical behavior, preserve the old record
and add a new validation record rather than editing the old one.
