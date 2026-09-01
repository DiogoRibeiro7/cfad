# Post-result computational timing protocol

Status: **frozen before timing results**.

## Purpose

The primary comparative validation study is statistically frozen. Its
`mean_seconds_per_window` field is invalid because the original runner timed one
combined `score_all()` call and divided the elapsed time equally across the seven
methods. This protocol repairs only that secondary computational-cost result.
It does not rerun, replace, reinterpret, or tune any statistical score, AUC,
robustness result, or confirmatory decision.

## Methods

Measure the same seven methods used by the comparative study:

- kurtosis distance
- skewness distance
- joint skewness/kurtosis distance
- empirical-reference ECF L2 distance
- energy distance
- Gaussian-kernel MMD
- Wasserstein-1 distance

Each timing call includes the method's own required preprocessing from the raw
reference/sample arrays, including standardisation and any method-specific
reference quantity such as the Gaussian-kernel bandwidth. Shared work is not
precomputed across methods. This estimates standalone wall-clock cost for an
independent call to each statistic.

## Inputs

Use deterministic Gaussian draws with reference size 300 and sample sizes
`30`, `60`, and `120`. For each window size, generate 200 fixed input pairs from
a dedicated seed namespace starting at `5_001_000`. The same fixed input pairs
are used for every method at that window size.

These timing inputs are not inferential Monte Carlo replicates and do not alter
the frozen comparative study.

## Timing procedure

For each method and window size:

1. Run 10 untimed warm-up calls on the first deterministic input pair.
2. Time each of the 200 fixed input pairs once with `time.perf_counter_ns()`.
3. Record elapsed nanoseconds per standalone method call.
4. Report the median, arithmetic mean, interquartile range, and 95th percentile
   across the 200 calls.

The primary computational-cost estimand is the **median microseconds per
standalone method call**. Mean and tail summaries are secondary diagnostics.

## Environment provenance

Record Python, NumPy, SciPy, platform, processor string, runner OS, Git commit,
and workflow-run identifiers. Timing values are environment-specific and are not
claimed to be machine-independent constants.

## Integrity checks

The timing runner must import the existing method implementations from
`benchmarks/validation_study_benchmark.py`; it must not duplicate or modify their
mathematics.

The frozen statistical files are inputs only. The timing study must not write to
or regenerate `validation_scores.csv`, `validation_auc.csv`,
`validation_null_robustness.csv`, `validation_confirmatory_screen.csv`, or the
primary result record.

## Interpretation

This study can compare computational burden among methods under one documented
runner environment. It cannot change the registered statistical conclusion that
no omnibus method passed the full window-60 confirmatory screen.
