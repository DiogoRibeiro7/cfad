# Detector Guide

This guide explains how to choose inputs, tune the main parameters, and read the
detector output. It complements the generated [API Reference](api.md).

## Input Data

CFAD expects a one-dimensional return series. In most financial workflows that
means percentage, arithmetic, or log returns computed from a price series before
calling the detector.

```python
import numpy as np

prices = np.asarray([100.0, 101.0, 100.5, 102.0])
log_returns = np.diff(np.log(prices))
```

The detector checks for finite numeric values. Handle missing prices, trading
halts, corporate actions, and calendar alignment before passing returns to
CFAD.

## Parameter Summary

| Parameter | Default | Effect |
|---|---:|---|
| `window` | `60` | Number of returns in each rolling ECF estimate. Larger windows reduce noise but delay local changes. |
| `xi_range` / `xi_min`, `xi_max` | `(-10, 10)` | Real-frequency interval used to compare characteristic functions. Higher cutoffs emphasize finer distributional features but can increase variance. |
| `n_xi` | `128` | Number of real-frequency grid points. Larger grids are smoother but slower. |
| `step` | `1` | Distance between rolling windows. Larger steps reduce runtime and temporal resolution. |
| `calibration_frac` | `0.3` | Prefix fraction used to estimate in-control score mean and scale. This prefix should be credible in-control data. |
| `k` | `0.5` | Page-CUSUM reference value after score standardization. Larger values suppress small sustained deviations. |
| `h` | `5.0` | CUSUM alarm threshold. Larger values reduce alarms and increase detection delay. |

## Choosing a Window

The window controls the tradeoff between score variance and detection delay.

- Use shorter windows when changes are expected to be abrupt and local.
- Use longer windows when return distributions are noisy and the expected
  change is persistent.
- Avoid interpreting a result from one arbitrary window as confirmatory
  evidence. Check sensitivity across plausible windows.

```python
from cfad.sensitivity import window_sensitivity

window_df = window_sensitivity(returns, windows=[40, 60, 90, 120], step=5)
print(window_df)
```

## Choosing the Frequency Range

Characteristic functions encode distributional information over frequency. In
CFAD, the score is an integrated discrepancy over a finite real-frequency grid.

Low to moderate frequencies usually carry more stable finite-sample information.
Very high cutoffs may react to tail and fine-shape differences, but they can
also amplify estimation noise.

```python
from cfad.sensitivity import frequency_sensitivity

freq_df = frequency_sensitivity(
    returns,
    xi_max_values=[5.0, 10.0, 20.0, 40.0],
    window=80,
    step=5,
)
print(freq_df)
```

## Calibration Discipline

`calibration_frac` estimates `mu0` and `sigma0` from the initial part of the
score sequence. That prefix defines the in-control reference for CUSUM.

Good calibration practice:

- choose a prefix that is plausible in-control for the question being asked;
- avoid selecting the prefix after inspecting alarms;
- use walk-forward calibration for temporal evaluation;
- report `mu0`, `sigma0`, `window`, `xi_range`, `n_xi`, `k`, and `h` with any
  result.

For train-only evaluation, prefer `WalkForwardBacktest` over a single full-series
`detect()` run.

## Reading Alarms

`alarm_indices` are indices into the score sequence, not directly into the
original return vector. Use `window_end_indices` to map alarms back to input
positions.

```python
alarm_score_positions = report.alarm_indices
alarm_return_positions = report.window_end_indices[alarm_score_positions] - 1
```

If the input was a pandas `Series`, use:

```python
print(report.alarm_dates)
```

Each alarm resets both CUSUM branches. Multiple alarms after one persistent
change may therefore indicate continued deviation, not multiple independent
events.

## Batch vs Stream Processing

`RollingDetector` scores a complete array. `StreamDetector` ingests one return
at a time and uses the same ECF shape score.

```python
from cfad.detection import StreamDetector

stream = StreamDetector(
    window=60,
    xi_min=-10.0,
    xi_max=10.0,
    n_xi=128,
    warmup=80,
    k=0.5,
    h=5.0,
)

for r in returns:
    state = stream.update(float(r))
    if state["alarm"]:
        print("alarm at observation", state["n_obs"])
```

The stream detector needs enough observations to fill the rolling window and
complete calibration before it emits meaningful scores.

## Common Interpretation Errors

Avoid these mistakes:

- treating any alarm as proof of a structural break;
- tuning `h` until a desired historical event is detected, then reporting the
  alarm as out-of-sample evidence;
- comparing raw alarm counts across different `window`, `step`, or `h`
  settings;
- describing the empirical score as a contour-residue, pole, or branch-cut
  detector.

The score is a finite-window ECF discrepancy. It can be useful, but its evidence
comes from validation against controlled nulls, alternatives, and simpler
baselines.
