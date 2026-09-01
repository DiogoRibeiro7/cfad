# Quickstart

This page shows complete, runnable examples for the common workflows. The
examples use synthetic returns so they are deterministic and do not require
network access.

## 1. Run the Default Detector

`detect()` is the high-level entry point. It expects a one-dimensional return
series, not prices. Convert prices to returns before calling it.

```python
import numpy as np
from cfad import detect

rng = np.random.default_rng(42)
returns = np.concatenate(
    [
        rng.normal(0.0, 0.01, 300),
        rng.standard_t(df=4.0, size=180) * 0.01 / np.sqrt(2.0),
    ]
)

report = detect(
    returns,
    window=60,
    xi_range=(-10.0, 10.0),
    n_xi=128,
    step=1,
    calibration_frac=0.35,
    k=0.5,
    h=5.0,
)

print(report.summary())
print("First alarm window:", report.alarm_indices[:1])
```

The returned `AnomalyReport` contains:

| Attribute | Meaning |
|---|---|
| `scores` | Rolling ECF shape scores, one per evaluated window. |
| `cusum_pos` | Positive Page-CUSUM branch on standardized scores. |
| `cusum_neg` | Negative Page-CUSUM branch on standardized scores. |
| `alarm_indices` | Positions in the score sequence where an alarm fired. |
| `window_end_indices` | End positions of the rolling windows in the input series. |
| `mu0`, `sigma0` | Calibration mean and standard deviation estimated from the prefix. |

## 2. Preserve Dates With pandas

If you pass a pandas `Series`, CFAD keeps the date index and exposes alarm
dates through `report.alarm_dates`.

```python
import numpy as np
import pandas as pd
from cfad import detect

rng = np.random.default_rng(7)
dates = pd.bdate_range("2024-01-01", periods=500)
returns = pd.Series(rng.normal(0.0, 0.01, len(dates)), index=dates)
returns.iloc[360:390] = rng.standard_t(df=3.0, size=30) * 0.02

report = detect(returns, window=80, step=2, h=5.5)

print(report.summary())
print(report.alarm_dates)
```

Alarm dates refer to the final observation in the rolling window that triggered
the alarm.

## 3. Inspect Scores as a DataFrame

For analysis, build a diagnostic table from the report.

```python
import pandas as pd

score_dates = returns.index[report.window_end_indices - 1]
diagnostics = pd.DataFrame(
    {
        "score": report.scores,
        "cusum_pos": report.cusum_pos,
        "cusum_neg": report.cusum_neg,
    },
    index=score_dates,
)
diagnostics["alarm"] = False
diagnostics.iloc[report.alarm_indices, diagnostics.columns.get_loc("alarm")] = True

print(diagnostics.tail())
```

Use the score path, not just the binary alarms, when assessing whether a run is
stable. A detector that fires only after extreme threshold tuning is usually not
strong evidence of a robust distributional change.

## 4. Compare Characteristic-Function Models

`compare_models()` fits Gaussian and NIG models and compares real-frequency ECF
distance. It is descriptive model-comparison evidence, not a population
singularity test.

```python
import numpy as np
from scipy.stats import t
from cfad import compare_models

rng = np.random.default_rng(11)
returns = t.rvs(df=4, loc=0.0, scale=0.01, size=800, random_state=rng)

result = compare_models(returns)

print("winner:", result["winner"])
print("Gaussian ECF L2:", result["gaussian"]["ecf_l2"])
print("NIG ECF L2:", result["nig"]["ecf_l2"])
```

## 5. Run a Walk-Forward Backtest

Use `WalkForwardBacktest` when you need train-only calibration. Each fold
estimates calibration parameters on training scores, then applies that frozen
calibration to the test fold.

```python
import numpy as np
from cfad.backtest import WalkForwardBacktest

rng = np.random.default_rng(99)
returns = np.concatenate(
    [
        rng.normal(0.0, 0.01, 500),
        rng.standard_t(df=4.0, size=300) * 0.01 / np.sqrt(2.0),
    ]
)

backtest = WalkForwardBacktest(
    detector_kwargs={
        "window": 80,
        "xi_min": -12.0,
        "xi_max": 12.0,
        "n_xi": 192,
        "step": 2,
        "calibration_frac": 0.35,
        "k": 0.5,
        "h": 5.0,
    },
    n_folds=4,
    train_frac=0.6,
    expanding=True,
)

result = backtest.run(returns)

print(result.summary())
print(result.to_dataframe().head())
```

## 6. Sweep Parameters Before Trusting Alarms

Threshold and window choices materially affect the output. Use sensitivity
helpers to check whether a finding is stable.

```python
from cfad.sensitivity import frequency_sensitivity, threshold_sensitivity

freq = frequency_sensitivity(returns, window=80, step=5)
thresholds = threshold_sensitivity(returns, window=80, step=5)

print(freq)
print(thresholds)
```

Treat these sweeps as diagnostics. They help reveal fragile settings; they do
not validate the detector by themselves.
