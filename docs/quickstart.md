# Quickstart

This page shows three complete examples that cover the main entry points.

## Example 1: Minimal `detect()` Run

```python
import numpy as np
from cfad import detect

rng = np.random.default_rng(42)
returns = rng.normal(0.0, 0.01, 500)

report = detect(returns, window=60, step=5, h=4.0)
print(report.summary())
```

## Example 2: `compare_models()` on Student-t Returns

```python
import numpy as np
from scipy.stats import t
from cfad import compare_models

rng = np.random.default_rng(7)
returns = t.rvs(df=4, loc=0.0, scale=0.01, size=800, random_state=rng)

result = compare_models(returns)
print(result)
```

Interpretation: if `winner` is `"nig"`, the empirical characteristic function
is closer to a non-analytic heavy-tailed model than to the Gaussian baseline for
that sample.

## Example 3: Custom `RollingDetector`

```python
import numpy as np
from cfad.detection import RollingDetector

rng = np.random.default_rng(99)
returns = rng.normal(0.0, 0.01, 600)
returns[350:360] += 0.08

detector = RollingDetector(
    window=80,
    xi_min=-12.0,
    xi_max=12.0,
    n_xi=192,
    height=0.15,
    step=2,
    calibration_frac=0.35,
    h=4.0,
)
report = detector.fit_transform(returns)

print("Alarm indices:", report.alarm_indices)
```
