"""Monte Carlo ROC benchmark for CFAD under null vs regime-shift alternatives."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfad import detect
from cfad.utils import simulate_levy_returns

N_SIM = 200
T = 400
WINDOW = 60
STEP = 5
CALIBRATION_FRAC = 0.4
H_GRID = np.linspace(1.0, 10.0, 30)
TAIL_WINDOWS = 60


def positive_detection(report) -> bool:
    """True when any alarm occurs in the last 60 windows."""
    alarms = np.asarray(report.alarm_indices, dtype=np.int64)
    if alarms.size == 0:
        return False
    n_windows = int(len(report.scores))
    cutoff = max(0, n_windows - TAIL_WINDOWS)
    return bool(np.any(alarms >= cutoff))


def simulate_null_series(rng: np.random.Generator) -> np.ndarray:
    """Generate a null Gaussian return series."""
    return rng.normal(0.0, 0.01, T)


def simulate_alt_series(rng: np.random.Generator) -> np.ndarray:
    """Generate alternative series with a final Lévy-stable regime shift."""
    left = rng.normal(0.0, 0.01, 300)
    levy_seed = int(rng.integers(0, 2**32 - 1))
    right = simulate_levy_returns(100, alpha=1.5, scale=0.012, seed=levy_seed)
    return np.concatenate([left, right])


def compute_rates_for_h(
    h: float, null_series: list[np.ndarray], alt_series: list[np.ndarray]
) -> tuple[float, float]:
    """Compute (FPR, TPR) for one threshold h."""
    null_positive = 0
    alt_positive = 0

    for series in null_series:
        report = detect(
            series,
            window=WINDOW,
            step=STEP,
            calibration_frac=CALIBRATION_FRAC,
            h=float(h),
        )
        if positive_detection(report):
            null_positive += 1

    for series in alt_series:
        report = detect(
            series,
            window=WINDOW,
            step=STEP,
            calibration_frac=CALIBRATION_FRAC,
            h=float(h),
        )
        if positive_detection(report):
            alt_positive += 1

    fpr = null_positive / float(N_SIM)
    tpr = alt_positive / float(N_SIM)
    return fpr, tpr


def main() -> None:
    """Run the ROC benchmark, print AUC, and save outputs in benchmarks/."""
    rng = np.random.default_rng(0)
    null_series = [simulate_null_series(rng) for _ in range(N_SIM)]
    alt_series = [simulate_alt_series(rng) for _ in range(N_SIM)]

    fprs: list[float] = []
    tprs: list[float] = []
    for h in H_GRID:
        fpr, tpr = compute_rates_for_h(float(h), null_series, alt_series)
        fprs.append(fpr)
        tprs.append(tpr)

    fpr_arr = np.asarray(fprs, dtype=np.float64)
    tpr_arr = np.asarray(tprs, dtype=np.float64)
    order = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[order]
    tpr_sorted = tpr_arr[order]
    auc = float(np.trapezoid(tpr_sorted, fpr_sorted))

    print(f"AUC = {auc:.3f}")

    out_dir = Path(__file__).resolve().parent
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_sorted, tpr_sorted, marker="o", linewidth=1.5, label=f"CFAD (AUC={auc:.3f})")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("CFAD Detector ROC — Gaussian vs Lévy-stable regime shift")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)

    (out_dir / "roc_auc.txt").write_text(f"{auc}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
