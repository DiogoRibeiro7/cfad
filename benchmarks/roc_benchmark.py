"""ROC benchmark for CFAD using synthetic normal and jump return series."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cfad import detect
from cfad.utils import simulate_levy_returns


def simulate_series(n: int, jump: bool = False) -> np.ndarray:
    rng = np.random.default_rng()
    returns = rng.normal(loc=0.0, scale=0.01, size=n)
    if jump:
        jump_index = n // 2
        jump_value = simulate_levy_returns(1, alpha=1.7, beta=0.0, scale=0.2)[0]
        returns[jump_index] += jump_value
    return returns


def compute_cusum_alarm(
    scores: np.ndarray,
    mu0: float,
    sigma0: float,
    h: float,
    k: float = 0.5,
) -> bool:
    s_pos = 0.0
    s_neg = 0.0
    for score in scores:
        z = (score - mu0) / sigma0
        s_pos = max(0.0, s_pos + z - k)
        s_neg = max(0.0, s_neg - z - k)
        if s_pos > h or s_neg > h:
            return True
    return False


def compute_roc(
    labels: np.ndarray, scores: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    tprs = []
    fprs = []
    for thresh in thresholds:
        preds = scores >= thresh
        tp = np.sum(preds & labels)
        fp = np.sum(preds & ~labels)
        fn = np.sum(~preds & labels)
        tn = np.sum(~preds & ~labels)
        tprs.append(tp / max(1, tp + fn))
        fprs.append(fp / max(1, fp + tn))
    return np.array(fprs), np.array(tprs)


def auc_from_curve(fprs: np.ndarray, tprs: np.ndarray) -> float:
    order = np.argsort(fprs)
    return float(np.trapz(tprs[order], fprs[order]))


def main() -> None:
    n_series = 500
    series_length = 200
    thresholds = np.linspace(2.0, 8.0, 13)
    scores = []
    labels = []

    for jump in [False, True]:
        for _ in range(n_series):
            returns = simulate_series(series_length, jump=jump)
            report = detect(returns, window=60, h=10.0)
            scores.append(np.max(report.cusum_pos))
            labels.append(jump)

    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=bool)
    fprs, tprs = compute_roc(labels_arr, scores_arr, thresholds)
    auc_score = auc_from_curve(fprs, tprs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fprs, tprs, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"CFAD ROC curve (AUC = {auc_score:.3f})")
    ax.grid(True, alpha=0.3)
    out_path = Path(__file__).resolve().parent / "roc_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved ROC curve to {out_path}")


if __name__ == "__main__":
    main()
    main()
