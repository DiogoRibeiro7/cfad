"""Frozen comparative finite-window validation benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import kurtosis, rankdata, skew, skewnorm

from cfad.empirical_cf import ecf_at

PROTOCOL_PATH = Path("paper/validation_study_protocol.md")
RESULTS_DIR = Path("benchmarks/results_validation_study")
WINDOWS = (30, 60, 120)
N_REP = 2000
SEED_BASE = 4_001_000
SEED_STRIDE = 100_000
XI = np.linspace(-10.0, 10.0, 128, dtype=np.float64)

METHODS = (
    "kurtosis_distance",
    "skewness_distance",
    "joint_moment_distance",
    "empirical_ecf_l2",
    "energy_distance",
    "gaussian_mmd",
    "wasserstein_1",
)

SHAPE_CELLS = (
    ("gaussian", "heavier_tails"),
    ("student_t5", "lighter_tails"),
    ("gaussian", "positive_skew"),
    ("gaussian", "negative_skew"),
    ("gaussian", "bimodality"),
    ("gaussian", "contamination"),
)

NEGATIVE_CONTROL_CELLS = tuple(
    (law, alt)
    for law in ("gaussian", "student_t5", "skew4")
    for alt in ("mean_shift", "variance_shift")
)


def standardize(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    sd = float(np.std(values, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("sample standard deviation must be positive")
    return (values - float(np.mean(values))) / sd


def _t_unit(rng: np.random.Generator, df: float, n: int) -> np.ndarray:
    return rng.standard_t(df, n) / math.sqrt(df / (df - 2.0))


def draw_law(law: str, rng: np.random.Generator, n: int) -> np.ndarray:
    if law == "gaussian":
        return rng.normal(size=n)
    if law == "student_t5":
        return _t_unit(rng, 5.0, n)
    if law == "skew4":
        raw = np.asarray(skewnorm.rvs(4.0, size=n, random_state=rng), dtype=float)
        return standardize(raw)
    raise ValueError(f"unknown law: {law}")


def draw_alternative(
    source_law: str,
    alternative: str,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    if alternative == "heavier_tails":
        return _t_unit(rng, 4.0, n)
    if alternative == "lighter_tails":
        return rng.normal(size=n)
    if alternative == "positive_skew":
        raw = np.asarray(skewnorm.rvs(8.0, size=n, random_state=rng), dtype=float)
        return standardize(raw)
    if alternative == "negative_skew":
        raw = np.asarray(skewnorm.rvs(-8.0, size=n, random_state=rng), dtype=float)
        return standardize(raw)
    if alternative == "bimodality":
        labels = rng.integers(0, 2, size=n)
        raw = rng.normal(np.where(labels == 0, -1.0, 1.0), 0.45)
        return standardize(raw)
    if alternative == "contamination":
        mask = rng.random(n) < 0.05
        raw = rng.normal(size=n)
        raw[mask] = rng.normal(scale=5.0, size=int(mask.sum()))
        return standardize(raw)
    if alternative == "mean_shift":
        return draw_law(source_law, rng, n) + 1.5
    if alternative == "variance_shift":
        return draw_law(source_law, rng, n) * 1.75
    raise ValueError(f"unknown alternative: {alternative}")


def ecf_l2(reference: np.ndarray, sample: np.ndarray) -> float:
    ref_phi = ecf_at(reference, XI)
    sample_phi = ecf_at(sample, XI)
    area = float(np.trapezoid(np.abs(ref_phi - sample_phi) ** 2, XI))
    return math.sqrt(max(area / 20.0, 0.0))


def _mean_abs_pairwise(x: np.ndarray, y: np.ndarray) -> float:
    xs = np.sort(np.asarray(x, dtype=np.float64))
    ys = np.sort(np.asarray(y, dtype=np.float64))
    total = 0.0
    prefix = np.concatenate(([0.0], np.cumsum(ys)))
    for value in xs:
        idx = int(np.searchsorted(ys, value, side="right"))
        left = value * idx - prefix[idx]
        right = (prefix[-1] - prefix[idx]) - value * (ys.size - idx)
        total += left + right
    return total / (xs.size * ys.size)


def energy_distance(reference: np.ndarray, sample: np.ndarray) -> float:
    cross = _mean_abs_pairwise(reference, sample)
    xx = _mean_abs_pairwise(reference, reference)
    yy = _mean_abs_pairwise(sample, sample)
    return max(2.0 * cross - xx - yy, 0.0)


def reference_bandwidth(reference: np.ndarray) -> float:
    values = np.asarray(reference, dtype=np.float64)
    diffs = np.abs(values[:, None] - values[None, :])
    tri = diffs[np.triu_indices(values.size, k=1)]
    positive = tri[tri > 0]
    if positive.size == 0:
        return 1.0
    return max(float(np.median(positive)), 1e-12)


def gaussian_mmd(reference: np.ndarray, sample: np.ndarray, bandwidth: float) -> float:
    gamma = 1.0 / (2.0 * bandwidth**2)
    xx = np.exp(-gamma * (reference[:, None] - reference[None, :]) ** 2)
    yy = np.exp(-gamma * (sample[:, None] - sample[None, :]) ** 2)
    xy = np.exp(-gamma * (reference[:, None] - sample[None, :]) ** 2)
    value = float(np.mean(xx) + np.mean(yy) - 2.0 * np.mean(xy))
    return math.sqrt(max(value, 0.0))


def wasserstein_1(reference: np.ndarray, sample: np.ndarray) -> float:
    x = np.sort(np.asarray(reference, dtype=np.float64))
    y = np.sort(np.asarray(sample, dtype=np.float64))
    grid = np.unique(np.concatenate([x, y]))
    if grid.size < 2:
        return 0.0
    cdf_x = np.searchsorted(x, grid, side="right") / x.size
    cdf_y = np.searchsorted(y, grid, side="right") / y.size
    return float(np.sum(np.abs(cdf_x[:-1] - cdf_y[:-1]) * np.diff(grid)))


def score_all(reference_raw: np.ndarray, sample_raw: np.ndarray) -> dict[str, float]:
    reference = standardize(reference_raw)
    sample = standardize(sample_raw)
    ref_skew = float(skew(reference, bias=False))
    ref_kurt = float(kurtosis(reference, fisher=True, bias=False))
    sample_skew = float(skew(sample, bias=False))
    sample_kurt = float(kurtosis(sample, fisher=True, bias=False))
    bandwidth = reference_bandwidth(reference)
    return {
        "kurtosis_distance": abs(sample_kurt - ref_kurt),
        "skewness_distance": abs(sample_skew - ref_skew),
        "joint_moment_distance": math.hypot(
            sample_skew - ref_skew, sample_kurt - ref_kurt
        ),
        "empirical_ecf_l2": ecf_l2(reference, sample),
        "energy_distance": energy_distance(reference, sample),
        "gaussian_mmd": gaussian_mmd(reference, sample, bandwidth),
        "wasserstein_1": wasserstein_1(reference, sample),
    }


def auc_from_scores(negative: np.ndarray, positive: np.ndarray) -> float:
    neg = np.asarray(negative, dtype=np.float64)
    pos = np.asarray(positive, dtype=np.float64)
    ranks = rankdata(np.concatenate([neg, pos]), method="average")
    rank_sum_pos = float(np.sum(ranks[neg.size :]))
    u = rank_sum_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (neg.size * pos.size))


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.asarray(a, dtype=np.float64))
    y = np.sort(np.asarray(b, dtype=np.float64))
    grid = np.unique(np.concatenate([x, y]))
    fx = np.searchsorted(x, grid, side="right") / x.size
    fy = np.searchsorted(y, grid, side="right") / y.size
    return float(np.max(np.abs(fx - fy)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if "Frozen design before any new comparative benchmark results." not in (
        PROTOCOL_PATH.read_text(encoding="utf-8")
    ):
        raise RuntimeError("comparative protocol is not frozen")

    null_rows: list[dict[str, Any]] = []
    alt_rows: list[dict[str, Any]] = []
    timing: dict[str, list[float]] = {method: [] for method in METHODS}

    null_laws = ("gaussian", "student_t5", "skew4")
    cell_index = 0
    for window in WINDOWS:
        for law in null_laws:
            for replicate in range(N_REP):
                seed = SEED_BASE + cell_index * SEED_STRIDE + replicate
                rng = np.random.default_rng(seed)
                reference = draw_law(law, rng, 300)
                sample = draw_law(law, rng, window)
                started = time.perf_counter()
                scores = score_all(reference, sample)
                elapsed = time.perf_counter() - started
                for method, value in scores.items():
                    timing[method].append(elapsed / len(METHODS))
                    null_rows.append(
                        {
                            "window": window,
                            "reference_law": law,
                            "replicate": replicate,
                            "seed": seed,
                            "method": method,
                            "score": value,
                        }
                    )
            cell_index += 1

        registered_cells = SHAPE_CELLS + NEGATIVE_CONTROL_CELLS
        for source_law, alternative in registered_cells:
            for replicate in range(N_REP):
                seed = SEED_BASE + cell_index * SEED_STRIDE + replicate
                rng = np.random.default_rng(seed)
                reference = draw_law(source_law, rng, 300)
                sample = draw_alternative(source_law, alternative, rng, window)
                scores = score_all(reference, sample)
                for method, value in scores.items():
                    alt_rows.append(
                        {
                            "window": window,
                            "reference_law": source_law,
                            "alternative": alternative,
                            "replicate": replicate,
                            "seed": seed,
                            "method": method,
                            "score": value,
                        }
                    )
            cell_index += 1

    def null_scores(method: str, law: str, window: int) -> np.ndarray:
        return np.asarray(
            [
                float(row["score"])
                for row in null_rows
                if row["method"] == method
                and row["reference_law"] == law
                and row["window"] == window
            ],
            dtype=np.float64,
        )

    def changed_scores(
        method: str, law: str, alternative: str, window: int
    ) -> np.ndarray:
        return np.asarray(
            [
                float(row["score"])
                for row in alt_rows
                if row["method"] == method
                and row["reference_law"] == law
                and row["alternative"] == alternative
                and row["window"] == window
            ],
            dtype=np.float64,
        )

    auc_rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        for source_law, alternative in SHAPE_CELLS + NEGATIVE_CONTROL_CELLS:
            for method in METHODS:
                auc_rows.append(
                    {
                        "window": window,
                        "reference_law": source_law,
                        "alternative": alternative,
                        "method": method,
                        "auc": auc_from_scores(
                            null_scores(method, source_law, window),
                            changed_scores(method, source_law, alternative, window),
                        ),
                    }
                )

    robustness_rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        for method in METHODS:
            for left, right in (
                ("gaussian", "student_t5"),
                ("gaussian", "skew4"),
                ("student_t5", "skew4"),
            ):
                a = null_scores(method, left, window)
                b = null_scores(method, right, window)
                median_a = float(np.median(a))
                median_b = float(np.median(b))
                ratio = max(median_a, median_b) / max(min(median_a, median_b), 1e-15)
                robustness_rows.append(
                    {
                        "window": window,
                        "method": method,
                        "law_a": left,
                        "law_b": right,
                        "median_ratio": ratio,
                        "ks_statistic": ks_statistic(a, b),
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    for method in METHODS:
        shape60 = [
            float(row["auc"])
            for row in auc_rows
            if row["method"] == method
            and row["window"] == 60
            and row["alternative"] in {cell[1] for cell in SHAPE_CELLS}
        ]
        summary_rows.append(
            {
                "method": method,
                "mean_shape_auc_w60": float(np.mean(shape60)),
                "worst_shape_auc_w60": float(np.min(shape60)),
                "mean_seconds_per_window": float(np.mean(timing[method])),
            }
        )

    targeted = {
        row["method"]: row
        for row in summary_rows
        if row["method"] in {
            "kurtosis_distance",
            "skewness_distance",
            "joint_moment_distance",
        }
    }
    best_targeted_mean = max(float(row["mean_shape_auc_w60"]) for row in targeted.values())
    confirmatory_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        method = str(row["method"])
        if method not in {
            "empirical_ecf_l2",
            "energy_distance",
            "gaussian_mmd",
            "wasserstein_1",
        }:
            continue
        robustness = [
            r
            for r in robustness_rows
            if r["method"] == method and r["window"] == 60
        ]
        controls = [
            float(r["auc"])
            for r in auc_rows
            if r["method"] == method
            and r["window"] == 60
            and r["alternative"] in {"mean_shift", "variance_shift"}
        ]
        checks = {
            "worst_shape_auc": float(row["worst_shape_auc_w60"]) >= 0.70,
            "mean_within_best_targeted": float(row["mean_shape_auc_w60"])
            >= best_targeted_mean - 0.02,
            "null_median_ratio": max(float(r["median_ratio"]) for r in robustness) <= 2.0,
            "null_ks": max(float(r["ks_statistic"]) for r in robustness) <= 0.35,
            "location_scale_specificity": all(0.40 <= value <= 0.60 for value in controls),
        }
        confirmatory_rows.append(
            {
                "method": method,
                "passes_all": all(checks.values()),
                **checks,
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(RESULTS_DIR / "validation_null_scores.csv", null_rows)
    _write_csv(RESULTS_DIR / "validation_changed_scores.csv", alt_rows)
    _write_csv(RESULTS_DIR / "validation_auc.csv", auc_rows)
    _write_csv(RESULTS_DIR / "validation_null_robustness.csv", robustness_rows)
    _write_csv(RESULTS_DIR / "validation_method_summary.csv", summary_rows)
    _write_csv(RESULTS_DIR / "validation_confirmatory_screen.csv", confirmatory_rows)
    manifest = {
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "git_sha": os.getenv("GITHUB_SHA"),
        "replicates_per_cell": N_REP,
        "seed_base": SEED_BASE,
        "seed_stride": SEED_STRIDE,
        "windows": list(WINDOWS),
        "shape_cells": list(SHAPE_CELLS),
        "negative_control_cells": list(NEGATIVE_CONTROL_CELLS),
    }
    (RESULTS_DIR / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(confirmatory_rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
