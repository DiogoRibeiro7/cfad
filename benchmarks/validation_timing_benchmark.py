"""Post-result timing correction for the comparative validation study."""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import kurtosis, skew

from benchmarks.validation_study_benchmark import (
    ecf_l2,
    energy_distance,
    gaussian_mmd,
    reference_bandwidth,
    standardize,
    wasserstein_1,
)

PROTOCOL_PATH = Path("paper/validation_timing_protocol.md")
RESULTS_DIR = Path("benchmarks/results_validation_timing")
WINDOWS = (30, 60, 120)
N_INPUTS = 200
N_WARMUP = 10
SEED_BASE = 5_001_000


def _kurtosis_distance(reference_raw: np.ndarray, sample_raw: np.ndarray) -> float:
    reference = standardize(reference_raw)
    sample = standardize(sample_raw)
    return abs(
        float(kurtosis(sample, fisher=True, bias=False))
        - float(kurtosis(reference, fisher=True, bias=False))
    )


def _skewness_distance(reference_raw: np.ndarray, sample_raw: np.ndarray) -> float:
    reference = standardize(reference_raw)
    sample = standardize(sample_raw)
    return abs(float(skew(sample, bias=False)) - float(skew(reference, bias=False)))


def _joint_moment_distance(reference_raw: np.ndarray, sample_raw: np.ndarray) -> float:
    reference = standardize(reference_raw)
    sample = standardize(sample_raw)
    ds = float(skew(sample, bias=False)) - float(skew(reference, bias=False))
    dk = float(kurtosis(sample, fisher=True, bias=False)) - float(
        kurtosis(reference, fisher=True, bias=False)
    )
    return float(np.hypot(ds, dk))


def _empirical_ecf_l2(reference_raw: np.ndarray, sample_raw: np.ndarray) -> float:
    return ecf_l2(standardize(reference_raw), standardize(sample_raw))


def _energy_distance(reference_raw: np.ndarray, sample_raw: np.ndarray) -> float:
    return energy_distance(standardize(reference_raw), standardize(sample_raw))


def _gaussian_mmd(reference_raw: np.ndarray, sample_raw: np.ndarray) -> float:
    reference = standardize(reference_raw)
    sample = standardize(sample_raw)
    return gaussian_mmd(reference, sample, reference_bandwidth(reference))


def _wasserstein_1(reference_raw: np.ndarray, sample_raw: np.ndarray) -> float:
    return wasserstein_1(standardize(reference_raw), standardize(sample_raw))


METHODS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "kurtosis_distance": _kurtosis_distance,
    "skewness_distance": _skewness_distance,
    "joint_moment_distance": _joint_moment_distance,
    "empirical_ecf_l2": _empirical_ecf_l2,
    "energy_distance": _energy_distance,
    "gaussian_mmd": _gaussian_mmd,
    "wasserstein_1": _wasserstein_1,
}


def deterministic_inputs(window: int) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(N_INPUTS):
        rng = np.random.default_rng(SEED_BASE + window * 10_000 + index)
        pairs.append((rng.normal(size=300), rng.normal(size=window)))
    return pairs


def summarize_ns(values: list[int]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    return {
        "median_us": float(np.median(arr) / 1_000.0),
        "mean_us": float(np.mean(arr) / 1_000.0),
        "iqr_us": float((q75 - q25) / 1_000.0),
        "p95_us": float(np.percentile(arr, 95.0) / 1_000.0),
    }


def run_timing() -> tuple[list[dict[str, object]], dict[str, object]]:
    if "frozen before timing results" not in PROTOCOL_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("timing protocol is not frozen")
    rows: list[dict[str, object]] = []
    for window in WINDOWS:
        pairs = deterministic_inputs(window)
        for name, method in METHODS.items():
            for _ in range(N_WARMUP):
                method(*pairs[0])
            elapsed_ns: list[int] = []
            for reference, sample in pairs:
                started = time.perf_counter_ns()
                value = method(reference, sample)
                elapsed_ns.append(time.perf_counter_ns() - started)
                if not np.isfinite(value):
                    raise RuntimeError(f"non-finite timing-call result for {name}")
            rows.append({"method": name, "window": window, **summarize_ns(elapsed_ns)})
    provenance: dict[str, object] = {
        "protocol": str(PROTOCOL_PATH),
        "windows": list(WINDOWS),
        "n_inputs": N_INPUTS,
        "n_warmup": N_WARMUP,
        "seed_base": SEED_BASE,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "runner_os": os.environ.get("RUNNER_OS"),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
    }
    return rows, provenance


def main() -> None:
    rows, provenance = run_timing()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "validation_method_timing.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS_DIR / "timing_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
