"""Frozen v3 score-validation benchmark for CFAD."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kurtosis, rankdata, skew, skewnorm

from cfad.empirical_cf import ecf_at

PROTOCOL_PATH = Path(__file__).with_name("empirical_ecf_protocol_v3.json")
RESULTS_DIR = Path(__file__).with_name("results_v3")


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen-before-v3-results":
        raise RuntimeError("v3 protocol is not frozen")
    if protocol.get("protocol_version") != "3.0":
        raise RuntimeError("unexpected v3 protocol version")
    return protocol


def _rescale_t(draws: np.ndarray, df: float, sigma: float) -> np.ndarray:
    return np.asarray(draws, dtype=np.float64) * sigma / math.sqrt(df / (df - 2.0))


def generate_series(scenario: str, seed: int, protocol: dict[str, Any]) -> np.ndarray:
    cfg = protocol["series"]
    n = int(cfg["n_observations"])
    change = int(cfg["change_point"])
    sigma = float(cfg["base_sigma"])
    rng = np.random.default_rng(seed)

    if scenario == "null_gaussian":
        return rng.normal(0.0, sigma, n).astype(np.float64)
    if scenario == "null_student_t":
        return _rescale_t(rng.standard_t(5.0, n), 5.0, sigma)

    left = rng.normal(0.0, sigma, change).astype(np.float64)
    post_n = n - change
    if scenario == "shape_student_t":
        right = _rescale_t(rng.standard_t(4.0, post_n), 4.0, sigma)
    elif scenario == "shape_skew":
        raw = np.asarray(skewnorm.rvs(8.0, size=post_n, random_state=rng), dtype=float)
        right = (raw - np.mean(raw)) / np.std(raw, ddof=1) * sigma
    elif scenario == "mean_shift":
        right = rng.normal(1.5 * sigma, sigma, post_n)
    elif scenario == "variance_shift":
        right = rng.normal(0.0, 1.75 * sigma, post_n)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return np.concatenate([left, right]).astype(np.float64)


def standardize(sample: np.ndarray) -> np.ndarray:
    values = np.asarray(sample, dtype=np.float64)
    centre = float(np.mean(values))
    scale = float(np.std(values, ddof=1))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("sample standard deviation must be positive")
    return (values - centre) / scale


def normalized_l2(phi_a: np.ndarray, phi_b: np.ndarray, xi: np.ndarray) -> float:
    integrand = np.abs(np.asarray(phi_a) - np.asarray(phi_b)) ** 2
    area = float(np.trapezoid(integrand, xi))
    width = float(xi[-1] - xi[0])
    return math.sqrt(max(area / width, 0.0))


def score_windows(values: np.ndarray, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    series = protocol["series"]
    freq = protocol["frequency_grid"]
    ref_n = int(series["reference_observations"])
    window = int(series["evaluation_window"])
    step = int(series["evaluation_step"])
    xi = np.linspace(
        float(freq["standardized_xi_min"]),
        float(freq["standardized_xi_max"]),
        int(freq["n_xi"]),
        dtype=np.float64,
    )
    raw_xi = np.linspace(-10.0, 10.0, int(freq["n_xi"]), dtype=np.float64)

    reference = np.asarray(values[:ref_n], dtype=np.float64)
    reference_z = standardize(reference)
    reference_ecf = ecf_at(reference_z, xi)
    ref_kurt = float(kurtosis(reference_z, fisher=True, bias=False))
    ref_skew = float(skew(reference_z, bias=False))

    rows: list[dict[str, Any]] = []
    for start in range(ref_n, len(values) - window + 1, step):
        end = start + window
        sample = np.asarray(values[start:end], dtype=np.float64)
        sample_z = standardize(sample)

        raw_ecf = ecf_at(sample, raw_xi)
        mu = float(np.mean(sample))
        sigma = float(np.std(sample, ddof=1))
        raw_gaussian = np.exp(1j * raw_xi * mu - 0.5 * sigma**2 * raw_xi**2)

        z_ecf = ecf_at(sample_z, xi)
        std_gaussian = np.exp(-0.5 * xi**2)
        rows.extend(
            [
                {"method": "legacy_raw_gaussian", "start": start, "end": end,
                 "score": normalized_l2(raw_ecf, raw_gaussian, raw_xi)},
                {"method": "standardized_gaussian", "start": start, "end": end,
                 "score": normalized_l2(z_ecf, std_gaussian, xi)},
                {"method": "standardized_empirical_reference", "start": start,
                 "end": end, "score": normalized_l2(z_ecf, reference_ecf, xi)},
                {"method": "rolling_excess_kurtosis_distance", "start": start,
                 "end": end, "score": abs(float(kurtosis(sample_z, fisher=True,
                 bias=False)) - ref_kurt)},
                {"method": "rolling_skewness_distance", "start": start,
                 "end": end, "score": abs(float(skew(sample_z, bias=False)) - ref_skew)},
            ]
        )
    return rows


def auc_from_scores(negative: np.ndarray, positive: np.ndarray) -> float:
    neg = np.asarray(negative, dtype=np.float64)
    pos = np.asarray(positive, dtype=np.float64)
    if neg.size == 0 or pos.size == 0:
        raise ValueError("AUC requires positive and negative scores")
    combined = np.concatenate([neg, pos])
    ranks = rankdata(combined, method="average")
    rank_sum_pos = float(np.sum(ranks[neg.size :]))
    u = rank_sum_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.asarray(a, dtype=np.float64))
    y = np.sort(np.asarray(b, dtype=np.float64))
    grid = np.sort(np.unique(np.concatenate([x, y])))
    fx = np.searchsorted(x, grid, side="right") / x.size
    fy = np.searchsorted(y, grid, side="right") / y.size
    return float(np.max(np.abs(fx - fy)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = load_protocol()
    methods = list(protocol["methods"])
    scenarios = list(protocol["scenarios"])
    mc = protocol["monte_carlo"]
    n_rep = int(mc["replicates_per_scenario"])
    seed_base = int(mc["seed_base"])
    stride = int(mc["scenario_seed_stride"])
    change = int(protocol["series"]["change_point"])

    all_rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        for replicate in range(n_rep):
            seed = seed_base + scenario_index * stride + replicate
            values = generate_series(scenario, seed, protocol)
            for row in score_windows(values, protocol):
                row.update({"scenario": scenario, "replicate": replicate, "seed": seed})
                all_rows.append(row)

    def scores(method: str, scenario: str, post_only: bool = True) -> np.ndarray:
        return np.asarray([
            float(row["score"])
            for row in all_rows
            if row["method"] == method and row["scenario"] == scenario
            and (not post_only or int(row["start"]) >= change)
        ])

    auc_rows: list[dict[str, Any]] = []
    for method in methods:
        null = scores(method, "null_gaussian")
        for scenario in ["shape_student_t", "shape_skew", "mean_shift", "variance_shift"]:
            value = auc_from_scores(null, scores(method, scenario))
            auc_rows.append({"method": method, "scenario": scenario, "auc": value})

    robustness_rows: list[dict[str, Any]] = []
    for method in methods:
        gauss = scores(method, "null_gaussian")
        student = scores(method, "null_student_t")
        gauss_med = float(np.median(gauss))
        student_med = float(np.median(student))
        ratio = student_med / gauss_med if gauss_med > 0.0 else math.inf
        robustness_rows.append({
            "method": method,
            "gaussian_median": gauss_med,
            "student_t_median": student_med,
            "student_t_to_gaussian_median_ratio": ratio,
            "ks_statistic": ks_statistic(gauss, student),
        })

    by_auc = {(r["method"], r["scenario"]): float(r["auc"]) for r in auc_rows}
    primary = str(protocol["success_screen"]["primary_method"])
    kurt_method = "rolling_excess_kurtosis_distance"
    std_gauss = "standardized_gaussian"
    screen_cfg = protocol["success_screen"]
    shape_scenarios = ["shape_student_t", "shape_skew"]
    primary_shape = [by_auc[(primary, s)] for s in shape_scenarios]
    kurt_shape = [by_auc[(kurt_method, s)] for s in shape_scenarios]
    std_gauss_shape = [by_auc[(std_gauss, s)] for s in shape_scenarios]
    robustness = next(r for r in robustness_rows if r["method"] == primary)
    mean_lo, mean_hi = map(float, screen_cfg["mean_shift_auc_range"])
    var_lo, var_hi = map(float, screen_cfg["variance_shift_auc_range"])
    tolerance = float(
        screen_cfg[
            "require_standardized_empirical_auc_not_worse_than_standardized_gaussian_by_more_than"
        ]
    )
    checks = {
        "shape_auc": all(
            auc >= float(screen_cfg["min_auc_each_shape_alternative"])
            for auc in primary_shape
        ),
        "kurtosis_advantage": float(np.mean(primary_shape) - np.mean(kurt_shape))
        >= float(screen_cfg["min_average_auc_advantage_over_kurtosis"]),
        "null_median_ratio": float(robustness["student_t_to_gaussian_median_ratio"])
        <= float(screen_cfg["max_null_median_ratio_student_t_to_gaussian"]),
        "null_ks": float(robustness["ks_statistic"])
        <= float(screen_cfg["max_null_ks_statistic_gaussian_vs_student_t"]),
        "mean_specificity": mean_lo <= by_auc[(primary, "mean_shift")] <= mean_hi,
        "variance_specificity": var_lo <= by_auc[(primary, "variance_shift")] <= var_hi,
        "not_worse_than_standardized_gaussian": all(
            empirical >= gaussian - tolerance
            for empirical, gaussian in zip(primary_shape, std_gauss_shape)
        ),
    }
    screen = {
        "passes_all": bool(all(checks.values())),
        "checks": checks,
        "primary_average_shape_auc": float(np.mean(primary_shape)),
        "kurtosis_average_shape_auc": float(np.mean(kurt_shape)),
        "average_auc_advantage_over_kurtosis": float(np.mean(primary_shape) - np.mean(kurt_shape)),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(RESULTS_DIR / "empirical_ecf_v3_scores.csv", all_rows)
    _write_csv(RESULTS_DIR / "empirical_ecf_v3_auc.csv", auc_rows)
    _write_csv(RESULTS_DIR / "empirical_ecf_v3_null_robustness.csv", robustness_rows)
    (RESULTS_DIR / "empirical_ecf_v3_screen.json").write_text(
        json.dumps(screen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "git_sha": os.getenv("GITHUB_SHA"),
        "protocol_status": protocol["status"],
    }
    (RESULTS_DIR / "empirical_ecf_v3_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(screen, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
