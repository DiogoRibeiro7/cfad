"""Prospectively frozen Monte Carlo benchmark for the corrected CFAD detector."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import kurtosis, skewnorm

from cfad.detection import RollingDetector

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path(__file__).with_name("corrected_protocol.json")
RESULTS_DIR = Path(__file__).with_name("results")


@dataclass(frozen=True)
class ScoredSeries:
    """Standardized rolling statistic and its observation endpoints."""

    z: np.ndarray
    end_indices: np.ndarray
    n_calibration: int


def load_protocol() -> dict[str, Any]:
    """Load the frozen benchmark protocol."""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen-before-corrected-results":
        raise RuntimeError("benchmark protocol is not marked frozen")
    return protocol


def _rescale_theoretical_t(draws: np.ndarray, df: float, sigma: float) -> np.ndarray:
    """Rescale Student-t draws to the requested theoretical standard deviation."""
    return np.asarray(draws, dtype=np.float64) * sigma / math.sqrt(df / (df - 2.0))


def generate_series(
    scenario: str,
    seed: int,
    protocol: dict[str, Any],
) -> np.ndarray:
    """Generate one series from a frozen scenario definition."""
    cfg = protocol["series"]
    n = int(cfg["n_observations"])
    change = int(cfg["change_point"])
    sigma = float(cfg["base_sigma"])
    rng = np.random.default_rng(seed)

    if scenario == "null_gaussian":
        return rng.normal(0.0, sigma, n).astype(np.float64)

    if scenario == "null_student_t":
        return _rescale_theoretical_t(rng.standard_t(5.0, n), 5.0, sigma)

    left = rng.normal(0.0, sigma, change).astype(np.float64)
    post_n = n - change

    if scenario == "shape_student_t":
        right = _rescale_theoretical_t(rng.standard_t(4.0, post_n), 4.0, sigma)
    elif scenario == "shape_skew":
        raw = np.asarray(
            skewnorm.rvs(8.0, size=post_n, random_state=rng),
            dtype=np.float64,
        )
        right = (raw - float(np.mean(raw))) / float(np.std(raw, ddof=1)) * sigma
    elif scenario == "mean_shift":
        right = rng.normal(1.5 * sigma, sigma, post_n).astype(np.float64)
    elif scenario == "variance_shift":
        right = rng.normal(0.0, 1.75 * sigma, post_n).astype(np.float64)
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    return np.concatenate([left, right]).astype(np.float64)


def _rolling_windows(values: np.ndarray, window: int, step: int) -> np.ndarray:
    """Return stepped rolling windows without copying when possible."""
    return sliding_window_view(values, window_shape=window)[::step]


def _baseline_statistic(
    values: np.ndarray,
    method: str,
    window: int,
    step: int,
) -> np.ndarray:
    """Compute one rolling baseline statistic."""
    windows = _rolling_windows(values, window, step)
    if method == "rolling_mean":
        return np.mean(windows, axis=1, dtype=np.float64)
    if method == "rolling_log_variance":
        variance = np.var(windows, axis=1, ddof=1, dtype=np.float64)
        return np.log(np.maximum(variance, 1e-18))
    if method == "rolling_excess_kurtosis":
        return np.asarray(
            kurtosis(windows, axis=1, fisher=True, bias=False),
            dtype=np.float64,
        )
    raise ValueError(f"unknown baseline method: {method}")


def score_series(
    values: np.ndarray,
    method: str,
    protocol: dict[str, Any],
) -> ScoredSeries:
    """Compute and internally standardize one method's rolling statistic."""
    cfg = protocol["detector"]
    window = int(cfg["window"])
    step = int(cfg["step"])

    if method == "cfad_ecf_shape":
        detector = RollingDetector(
            window=window,
            xi_min=-float(cfg["xi_max"]),
            xi_max=float(cfg["xi_max"]),
            n_xi=int(cfg["n_xi"]),
            step=step,
            calibration_frac=float(cfg["calibration_frac"]),
            k=float(cfg["cusum_k"]),
            h=5.0,
        )
        statistic, end_indices = detector.score_windows(values)
    else:
        statistic = _baseline_statistic(values, method, window, step)
        end_indices = np.arange(
            window,
            window + step * len(statistic),
            step,
            dtype=np.int64,
        )

    n_cal = max(10, int(float(cfg["calibration_frac"]) * len(statistic)))
    n_cal = min(n_cal, len(statistic))
    calibration = np.asarray(statistic[:n_cal], dtype=np.float64)
    mu = float(np.mean(calibration))
    sigma = float(np.std(calibration, ddof=1)) if n_cal > 1 else 0.0
    sigma = max(sigma, 1e-12)
    z = (np.asarray(statistic, dtype=np.float64) - mu) / sigma
    return ScoredSeries(z=z, end_indices=end_indices, n_calibration=n_cal)


def cusum_alarm_indices(z: np.ndarray, k: float, h: float) -> np.ndarray:
    """Apply a reset-on-alarm two-sided Page-CUSUM to standardized scores."""
    positive = 0.0
    negative = 0.0
    alarms: list[int] = []
    for index, value in enumerate(np.asarray(z, dtype=np.float64)):
        positive = max(0.0, positive + float(value) - k)
        negative = max(0.0, negative - float(value) - k)
        if positive > h or negative > h:
            alarms.append(index)
            positive = 0.0
            negative = 0.0
    return np.asarray(alarms, dtype=np.int64)


def has_post_calibration_alarm(scored: ScoredSeries, alarms: np.ndarray) -> bool:
    """Return whether any alarm occurs after the calibration score block."""
    return bool(np.any(alarms >= scored.n_calibration))


def calibrate_thresholds(protocol: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Choose method-specific h on a disjoint Gaussian-null simulation set."""
    mc = protocol["monte_carlo"]
    methods = list(protocol["methods"])
    n_rep = int(mc["threshold_calibration_replicates"])
    seed_base = int(mc["threshold_seed_base"])
    h_grid = [float(value) for value in mc["h_grid"]]
    target = float(mc["target_series_false_alarm_rate"])
    k = float(protocol["detector"]["cusum_k"])

    scored_by_method: dict[str, list[ScoredSeries]] = {method: [] for method in methods}
    for replicate in range(n_rep):
        values = generate_series("null_gaussian", seed_base + replicate, protocol)
        for method in methods:
            scored_by_method[method].append(score_series(values, method, protocol))

    selected: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for method in methods:
        candidates: list[tuple[float, float]] = []
        for h in h_grid:
            positives = 0
            for scored in scored_by_method[method]:
                alarms = cusum_alarm_indices(scored.z, k=k, h=h)
                positives += int(has_post_calibration_alarm(scored, alarms))
            rate = positives / n_rep
            rows.append(
                {
                    "method": method,
                    "h": h,
                    "calibration_false_alarm_rate": rate,
                    "target": target,
                }
            )
            candidates.append((h, rate))

        best_h, _ = min(candidates, key=lambda item: (abs(item[1] - target), -item[0]))
        selected[method] = float(best_h)

    return selected, rows


def wilson_interval(successes: int, total: int, z_value: float = 1.959963984540054) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    denom = 1.0 + z_value**2 / total
    centre = (p + z_value**2 / (2.0 * total)) / denom
    half = (
        z_value
        * math.sqrt(p * (1.0 - p) / total + z_value**2 / (4.0 * total**2))
        / denom
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate_method_on_scenario(
    method: str,
    scenario: str,
    h: float,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one frozen method/scenario cell over disjoint Monte Carlo seeds."""
    mc = protocol["monte_carlo"]
    series_cfg = protocol["series"]
    n_rep = int(mc["evaluation_replicates"])
    seed_base = int(mc["evaluation_seed_base"])
    change = int(series_cfg["change_point"])
    horizon = int(series_cfg["post_change_horizon"])
    k = float(protocol["detector"]["cusum_k"])

    null_scenarios = {"null_gaussian", "null_student_t"}
    null_alarms = 0
    pre_change_alarms = 0
    detections = 0
    delays: list[int] = []

    scenario_offset = list(protocol["scenarios"].keys()).index(scenario) * 100000
    for replicate in range(n_rep):
        seed = seed_base + scenario_offset + replicate
        values = generate_series(scenario, seed, protocol)
        scored = score_series(values, method, protocol)
        alarms = cusum_alarm_indices(scored.z, k=k, h=h)
        valid = alarms[alarms >= scored.n_calibration]
        alarm_endpoints = scored.end_indices[valid] if valid.size else np.zeros(0, dtype=np.int64)

        if scenario in null_scenarios:
            null_alarms += int(alarm_endpoints.size > 0)
            continue

        pre_change_alarms += int(np.any(alarm_endpoints < change))
        eligible = alarm_endpoints[
            (alarm_endpoints >= change) & (alarm_endpoints <= change + horizon)
        ]
        if eligible.size:
            detections += 1
            delays.append(int(eligible[0] - change))

    result: dict[str, Any] = {
        "method": method,
        "scenario": scenario,
        "h": h,
        "n": n_rep,
    }
    if scenario in null_scenarios:
        rate = null_alarms / n_rep
        lo, hi = wilson_interval(null_alarms, n_rep)
        result.update(
            {
                "false_alarm_rate": rate,
                "false_alarm_ci_low": lo,
                "false_alarm_ci_high": hi,
                "pre_change_false_alarm_rate": math.nan,
                "power": math.nan,
                "power_ci_low": math.nan,
                "power_ci_high": math.nan,
                "median_delay": math.nan,
                "delay_q25": math.nan,
                "delay_q75": math.nan,
            }
        )
    else:
        power = detections / n_rep
        lo, hi = wilson_interval(detections, n_rep)
        delay_array = np.asarray(delays, dtype=np.float64)
        result.update(
            {
                "false_alarm_rate": math.nan,
                "false_alarm_ci_low": math.nan,
                "false_alarm_ci_high": math.nan,
                "pre_change_false_alarm_rate": pre_change_alarms / n_rep,
                "power": power,
                "power_ci_low": lo,
                "power_ci_high": hi,
                "median_delay": float(np.median(delay_array)) if delay_array.size else math.nan,
                "delay_q25": float(np.quantile(delay_array, 0.25)) if delay_array.size else math.nan,
                "delay_q75": float(np.quantile(delay_array, 0.75)) if delay_array.size else math.nan,
            }
        )
    return result


def publication_screen(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    """Apply the prospectively frozen publication-screen criteria."""
    criteria = protocol["publication_screen"]
    by_key = {(row["method"], row["scenario"]): row for row in rows}
    cfad = "cfad_ecf_shape"
    shape_scenarios = ["shape_student_t", "shape_skew"]

    null_ok = all(
        by_key[(cfad, scenario)]["false_alarm_rate"]
        <= float(criteria["max_null_false_alarm_rate_each"])
        for scenario in ("null_gaussian", "null_student_t")
    )
    power_ok = all(
        by_key[(cfad, scenario)]["power"]
        >= float(criteria["min_power_each_shape_alternative"])
        for scenario in shape_scenarios
    )
    delay_ok = (
        by_key[(cfad, "shape_student_t")]["median_delay"]
        <= float(criteria["max_median_delay_shape_student_t"])
        and by_key[(cfad, "shape_skew")]["median_delay"]
        <= float(criteria["max_median_delay_shape_skew"])
    )

    cfad_average_power = float(
        np.mean([by_key[(cfad, scenario)]["power"] for scenario in shape_scenarios])
    )
    moment_methods = ["rolling_mean", "rolling_log_variance"]
    best_moment_average = max(
        float(np.mean([by_key[(method, scenario)]["power"] for scenario in shape_scenarios]))
        for method in moment_methods
    )
    moment_advantage = cfad_average_power - best_moment_average
    moment_ok = moment_advantage >= float(
        criteria["min_average_power_advantage_over_best_moment_baseline"]
    )

    kurtosis_method = "rolling_excess_kurtosis"
    kurtosis_power_ok = all(
        by_key[(cfad, scenario)]["power"]
        >= by_key[(kurtosis_method, scenario)]["power"]
        - float(criteria["kurtosis_noninferiority_power_margin"])
        for scenario in shape_scenarios
    )
    kurtosis_delay_ok = all(
        (
            math.isnan(by_key[(kurtosis_method, scenario)]["median_delay"])
            or by_key[(cfad, scenario)]["median_delay"]
            <= by_key[(kurtosis_method, scenario)]["median_delay"]
            + float(criteria["kurtosis_noninferiority_delay_margin_observations"])
        )
        for scenario in shape_scenarios
    )

    checks = {
        "null_false_alarm_control": bool(null_ok),
        "shape_power": bool(power_ok),
        "shape_delay": bool(delay_ok),
        "moment_baseline_advantage": bool(moment_ok),
        "kurtosis_power_noninferiority": bool(kurtosis_power_ok),
        "kurtosis_delay_noninferiority": bool(kurtosis_delay_ok),
    }
    return {
        "passes_all": all(checks.values()),
        "checks": checks,
        "cfad_average_shape_power": cfad_average_power,
        "best_moment_average_shape_power": best_moment_average,
        "average_power_advantage_over_best_moment_baseline": moment_advantage,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV with stable column order."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_power(rows: list[dict[str, Any]], output: Path) -> None:
    """Plot shape-change power by method."""
    methods = sorted({row["method"] for row in rows})
    scenarios = ["shape_student_t", "shape_skew"]
    x = np.arange(len(scenarios), dtype=np.float64)
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, method in enumerate(methods):
        values = [
            next(row["power"] for row in rows if row["method"] == method and row["scenario"] == scenario)
            for scenario in scenarios
        ]
        ax.bar(x + (index - 1.5) * width, values, width=width, label=method)
    ax.set_xticks(x, ["Student-t shape shift", "Skew shape shift"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Detection probability within frozen horizon")
    ax.set_title("Corrected CFAD benchmark: shape-change power")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_false_alarms(rows: list[dict[str, Any]], output: Path) -> None:
    """Plot stationary false-alarm rates by method."""
    methods = sorted({row["method"] for row in rows})
    scenarios = ["null_gaussian", "null_student_t"]
    x = np.arange(len(scenarios), dtype=np.float64)
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, method in enumerate(methods):
        values = [
            next(
                row["false_alarm_rate"]
                for row in rows
                if row["method"] == method and row["scenario"] == scenario
            )
            for scenario in scenarios
        ]
        ax.bar(x + (index - 1.5) * width, values, width=width, label=method)
    ax.axhline(0.10, linestyle="--", linewidth=1.0, label="10% publication screen")
    ax.set_xticks(x, ["Gaussian stationary", "Student-t stationary"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Series-level false-alarm rate")
    ax.set_title("Corrected CFAD benchmark: stationary false alarms")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    """Run the complete frozen benchmark and save machine-readable evidence."""
    protocol = load_protocol()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    thresholds, threshold_rows = calibrate_thresholds(protocol)
    rows: list[dict[str, Any]] = []
    for method in protocol["methods"]:
        for scenario in protocol["scenarios"]:
            rows.append(
                evaluate_method_on_scenario(
                    method=method,
                    scenario=scenario,
                    h=thresholds[method],
                    protocol=protocol,
                )
            )

    screen = publication_screen(rows, protocol)
    payload = {
        "protocol": protocol,
        "selected_thresholds": thresholds,
        "publication_screen": screen,
        "results": rows,
    }
    (RESULTS_DIR / "corrected_results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_csv(RESULTS_DIR / "corrected_summary.csv", rows)
    write_csv(RESULTS_DIR / "threshold_calibration.csv", threshold_rows)
    plot_power(rows, RESULTS_DIR / "corrected_power.png")
    plot_false_alarms(rows, RESULTS_DIR / "corrected_false_alarms.png")

    print(json.dumps(screen, indent=2))
    print("Selected thresholds:", json.dumps(thresholds, sort_keys=True))
    print(f"Wrote benchmark evidence to {RESULTS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
