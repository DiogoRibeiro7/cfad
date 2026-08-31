"""Prospectively frozen v2 sequential-calibration benchmark for CFAD."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import kurtosis, skewnorm

from cfad.detection import RollingDetector

PROTOCOL_PATH = Path(__file__).with_name("sequential_calibration_protocol_v2.json")
RESULTS_DIR = Path(__file__).with_name("results_v2")
V1_RECORD_PATH = Path(__file__).with_name("v1_failed_calibration_record.json")


@dataclass(frozen=True)
class ScoredSeries:
    """Standardized rolling statistic, endpoints, and calibration size."""

    z: np.ndarray
    end_indices: np.ndarray
    n_calibration: int


def load_protocol() -> dict[str, Any]:
    """Load and validate the frozen v2 protocol."""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen-before-v2-results":
        raise RuntimeError("v2 protocol is not marked frozen-before-v2-results")
    if protocol.get("protocol_version") != "2.0":
        raise RuntimeError("unexpected v2 protocol version")
    return protocol


def _rescale_theoretical_t(draws: np.ndarray, df: float, sigma: float) -> np.ndarray:
    """Rescale Student-t draws to the requested theoretical standard deviation."""
    return np.asarray(draws, dtype=np.float64) * sigma / math.sqrt(df / (df - 2.0))


def generate_series(
    scenario: str,
    seed: int,
    protocol: dict[str, Any],
) -> np.ndarray:
    """Generate one series from a frozen v2 scenario definition."""
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
    """Return stepped rolling windows without unnecessary copies."""
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
    *,
    step_override: int | None = None,
) -> ScoredSeries:
    """Compute a rolling statistic and standardize it from its initial block."""
    cfg = protocol["detector"]
    window = int(cfg["window"])
    step = int(cfg["step"] if step_override is None else step_override)

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

    statistic = np.asarray(statistic, dtype=np.float64)
    if statistic.size < 3:
        raise ValueError("at least three rolling statistics are required")

    n_cal = max(2, int(float(cfg["calibration_frac"]) * len(statistic)))
    n_cal = min(n_cal, len(statistic) - 1)
    calibration = statistic[:n_cal]
    mu0 = float(np.mean(calibration))
    sigma0 = float(np.std(calibration, ddof=1)) if n_cal > 1 else 0.0
    sigma0 = max(sigma0, 1e-12)
    z = (statistic - mu0) / sigma0
    if not np.all(np.isfinite(z)):
        raise RuntimeError(f"non-finite standardized scores for method {method}")

    return ScoredSeries(
        z=np.asarray(z, dtype=np.float64),
        end_indices=np.asarray(end_indices, dtype=np.int64),
        n_calibration=n_cal,
    )


def max_page_cusum_path(z: np.ndarray, k: float) -> float:
    """Maximum two-sided Page-CUSUM path statistic without alarm resets."""
    s_pos = 0.0
    s_neg = 0.0
    path_max = 0.0
    for value in np.asarray(z, dtype=np.float64):
        s_pos = max(0.0, s_pos + float(value) - k)
        s_neg = max(0.0, s_neg - float(value) - k)
        path_max = max(path_max, s_pos, s_neg)
    return float(path_max)


def monitored_path_max(scored: ScoredSeries, k: float) -> float:
    """Return the null path maximum after the score-calibration block."""
    return max_page_cusum_path(scored.z[scored.n_calibration :], k=k)


def first_page_cusum_alarm(
    scored: ScoredSeries,
    *,
    k: float,
    h: float,
) -> tuple[int, int] | None:
    """Return the first monitored alarm as (score index, exclusive endpoint)."""
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be finite and positive")

    s_pos = 0.0
    s_neg = 0.0
    for index in range(scored.n_calibration, len(scored.z)):
        value = float(scored.z[index])
        s_pos = max(0.0, s_pos + value - k)
        s_neg = max(0.0, s_neg - value - k)
        if s_pos > h or s_neg > h:
            return index, int(scored.end_indices[index])
    return None


def classify_changed_alarm(
    endpoint: int | None,
    *,
    change_point: int,
    horizon: int,
) -> tuple[str, int | None]:
    """Classify the first alarm using half-open rolling-window endpoints."""
    if endpoint is None:
        return "none", None
    if endpoint <= change_point:
        return "pre_change_false_alarm", None
    if endpoint <= change_point + horizon:
        return "detection", int(endpoint - change_point)
    return "late_alarm", None


def empirical_threshold(maxima: np.ndarray, quantile: float) -> float:
    """Select the frozen empirical threshold using the higher order statistic."""
    values = np.asarray(maxima, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("maxima must be a non-empty finite one-dimensional array")
    threshold = float(np.quantile(values, quantile, method="higher"))
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise RuntimeError("calibrated threshold must be finite and positive")
    return threshold


def calibrate_thresholds(
    protocol: dict[str, Any],
    *,
    step_override: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Calibrate method-specific thresholds from Gaussian-null path maxima."""
    mc = protocol["monte_carlo"]
    methods = list(protocol["methods"])
    n_rep = int(mc["threshold_calibration_replicates"])
    seed_base = int(mc["threshold_seed_base"])
    quantile = float(mc["threshold_quantile"])
    k = float(protocol["detector"]["cusum_k"])
    step = int(protocol["detector"]["step"] if step_override is None else step_override)

    maxima: dict[str, list[float]] = {method: [] for method in methods}
    rows: list[dict[str, Any]] = []
    for replicate in range(n_rep):
        seed = seed_base + replicate
        values = generate_series("null_gaussian", seed, protocol)
        for method in methods:
            scored = score_series(
                values,
                method,
                protocol,
                step_override=step_override,
            )
            maximum = monitored_path_max(scored, k=k)
            maxima[method].append(maximum)
            rows.append(
                {
                    "step": step,
                    "method": method,
                    "replicate": replicate,
                    "seed": seed,
                    "n_calibration_scores": scored.n_calibration,
                    "first_monitor_endpoint": int(
                        scored.end_indices[scored.n_calibration]
                    ),
                    "max_path_statistic": maximum,
                }
            )

    thresholds = {
        method: empirical_threshold(np.asarray(values), quantile)
        for method, values in maxima.items()
    }
    return thresholds, rows


def wilson_interval(
    successes: int,
    total: int,
    z_value: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a Wilson 95% interval for a binomial proportion."""
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


def evaluate(
    protocol: dict[str, Any],
    thresholds: dict[str, float],
    *,
    step_override: int | None = None,
    scenarios: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate frozen thresholds on independent null and changed-series seeds."""
    methods = list(protocol["methods"])
    scenario_names = list(protocol["scenarios"] if scenarios is None else scenarios)
    all_scenarios = list(protocol["scenarios"])
    mc = protocol["monte_carlo"]
    n_rep = int(mc["evaluation_replicates"])
    seed_base = int(mc["evaluation_seed_base"])
    change = int(protocol["series"]["change_point"])
    horizon = int(protocol["series"]["post_change_horizon"])
    k = float(protocol["detector"]["cusum_k"])
    step = int(protocol["detector"]["step"] if step_override is None else step_override)
    null_scenarios = {"null_gaussian", "null_student_t"}

    state: dict[tuple[str, str], dict[str, Any]] = {}
    for method in methods:
        for scenario in scenario_names:
            state[(method, scenario)] = {
                "null_alarms": 0,
                "pre_change_false_alarms": 0,
                "detections": 0,
                "late_alarms": 0,
                "no_alarms": 0,
                "delays": [],
            }

    for scenario in scenario_names:
        scenario_offset = all_scenarios.index(scenario) * 100000
        for replicate in range(n_rep):
            seed = seed_base + scenario_offset + replicate
            values = generate_series(scenario, seed, protocol)
            for method in methods:
                scored = score_series(
                    values,
                    method,
                    protocol,
                    step_override=step_override,
                )
                alarm = first_page_cusum_alarm(
                    scored,
                    k=k,
                    h=float(thresholds[method]),
                )
                endpoint = None if alarm is None else alarm[1]
                cell = state[(method, scenario)]

                if scenario in null_scenarios:
                    cell["null_alarms"] += int(endpoint is not None)
                    cell["no_alarms"] += int(endpoint is None)
                    continue

                classification, delay = classify_changed_alarm(
                    endpoint,
                    change_point=change,
                    horizon=horizon,
                )
                if classification == "pre_change_false_alarm":
                    cell["pre_change_false_alarms"] += 1
                elif classification == "detection":
                    cell["detections"] += 1
                    cell["delays"].append(int(delay))
                elif classification == "late_alarm":
                    cell["late_alarms"] += 1
                else:
                    cell["no_alarms"] += 1

    rows: list[dict[str, Any]] = []
    for method in methods:
        for scenario in scenario_names:
            cell = state[(method, scenario)]
            row: dict[str, Any] = {
                "step": step,
                "method": method,
                "scenario": scenario,
                "h": float(thresholds[method]),
                "n": n_rep,
            }
            if scenario in null_scenarios:
                successes = int(cell["null_alarms"])
                rate = successes / n_rep
                lo, hi = wilson_interval(successes, n_rep)
                row.update(
                    {
                        "false_alarm_rate": rate,
                        "false_alarm_ci_low": lo,
                        "false_alarm_ci_high": hi,
                        "pre_change_false_alarm_rate": None,
                        "power": None,
                        "power_ci_low": None,
                        "power_ci_high": None,
                        "late_alarm_rate": None,
                        "no_alarm_rate": cell["no_alarms"] / n_rep,
                        "median_delay": None,
                        "delay_q25": None,
                        "delay_q75": None,
                    }
                )
            else:
                detections = int(cell["detections"])
                power = detections / n_rep
                lo, hi = wilson_interval(detections, n_rep)
                delays = np.asarray(cell["delays"], dtype=np.float64)
                row.update(
                    {
                        "false_alarm_rate": None,
                        "false_alarm_ci_low": None,
                        "false_alarm_ci_high": None,
                        "pre_change_false_alarm_rate": cell["pre_change_false_alarms"]
                        / n_rep,
                        "power": power,
                        "power_ci_low": lo,
                        "power_ci_high": hi,
                        "late_alarm_rate": cell["late_alarms"] / n_rep,
                        "no_alarm_rate": cell["no_alarms"] / n_rep,
                        "median_delay": (
                            float(np.median(delays)) if delays.size else None
                        ),
                        "delay_q25": float(np.quantile(delays, 0.25))
                        if delays.size
                        else None,
                        "delay_q75": float(np.quantile(delays, 0.75))
                        if delays.size
                        else None,
                    }
                )
            rows.append(row)
    return rows


def gaussian_validation_gate(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen independent Gaussian-null calibration-attainment gate."""
    target = float(protocol["monte_carlo"]["target_series_false_alarm_rate"])
    max_rate = float(protocol["publication_screen"]["max_null_false_alarm_rate_each"])
    gaussian = {
        row["method"]: row for row in rows if row["scenario"] == "null_gaussian"
    }
    methods: dict[str, Any] = {}
    for method in protocol["methods"]:
        row = gaussian[method]
        contains_target = (
            float(row["false_alarm_ci_low"])
            <= target
            <= float(row["false_alarm_ci_high"])
        )
        point_ok = float(row["false_alarm_rate"]) <= max_rate
        methods[method] = {
            "false_alarm_rate": row["false_alarm_rate"],
            "false_alarm_ci_low": row["false_alarm_ci_low"],
            "false_alarm_ci_high": row["false_alarm_ci_high"],
            "contains_target": bool(contains_target),
            "point_estimate_within_limit": bool(point_ok),
            "passes": bool(contains_target and point_ok),
        }
    return {
        "target": target,
        "methods": methods,
        "cfad_passes": bool(methods["cfad_ecf_shape"]["passes"]),
        "all_methods_pass": bool(all(item["passes"] for item in methods.values())),
    }


def _row(rows: list[dict[str, Any]], method: str, scenario: str) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["method"] == method and row["scenario"] == scenario
    )


def _delay_within(value: Any, limit: float) -> bool:
    return value is not None and np.isfinite(float(value)) and float(value) <= limit


def publication_screen(
    rows: list[dict[str, Any]],
    validation_gate: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Apply the prospectively frozen v2 publication screen."""
    criteria = protocol["publication_screen"]
    cfad = "cfad_ecf_shape"
    shape_scenarios = ["shape_student_t", "shape_skew"]

    calibration_ok = bool(validation_gate["all_methods_pass"])
    null_ok = all(
        float(_row(rows, cfad, scenario)["false_alarm_rate"])
        <= float(criteria["max_null_false_alarm_rate_each"])
        for scenario in ("null_gaussian", "null_student_t")
    )
    power_ok = all(
        float(_row(rows, cfad, scenario)["power"])
        >= float(criteria["min_power_each_shape_alternative"])
        for scenario in shape_scenarios
    )
    delay_ok = _delay_within(
        _row(rows, cfad, "shape_student_t")["median_delay"],
        float(criteria["max_median_delay_shape_student_t"]),
    ) and _delay_within(
        _row(rows, cfad, "shape_skew")["median_delay"],
        float(criteria["max_median_delay_shape_skew"]),
    )

    cfad_average_power = float(
        np.mean(
            [float(_row(rows, cfad, scenario)["power"]) for scenario in shape_scenarios]
        )
    )
    moment_methods = ["rolling_mean", "rolling_log_variance"]
    best_moment_average = max(
        float(
            np.mean(
                [
                    float(_row(rows, method, scenario)["power"])
                    for scenario in shape_scenarios
                ]
            )
        )
        for method in moment_methods
    )
    moment_advantage = cfad_average_power - best_moment_average
    moment_ok = moment_advantage >= float(
        criteria["min_average_power_advantage_over_best_moment_baseline"]
    )

    kurtosis_method = "rolling_excess_kurtosis"
    kurtosis_power_ok = all(
        float(_row(rows, cfad, scenario)["power"])
        >= float(_row(rows, kurtosis_method, scenario)["power"])
        - float(criteria["kurtosis_noninferiority_power_margin"])
        for scenario in shape_scenarios
    )

    kurtosis_delay_ok = True
    delay_margin = float(criteria["kurtosis_noninferiority_delay_margin_observations"])
    for scenario in shape_scenarios:
        cfad_delay = _row(rows, cfad, scenario)["median_delay"]
        kurtosis_delay = _row(rows, kurtosis_method, scenario)["median_delay"]
        if cfad_delay is None:
            kurtosis_delay_ok = False
            break
        if (
            kurtosis_delay is not None
            and float(cfad_delay) > float(kurtosis_delay) + delay_margin
        ):
            kurtosis_delay_ok = False
            break

    checks = {
        "gaussian_calibration_validation": calibration_ok,
        "null_false_alarm_control": bool(null_ok),
        "shape_power": bool(power_ok),
        "shape_delay": bool(delay_ok),
        "moment_baseline_advantage": bool(moment_ok),
        "kurtosis_power_noninferiority": bool(kurtosis_power_ok),
        "kurtosis_delay_noninferiority": bool(kurtosis_delay_ok),
    }
    return {
        "passes_all": bool(all(checks.values())),
        "checks": checks,
        "cfad_average_shape_power": cfad_average_power,
        "best_moment_average_shape_power": best_moment_average,
        "average_power_advantage_over_best_moment_baseline": moment_advantage,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV with a stable union of keys."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Run the frozen v2 calibration, validation, evaluation, and diagnostic."""
    protocol = load_protocol()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    thresholds, maxima_rows = calibrate_thresholds(protocol)
    primary_rows = evaluate(protocol, thresholds)
    gate = gaussian_validation_gate(primary_rows, protocol)
    screen = publication_screen(primary_rows, gate, protocol)

    diagnostic_step = 60
    diagnostic_thresholds, diagnostic_maxima = calibrate_thresholds(
        protocol,
        step_override=diagnostic_step,
    )
    diagnostic_rows = evaluate(
        protocol,
        diagnostic_thresholds,
        step_override=diagnostic_step,
        scenarios=["null_gaussian", "null_student_t"],
    )

    threshold_rows = []
    for method in protocol["methods"]:
        method_maxima = np.asarray(
            [
                row["max_path_statistic"]
                for row in maxima_rows
                if row["method"] == method
            ],
            dtype=np.float64,
        )
        h = float(thresholds[method])
        threshold_rows.append(
            {
                "step": int(protocol["detector"]["step"]),
                "method": method,
                "h": h,
                "calibration_exceedance_rate": float(np.mean(method_maxima > h)),
                "calibration_max_median": float(np.median(method_maxima)),
                "calibration_max_q95_higher": h,
            }
        )

    result_payload = {
        "protocol": protocol,
        "thresholds": thresholds,
        "gaussian_validation_gate": gate,
        "publication_screen": screen,
        "results": primary_rows,
        "non_overlapping_diagnostic": {
            "step": diagnostic_step,
            "thresholds": diagnostic_thresholds,
            "results": diagnostic_rows,
        },
    }

    (RESULTS_DIR / "sequential_calibration_v2_results.json").write_text(
        json.dumps(result_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(RESULTS_DIR / "sequential_calibration_v2_summary.csv", primary_rows)
    _write_csv(RESULTS_DIR / "sequential_calibration_v2_thresholds.csv", threshold_rows)
    _write_csv(RESULTS_DIR / "sequential_calibration_v2_maxima.csv", maxima_rows)
    _write_csv(
        RESULTS_DIR / "sequential_calibration_v2_nonoverlap_summary.csv",
        diagnostic_rows,
    )
    _write_csv(
        RESULTS_DIR / "sequential_calibration_v2_nonoverlap_maxima.csv",
        diagnostic_maxima,
    )

    manifest = {
        "protocol_version": protocol["protocol_version"],
        "protocol_status": protocol["status"],
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "git_sha": os.getenv("GITHUB_SHA"),
        "v1_failed_record_sha256": (
            _sha256(V1_RECORD_PATH) if V1_RECORD_PATH.exists() else None
        ),
        "result_files": sorted(path.name for path in RESULTS_DIR.iterdir()),
    }
    (RESULTS_DIR / "sequential_calibration_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {"thresholds": thresholds, "gate": gate, "screen": screen},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()