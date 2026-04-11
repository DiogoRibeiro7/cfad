"""Benchmark the full CFAD detect() pipeline over multiple series lengths."""

from __future__ import annotations

import sys
import timeit
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfad import detect

WINDOW = 60
STEP = 5
LENGTHS = [500, 2000, 10000]


def n_windows(t: int, window: int = WINDOW, step: int = STEP) -> int:
    """Return number of rolling windows used by detect()."""
    return (t - window) // step + 1


def benchmark_length(t: int) -> tuple[float, float, float]:
    """Time detect() at length `t` and return mean/std/ms-per-window."""
    returns = np.random.default_rng(0).normal(0.0, 0.01, t)
    times = timeit.repeat(
        stmt=lambda: detect(returns, window=WINDOW, step=STEP),
        number=5,
        repeat=3,
    )
    per_call_ms = (np.asarray(times, dtype=np.float64) / 5.0) * 1000.0
    mean_ms = float(np.mean(per_call_ms))
    std_ms = float(np.std(per_call_ms))
    ms_per_window = mean_ms / float(n_windows(t))
    return mean_ms, std_ms, ms_per_window


def format_table(rows: list[tuple[int, float, float, float]]) -> str:
    """Build the benchmark report table."""
    lines = [
        "T       | mean (ms) | std (ms)  | ms/window",
        "--------|-----------|-----------|----------",
    ]
    for t, mean_ms, std_ms, ms_per_window in rows:
        lines.append(
            f"{t:>7} | {mean_ms:>9.1f} | {std_ms:>9.1f} | {ms_per_window:>8.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Run timing benchmark and save results to benchmarks/timing_results.txt."""
    rows: list[tuple[int, float, float, float]] = []
    for t in LENGTHS:
        mean_ms, std_ms, ms_per_window = benchmark_length(t)
        rows.append((t, mean_ms, std_ms, ms_per_window))

    table = format_table(rows)
    print(table)

    output_path = Path(__file__).resolve().parent / "timing_results.txt"
    output_path.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
