"""Benchmark script for CFAD detection performance.

Compares runtime for the C extension (when available) and the pure-Python CUSUM
fallback over increasing sequence lengths.
"""

from __future__ import annotations

import timeit

import numpy as np

from cfad import api, detection


def detect_with_mode(returns: np.ndarray, use_c_ext: bool) -> None:
    original = detection._HAS_C_EXT
    detection._HAS_C_EXT = use_c_ext and hasattr(detection, "_cusum_c")
    try:
        api.detect(returns, window=60, xi_range=(-10.0, 10.0), n_xi=128, h=5.0)
    finally:
        detection._HAS_C_EXT = original


def benchmark_length(length: int, use_c_ext: bool, repeat: int = 3) -> float:
    rng = np.random.default_rng(42)
    returns = rng.normal(loc=0.0, scale=0.01, size=length)
    timer = timeit.Timer(lambda: detect_with_mode(returns, use_c_ext))
    times = timer.repeat(repeat=repeat, number=1)
    return float(np.mean(times))


def main() -> None:
    lengths = [500, 2000, 10000]
    modes = [(True, "C extension"), (False, "Python fallback")]

    print("CFAD detection benchmark")
    print(f"{'length':>8} {'mode':>16} {'time (s)':>12}")
    print("" + "-" * 38)

    for use_c_ext, label in modes:
        if use_c_ext and not hasattr(detection, "_cusum_c"):
            print(f"{'n/a':>8} {label:>16} {'(not available)':>12}")
            continue
        for length in lengths:
            elapsed = benchmark_length(length, use_c_ext)
            print(f"{length:>8} {label:>16} {elapsed:12.4f}")


if __name__ == "__main__":
    main()
    main()
