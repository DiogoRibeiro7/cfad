import numpy as np

from benchmarks import validation_timing_benchmark as timing
from benchmarks.validation_study_benchmark import score_all


def test_timing_methods_match_registered_method_names() -> None:
    assert tuple(timing.METHODS) == (
        "kurtosis_distance",
        "skewness_distance",
        "joint_moment_distance",
        "empirical_ecf_l2",
        "energy_distance",
        "gaussian_mmd",
        "wasserstein_1",
    )


def test_deterministic_inputs_are_reproducible() -> None:
    first = timing.deterministic_inputs(30)
    second = timing.deterministic_inputs(30)
    assert len(first) == timing.N_INPUTS
    assert np.array_equal(first[0][0], second[0][0])
    assert np.array_equal(first[0][1], second[0][1])


def test_seed_mapping_is_protocol_exact() -> None:
    reference, sample = timing.deterministic_inputs(60)[7]
    rng = np.random.default_rng(timing.SEED_BASE + 60 * 10_000 + 7)
    assert np.array_equal(reference, rng.normal(size=300))
    assert np.array_equal(sample, rng.normal(size=60))


def test_each_method_matches_frozen_score_all() -> None:
    reference, sample = timing.deterministic_inputs(60)[0]
    expected = score_all(reference, sample)
    for name, method in timing.METHODS.items():
        assert np.isclose(method(reference, sample), expected[name], rtol=1e-12, atol=1e-12)


def test_each_method_returns_finite_value() -> None:
    reference, sample = timing.deterministic_inputs(60)[0]
    for method in timing.METHODS.values():
        assert np.isfinite(method(reference, sample))


def test_summary_reports_nonnegative_timings() -> None:
    summary = timing.summarize_ns([1_000, 2_000, 3_000, 4_000])
    assert summary["median_us"] == 2.5
    assert all(value >= 0 for value in summary.values())
