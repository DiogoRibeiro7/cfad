import numpy as np

from benchmarks.validation_study_benchmark import (
    NEGATIVE_CONTROL_CELLS,
    SHAPE_CELLS,
    auc_from_scores,
    energy_distance,
    standardize,
    wasserstein_1,
)


def test_auc_perfect_separation() -> None:
    assert auc_from_scores(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0


def test_standardize_removes_location_and_scale() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    z1 = standardize(x)
    z2 = standardize(3.5 + 7.0 * x)
    assert np.allclose(z1, z2)


def test_identity_distances_are_zero() -> None:
    x = np.array([-1.0, 0.0, 0.5, 2.0])
    assert np.isclose(energy_distance(x, x), 0.0)
    assert np.isclose(wasserstein_1(x, x), 0.0)


def test_registered_shape_cell_map() -> None:
    assert len(SHAPE_CELLS) == 6
    assert ("student_t5", "lighter_tails") in SHAPE_CELLS
    assert ("gaussian", "heavier_tails") in SHAPE_CELLS


def test_negative_controls_cover_all_reference_laws() -> None:
    laws = {law for law, _ in NEGATIVE_CONTROL_CELLS}
    assert laws == {"gaussian", "student_t5", "skew4"}
    assert len(NEGATIVE_CONTROL_CELLS) == 6
