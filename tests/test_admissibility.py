"""Tests for the admissibility analysis helpers."""

import numpy as np
import pytest

from src.admissibility import (
    evaluate_modes,
    heuristic_values,
    overestimation_cdf,
    overestimation_metrics,
    reconstruct_split_assignment,
    summarize_by_mode,
)


def test_split_assignment_matches_pipeline() -> None:
    assignment = reconstruct_split_assignment(10, seed=42)

    assert assignment.tolist() == [
        "train",
        "val",
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "test",
        "val",
    ]


def test_split_counts_are_deterministic_and_disjoint() -> None:
    first = reconstruct_split_assignment(1876, seed=42)
    second = reconstruct_split_assignment(1876, seed=42)

    np.testing.assert_array_equal(first, second)
    assert int(np.sum(first == "train")) == 1313
    assert int(np.sum(first == "val")) == 281
    assert int(np.sum(first == "test")) == 282
    assert set(first) == {"train", "val", "test"}


def test_heuristic_values_apply_modes() -> None:
    predictions = np.array([2.0, 4.0])
    manhattan = np.array([1.0, 3.0])

    np.testing.assert_allclose(
        heuristic_values(predictions, manhattan),
        predictions,
    )
    np.testing.assert_allclose(
        heuristic_values(predictions, manhattan, alpha=0.5, mode="scaled"),
        [1.0, 2.0],
    )
    np.testing.assert_allclose(
        heuristic_values(
            predictions, manhattan, alpha=1.0, mode="clipped_manhattan"
        ),
        [1.0, 3.0],
    )

    with pytest.raises(ValueError):
        heuristic_values(predictions, manhattan, mode="bad")
    with pytest.raises(ValueError):
        heuristic_values(predictions, manhattan, alpha=-0.1)


def test_overestimation_metrics_hand_computed() -> None:
    predictions = np.array([1.0, 2.0, 3.0, 4.0])
    true_cost = np.array([1.0, 2.0, 2.0, 4.0])
    manhattan = np.array([1.0, 1.0, 1.0, 1.0])

    raw = overestimation_metrics(predictions, true_cost, manhattan)
    assert raw["n_cells"] == 4
    assert raw["fraction_over"] == pytest.approx(0.25)
    assert raw["mean_overestimation"] == pytest.approx(0.25)
    assert raw["max_overestimation"] == pytest.approx(1.0)
    assert raw["mae"] == pytest.approx(0.25)
    assert raw["rmse"] == pytest.approx(0.5)
    assert 0.0 <= raw["p95_overestimation"] <= 1.0

    scaled = overestimation_metrics(
        predictions, true_cost, manhattan, alpha=0.5, mode="scaled"
    )
    assert scaled["fraction_over"] == 0.0
    assert scaled["max_overestimation"] == 0.0

    clipped = overestimation_metrics(
        predictions,
        true_cost,
        manhattan,
        alpha=1.0,
        mode="clipped_manhattan",
    )
    assert clipped["fraction_over"] == 0.0
    assert clipped["max_overestimation"] == 0.0


def test_clipped_mode_is_admissible_for_random_predictions() -> None:
    rng = np.random.default_rng(0)
    predictions = rng.uniform(0.0, 20.0, size=1000)
    true_cost = rng.uniform(0.0, 20.0, size=1000)
    manhattan = rng.uniform(0.0, 1.0, size=1000) * true_cost

    clipped = heuristic_values(
        predictions,
        manhattan,
        alpha=1.0,
        mode="clipped_manhattan",
    )

    assert np.all(clipped <= true_cost + 1e-9)


def test_evaluate_modes_returns_one_row_per_configuration() -> None:
    predictions = np.array([1.0, 2.0])
    true_cost = np.array([1.0, 2.0])
    manhattan = np.array([1.0, 1.0])

    rows = evaluate_modes(
        predictions,
        true_cost,
        manhattan,
        alphas=(0.5, 1.0),
    )

    assert len(rows) == 5
    assert [row["mode"] for row in rows] == [
        "raw",
        "scaled",
        "clipped_manhattan",
        "scaled",
        "clipped_manhattan",
    ]


def test_cdf_is_sorted_and_monotonic() -> None:
    predictions = np.array([1.0, 2.0, 3.0])
    true_cost = np.array([1.0, 1.0, 1.0])
    manhattan = np.array([1.0, 1.0, 1.0])

    values, cumulative = overestimation_cdf(
        predictions, true_cost, manhattan
    )

    assert np.all(np.diff(values) >= 0.0)
    assert cumulative[0] > 0.0
    assert cumulative[-1] == pytest.approx(1.0)


def test_overestimation_metrics_rejects_empty_arrays() -> None:
    with pytest.raises(ValueError):
        overestimation_metrics(
            np.array([]), np.array([]), np.array([])
        )


def test_summarize_by_mode_validates_required_keys() -> None:
    rows = evaluate_modes(
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0]),
        alphas=(1.0,),
    )

    assert summarize_by_mode(rows) == rows
    with pytest.raises(ValueError):
        summarize_by_mode([{"mode": "raw"}])
