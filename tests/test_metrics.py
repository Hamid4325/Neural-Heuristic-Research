"""Unit tests for evaluation metrics (regression, classification, search, admissibility).

Part of Deliverable 2 for Neural Heuristic Research.
Author: Abdullah Zia
"""

import numpy as np
import pytest
from src.metrics import (
    regression_metrics,
    admissibility_metrics,
    classification_metrics,
    search_aggregate,
)


def test_regression_metrics_perfect():
    """Test perfect predictions yield zero error and unit correlation."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = regression_metrics(y, y, grid_size=20.0)

    assert res["mse"] == pytest.approx(0.0)
    assert res["rmse"] == pytest.approx(0.0)
    assert res["mae"] == pytest.approx(0.0)
    assert res["r2"] == pytest.approx(1.0)
    assert res["pearson"] == pytest.approx(1.0)
    assert res["spearman"] == pytest.approx(1.0)
    assert res["mean_signed_error"] == pytest.approx(0.0)
    assert res["mae_cells"] == pytest.approx(0.0)
    assert res["rmse_cells"] == pytest.approx(0.0)


def test_regression_metrics_known_values():
    """Test regression metrics on known hand-calculated values."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 4.0])
    # errors: [+1, 0, +1]
    # abs_errors: [1, 0, 1]
    # mse: (1 + 0 + 1) / 3 = 2/3
    # mae: (1 + 0 + 1) / 3 = 2/3
    # signed: 2/3
    res = regression_metrics(y_true, y_pred, grid_size=30.0)

    assert res["mse"] == pytest.approx(2.0 / 3.0)
    assert res["rmse"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert res["mae"] == pytest.approx(2.0 / 3.0)
    assert res["mean_signed_error"] == pytest.approx(2.0 / 3.0)
    assert res["mae_cells"] == pytest.approx((2.0 / 3.0) * 30.0)
    assert res["max_absolute_error_cells"] == pytest.approx(1.0 * 30.0)


def test_regression_mismatched_length():
    """Test length mismatch raises ValueError."""
    with pytest.raises(ValueError):
        regression_metrics([1.0, 2.0], [1.0])


def test_admissibility_metrics_flags_violations():
    """Test admissibility flags overestimation above epsilon_cells."""
    # N = 20
    # true: 1.0 -> 20 cells
    # pred1: 1.02 -> 20.4 cells (+0.4 cells <= 0.5 epsilon -> no violation)
    # pred2: 1.05 -> 21.0 cells (+1.0 cells > 0.5 epsilon -> VIOLATION)
    # pred3: 0.90 -> 18.0 cells (-2.0 cells <= 0.5 -> no violation)
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([1.02, 1.05, 0.90])

    res = admissibility_metrics(y_true, y_pred, grid_size=20.0, epsilon_cells=0.5)

    assert res["violation_count"] == 1
    assert res["violation_rate"] == pytest.approx(1.0 / 3.0)
    assert res["max_overestimation_cells"] == pytest.approx(1.0)
    # positive overestimations: [0.4, 1.0, 0.0] -> mean = 1.4 / 3
    assert res["mean_overestimation_cells"] == pytest.approx(1.4 / 3.0)


def test_admissibility_metrics_all_admissible():
    """Test when all predictions are <= ground truth, violation rate is 0."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.9, 1.8, 2.5])
    res = admissibility_metrics(y_true, y_pred, grid_size=20.0, epsilon_cells=0.5)

    assert res["violation_count"] == 0
    assert res["violation_rate"] == pytest.approx(0.0)
    assert res["mean_overestimation_cells"] == pytest.approx(0.0)


def test_classification_metrics_perfect():
    """Test 10-bin classification metrics on perfect probability predictions."""
    n_bins = 10
    n_samples = 50
    # 5 samples per bin
    y_true = np.repeat(np.arange(n_bins), 5)
    y_proba = np.zeros((n_samples, n_bins))
    for i, label in enumerate(y_true):
        y_proba[i, label] = 1.0

    res = classification_metrics(y_true, y_proba, n_bins=10)

    assert res["macro_precision"] == pytest.approx(1.0)
    assert res["macro_recall"] == pytest.approx(1.0)
    assert res["macro_f1"] == pytest.approx(1.0)
    assert res["weighted_f1"] == pytest.approx(1.0)
    assert res["auc_roc_ovr_macro"] == pytest.approx(1.0)
    assert res["n_bins"] == 10


def test_search_aggregate():
    """Test search_aggregate correctly calculates metrics and baseline node reduction."""
    rows = [
        # Maze 1
        {"maze_id": 1, "heuristic": "manhattan", "cost": 10.0, "opt_cost": 10.0, "nodes_expanded": 100, "nodes_generated": 150, "runtime_ms": 5.0, "found": True},
        {"maze_id": 1, "heuristic": "neural", "cost": 10.0, "opt_cost": 10.0, "nodes_expanded": 50, "nodes_generated": 80, "runtime_ms": 3.0, "found": True},
        # Maze 2
        {"maze_id": 2, "heuristic": "manhattan", "cost": 20.0, "opt_cost": 20.0, "nodes_expanded": 200, "nodes_generated": 300, "runtime_ms": 10.0, "found": True},
        {"maze_id": 2, "heuristic": "neural", "cost": 20.0, "opt_cost": 20.0, "nodes_expanded": 100, "nodes_generated": 160, "runtime_ms": 6.0, "found": True},
    ]

    summary = search_aggregate(rows, baseline="manhattan")

    assert "manhattan" in summary
    assert "neural" in summary

    # Manhattan
    assert summary["manhattan"]["mean_nodes_expanded"] == pytest.approx(150.0)
    assert summary["manhattan"]["success_rate"] == pytest.approx(1.0)
    assert summary["manhattan"]["mean_relative_node_reduction_vs_baseline"] == pytest.approx(0.0)

    # Neural achieved 50% node reduction on both mazes: (1 - 50/100) = 0.5, (1 - 100/200) = 0.5
    assert summary["neural"]["mean_nodes_expanded"] == pytest.approx(75.0)
    assert summary["neural"]["mean_relative_node_reduction_vs_baseline"] == pytest.approx(0.5)
    assert summary["neural"]["mean_optimality_ratio"] == pytest.approx(1.0)
