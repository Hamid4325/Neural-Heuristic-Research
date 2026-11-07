"""Unit tests for scikit-learn fallback heuristic wrapper.

Part of Deliverable 2 for Neural Heuristic Research.
Author: Abdullah Zia
"""

import os
import numpy as np
import pytest
from src.search import astar
from src.sklearn_heuristic import SklearnNeuralHeuristic, compute_grid_features


def test_compute_grid_features():
    """Test feature tensor construction matches (N, N, 29) contract."""
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[0, :] = 1  # top boundary wall
    goal = (9, 9)

    feats = compute_grid_features(grid, goal)
    assert feats.shape == (10, 10, 29)
    assert feats.dtype == np.float32

    # Coordinates
    assert feats[0, 0, 0] == 0.0  # x / 10
    assert feats[0, 0, 1] == 0.0  # y / 10
    assert feats[9, 9, 0] == 0.9
    assert feats[9, 9, 1] == 0.9

    # Goal offset at goal cell should be 0.0
    assert feats[9, 9, 2] == 0.0
    assert feats[9, 9, 3] == 0.0


def test_sklearn_heuristic_modes():
    """Test raw, scaled, and clipped_manhattan modes."""
    reg_path = "models/fallback_mlp_regressor.joblib"
    if not os.path.exists(reg_path):
        pytest.skip("Fallback regressor not trained yet")

    grid = np.zeros((10, 10), dtype=np.uint8)
    goal = (9, 9)

    h_raw = SklearnNeuralHeuristic(reg_path, grid, goal, alpha=1.0, mode="raw")
    h_scaled = SklearnNeuralHeuristic(reg_path, grid, goal, alpha=0.8, mode="scaled")
    h_clipped = SklearnNeuralHeuristic(reg_path, grid, goal, alpha=1.0, mode="clipped_manhattan")

    # Evaluate at start (0, 0)
    v_raw = h_raw(0, 0)
    v_scaled = h_scaled(0, 0)
    v_clipped = h_clipped(0, 0)

    assert v_raw >= 0.0
    assert v_scaled == pytest.approx(0.8 * v_raw, rel=1e-4)
    manhattan = 18.0
    assert v_clipped <= manhattan


def test_search_with_sklearn_heuristic():
    """Test A* search successfully runs with SklearnNeuralHeuristic."""
    reg_path = "models/fallback_mlp_regressor.joblib"
    if not os.path.exists(reg_path):
        pytest.skip("Fallback regressor not trained yet")

    grid = np.zeros((10, 10), dtype=np.uint8)
    goal = (9, 9)

    h_neural = SklearnNeuralHeuristic(reg_path, grid, goal, alpha=0.9, mode="scaled")
    res = astar(grid, (0, 0), goal, h_neural)

    assert res.found
    assert res.cost == 18.0
    assert res.path[0] == (0, 0)
    assert res.path[-1] == (9, 9)
