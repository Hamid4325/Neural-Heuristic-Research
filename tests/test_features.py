"""Tests for the shared per-cell feature builder."""

import numpy as np

from src.features import (
    N_FEATURES,
    WINDOW,
    build_features_for_maze,
    feature_names,
    features_for_grid,
)


def test_feature_dimensions() -> None:
    assert WINDOW == 5
    assert N_FEATURES == 29
    assert len(feature_names()) == N_FEATURES


def test_features_for_grid_shape_and_metadata() -> None:
    grid = np.zeros((5, 5), dtype=np.uint8)
    goal = (0, 0)

    feats = features_for_grid(grid, goal)

    assert feats.shape == (5, 5, N_FEATURES)
    assert feats.dtype == np.float32

    # Column/row coordinates normalised by N.
    assert np.isclose(feats[0, 0, 0], 0.0)
    assert np.isclose(feats[0, 0, 1], 0.0)
    assert np.isclose(feats[0, 4, 0], 4.0 / 5.0)
    assert np.isclose(feats[4, 0, 1], 4.0 / 5.0)

    # Goal-relative offsets.
    assert np.isclose(feats[0, 4, 2], 4.0 / 5.0)
    assert np.isclose(feats[4, 0, 3], 4.0 / 5.0)


def test_boundary_padding_is_wall() -> None:
    grid = np.zeros((5, 5), dtype=np.uint8)

    feats = features_for_grid(grid, (2, 2))
    window = feats[0, 0, 4:].reshape(WINDOW, WINDOW)

    expected = np.ones((WINDOW, WINDOW), dtype=np.float32)
    expected[2:, 2:] = 0.0
    np.testing.assert_array_equal(window, expected)


def test_interior_wall_window() -> None:
    grid = np.zeros((7, 7), dtype=np.uint8)
    grid[3, 3] = 1

    feats = features_for_grid(grid, (3, 3))
    window = feats[3, 3, 4:].reshape(WINDOW, WINDOW)

    assert window[2, 2] == 1.0
    assert window.sum() == 1.0


def test_build_features_matches_grid_features() -> None:
    grid = np.zeros((5, 5), dtype=np.uint8)
    goal = (4, 4)
    rows, cols = np.mgrid[0:5, 0:5]
    h_star = (np.abs(rows - 4) + np.abs(cols - 4)).astype(np.float32)

    x, y = build_features_for_maze(grid, goal, h_star)
    full = features_for_grid(grid, goal)
    mask = h_star >= 0.0

    np.testing.assert_allclose(x, full[mask])
    np.testing.assert_allclose(y, (h_star[mask] / 5.0).astype(np.float32))
    assert x.shape == (25, N_FEATURES)


def test_invalid_cells_are_excluded() -> None:
    grid = np.zeros((5, 5), dtype=np.uint8)
    goal = (0, 0)
    h_star = np.full((5, 5), -1.0, dtype=np.float32)
    h_star[0, 0] = 0.0

    x, y = build_features_for_maze(grid, goal, h_star)

    assert x.shape == (1, N_FEATURES)
    assert y[0] == 0.0
