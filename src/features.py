"""Shared per-cell feature builder for training and search-time inference.

The feature vector contains 29 values:

1. Normalised coordinates ``x / N`` and ``y / N`` (2).
2. Goal-relative offsets ``(x - x_goal) / N`` and ``(y - y_goal) / N`` (2).
3. A 5x5 binary wall window centred on the cell, padded with 1 outside the
   grid (25).
"""

import numpy as np

WINDOW = 5
HALF = WINDOW // 2
N_FEATURES = 2 + 2 + WINDOW * WINDOW
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def feature_names() -> list[str]:
    """Return the 29 feature names in column order."""
    return ["coord_x", "coord_y", "goal_dx", "goal_dy"] + [
        f"w{i // WINDOW},{i % WINDOW}" for i in range(WINDOW * WINDOW)
    ]


def features_for_grid(grid: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    """Build the ``(N, N, 29)`` feature tensor for every cell in a maze.

    ``grid`` uses the convention 0 = free cell, 1 = wall. The tensor is built
    for all cells; masking of walls and unreachable cells is the caller's job.
    """
    n = int(grid.shape[0])

    rows, cols = np.mgrid[0:n, 0:n].astype(np.float32)
    coord_x = (cols / n).astype(np.float32)
    coord_y = (rows / n).astype(np.float32)

    goal_r, goal_c = int(goal[0]), int(goal[1])
    off_x = ((cols - goal_c) / n).astype(np.float32)
    off_y = ((rows - goal_r) / n).astype(np.float32)

    padded = np.pad(grid, HALF, mode="constant", constant_values=1)
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (WINDOW, WINDOW)
    )
    wall_features = windows.reshape(n, n, WINDOW * WINDOW).astype(np.float32)

    return np.concatenate(
        [
            coord_x[..., None],
            coord_y[..., None],
            off_x[..., None],
            off_y[..., None],
            wall_features,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def build_features_for_maze(
    grid: np.ndarray,
    goal: tuple[int, int],
    h_star: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` for all reachable free cells of a single maze.

    ``h_star`` is the exact cost-to-goal array; ``-1.0`` (or any negative
    value) marks walls and unreachable cells. The regression target is the
    normalised cost ``h*(n) / N``.
    """
    n = int(grid.shape[0])
    features = features_for_grid(grid, goal)

    mask = h_star >= 0.0
    x = features[mask]
    y = (h_star[mask] / n).astype(np.float32)
    return x, y
