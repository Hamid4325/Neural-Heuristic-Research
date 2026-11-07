"""Scikit-learn fallback neural heuristic wrapper.

Part of Deliverable 2 for Neural Heuristic Research.
Author: Abdullah Zia
"""

from typing import Any, Optional, Union
import joblib
import numpy as np


def compute_grid_features(grid: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    """Compute (N, N, 29) feature tensor for all cells of a maze.

    Features:
      0..1: Normalised cell coordinates (x/N, y/N)
      2..3: Goal-relative offsets (dx/N, dy/N)
      4..28: Local 5x5 wall context (padded with 1 at boundaries)
    """
    n = grid.shape[0]
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)

    coord_x = (xs / n).astype(np.float32)
    coord_y = (ys / n).astype(np.float32)

    goal_r, goal_c = int(goal[0]), int(goal[1])
    off_x = ((xs - goal_c) / n).astype(np.float32)
    off_y = ((ys - goal_r) / n).astype(np.float32)

    # 5x5 wall context padded with 1 at borders
    padded = np.pad(grid, 2, mode="constant", constant_values=1)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (5, 5))
    wall_feats = windows.reshape(n, n, 25).astype(np.float32)

    feats = np.concatenate(
        [
            coord_x[..., None],
            coord_y[..., None],
            off_x[..., None],
            off_y[..., None],
            wall_feats,
        ],
        axis=-1,
    )
    return feats


class SklearnNeuralHeuristic:
    """Scikit-learn fallback neural heuristic implementing the Heuristic protocol.

    Precomputes features and batch predictions for all cells of the maze upon initialization,
    denormalizes by grid dimension N, clamps to >= 0, and applies the specified mode.
    """
    name: str = "neural_fallback"

    def __init__(
        self,
        regressor_path_or_model: Union[str, Any],
        grid: np.ndarray,
        goal: tuple[int, int],
        alpha: float = 1.0,
        mode: str = "raw",  # 'raw', 'scaled', 'clipped_manhattan'
    ):
        self.alpha = float(alpha)
        self.mode = mode
        self.grid = grid
        self.goal = goal
        self.n = grid.shape[0]

        if isinstance(regressor_path_or_model, str):
            self.model = joblib.load(regressor_path_or_model)
        else:
            self.model = regressor_path_or_model

        self._pred_grid = self._predict_all_cells()

    def _predict_all_cells(self) -> np.ndarray:
        """Compute full N x N heuristic distance array."""
        feats = compute_grid_features(self.grid, self.goal)
        feats_flat = feats.reshape(-1, 29)

        # Batch predict normalized h*/N
        norm_preds = self.model.predict(feats_flat)
        norm_grid = norm_preds.reshape(self.n, self.n)

        # Denormalise by grid dimension N
        raw_pred_cells = norm_grid * self.n

        # Non-negativity clamp
        clamped_preds = np.maximum(0.0, raw_pred_cells)
        return clamped_preds

    def __call__(self, r: int, c: int) -> float:
        # If coordinates are outside grid or inside a wall, return Manhattan distance
        if not (0 <= r < self.n and 0 <= c < self.n) or self.grid[r, c] == 1:
            return float(abs(r - self.goal[0]) + abs(c - self.goal[1]))

        val = float(self._pred_grid[r, c])

        if self.mode == "raw":
            return max(0.0, val)
        elif self.mode == "scaled":
            return max(0.0, self.alpha * val)
        elif self.mode == "clipped_manhattan":
            manhattan = float(abs(r - self.goal[0]) + abs(c - self.goal[1]))
            return min(max(0.0, self.alpha * val), manhattan)
        else:
            raise ValueError(f"Unknown heuristic mode '{self.mode}'; expected 'raw', 'scaled', or 'clipped_manhattan'.")
