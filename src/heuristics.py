"""Heuristic wrappers used by the A* search implementation."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from src.features import N_FEATURES, features_for_grid
from src.model import ModelConfig, NeuralHeuristicNet


class Heuristic(Protocol):
    """Callable interface expected by :func:`src.search.astar`."""

    name: str

    def __call__(self, r: int, c: int) -> float:
        ...


class ManhattanHeuristic:
    """Classical admissible heuristic for 4-connected uniform-cost grids."""

    name = "manhattan"

    def __init__(self, goal: tuple[int, int]):
        self.goal = (int(goal[0]), int(goal[1]))

    def __call__(self, r: int, c: int) -> float:
        return float(abs(r - self.goal[0]) + abs(c - self.goal[1]))


class OracleHeuristic:
    """Exact cost-to-goal heuristic computed by Dijkstra during labelling."""

    name = "oracle"

    def __init__(self, h_star: np.ndarray):
        self.h_star = np.asarray(h_star)

    def __call__(self, r: int, c: int) -> float:
        rows, cols = self.h_star.shape
        if not (0 <= r < rows and 0 <= c < cols):
            return float("inf")
        value = float(self.h_star[r, c])
        if not np.isfinite(value) or value < 0.0:
            return float("inf")
        return value


class NeuralHeuristic:
    """Learned heuristic backed by a trained :class:`NeuralHeuristicNet`.

    Predictions are made for every cell of one maze in a single batched pass.
    The checkpoint stores the target standardisation statistics, so the model
    output is converted back to costs in grid-cell units.
    """

    name = "neural"
    _VALID_MODES = {"raw", "scaled", "clipped_manhattan"}

    def __init__(
        self,
        checkpoint_path: str,
        grid: np.ndarray,
        goal: tuple[int, int],
        alpha: float = 1.0,
        mode: str = "raw",
        batch_size: int = 8192,
    ):
        if mode not in self._VALID_MODES:
            choices = ", ".join(sorted(self._VALID_MODES))
            raise ValueError(f"Unknown mode {mode!r}; choose one of: {choices}")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")

        self.grid = np.asarray(grid)
        self.goal = (int(goal[0]), int(goal[1]))
        self.alpha = float(alpha)
        self.mode = mode
        self.batch_size = int(batch_size)
        self.n = int(self.grid.shape[0])
        self._manhattan = ManhattanHeuristic(self.goal)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = dict(checkpoint["model_config"])
        if "hidden_dims" in model_config:
            model_config["hidden_dims"] = tuple(model_config["hidden_dims"])

        self.model = NeuralHeuristicNet(ModelConfig(**model_config))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.target_mean = float(checkpoint.get("target_mean", 0.0))
        self.target_std = float(checkpoint.get("target_std", 1.0))
        self.bin_edges = np.asarray(checkpoint.get("bin_edges", []), dtype=np.float32)

        self._predicted_cells = self._predict_all()

    def _predict_all(self) -> np.ndarray:
        features = features_for_grid(self.grid, self.goal).reshape(-1, N_FEATURES)

        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), self.batch_size):
                batch = torch.from_numpy(features[start : start + self.batch_size])
                y_std, _ = self.model(batch)
                batches.append(y_std.squeeze(-1).cpu().numpy())

        if not batches:
            return np.zeros((self.n, self.n), dtype=np.float32)

        y_std = np.concatenate(batches).astype(np.float32)
        y_normalised = y_std * self.target_std + self.target_mean
        h_cells = np.maximum(y_normalised, 0.0) * self.n
        return h_cells.reshape(self.n, self.n).astype(np.float32)

    def __call__(self, r: int, c: int) -> float:
        manhattan = self._manhattan(r, c)
        if not (0 <= r < self.n and 0 <= c < self.n):
            return manhattan
        if int(self.grid[r, c]) != 0:
            return manhattan

        raw = float(self._predicted_cells[r, c])
        if not np.isfinite(raw):
            return manhattan

        if self.mode == "raw":
            return raw
        if self.mode == "scaled":
            return self.alpha * raw
        return min(self.alpha * raw, manhattan)
