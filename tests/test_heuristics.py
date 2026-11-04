"""Tests for Manhattan, oracle and neural heuristic wrappers."""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from src.heuristics import ManhattanHeuristic, NeuralHeuristic, OracleHeuristic
from src.model import ModelConfig, NeuralHeuristicNet


def _make_checkpoint(tmp_path, hidden_dims: tuple[int, ...] = (8,)):
    config = ModelConfig(
        input_dim=29,
        hidden_dims=hidden_dims,
        dropout=0.0,
        activation="relu",
        n_bins=10,
    )
    net = NeuralHeuristicNet(config)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state": net.state_dict(),
            "model_config": asdict(config),
            "target_mean": 0.0,
            "target_std": 1.0,
            "bin_edges": [float(v) for v in np.linspace(0.0, 3.0, 11)[1:-1]],
            "config": {},
            "git_commit": "test",
            "seed": 42,
        },
        path,
    )
    return path


def test_manhattan_values() -> None:
    heuristic = ManhattanHeuristic((2, 2))

    assert heuristic(0, 0) == 4.0
    assert heuristic(2, 2) == 0.0
    assert heuristic(4, 1) == 3.0


def test_oracle_values() -> None:
    h_star = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.float32)
    heuristic = OracleHeuristic(h_star)

    assert heuristic(0, 0) == 0.0
    assert heuristic(2, 2) == 4.0


def test_neural_predictions_are_finite_and_non_negative(tmp_path) -> None:
    grid = np.zeros((6, 6), dtype=np.uint8)
    checkpoint = _make_checkpoint(tmp_path)
    heuristic = NeuralHeuristic(str(checkpoint), grid, (5, 5))

    for r in range(6):
        for c in range(6):
            value = heuristic(r, c)
            assert np.isfinite(value)
            assert value >= 0.0


def test_neural_scaled_mode(tmp_path) -> None:
    grid = np.zeros((6, 6), dtype=np.uint8)
    checkpoint = _make_checkpoint(tmp_path)
    raw = NeuralHeuristic(str(checkpoint), grid, (5, 5), alpha=1.0)
    scaled = NeuralHeuristic(
        str(checkpoint), grid, (5, 5), alpha=0.5, mode="scaled"
    )

    for r in range(6):
        for c in range(6):
            assert np.isclose(scaled(r, c), 0.5 * raw(r, c), atol=1e-6)


def test_neural_clipped_manhattan_mode(tmp_path) -> None:
    grid = np.zeros((6, 6), dtype=np.uint8)
    checkpoint = _make_checkpoint(tmp_path)
    raw = NeuralHeuristic(str(checkpoint), grid, (5, 5), alpha=1.0)
    clipped = NeuralHeuristic(
        str(checkpoint), grid, (5, 5), alpha=1.0, mode="clipped_manhattan"
    )

    for r in range(6):
        for c in range(6):
            manhattan = abs(r - 5) + abs(c - 5)
            assert np.isclose(clipped(r, c), min(raw(r, c), manhattan), atol=1e-6)


def test_walls_and_out_of_bounds_fall_back_to_manhattan(tmp_path) -> None:
    grid = np.zeros((6, 6), dtype=np.uint8)
    grid[0, 0] = 1
    checkpoint = _make_checkpoint(tmp_path)
    heuristic = NeuralHeuristic(str(checkpoint), grid, (5, 5))

    assert heuristic(0, 0) == 10.0
    assert heuristic(-1, 0) == 11.0
    assert heuristic(0, -1) == 11.0


def test_unknown_mode_raises(tmp_path) -> None:
    grid = np.zeros((4, 4), dtype=np.uint8)
    checkpoint = _make_checkpoint(tmp_path)

    with pytest.raises(ValueError):
        NeuralHeuristic(str(checkpoint), grid, (3, 3), mode="bad")
