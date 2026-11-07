"""Unit tests for evaluation harness and split reconstruction.

Part of Deliverable 2 for Neural Heuristic Research.
Author: Abdullah Zia
"""

import os
import numpy as np
import pytest
from src.evaluate import (
    parse_filename_metadata,
    reconstruct_split,
    run_search_evaluation,
)
from src.search import ManhattanHeuristic, OracleHeuristic, ZeroHeuristic


def test_parse_filename_metadata():
    """Test regex parsing of generator, size, and density from filenames."""
    gen, size, density = parse_filename_metadata("mazes_backtracker_100_10.npz")
    assert gen == "backtracker"
    assert size == 100
    assert density == 10

    gen, size, density = parse_filename_metadata("mazes_perlin_40_30.npz")
    assert gen == "perlin"
    assert size == 40
    assert density == 30


def test_deterministic_split_counts():
    """Verify Appendix A split yields exactly 1,313 train, 281 val, 282 test mazes."""
    if not os.path.exists("data/cleaned"):
        pytest.skip("data/cleaned not present")

    files, assignment = reconstruct_split("data/cleaned", seed=42)
    counts = {s: int(np.sum(assignment == s)) for s in ("train", "val", "test")}

    assert counts["train"] == 1313
    assert counts["val"] == 281
    assert counts["test"] == 282


def test_run_search_evaluation_mock(tmp_path):
    """Test evaluation harness on mock mazes produces required CSV and JSON files."""
    grid = np.zeros((5, 5), dtype=np.uint8)
    h_star = np.abs(np.arange(5)[:, None] - 4) + np.abs(np.arange(5)[None, :] - 4)

    mock_mazes = [
        {
            "maze_id": 0,
            "file_name": "mazes_backtracker_20_10.npz",
            "local_index": 0,
            "grid": grid,
            "start": (0, 0),
            "goal": (4, 4),
            "h_star": h_star.astype(np.float32),
            "opt_cost": 8.0,
        }
    ]

    heuristics = {
        "dijkstra": lambda m: ZeroHeuristic(),
        "manhattan": lambda m: ManhattanHeuristic(m["goal"]),
        "oracle": lambda m: OracleHeuristic(m["h_star"]),
    }

    out_dir = str(tmp_path / "eval_test")
    df = run_search_evaluation(mock_mazes, heuristics, output_dir=out_dir, split="test")

    assert len(df) == 3
    assert os.path.exists(os.path.join(out_dir, "test_per_maze.csv"))
    assert os.path.exists(os.path.join(out_dir, "test_summary.json"))

    expected_cols = {
        "split", "maze_id", "file_name", "local_index", "generator", "size", "density",
        "heuristic", "alpha", "found", "cost", "opt_cost", "optimality_ratio",
        "nodes_expanded", "nodes_generated", "heuristic_calls", "runtime_ms", "peak_open"
    }
    assert expected_cols.issubset(set(df.columns))
