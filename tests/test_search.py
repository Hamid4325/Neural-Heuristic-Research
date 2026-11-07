"""Unit tests for deterministic A* search engine.

Part of Deliverable 2 for Neural Heuristic Research.
Author: Abdullah Zia
"""

import numpy as np
import pytest
from src.search import (
    SearchResult,
    astar,
    ManhattanHeuristic,
    OracleHeuristic,
    ZeroHeuristic,
)


def test_empty_grid_optimal_cost():
    """Test A* on an empty 3x3 grid reaches goal with optimal Manhattan cost 4.0."""
    grid = np.zeros((3, 3), dtype=np.uint8)
    result = astar(grid, (0, 0), (2, 2), ManhattanHeuristic((2, 2)))
    assert result.found
    assert result.cost == 4.0
    assert result.path[0] == (0, 0)
    assert result.path[-1] == (2, 2)
    assert len(result.path) == 5


def test_blocked_grid_returns_not_found():
    """Test fully blocked grid returns found=False and infinite cost."""
    grid = np.ones((3, 3), dtype=np.uint8)
    grid[0, 0] = 0
    result = astar(grid, (0, 0), (2, 2), ManhattanHeuristic((2, 2)))
    assert not result.found
    assert result.cost == float("inf")
    assert result.path == []


def test_start_equals_goal():
    """Test search when start equals goal returns cost 0 and single-cell path."""
    grid = np.zeros((3, 3), dtype=np.uint8)
    result = astar(grid, (1, 1), (1, 1), ManhattanHeuristic((1, 1)))
    assert result.found
    assert result.cost == 0.0
    assert result.path == [(1, 1)]
    assert result.nodes_expanded == 0


def test_deterministic_node_counts():
    """Test that two runs on the same grid produce identical expanded and generated counts."""
    grid = np.zeros((5, 5), dtype=np.uint8)
    r1 = astar(grid, (0, 0), (4, 4), ManhattanHeuristic((4, 4)))
    r2 = astar(grid, (0, 0), (4, 4), ManhattanHeuristic((4, 4)))
    assert r1.nodes_expanded == r2.nodes_expanded
    assert r1.nodes_generated == r2.nodes_generated
    assert r1.peak_open == r2.peak_open
    assert r1.cost == r2.cost


def test_oracle_never_expands_more_than_manhattan():
    """Test that an oracle heuristic never expands more nodes than Manhattan."""
    grid = np.zeros((5, 5), dtype=np.uint8)
    # Perfect distance h_star on empty grid is exact Manhattan distance to (4, 4)
    r_coords = np.arange(5)[:, None]
    c_coords = np.arange(5)[None, :]
    h_star = np.abs(r_coords - 4) + np.abs(c_coords - 4)
    h_star = h_star.astype(np.float32)

    oracle = OracleHeuristic(h_star)
    r_oracle = astar(grid, (0, 0), (4, 4), oracle)
    r_manhattan = astar(grid, (0, 0), (4, 4), ManhattanHeuristic((4, 4)))

    assert r_oracle.found
    assert r_manhattan.found
    assert r_oracle.cost == r_manhattan.cost
    assert r_oracle.nodes_expanded <= r_manhattan.nodes_expanded


def test_obstacle_navigation():
    """Test search navigates around a wall barrier to reach goal."""
    # 5x5 grid with a vertical wall down column 2, gap at bottom row 4
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[0:4, 2] = 1  # wall blocks rows 0..3 at col 2

    result = astar(grid, (0, 0), (0, 4), ManhattanHeuristic((0, 4)))
    assert result.found
    # Moving (0,0) -> (4,0) -> (4,2) -> (4,4) -> (0,4) or similar:
    # Must go down around row 4: (0,0) down to (4,1), past col 2 at row 4, then up to (0,4)
    # Distance: 4 down + 4 right + 4 up = 12 steps
    assert result.cost == 12.0
    for r, c in result.path:
        assert grid[r, c] == 0


def test_start_or_goal_in_wall():
    """Test starting inside a wall or targeting a wall returns not found."""
    grid = np.zeros((3, 3), dtype=np.uint8)
    grid[0, 0] = 1
    r1 = astar(grid, (0, 0), (2, 2), ManhattanHeuristic((2, 2)))
    assert not r1.found
    assert r1.cost == float("inf")

    grid[0, 0] = 0
    grid[2, 2] = 1
    r2 = astar(grid, (0, 0), (2, 2), ManhattanHeuristic((2, 2)))
    assert not r2.found
    assert r2.cost == float("inf")


def test_dijkstra_baseline():
    """Test that zero heuristic behaves as Dijkstra and finds the optimal path."""
    grid = np.zeros((4, 4), dtype=np.uint8)
    grid[1, 1] = 1
    dijkstra = ZeroHeuristic()
    res_dijkstra = astar(grid, (0, 0), (3, 3), dijkstra)
    res_manhattan = astar(grid, (0, 0), (3, 3), ManhattanHeuristic((3, 3)))

    assert res_dijkstra.found
    assert res_dijkstra.cost == res_manhattan.cost
    # Dijkstra explores uniformly, expanding >= nodes than Manhattan
    assert res_dijkstra.nodes_expanded >= res_manhattan.nodes_expanded
