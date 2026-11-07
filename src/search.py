"""A* search implementation with pluggable heuristics.

Part of Deliverable 2 for Neural Heuristic Research.
Author: Abdullah Zia
"""

from dataclasses import dataclass
from typing import Callable, Optional
import heapq
import time
import numpy as np


@dataclass
class SearchResult:
    """Result container for A* search."""
    found: bool
    path: list[tuple[int, int]]
    cost: float
    nodes_expanded: int
    nodes_generated: int
    heuristic_calls: int
    runtime_ms: float
    peak_open: int


class ManhattanHeuristic:
    """Standard Manhattan distance heuristic to goal."""
    name: str = "manhattan"

    def __init__(self, goal: tuple[int, int]):
        self.goal = goal

    def __call__(self, r: int, c: int) -> float:
        return float(abs(r - self.goal[0]) + abs(c - self.goal[1]))


class OracleHeuristic:
    """Perfect distance oracle heuristic from precomputed h_star grid."""
    name: str = "oracle"

    def __init__(self, h_star: np.ndarray):
        self.h_star = h_star

    def __call__(self, r: int, c: int) -> float:
        return float(self.h_star[r, c])


class ZeroHeuristic:
    """Dijkstra baseline heuristic returning 0.0 everywhere."""
    name: str = "dijkstra"

    def __call__(self, r: int, c: int) -> float:
        return 0.0


def astar(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    heuristic: Callable[[int, int], float],
    tie_break: str = "h",
) -> SearchResult:
    """Execute A* search on a 4-connected 2D grid with uniform move cost 1.0.

    Parameters
    ----------
    grid : np.ndarray
        2D binary grid where 0 indicates free space and 1 indicates obstacle/wall.
    start : tuple[int, int]
        (row, col) coordinates of start cell.
    goal : tuple[int, int]
        (row, col) coordinates of goal cell.
    heuristic : Callable[[int, int], float]
        Callable returning estimated distance from (r, c) to goal.
    tie_break : str, optional
        Tie-breaking rule for open set: 'h' prefers smaller h value; default 'h'.

    Returns
    -------
    SearchResult
        Named dataclass recording search outcomes and execution metrics.
    """
    start_time = time.perf_counter()
    H, W = grid.shape

    # Boundary and obstacle check on start and goal
    if not (0 <= start[0] < H and 0 <= start[1] < W and 0 <= goal[0] < H and 0 <= goal[1] < W):
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchResult(
            found=False,
            path=[],
            cost=float("inf"),
            nodes_expanded=0,
            nodes_generated=0,
            heuristic_calls=0,
            runtime_ms=elapsed_ms,
            peak_open=0,
        )

    if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchResult(
            found=False,
            path=[],
            cost=float("inf"),
            nodes_expanded=0,
            nodes_generated=0,
            heuristic_calls=0,
            runtime_ms=elapsed_ms,
            peak_open=0,
        )

    # Edge case: start == goal
    if start == goal:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchResult(
            found=True,
            path=[start],
            cost=0.0,
            nodes_expanded=0,
            nodes_generated=1,
            heuristic_calls=0,
            runtime_ms=elapsed_ms,
            peak_open=1,
        )

    g_score = np.full((H, W), np.inf, dtype=np.float64)
    g_score[start[0], start[1]] = 0.0

    closed = np.zeros((H, W), dtype=bool)
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    h_start = float(heuristic(start[0], start[1]))
    heuristic_calls = 1

    # Monotonically increasing counter for deterministic tie breaking
    counter = 0
    open_heap: list = []

    # Format of heap elements depends on tie_break:
    # If 'h': (f_score, h_score, counter, r, c)
    # If 'lifo': (f_score, -counter, r, c)
    # Default: (f_score, counter, r, c)
    if tie_break == "h":
        heapq.heappush(open_heap, (h_start, h_start, counter, start[0], start[1]))
    elif tie_break == "lifo":
        heapq.heappush(open_heap, (h_start, -counter, start[0], start[1]))
    else:
        heapq.heappush(open_heap, (h_start, counter, start[0], start[1]))

    nodes_generated = 1
    nodes_expanded = 0
    peak_open = 1

    # 4-connected grid directions: Up, Down, Left, Right
    DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    found = False

    while open_heap:
        if tie_break == "h":
            f, h, _, r, c = heapq.heappop(open_heap)
        else:
            f, _, r, c = heapq.heappop(open_heap)

        if closed[r, c]:
            continue

        closed[r, c] = True
        nodes_expanded += 1

        if (r, c) == goal:
            found = True
            break

        current_g = g_score[r, c]

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if grid[nr, nc] == 1:
                continue
            if closed[nr, nc]:
                continue

            tentative_g = current_g + 1.0
            if tentative_g < g_score[nr, nc]:
                g_score[nr, nc] = tentative_g
                parent[(nr, nc)] = (r, c)
                h_val = float(heuristic(nr, nc))
                heuristic_calls += 1
                f_val = tentative_g + h_val
                counter += 1
                nodes_generated += 1

                if tie_break == "h":
                    heapq.heappush(open_heap, (f_val, h_val, counter, nr, nc))
                elif tie_break == "lifo":
                    heapq.heappush(open_heap, (f_val, -counter, nr, nc))
                else:
                    heapq.heappush(open_heap, (f_val, counter, nr, nc))

        if len(open_heap) > peak_open:
            peak_open = len(open_heap)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    if found:
        path = []
        curr = goal
        while curr in parent:
            path.append(curr)
            curr = parent[curr]
        path.append(start)
        path.reverse()
        cost = float(g_score[goal[0], goal[1]])
        return SearchResult(
            found=True,
            path=path,
            cost=cost,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            heuristic_calls=heuristic_calls,
            runtime_ms=elapsed_ms,
            peak_open=peak_open,
        )

    return SearchResult(
        found=False,
        path=[],
        cost=float("inf"),
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        heuristic_calls=heuristic_calls,
        runtime_ms=elapsed_ms,
        peak_open=peak_open,
    )
