"""
compute_labels.py
Phase 2 (Ground truth): compute the exact shortest-path cost h*(n) from
EVERY cell n to the goal, using Dijkstra's algorithm.

Key idea: the maze is undirected with uniform move cost, so running ONE
Dijkstra search outward from the goal gives h*(n) for every cell at once.
This is far cheaper than one search per cell.

Input  (from generate_mazes.py): data/raw/*.npz
Output (one file per input):     data/labeled/*.npz with arrays
    grids     (N, size, size) uint8    0 = free, 1 = wall
    starts    (N, 2) int
    goals     (N, 2) int
    h_star    (N, size, size) float32  exact cost to goal; inf = unreachable or wall
    opt_cost  (N,) float32             h*(start), i.e. optimal start->goal cost (inf if none)
A summary is written to data_scripts/label_log.json

Usage:
  python data_scripts/compute_labels.py
  python data_scripts/compute_labels.py --workers 4
"""

import argparse
import glob
import heapq
import json
import os
from multiprocessing import Pool

import numpy as np

MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))  # 4-connected grid
STEP_COST = 1.0  # uniform traversal cost (terrain weights could be added here)


def dijkstra_from_goal(grid, goal):
    """Return an array of shortest distances from `goal` to every cell.

    Walls and cells that cannot be reached keep the value inf.
    """
    h, w = grid.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    gr, gc = int(goal[0]), int(goal[1])
    if grid[gr, gc] == 1:  # goal inside a wall: nothing is reachable
        return dist

    dist[gr, gc] = 0.0
    heap = [(0.0, gr, gc)]  # (distance, row, col)
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r, c]:  # stale heap entry, a shorter path was already found
            continue
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                nd = d + STEP_COST
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))
    return dist


def label_file(args):
    in_path, out_dir = args
    data = np.load(in_path)
    grids, starts, goals = data["grids"], data["starts"], data["goals"]

    h_star = np.stack([dijkstra_from_goal(g, t) for g, t in zip(grids, goals)])
    opt_cost = np.array(
        [h_star[i, s[0], s[1]] for i, s in enumerate(starts)], dtype=np.float32
    )

    out_path = os.path.join(out_dir, os.path.basename(in_path))
    np.savez_compressed(
        out_path,
        grids=grids,
        starts=starts,
        goals=goals,
        h_star=h_star,
        opt_cost=opt_cost,
    )

    free = grids == 0
    reachable = np.isfinite(h_star) & free
    return {
        "file": os.path.basename(in_path),
        "mazes": int(len(grids)),
        "mazes_start_unreachable": int(np.sum(~np.isfinite(opt_cost))),
        "free_cells": int(free.sum()),
        "reachable_cells": int(reachable.sum()),
        "unreachable_free_cells": int(free.sum() - reachable.sum()),
        "max_h_star": float(h_star[np.isfinite(h_star)].max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/labeled")
    parser.add_argument("--log", default="data_scripts/label_log.json")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw, "*.npz")))
    if not files:
        raise SystemExit(f"No .npz files in {args.raw}. Run generate_mazes.py first.")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    jobs = [(f, args.out) for f in files]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(label_file, jobs)
    else:
        results = []
        for j in jobs:
            r = label_file(j)
            results.append(r)
            print(f"{r['file']}: {r['mazes']} mazes labeled, "
                  f"{r['mazes_start_unreachable']} with unreachable start")

    summary = {
        "total_mazes": sum(r["mazes"] for r in results),
        "total_mazes_start_unreachable": sum(r["mazes_start_unreachable"] for r in results),
        "total_free_cells": sum(r["free_cells"] for r in results),
        "total_unreachable_free_cells": sum(r["unreachable_free_cells"] for r in results),
        "files": results,
    }
    with open(args.log, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Labeled {summary['total_mazes']} mazes -> {args.out}")
    print(f"Mazes with no start->goal path: {summary['total_mazes_start_unreachable']}")
    print(f"Free cells that cannot reach the goal: {summary['total_unreachable_free_cells']}")


if __name__ == "__main__":
    main()
