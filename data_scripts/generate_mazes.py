"""
generate_mazes.py
Phase 2 (Collection): procedural generation of 2D grid mazes.

Two generators are used:
  - "backtracker": Recursive Backtracking (long corridors and dead ends)
  - "perlin":      thresholded Perlin noise (smooth, continuous walls)

Grid convention: 0 = free cell, 1 = wall.

Output (per size/density/method configuration), in data/raw/:
  mazes_<method>_<size>_<density%>.npz  with arrays
      grids  (N, size, size) uint8
      starts (N, 2) int  (row, col)
      goals  (N, 2) int  (row, col)
and a log of all parameters in data_scripts/generation_log.json

Usage:
  python data_scripts/generate_mazes.py
  python data_scripts/generate_mazes.py --num-per-config 10 --seed 7
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np

SIZES = [20, 40, 60, 80, 100]
DENSITIES = [0.10, 0.20, 0.30, 0.40]
METHODS = ["backtracker", "perlin"]


# ----------------------------------------------------------------------
# Generator 1: Recursive Backtracking
# ----------------------------------------------------------------------
def recursive_backtracker(size, rng):
    """Carve a perfect maze with an iterative depth-first search.

    Iterative (explicit stack) instead of recursive, so large grids do not
    hit Python's recursion limit. Cells at odd coordinates are rooms; the
    cell between two rooms is the wall that gets removed.
    """
    n = size if size % 2 == 1 else size + 1  # algorithm needs odd dimensions
    grid = np.ones((n, n), dtype=np.uint8)

    r, c = 2 * rng.integers(n // 2) + 1, 2 * rng.integers(n // 2) + 1
    grid[r, c] = 0
    stack = [(r, c)]

    while stack:
        r, c = stack[-1]
        neighbours = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == 1:
                neighbours.append((nr, nc, dr, dc))
        if neighbours:
            nr, nc, dr, dc = neighbours[rng.integers(len(neighbours))]
            grid[r + dr // 2, c + dc // 2] = 0  # knock down wall between rooms
            grid[nr, nc] = 0
            stack.append((nr, nc))
        else:
            stack.pop()  # dead end: backtrack

    return grid[:size, :size]


# ----------------------------------------------------------------------
# Generator 2: Perlin noise
# ----------------------------------------------------------------------
def _perlin_layer(shape, scale, rng):
    """One octave of 2D gradient (Perlin) noise, implemented in NumPy."""
    h, w = shape
    gh, gw = int(np.ceil(h / scale)) + 2, int(np.ceil(w / scale)) + 2
    angles = rng.uniform(0, 2 * np.pi, (gh, gw))
    gx, gy = np.cos(angles), np.sin(angles)  # random unit gradient per lattice point

    ys, xs = np.mgrid[0:h, 0:w]
    ys, xs = ys / scale, xs / scale
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    dy, dx = ys - y0, xs - x0

    def corner(iy, ix, oy, ox):
        return gx[iy, ix] * (dx - ox) + gy[iy, ix] * (dy - oy)

    n00 = corner(y0, x0, 0, 0)
    n10 = corner(y0, x0 + 1, 0, 1)
    n01 = corner(y0 + 1, x0, 1, 0)
    n11 = corner(y0 + 1, x0 + 1, 1, 1)

    fade = lambda t: t ** 3 * (t * (t * 6 - 15) + 10)  # smooth interpolation curve
    u, v = fade(dx), fade(dy)
    top = n00 * (1 - u) + n10 * u
    bottom = n01 * (1 - u) + n11 * u
    return top * (1 - v) + bottom * v


def perlin_maze(size, density, rng, octaves=3):
    """Threshold fractal Perlin noise so that exactly `density` of cells are walls."""
    base_scale = max(3.0, rng.uniform(size / 12, size / 6))
    noise = np.zeros((size, size))
    for o in range(octaves):
        noise += _perlin_layer((size, size), base_scale / 2 ** o, rng) / 2 ** o
    threshold = np.quantile(noise, 1 - density)
    return (noise > threshold).astype(np.uint8)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def adjust_density(grid, target, rng):
    """Randomly add/remove wall cells until the wall fraction matches target."""
    flat = grid.reshape(-1).copy()
    diff = int(flat.sum()) - int(round(target * flat.size))
    if diff > 0:  # too many walls -> remove some
        idx = rng.choice(np.flatnonzero(flat == 1), size=diff, replace=False)
        flat[idx] = 0
    elif diff < 0:  # too few walls -> add some
        idx = rng.choice(np.flatnonzero(flat == 0), size=-diff, replace=False)
        flat[idx] = 1
    return flat.reshape(grid.shape)


def pick_start_goal(grid, rng):
    """Random free start and goal, at least size//2 apart (Manhattan)."""
    free = np.argwhere(grid == 0)
    start = free[rng.integers(len(free))]
    dist = np.abs(free - start).sum(axis=1)
    far = free[dist >= grid.shape[0] // 2]
    pool = far if len(far) > 0 else free[dist > 0]
    goal = pool[rng.integers(len(pool))]
    return start, goal


def generate_one(method, size, density, rng):
    if method == "backtracker":
        grid = adjust_density(recursive_backtracker(size, rng), density, rng)
    else:
        grid = perlin_maze(size, density, rng)
    start, goal = pick_start_goal(grid, rng)
    return grid, start, goal


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-per-config", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--log", default="data_scripts/generation_log.json")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    log = {
        "seed": args.seed,
        "num_per_config": args.num_per_config,
        "sizes": SIZES,
        "densities": DENSITIES,
        "methods": METHODS,
        "created": datetime.now().isoformat(timespec="seconds"),
        "configs": [],
    }

    total = 0
    for m_id, method in enumerate(METHODS):
        for size in SIZES:
            for density in DENSITIES:
                # one independent, reproducible RNG per configuration
                rng = np.random.default_rng([args.seed, m_id, size, int(density * 100)])
                grids, starts, goals = [], [], []
                for _ in range(args.num_per_config):
                    g, s, t = generate_one(method, size, density, rng)
                    grids.append(g)
                    starts.append(s)
                    goals.append(t)

                grids = np.stack(grids)
                name = f"mazes_{method}_{size}_{int(density * 100)}.npz"
                np.savez_compressed(
                    os.path.join(args.out, name),
                    grids=grids,
                    starts=np.array(starts),
                    goals=np.array(goals),
                )
                total += len(grids)
                log["configs"].append(
                    {
                        "file": name,
                        "method": method,
                        "size": size,
                        "target_density": density,
                        "achieved_density_mean": round(float(grids.mean()), 4),
                        "count": len(grids),
                    }
                )
                print(f"{name}: {len(grids)} mazes, density {grids.mean():.3f}")

    log["total_mazes"] = total
    with open(args.log, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. {total} mazes saved to {args.out}, log at {args.log}")


if __name__ == "__main__":
    main()