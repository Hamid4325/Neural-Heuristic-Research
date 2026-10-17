"""
features.py
Phase 2 (Feature Engineering):
Transform cleaned mazes from data/cleaned/ into feature matrices for
neural-network training.

For every REACHABLE free cell n in a maze of size N x N, we build:

  Feature vector X(n)  (29 features):
    1. Normalised cell coordinates      x / N, y / N            (2)  -> [0, 1]
    2. Goal-relative offsets            dx / N, dy / N          (2)  -> (-1, 1)
    3. Local wall context               5 x 5 binary window     (25) -> {0, 1}
       (window is padded with 1 outside the grid, i.e. treated as wall)

  Target y(n):
    Normalised exact heuristic         h*(n) / N               (1)

Cells whose h*(n) is the -1.0 sentinel (walls / unreachable free
cells) are NOT emitted: they have no valid learning target.

Train/Val/Test split:
  70% / 15% / 15% split BY MAZE ID (whole topologies move together),
  so no cell from a maze appears in more than one split. This prevents
  data leakage across correlated cells of the same grid.

Output (data/processed/):
    train.npz, val.npz, test.npz  each with arrays X (M, 29) and y (M,)

Usage:
  python data_scripts/features.py
  python data_scripts/features.py --cleaned data/cleaned --out data/processed --seed 42
"""

import argparse
import glob
import logging
import os
from datetime import datetime

import numpy as np

WINDOW = 5                 # local wall context is 5x5
HALF = WINDOW // 2         # 2 cells of padding on each side
N_FEATURES = 2 + 2 + WINDOW * WINDOW  # coords(2) + goal offsets(2) + walls(25) = 29
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("features")


def build_features_for_maze(grid, goal, h_star):
    """Return (X, y) for all reachable free cells of a single maze.

    grid   : (N, N) uint8, 0 = free, 1 = wall
    goal   : (2,) int, (row, col) of the goal cell
    h_star : (N, N) float32, exact cost to goal; -1.0 marks invalid cells
    """
    n = grid.shape[0]

    # Coordinate grids: x = column index, y = row index (0 .. N-1).
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)

    # 1) Normalised cell coordinates -> [0, 1].
    coord_x = (xs / n).astype(np.float32)
    coord_y = (ys / n).astype(np.float32)

    # 2) Goal-relative offsets scaled to (-1, 1).
    goal_r, goal_c = int(goal[0]), int(goal[1])
    off_x = ((xs - goal_c) / n).astype(np.float32)
    off_y = ((ys - goal_r) / n).astype(np.float32)

    # 3) Local 5x5 wall window, padded with 1 (border treated as wall).
    padded = np.pad(grid, HALF, mode="constant", constant_values=1)
    # sliding_window_view -> (N, N, 5, 5) then flatten the window to 25 features.
    windows = np.lib.stride_tricks.sliding_window_view(padded, (WINDOW, WINDOW))
    wall_feats = windows.reshape(n, n, WINDOW * WINDOW).astype(np.float32)

    # Assemble the full feature tensor: (N, N, 29).
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

    # Keep only reachable free cells (h*(n) >= 0 marks a valid target).
    mask = h_star >= 0.0
    x = feats[mask]                 # (M, 29)
    y = (h_star[mask] / n).astype(np.float32)  # (M,) normalised target
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Feature engineering for Neural Heuristic Search")
    parser.add_argument("--cleaned", default="data/cleaned", help="Input directory of cleaned npz files")
    parser.add_argument("--out", default="data/processed", help="Output directory for split npz files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/val/test split")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.cleaned, "*.npz")))
    if not files:
        raise SystemExit(f"Error: No .npz files found in '{args.cleaned}'. Run clean_data.py first.")
    os.makedirs(args.out, exist_ok=True)

    # -------------------------------------------------------------------------
    # Discover all mazes and assign a global maze id (file order, then index).
    # -------------------------------------------------------------------------
    maze_index = []  # (file_path, local_idx, size)
    for file_path in files:
        data = np.load(file_path)
        num = len(data["grids"])
        for i in range(num):
            maze_index.append((file_path, i, int(data["grids"][i].shape[0])))
    total_mazes = len(maze_index)

    # -------------------------------------------------------------------------
    # Split maze ids 70 / 15 / 15 (whole mazes stay together -> no leakage).
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(total_mazes)
    n_train = int(round(SPLITS["train"] * total_mazes))
    n_val = int(round(SPLITS["val"] * total_mazes))
    assignment = np.empty(total_mazes, dtype=object)
    assignment[shuffled[:n_train]] = "train"
    assignment[shuffled[n_train:n_train + n_val]] = "val"
    assignment[shuffled[n_train + n_val:]] = "test"

    log.info("Found %d mazes across %d files", total_mazes, len(files))
    log.info("Split by maze id: train=%d val=%d test=%d",
             int(np.sum(assignment == "train")),
             int(np.sum(assignment == "val")),
             int(np.sum(assignment == "test")))

    # -------------------------------------------------------------------------
    # Build features for every maze and collect them into the right split.
    # -------------------------------------------------------------------------
    buckets = {name: {"X": [], "y": []} for name in SPLITS}
    global_id = 0

    for file_path in files:
        data = np.load(file_path)
        grids, goals, h_star = data["grids"], data["goals"], data["h_star"]
        for i in range(len(grids)):
            split = assignment[global_id]
            x, y = build_features_for_maze(grids[i], goals[i], h_star[i])
            buckets[split]["X"].append(x)
            buckets[split]["y"].append(y)
            global_id += 1
        log.info("Processed %-32s (%d mazes)", os.path.basename(file_path), len(grids))

    # -------------------------------------------------------------------------
    # Concatenate, write outputs, and log a summary.
    # -------------------------------------------------------------------------
    summary = {}
    for name in ("train", "val", "test"):
        X = np.concatenate(buckets[name]["X"], axis=0).astype(np.float32)
        y = np.concatenate(buckets[name]["y"], axis=0).astype(np.float32)

        # Shuffle within the split so batches are not grouped by maze.
        order = rng.permutation(len(X))
        X, y = X[order], y[order]

        out_path = os.path.join(args.out, f"{name}.npz")
        np.savez_compressed(out_path, X=X, y=y)
        summary[name] = {"samples": int(len(X)), "features": int(X.shape[1])}
        log.info("Saved %-11s -> %-28s X=%s y=%s", name, out_path, X.shape, y.shape)

    log.info("Feature engineering complete. Feature dim = %d", N_FEATURES)
    log.info("Summary: %s", summary)
    log.info("Finished at %s", datetime.now().isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()
