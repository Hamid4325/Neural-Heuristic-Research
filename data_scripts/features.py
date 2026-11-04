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
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Allow running this script directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import (  # noqa: E402
    N_FEATURES,
    SPLITS,
    WINDOW,
    build_features_for_maze,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("features")


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
