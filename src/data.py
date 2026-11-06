"""Data access for Deliverable 2: cell-level loaders, split manifest and maze loaders.

Owner: Muhammad Abdullah Saeed. Implements the frozen contract in plan Section 3.4.

Two views of the same 70/15/15 split (made BY MAZE, so no topology leaks across splits):

* Cell view   -> `load_cell_split`   : (X, y) arrays used to TRAIN the network.
* Maze view   -> `load_maze_split`   : whole mazes used to run A* on the exact test mazes.

The manifest algorithm reproduces `data_scripts/features.py` exactly (same sorted file
order, same seed, same rounding), so both views describe the same mazes.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Iterable, Iterator

import numpy as np

from src.utils import load_json

log = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
DEFAULT_SEED = 42
N_FEATURES = 29
MAZE_KEYS = ("grids", "starts", "goals", "h_star", "opt_cost")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_split(split: str) -> None:
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")


def _cleaned_files(cleaned_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(cleaned_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No .npz files in '{cleaned_dir}'. Run data_scripts/clean_data.py first "
            "(from the repository root)."
        )
    return files


# ---------------------------------------------------------------------------
# Cell view (training data)
# ---------------------------------------------------------------------------
def load_cell_split(split: str, data_dir: str = "data/processed") -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for split in {train, val, test}.

    X: (M, 29) float32 features; y: (M,) float32 target = h*(n) / N (range about [0, 3.18]).
    """
    _check_split(split)
    path = os.path.join(data_dir, f"{split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run data_scripts/features.py first (from the repository root)."
        )
    with np.load(path) as data:
        X, y = data["X"], data["y"]
    return X, y


# ---------------------------------------------------------------------------
# Split manifest
# ---------------------------------------------------------------------------
def build_split_manifest(cleaned_dir: str = "data/cleaned", seed: int = DEFAULT_SEED) -> dict:
    """Reproduce the Deliverable 1 70/15/15 split and return a manifest dict.

    Entries of each split are listed in global maze order (file order, then index inside
    the file), which is the same order data_scripts/features.py walks the mazes.
    """
    files = _cleaned_files(cleaned_dir)

    index: list[tuple[str, int]] = []  # position = global maze id
    for path in files:
        with np.load(path) as data:
            count = len(data["grids"])
        name = os.path.basename(path)
        index.extend((name, i) for i in range(count))
    total = len(index)

    # --- identical to data_scripts/features.py ---------------------------------
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(total)
    n_train = int(round(SPLIT_FRACTIONS["train"] * total))
    n_val = int(round(SPLIT_FRACTIONS["val"] * total))
    assignment = np.empty(total, dtype=object)
    assignment[shuffled[:n_train]] = "train"
    assignment[shuffled[n_train:n_train + n_val]] = "val"
    assignment[shuffled[n_train + n_val:]] = "test"
    # ---------------------------------------------------------------------------

    splits: dict[str, dict] = {}
    for name in SPLIT_NAMES:
        entries = [
            {"file": index[g][0], "local_index": index[g][1]}
            for g in range(total)
            if assignment[g] == name
        ]
        splits[name] = {"maze_count": len(entries), "entries": entries}

    return {"seed": seed, "total_mazes": total, "splits": splits}


# ---------------------------------------------------------------------------
# Maze view (search evaluation data)
# ---------------------------------------------------------------------------
def _iter_mazes(entries: Iterable[dict], cleaned_dir: str) -> Iterator[tuple[str, dict, int]]:
    """Yield (file_name, arrays_of_that_file, local_index) for each manifest entry.

    Only one cleaned file is kept in memory at a time (entries are grouped by file).
    """
    current_name = None
    current: dict = {}
    for entry in entries:
        name = entry["file"]
        if name != current_name:
            with np.load(os.path.join(cleaned_dir, name)) as data:
                current = {key: data[key] for key in MAZE_KEYS}
            current_name = name
        yield name, current, int(entry["local_index"])


def _object_array(items: list) -> np.ndarray:
    """1-D object array of arrays (grids have different sizes, so they cannot be stacked)."""
    out = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        out[i] = item
    return out


def load_maze_split(
    split: str,
    cleaned_dir: str = "data/cleaned",
    manifest_path: str = "data/processed/split_manifest.json",
) -> dict[str, np.ndarray]:
    """Return dict with keys: grids, starts, goals, h_star, opt_cost, file_names.

    * grids, h_star : 1-D object arrays; element i is an (N_i, N_i) array
                      (grid: 1 = wall, 0 = free; h_star: -1.0 marks walls/unreachable cells)
    * starts, goals : (M, 2) int arrays of (row, col)
    * opt_cost      : (M,) float array, optimal path cost start -> goal
    * file_names    : (M,) object array with the source cleaned file of each maze

    If the manifest file does not exist it is rebuilt deterministically with seed 42.
    """
    _check_split(split)
    if os.path.exists(manifest_path):
        manifest = load_json(manifest_path)
    else:
        log.warning("Manifest '%s' not found; rebuilding it from '%s' (seed %d).",
                    manifest_path, cleaned_dir, DEFAULT_SEED)
        manifest = build_split_manifest(cleaned_dir, DEFAULT_SEED)

    grids, starts, goals, h_star, opt_cost, file_names = [], [], [], [], [], []
    for name, arrays, i in _iter_mazes(manifest["splits"][split]["entries"], cleaned_dir):
        grids.append(arrays["grids"][i])
        starts.append(arrays["starts"][i])
        goals.append(arrays["goals"][i])
        h_star.append(arrays["h_star"][i])
        opt_cost.append(arrays["opt_cost"][i])
        file_names.append(name)

    return {
        "grids": _object_array(grids),
        "starts": np.asarray(starts).reshape(len(starts), 2),
        "goals": np.asarray(goals).reshape(len(goals), 2),
        "h_star": _object_array(h_star),
        "opt_cost": np.asarray(opt_cost),
        "file_names": _object_array(file_names),
    }


def manifest_cell_counts(manifest: dict, cleaned_dir: str = "data/cleaned") -> dict[str, int]:
    """Number of valid training cells (h* >= 0) each split of the manifest contains.

    Used to prove the manifest matches data/processed/{train,val,test}.npz exactly.
    """
    counts = {}
    for split in SPLIT_NAMES:
        total = 0
        for _, arrays, i in _iter_mazes(manifest["splits"][split]["entries"], cleaned_dir):
            total += int(np.count_nonzero(arrays["h_star"][i] >= 0.0))
        counts[split] = total
    return counts
