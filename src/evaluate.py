"""Evaluation harness for grid-maze heuristic search.

Part of Deliverable 2 for Neural Heuristic Research.
Author: Abdullah Zia
"""

import glob
import json
import os
import re
from typing import Any, Callable, Iterator, Optional, Union
import numpy as np
import pandas as pd

from src.search import astar, SearchResult, ManhattanHeuristic, OracleHeuristic, ZeroHeuristic
from src.metrics import search_aggregate


def reconstruct_split(cleaned_dir: str = "data/cleaned", seed: int = 42):
    """Reproduce the Deliverable 1 70/15/15 split deterministically across cleaned mazes.

    Yields exactly:
      train: 1,313 mazes
      val:     281 mazes
      test:    282 mazes
    """
    files = sorted(glob.glob(os.path.join(cleaned_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {cleaned_dir}. Regenerate dataset first.")

    total = 0
    for f in files:
        with np.load(f) as d:
            total += len(d["grids"])

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(total)
    n_train = int(round(0.70 * total))
    n_val = int(round(0.15 * total))

    assignment = np.empty(total, dtype=object)
    assignment[shuffled[:n_train]] = "train"
    assignment[shuffled[n_train : n_train + n_val]] = "val"
    assignment[shuffled[n_train + n_val :]] = "test"

    return files, assignment


def iter_split_mazes(
    split: str,
    cleaned_dir: str = "data/cleaned",
    seed: int = 42,
    manifest_path: Optional[str] = "data/processed/split_manifest.json",
) -> Iterator[dict[str, Any]]:
    """Yield individual maze records for a given split ('train', 'val', or 'test')."""
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        split_entries = manifest.get(split, [])
        for entry in split_entries:
            file_name = entry["file_name"]
            path = os.path.join(cleaned_dir, file_name)
            idx = entry["local_index"]
            with np.load(path) as d:
                yield {
                    "maze_id": entry.get("maze_id", f"{file_name}_{idx}"),
                    "file_name": file_name,
                    "local_index": idx,
                    "grid": d["grids"][idx],
                    "start": tuple(int(v) for v in d["starts"][idx]),
                    "goal": tuple(int(v) for v in d["goals"][idx]),
                    "h_star": d["h_star"][idx],
                    "opt_cost": float(d["opt_cost"][idx]),
                }
    else:
        files, assignment = reconstruct_split(cleaned_dir=cleaned_dir, seed=seed)
        gid = 0
        for path in files:
            file_name = os.path.basename(path)
            with np.load(path) as d:
                grids = d["grids"]
                starts = d["starts"]
                goals = d["goals"]
                h_star = d["h_star"]
                opt_cost = d["opt_cost"]
                for i in range(len(grids)):
                    if assignment[gid] == split:
                        yield {
                            "maze_id": gid,
                            "file_name": file_name,
                            "local_index": i,
                            "grid": grids[i],
                            "start": tuple(int(v) for v in starts[i]),
                            "goal": tuple(int(v) for v in goals[i]),
                            "h_star": h_star[i],
                            "opt_cost": float(opt_cost[i]),
                        }
                    gid += 1


def parse_filename_metadata(file_name: str) -> tuple[str, int, int]:
    """Extract (generator, size, density) from file name like 'mazes_backtracker_100_10.npz'."""
    pattern = r"mazes_([a-zA-Z0-9]+)_(\d+)_(\d+)\.npz"
    match = re.search(pattern, file_name)
    if match:
        generator = match.group(1)
        size = int(match.group(2))
        density = int(match.group(3))
        return generator, size, density
    return "unknown", 0, 0


def run_search_evaluation(
    mazes: Union[list[dict[str, Any]], Iterator[dict[str, Any]]],
    heuristics: dict[str, Any],
    output_dir: str,
    split: str,
    seed: int = 42,
    tie_break: str = "h",
) -> pd.DataFrame:
    """Run A* for each maze and heuristic.

    Write per-maze rows to output_dir/<split>_per_maze.csv and
    aggregate metrics to output_dir/<split>_summary.json.
    Return the per-maze DataFrame.

    Parameters
    ----------
    mazes : Union[list[dict[str, Any]], Iterator[dict[str, Any]]]
        Sequence of maze dicts containing 'maze_id', 'file_name', 'local_index',
        'grid', 'start', 'goal', 'h_star', 'opt_cost'.
    heuristics : dict[str, Any]
        Mapping from heuristic identifier to either a factory function (taking maze dict),
        or a tuple (factory, alpha).
    output_dir : str
        Directory where CSV and JSON results will be persisted.
    split : str
        Data split name ('train', 'val', or 'test').
    seed : int, default=42
        Seed for reproducibility.
    tie_break : str, default='h'
        Tie-breaking rule for A*.

    Returns
    -------
    pd.DataFrame
        DataFrame with individual per-maze search evaluation records.
    """
    os.makedirs(output_dir, exist_ok=True)
    maze_list = list(mazes)

    records = []

    for m in maze_list:
        file_name = m["file_name"]
        generator, size, density = parse_filename_metadata(file_name)
        grid = m["grid"]
        start = m["start"]
        goal = m["goal"]
        opt_cost = m["opt_cost"]
        maze_id = m["maze_id"]
        local_idx = m["local_index"]

        for h_key, h_spec in heuristics.items():
            # Extract factory and alpha
            alpha_val = 1.0
            if isinstance(h_spec, tuple) and len(h_spec) == 2:
                factory, alpha_val = h_spec
            elif callable(h_spec):
                factory = h_spec
                # Check if factory or object has alpha
                if hasattr(h_spec, "alpha"):
                    alpha_val = getattr(h_spec, "alpha")
            else:
                factory = lambda _: h_spec

            h_inst = factory(m)
            if hasattr(h_inst, "alpha"):
                alpha_val = getattr(h_inst, "alpha")

            result: SearchResult = astar(
                grid=grid,
                start=start,
                goal=goal,
                heuristic=h_inst,
                tie_break=tie_break,
            )

            optimality_ratio = (
                (result.cost / opt_cost)
                if (result.found and opt_cost > 0 and np.isfinite(result.cost))
                else (1.0 if (result.found and opt_cost == 0) else np.nan)
            )

            records.append({
                "split": split,
                "maze_id": maze_id,
                "file_name": file_name,
                "local_index": local_idx,
                "generator": generator,
                "size": size,
                "density": density,
                "heuristic": h_key,
                "alpha": float(alpha_val) if alpha_val is not None else 1.0,
                "found": bool(result.found),
                "cost": float(result.cost),
                "opt_cost": float(opt_cost),
                "optimality_ratio": float(optimality_ratio) if not np.isnan(optimality_ratio) else np.nan,
                "nodes_expanded": int(result.nodes_expanded),
                "nodes_generated": int(result.nodes_generated),
                "heuristic_calls": int(result.heuristic_calls),
                "runtime_ms": float(result.runtime_ms),
                "peak_open": int(result.peak_open),
            })

    df = pd.DataFrame(records)

    # Persist CSV
    csv_path = os.path.join(output_dir, f"{split}_per_maze.csv")
    df.to_csv(csv_path, index=False)

    # Aggregate metrics
    summary = search_aggregate(df, baseline="manhattan")
    json_path = os.path.join(output_dir, f"{split}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return df
