"""Run classical and oracle baselines (Dijkstra, Manhattan, Oracle) on split mazes.

Part of Deliverable 2 for Neural Heuristic Research.
Author: Abdullah Zia
"""

import argparse
import logging
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluate import iter_split_mazes, run_search_evaluation
from src.search import ManhattanHeuristic, OracleHeuristic, ZeroHeuristic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_baselines")


def main():
    parser = argparse.ArgumentParser(description="Run classical and oracle search baselines.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Maze dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--cleaned-dir",
        type=str,
        default="data/cleaned",
        help="Directory containing cleaned maze npz files",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/split_manifest.json",
        help="Path to split manifest if present",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation",
        help="Directory to persist CSV and JSON evaluation results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split reconstruction (default: 42)",
    )
    parser.add_argument(
        "--tie-break",
        type=str,
        default="h",
        choices=["h", "lifo", "fifo"],
        help="Tie-breaking strategy for A* search",
    )

    args = parser.parse_args()

    log.info(f"Loading {args.split} mazes from {args.cleaned_dir} (seed={args.seed})...")
    mazes = list(
        iter_split_mazes(
            split=args.split,
            cleaned_dir=args.cleaned_dir,
            seed=args.seed,
            manifest_path=args.manifest,
        )
    )
    log.info(f"Loaded {len(mazes)} mazes for split '{args.split}'.")

    if len(mazes) == 0:
        log.error("No mazes loaded. Please check data/cleaned directory.")
        sys.exit(1)

    heuristics = {
        "dijkstra": lambda m: ZeroHeuristic(),
        "manhattan": lambda m: ManhattanHeuristic(m["goal"]),
        "oracle": lambda m: OracleHeuristic(m["h_star"]),
    }

    log.info(f"Running search evaluation across {len(heuristics)} baseline heuristics...")
    df = run_search_evaluation(
        mazes=mazes,
        heuristics=heuristics,
        output_dir=args.output,
        split=args.split,
        seed=args.seed,
        tie_break=args.tie_break,
    )

    log.info(f"Completed baseline evaluation: {len(df)} total search trials.")
    log.info(f"Results saved to: {args.output}/{args.split}_per_maze.csv and {args.split}_summary.json")


if __name__ == "__main__":
    main()
