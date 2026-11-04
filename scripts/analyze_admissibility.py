"""Run the admissibility audit for a trained neural heuristic checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.admissibility import (
    evaluate_modes,
    overestimation_cdf,
    reconstruct_split_assignment,
)
from src.heuristics import NeuralHeuristic

DEFAULT_CHECKPOINT = ROOT / "models" / "neural_heuristic_seed42.pt"
DEFAULT_CLEANED = ROOT / "data" / "cleaned"
DEFAULT_OUT = ROOT / "results" / "admissibility"
DEFAULT_FIGURE = ROOT / "figures" / "admissibility_cdf.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--cleaned", default=str(DEFAULT_CLEANED))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--figure", default=str(DEFAULT_FIGURE))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 0.9, 1.0],
    )
    parser.add_argument(
        "--max-mazes",
        type=int,
        default=None,
        help="Optional limit for a smoke run.",
    )
    return parser.parse_args()


def list_maze_files(cleaned_dir: Path) -> list[Path]:
    files = sorted(cleaned_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files found in {cleaned_dir}")
    return files


def count_mazes(files: list[Path]) -> int:
    total = 0
    for path in files:
        with np.load(path) as data:
            total += len(data["grids"])
    return total


def iter_split_mazes(
    files: list[Path],
    assignment: np.ndarray,
    split: str,
):
    global_id = 0
    for path in files:
        with np.load(path) as data:
            grids = data["grids"]
            goals = data["goals"]
            h_star = data["h_star"]
            for local_index in range(len(grids)):
                if assignment[global_id] == split:
                    yield path, local_index, grids[local_index], goals[local_index], h_star[local_index]
                global_id += 1


def manhattan_field(shape: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.abs(rows - int(goal[0])) + np.abs(cols - int(goal[1]))


def collect_split_predictions(
    checkpoint: Path,
    files: list[Path],
    assignment: np.ndarray,
    split: str,
    max_mazes: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    predictions: list[np.ndarray] = []
    true_costs: list[np.ndarray] = []
    manhattan_costs: list[np.ndarray] = []
    maze_names: list[str] = []

    for path, local_index, grid, goal, h_star in iter_split_mazes(
        files, assignment, split
    ):
        if max_mazes is not None and len(maze_names) >= max_mazes:
            break

        heuristic = NeuralHeuristic(str(checkpoint), grid, goal)
        pred = heuristic.predictions()
        manhattan = manhattan_field(grid.shape, (int(goal[0]), int(goal[1])))
        mask = (grid == 0) & (h_star >= 0.0) & np.isfinite(pred)
        if not mask.any():
            continue

        predictions.append(pred[mask])
        true_costs.append(h_star[mask].astype(np.float64))
        manhattan_costs.append(manhattan[mask].astype(np.float64))
        maze_names.append(f"{path.name}:{local_index}")

    if not predictions:
        raise SystemExit("No val cells were collected; check the split and data path.")

    return (
        np.concatenate(predictions),
        np.concatenate(true_costs),
        np.concatenate(manhattan_costs),
        maze_names,
        len(maze_names),
    )


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    checkpoint: Path,
    seed: int,
    split: str,
    maze_count: int,
    cell_count: int,
    alphas: list[float],
) -> None:
    payload = {
        "checkpoint": str(checkpoint),
        "seed": seed,
        "split": split,
        "maze_count": maze_count,
        "cell_count": cell_count,
        "alphas": alphas,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_cdf(
    path: Path,
    predictions: np.ndarray,
    true_costs: np.ndarray,
    manhattan: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series = [
        ("raw", 1.0, "raw", "#dc2626"),
        ("scaled alpha=0.75", 0.75, "scaled", "#d97706"),
        ("clipped alpha=1.0", 1.0, "clipped_manhattan", "#2563eb"),
    ]
    for label, alpha, mode, color in series:
        values, cumulative = overestimation_cdf(
            predictions,
            true_costs,
            manhattan,
            alpha=alpha,
            mode=mode,
        )
        ax.plot(values, cumulative, label=label, color=color, linewidth=1.8)

    ax.axvline(0.0, color="#334155", linestyle="--", linewidth=1.0)
    ax.set_xlabel("overestimation $h(n) - h^*(n)$ (cells)")
    ax.set_ylabel("cumulative fraction of cells")
    ax.set_title("Admissibility CDF on the validation split")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def print_table(rows: list[dict[str, float | int | str]]) -> None:
    header = (
        f"{'mode':<20} {'alpha':>5} {'over%':>8} {'mean':>8} "
        f"{'p95':>8} {'max':>8} {'mae':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{str(row['mode']):<20} {float(row['alpha']):>5.2f} "
            f"{100.0 * float(row['fraction_over']):>7.2f}% "
            f"{float(row['mean_overestimation']):>8.3f} "
            f"{float(row['p95_overestimation']):>8.3f} "
            f"{float(row['max_overestimation']):>8.3f} "
            f"{float(row['mae']):>8.3f}"
        )


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    cleaned = Path(args.cleaned)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    files = list_maze_files(cleaned)
    total_mazes = count_mazes(files)
    assignment = reconstruct_split_assignment(total_mazes, seed=args.seed)

    predictions, true_costs, manhattan, maze_names, maze_count = (
        collect_split_predictions(
            checkpoint,
            files,
            assignment,
            "val",
            args.max_mazes,
        )
    )

    rows = evaluate_modes(predictions, true_costs, manhattan, alphas=args.alphas)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "admissibility.csv", rows)
    write_summary(
        out_dir / "admissibility_summary.json",
        checkpoint,
        args.seed,
        "val",
        maze_count,
        int(predictions.size),
        list(args.alphas),
    )
    plot_cdf(Path(args.figure), predictions, true_costs, manhattan)

    print(f"checkpoint: {checkpoint}")
    print(f"val mazes: {maze_count}  cells: {predictions.size}")
    print_table(rows)
    print(f"\nwrote {out_dir / 'admissibility.csv'}")
    print(f"wrote {Path(args.figure)}")


if __name__ == "__main__":
    main()
