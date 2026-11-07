"""Final training: the primary model (seed 42) and two replicates (seeds 43 and 44).

Owner: Muhammad Abdullah Saeed (plan Section 6.4, task 6.4.1).

Equivalent to the three commands of the plan, in one go:

    python -m src.train --config configs/final.yaml --seed 42 --output models/neural_heuristic_seed42.pt
    python -m src.train --config configs/final.yaml --seed 43 --max-train-cells 1000000 --output models/neural_heuristic_seed43.pt
    python -m src.train --config configs/final.yaml --seed 44 --max-train-cells 1000000 --output models/neural_heuristic_seed44.pt

The first seed uses the config as it is (full train split per epoch); the others use
`--replicate-cells` train cells per epoch to bound the CPU time. After each run the
checkpoint is re-loaded and checked against the contract in plan Section 3.5.

Outputs:
    models/neural_heuristic_seed<S>.pt                     checkpoints (git-ignored)
    results/training/neural_heuristic_seed<S>/             history.csv, config.yaml, metrics.json
    results/training/final_summary.{csv,json}              one row per seed + mean/std of val MAE

Usage (repository root):
    python scripts/train_final.py
    python scripts/train_final.py --resume                 # skip seeds that already finished
    python scripts/train_final.py --config configs/default.yaml --model standin --quick   # dry run
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.config import Config, load_config  # noqa: E402
from src.data import load_cell_split  # noqa: E402
from src.train import train  # noqa: E402
from src.tune import get_model_factory  # noqa: E402
from src.utils import configure_logging, load_json, save_csv, save_json  # noqa: E402

log = logging.getLogger("train_final")

QUICK_ROOT = os.path.join("results", "tuning_quick", "final_quick")  # already git-ignored
CONTRACT_KEYS = {"model_state", "model_config", "target_mean", "target_std",
                 "bin_edges", "config", "git_commit", "seed"}
SUMMARY_COLUMNS = [
    "seed", "role", "checkpoint", "train_cells_per_epoch", "epochs_planned", "epochs_run",
    "best_epoch", "stopped_early", "best_val_mae", "best_val_bin_accuracy",
    "best_val_macro_f1", "n_params", "train_seconds", "git_commit", "model_source",
]


def verify_checkpoint(path: str, seed: int, n_bins: int) -> dict:
    """Re-load a checkpoint with the default (safe) loader and check the contract keys."""
    ckpt = torch.load(path, map_location="cpu")
    missing = CONTRACT_KEYS - set(ckpt)
    if missing:
        raise ValueError(f"{path}: missing checkpoint keys {sorted(missing)}")
    if ckpt["seed"] != seed:
        raise ValueError(f"{path}: stored seed {ckpt['seed']} != expected {seed}")
    if len(ckpt["bin_edges"]) != n_bins - 1:
        raise ValueError(f"{path}: expected {n_bins - 1} bin edges, got {len(ckpt['bin_edges'])}")
    if not ckpt["target_std"] > 0:
        raise ValueError(f"{path}: target_std must be positive")
    return ckpt


def train_all(
    cfg: Config,
    seeds: list[int],
    *,
    models_dir: str,
    training_dir: str,
    replicate_cells: Optional[int],
    model_factory: Callable,
    model_source: str,
    data: tuple,
    resume: bool = False,
) -> list[dict]:
    """Train one model per seed; write final_summary.{csv,json}; return the summary rows."""
    rows = []
    for i, seed in enumerate(seeds):
        run_cfg = copy.deepcopy(cfg)
        run_cfg.train.seed = seed
        if i > 0 and replicate_cells:
            run_cfg.train.max_train_cells_per_epoch = replicate_cells
        stem = f"neural_heuristic_seed{seed}"
        checkpoint = os.path.join(models_dir, stem + ".pt")
        run_dir = os.path.join(training_dir, stem)
        metrics_path = os.path.join(run_dir, "metrics.json")

        metrics = None
        if resume and os.path.exists(checkpoint) and os.path.exists(metrics_path):
            metrics = load_json(metrics_path)
            if metrics.get("seed") != seed:
                metrics = None
            else:
                log.info("seed %d: finished earlier, reusing %s", seed, checkpoint)
        if metrics is None:
            log.info("===== seed %d (%s) =====", seed, "primary" if i == 0 else "replicate")
            summary = train(run_cfg, checkpoint, results_dir=run_dir,
                            model_factory=model_factory, data=data)
            metrics = {k: v for k, v in summary.items() if k != "history"}

        verify_checkpoint(checkpoint, seed, run_cfg.model.n_bins)
        cells = run_cfg.train.max_train_cells_per_epoch
        rows.append({
            "seed": seed, "role": "primary" if i == 0 else "replicate",
            "checkpoint": checkpoint.replace(os.sep, "/"),
            "train_cells_per_epoch": "all" if not cells else cells,
            "epochs_planned": run_cfg.train.epochs,
            "epochs_run": metrics["epochs_run"], "best_epoch": metrics["best_epoch"],
            "stopped_early": metrics["stopped_early"],
            "best_val_mae": metrics["best_val_mae"],
            "best_val_bin_accuracy": metrics["best_val_bin_accuracy"],
            "best_val_macro_f1": metrics["best_val_macro_f1"],
            "n_params": metrics["n_params"], "train_seconds": metrics["train_seconds"],
            "git_commit": metrics["git_commit"], "model_source": model_source,
        })

    maes = [r["best_val_mae"] for r in rows]
    aggregate = {
        "n_seeds": len(rows),
        "val_mae_mean": statistics.fmean(maes),
        "val_mae_std": statistics.stdev(maes) if len(maes) > 1 else 0.0,
    }
    os.makedirs(training_dir, exist_ok=True)
    save_csv(os.path.join(training_dir, "final_summary.csv"), rows, SUMMARY_COLUMNS)
    save_json(os.path.join(training_dir, "final_summary.json"),
              {"runs": rows, "aggregate": aggregate})

    log.info("===== Final models =====")
    for r in rows:
        log.info("seed %d (%s): val_mae %.4f  best_epoch %s  %s cells/epoch  %.0fs",
                 r["seed"], r["role"], r["best_val_mae"], r["best_epoch"],
                 r["train_cells_per_epoch"], r["train_seconds"])
    if math.isfinite(aggregate["val_mae_std"]):
        log.info("val_mae over seeds: %.4f +- %.4f", aggregate["val_mae_mean"],
                 aggregate["val_mae_std"])
    return rows


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the final models (seeds 42, 43, 44)")
    p.add_argument("--config", default="configs/final.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                   help="First seed = primary model (config as is); the rest = replicates")
    p.add_argument("--replicate-cells", type=int, default=1_000_000,
                   help="Train cells per epoch for the replicate seeds")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--training-dir", default="results/training")
    p.add_argument("--model", choices=("hamid", "standin"), default="hamid")
    p.add_argument("--quick", action="store_true",
                   help=f"Tiny dry run; writes to {QUICK_ROOT} instead of models/ and results/")
    p.add_argument("--resume", action="store_true", help="Skip seeds that already finished")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging()
    if args.model == "standin" and not args.quick:
        print("ERROR: checkpoints made with the stand-in network do not match Hamid's model, "
              "so --model standin is only allowed together with --quick (dry run).",
              file=sys.stderr)
        return 1
    if not os.path.exists(args.config):
        print(f"ERROR: {args.config} not found. After tuning, create it with "
              "`python scripts/make_final_config.py`.", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    models_dir, training_dir, replicate_cells = args.models_dir, args.training_dir, \
        args.replicate_cells
    if args.quick:
        cfg.train.epochs, cfg.train.max_train_cells_per_epoch = 2, 50_000
        cfg.train.max_val_cells, replicate_cells = 20_000, 30_000
        models_dir = os.path.join(QUICK_ROOT, "models")
        training_dir = os.path.join(QUICK_ROOT, "training")

    factory = get_model_factory(args.model)
    factory(cfg.model)  # clear error right away if src/model.py is missing

    log.info("Loading cell data from %s ...", cfg.data.processed_dir)
    X_tr, y_tr = load_cell_split("train", cfg.data.processed_dir)
    X_va, y_va = load_cell_split("val", cfg.data.processed_dir)
    train_all(cfg, args.seeds, models_dir=models_dir, training_dir=training_dir,
              replicate_cells=replicate_cells, model_factory=factory,
              model_source=args.model, data=(X_tr, y_tr, X_va, y_va), resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
