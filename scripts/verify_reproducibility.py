"""Verify that a final checkpoint can be reproduced by training again (plan Section 6.5).

Owner: Muhammad Abdullah Saeed.

It re-trains one of the final models from the exact configuration that run saved
(results/training/neural_heuristic_seed<S>/config.yaml, written by src.train) into a
temporary folder, then compares the new run with the stored one:

  * the per-epoch history (train loss and validation MAE),
  * and, when the whole run is repeated, the checkpoint itself (weights, target statistics,
    bin edges, best epoch).

A repeat of the whole run proves the training command reproduces the checkpoint. With
`--epochs N` only the first N epochs are repeated and compared (much faster, no weight
comparison). Differences up to `--tol` are accepted: results are bit-identical on the same
machine, but can differ slightly across machines with different CPU thread counts.

Usage (repository root):
    python scripts/verify_reproducibility.py --seed 44 --epochs 3     # quick evidence
    python scripts/verify_reproducibility.py --seed 44                # full repeat
    python scripts/verify_reproducibility.py --seed 44 --model standin --quick   # dry run
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import load_cell_split  # noqa: E402
from src.train import train  # noqa: E402
from src.tune import get_model_factory  # noqa: E402
from src.utils import configure_logging  # noqa: E402

QUICK_ROOT = os.path.join("results", "tuning_quick", "final_quick")  # already git-ignored


def read_history(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def verify(
    seed: int,
    *,
    models_dir: str,
    training_dir: str,
    model_factory: Callable,
    data: Optional[tuple] = None,
    epochs: Optional[int] = None,
    tol: float = 1e-5,
) -> list[tuple[bool, str]]:
    """Re-train seed `seed` and return a list of (passed, message) results."""
    stem = f"neural_heuristic_seed{seed}"
    run_dir = os.path.join(training_dir, stem)
    checkpoint = os.path.join(models_dir, stem + ".pt")
    for path in (os.path.join(run_dir, "config.yaml"), os.path.join(run_dir, "history.csv"),
                 checkpoint):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found (train the final models first)")

    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    full_repeat = epochs is None or epochs >= cfg.train.epochs
    if not full_repeat:
        cfg = copy.deepcopy(cfg)
        cfg.train.epochs = epochs

    original = read_history(os.path.join(run_dir, "history.csv"))
    with tempfile.TemporaryDirectory() as tmp:
        again = train(cfg, os.path.join(tmp, "repeat.pt"), results_dir=os.path.join(tmp, "run"),
                      model_factory=model_factory, data=data)
        new_history = again["history"]
        new_ckpt = torch.load(os.path.join(tmp, "repeat.pt"), map_location="cpu")
        old_ckpt = torch.load(checkpoint, map_location="cpu")

    results: list[tuple[bool, str]] = []
    n = min(len(original), len(new_history))
    results.append((n > 0, f"compared {n} epoch(s) of history"))
    for column, key in (("train_loss", "train_loss"), ("val_mae", "val_mae")):
        worst = max(abs(original[i][column] - new_history[i][key]) for i in range(n))
        results.append((worst <= tol, f"{column} matches in every epoch "
                                      f"(largest difference {worst:.2e}, tolerance {tol:.0e})"))

    if full_repeat:
        results.append((len(original) == len(new_history),
                        f"same number of epochs ({len(original)} stored, {len(new_history)} new)"))
        results.append((old_ckpt["best_epoch"] == new_ckpt["best_epoch"],
                        f"same best epoch ({old_ckpt['best_epoch']} vs {new_ckpt['best_epoch']})"))
        worst_w = max(float((old_ckpt["model_state"][k] - new_ckpt["model_state"][k]).abs().max())
                      for k in old_ckpt["model_state"])
        results.append((worst_w <= tol, f"all weights match (largest difference {worst_w:.2e})"))
        stats = max(abs(old_ckpt["target_mean"] - new_ckpt["target_mean"]),
                    abs(old_ckpt["target_std"] - new_ckpt["target_std"]))
        results.append((stats <= 1e-9, "target mean/std are identical"))
        results.append((old_ckpt["bin_edges"] == new_ckpt["bin_edges"], "bin edges are identical"))
    else:
        results.append((True, f"only the first {epochs} epochs were repeated "
                              "(weights not compared; omit --epochs for the full check)"))
    return results


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check that training reproduces a checkpoint")
    p.add_argument("--seed", type=int, default=44,
                   help="Which final model to repeat (44 is the cheapest: 1M cells per epoch)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Repeat and compare only the first N epochs")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--training-dir", default="results/training")
    p.add_argument("--model", choices=("hamid", "standin"), default="hamid")
    p.add_argument("--quick", action="store_true",
                   help=f"Dry run on the files of `train_final.py --quick` in {QUICK_ROOT}")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging(logging.WARNING)
    models_dir, training_dir = args.models_dir, args.training_dir
    if args.quick:
        models_dir = os.path.join(QUICK_ROOT, "models")
        training_dir = os.path.join(QUICK_ROOT, "training")
    elif args.model == "standin":
        print("ERROR: --model standin is only allowed together with --quick", file=sys.stderr)
        return 1

    try:
        factory = get_model_factory(args.model)
        cfg_path = os.path.join(training_dir, f"neural_heuristic_seed{args.seed}", "config.yaml")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"{cfg_path} not found (train the final models first)")
        cfg = load_config(cfg_path)
        factory(cfg.model)  # clear error right away if src/model.py is missing
        print(f"Re-training seed {args.seed} "
              f"({'first ' + str(args.epochs) + ' epochs' if args.epochs else 'full run'}) ...")
        X_tr, y_tr = load_cell_split("train", cfg.data.processed_dir)
        X_va, y_va = load_cell_split("val", cfg.data.processed_dir)
        results = verify(args.seed, models_dir=models_dir, training_dir=training_dir,
                         model_factory=factory, data=(X_tr, y_tr, X_va, y_va),
                         epochs=args.epochs, tol=args.tol)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    failed = 0
    for ok, message in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {message}")
        failed += 0 if ok else 1
    print("REPRODUCIBLE" if failed == 0 else f"{failed} CHECK(S) FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
