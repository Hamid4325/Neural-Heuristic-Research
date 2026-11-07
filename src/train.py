"""CPU training loop for the neural heuristic network.

Owner: Muhammad Abdullah Saeed. Implements plan Section 6.2 and the checkpoint
contract in Section 3.5.

What it does
------------
* Loads the cell-level train/val splits (src.data.load_cell_split).
* Standardises the target h*(n)/N with the TRAIN mean/std (both stored in the checkpoint).
* Trains with manual mini-batches over in-memory tensors (no DataLoader workers).
* Loss = SmoothL1(beta=0.1)(prediction, standardised target)
         + lambda_aux * CrossEntropy(bin_logits, distance_bin)
* AdamW, ReduceLROnPlateau on validation MAE, gradient clipping at 1.0,
  early stopping (patience 5), and a random subsample of the train cells every epoch.
* Saves the best checkpoint (by validation MAE), history.csv, config.yaml, metrics.json.

Usage (from the repository root)
--------------------------------
    python -m src.train --config configs/default.yaml --output models/run_default
    python -m src.train --config configs/final.yaml --seed 42 \
        --output models/neural_heuristic_seed42.pt

`--output` is either a directory (checkpoint saved as <dir>/best.pt, logs in <dir>) or a
path ending in .pt (checkpoint saved there, logs in results/training/<name>/).

Units: val_mae / val_rmse are reported on the ORIGINAL target scale, h*(n)/N.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import math
import os
import subprocess
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.config import Config, ModelConfig, load_config, save_config
from src.data import load_cell_split
from src.utils import Timer, configure_logging, save_csv, save_json, set_seed

log = logging.getLogger(__name__)

HISTORY_COLUMNS = [
    "epoch", "train_loss", "train_reg_loss", "train_aux_loss",
    "val_loss", "val_mae", "val_rmse", "val_bin_accuracy", "val_macro_f1",
    "learning_rate", "epoch_seconds",
]
EVAL_CHUNK = 65536
MODEL_FACTORY = Callable[[ModelConfig], nn.Module]


# ---------------------------------------------------------------------------
# Distance bins (auxiliary classification head)
# ---------------------------------------------------------------------------
def make_bin_edges(y: np.ndarray, n_bins: int, strategy: str = "quantile") -> list[float]:
    """Return the `n_bins - 1` internal bin edges, computed on TRAIN targets only.

    quantile : equal-frequency bins (every bin has about the same number of train cells)
    uniform  : equal-width bins between min(y) and max(y)
    Edges live in the h*(n)/N scale (not the standardised scale).
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    if strategy == "quantile":
        edges = np.quantile(y, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    elif strategy == "uniform":
        edges = np.linspace(float(y.min()), float(y.max()), n_bins + 1)[1:-1]
    else:
        raise ValueError(f"bin_strategy must be 'quantile' or 'uniform', got {strategy!r}")
    edges = np.asarray(edges, dtype=np.float64)
    if len(np.unique(edges)) != n_bins - 1:
        raise ValueError("Bin edges are not distinct; the target has too many tied values")
    return [float(e) for e in edges]


def assign_bins(y: np.ndarray, edges: list[float]) -> np.ndarray:
    """Bin index in [0, len(edges)] for every target value (int64)."""
    return np.searchsorted(np.asarray(edges), y, side="right").astype(np.int64)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-averaged F1 over all `n_classes` classes (a class with no support scores 0)."""
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes)
    cm = cm.reshape(n_classes, n_classes).astype(np.float64)
    tp = np.diag(cm)
    pred_total, true_total = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(tp, pred_total, out=np.zeros(n_classes), where=pred_total > 0)
    recall = np.divide(tp, true_total, out=np.zeros(n_classes), where=true_total > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n_classes), where=denom > 0)
    return float(f1.mean())


# ---------------------------------------------------------------------------
# Model / output helpers
# ---------------------------------------------------------------------------
def default_model_factory(model_cfg: ModelConfig) -> nn.Module:
    """Build Hamid's NeuralHeuristicNet from src/model.py (plan Section 3.1)."""
    try:
        from src.model import ModelConfig as NetConfig
        from src.model import NeuralHeuristicNet
    except ModuleNotFoundError as exc:
        if exc.name != "src.model":
            raise
        raise ModuleNotFoundError(
            "src/model.py not found. It is Hamid's file: wait until his PR is merged into "
            "deliverable-2, then run `git fetch origin` and `git rebase origin/deliverable-2`."
        ) from exc
    return NeuralHeuristicNet(NetConfig(**dataclasses.asdict(model_cfg)))


def resolve_outputs(
    output: str, results_dir: Optional[str] = None, results_root: str = "results"
) -> tuple[str, str]:
    """Return (checkpoint_path, run_dir) for an --output value (directory or .pt file)."""
    output = os.fspath(output)
    if output.lower().endswith(".pt"):
        stem = os.path.splitext(os.path.basename(output))[0]
        return output, results_dir or os.path.join(results_root, "training", stem)
    return os.path.join(output, "best.pt"), results_dir or output


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _reg_loss(pred: torch.Tensor, target: torch.Tensor, cfg: Config, reduction: str):
    if cfg.train.reg_loss == "smooth_l1":
        return F.smooth_l1_loss(pred, target, beta=cfg.train.smooth_l1_beta, reduction=reduction)
    if cfg.train.reg_loss == "mse":
        return F.mse_loss(pred, target, reduction=reduction)
    raise ValueError(f"reg_loss must be 'smooth_l1' or 'mse', got {cfg.train.reg_loss!r}")


def _check_outputs(pred: torch.Tensor, logits: torch.Tensor, batch: int, n_bins: int) -> None:
    if tuple(pred.shape) != (batch, 1) or tuple(logits.shape) != (batch, n_bins):
        raise ValueError(
            "Model must return (y_pred (B, 1), bin_logits (B, n_bins)); got "
            f"{tuple(pred.shape)} and {tuple(logits.shape)} for B={batch}, n_bins={n_bins}"
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y_std: torch.Tensor,
    y_raw: torch.Tensor,
    bins: torch.Tensor,
    cfg: Config,
    target_mean: float,
    target_std: float,
) -> dict:
    """Validation loss and metrics (MAE/RMSE in h*/N units, bin accuracy, macro-F1)."""
    model.eval()
    n = len(X)
    reg_sum = aux_sum = abs_sum = sq_sum = 0.0
    correct = 0
    preds_bin = []
    for start in range(0, n, EVAL_CHUNK):
        sl = slice(start, start + EVAL_CHUNK)
        pred, logits = model(X[sl])
        pred = pred.squeeze(1)
        reg_sum += _reg_loss(pred, y_std[sl], cfg, "sum").item()
        aux_sum += F.cross_entropy(logits, bins[sl], reduction="sum").item()
        err = (pred.double() * target_std + target_mean) - y_raw[sl].double()
        abs_sum += err.abs().sum().item()
        sq_sum += (err * err).sum().item()
        arg = logits.argmax(dim=1)
        correct += int((arg == bins[sl]).sum().item())
        preds_bin.append(arg.cpu().numpy())
    reg, aux = reg_sum / n, aux_sum / n
    return {
        "loss": reg + cfg.train.lambda_aux * aux,
        "mae": abs_sum / n,
        "rmse": math.sqrt(sq_sum / n),
        "bin_accuracy": correct / n,
        "macro_f1": macro_f1(bins.cpu().numpy(), np.concatenate(preds_bin), cfg.model.n_bins),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    cfg: Config,
    output: str,
    *,
    results_dir: Optional[str] = None,
    model_factory: Optional[MODEL_FACTORY] = None,
    data: Optional[tuple] = None,
) -> dict:
    """Train a model and return a summary dict.

    data: optional (X_train, y_train, X_val, y_val) numpy arrays. When None the cell
          splits are loaded from cfg.data.processed_dir (used by tests and tuning).
    """
    tc = cfg.train
    set_seed(tc.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    ckpt_path, run_dir = resolve_outputs(output, results_dir, tc.results_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    # ----- data -------------------------------------------------------------
    if data is None:
        X_tr, y_tr = load_cell_split("train", cfg.data.processed_dir)
        X_va, y_va = load_cell_split("val", cfg.data.processed_dir)
    else:
        X_tr, y_tr, X_va, y_va = data
    for name, X, y in (("train", X_tr, y_tr), ("val", X_va, y_va)):
        if X.ndim != 2 or X.shape[1] != cfg.model.input_dim or len(X) != len(y):
            raise ValueError(f"{name}: expected X (M, {cfg.model.input_dim}) and y (M,), "
                             f"got {X.shape} and {y.shape}")
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise ValueError(f"{name} split contains NaN or infinite values")

    if tc.max_val_cells and len(X_va) > tc.max_val_cells:
        # Fixed subset (independent of the training seed) so tuning runs are comparable.
        rng = np.random.default_rng(cfg.data.split_seed)
        keep = np.sort(rng.choice(len(X_va), size=tc.max_val_cells, replace=False))
        X_va, y_va = X_va[keep], y_va[keep]

    if tc.target_transform == "standardised":
        target_mean = float(y_tr.mean(dtype=np.float64))
        target_std = float(y_tr.std(dtype=np.float64))
    elif tc.target_transform == "raw":
        target_mean, target_std = 0.0, 1.0
    else:
        raise ValueError(f"target_transform must be 'raw' or 'standardised', "
                         f"got {tc.target_transform!r}")
    if target_std <= 0.0:
        raise ValueError("Target has zero variance")

    bin_edges = make_bin_edges(y_tr, cfg.model.n_bins, tc.bin_strategy)

    def tensor(a: np.ndarray, dtype=np.float32) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a, dtype=dtype))

    X_tr_t, X_va_t = tensor(X_tr), tensor(X_va)
    y_tr_t = tensor((y_tr - target_mean) / target_std)
    y_va_t, y_va_raw = tensor((y_va - target_mean) / target_std), tensor(y_va)
    bins_tr_t, bins_va_t = tensor(assign_bins(y_tr, bin_edges), np.int64), \
        tensor(assign_bins(y_va, bin_edges), np.int64)

    # ----- model / optimiser ------------------------------------------------
    model = (model_factory or default_model_factory)(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=tc.lr_factor, patience=tc.lr_patience
    )

    n_train = len(X_tr_t)
    per_epoch = n_train if not tc.max_train_cells_per_epoch or tc.max_train_cells_per_epoch <= 0 \
        else min(n_train, tc.max_train_cells_per_epoch)
    log.info("Training: %d train cells (%d per epoch), %d val cells, %d parameters, seed %d",
             n_train, per_epoch, len(X_va_t), n_params, tc.seed)

    generator = torch.Generator().manual_seed(tc.seed)
    history: list[dict] = []
    best_mae, best_epoch, bad_epochs, stopped_early = math.inf, 0, 0, False
    commit = git_commit()
    config_dict = json.loads(json.dumps(cfg.to_dict()))  # plain lists/floats only
    model_config = json.loads(json.dumps(dataclasses.asdict(cfg.model)))

    with Timer("training") as total_timer:
        for epoch in range(1, tc.epochs + 1):
            with Timer() as epoch_timer:
                model.train()
                perm = torch.randperm(n_train, generator=generator)[:per_epoch]
                loss_sum = reg_sum = aux_sum = 0.0
                for start in range(0, per_epoch, tc.batch_size):
                    idx = perm[start:start + tc.batch_size]
                    xb, yb, bb = X_tr_t[idx], y_tr_t[idx], bins_tr_t[idx]
                    pred, logits = model(xb)
                    _check_outputs(pred, logits, len(idx), cfg.model.n_bins)
                    reg = _reg_loss(pred.squeeze(1), yb, cfg, "mean")
                    aux = F.cross_entropy(logits, bb)
                    loss = reg + tc.lambda_aux * aux
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                    optimizer.step()
                    k = len(idx)
                    loss_sum += loss.item() * k
                    reg_sum += reg.item() * k
                    aux_sum += aux.item() * k
                val = evaluate(model, X_va_t, y_va_t, y_va_raw, bins_va_t, cfg,
                               target_mean, target_std)
            if not all(math.isfinite(v) for v in val.values()):
                raise FloatingPointError(f"Non-finite validation metrics at epoch {epoch}: {val}")

            lr_used = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / per_epoch,
                "train_reg_loss": reg_sum / per_epoch,
                "train_aux_loss": aux_sum / per_epoch,
                "val_loss": val["loss"],
                "val_mae": val["mae"],
                "val_rmse": val["rmse"],
                "val_bin_accuracy": val["bin_accuracy"],
                "val_macro_f1": val["macro_f1"],
                "learning_rate": lr_used,
                "epoch_seconds": epoch_timer.elapsed,
            }
            history.append(row)
            save_csv(os.path.join(run_dir, "history.csv"), history, HISTORY_COLUMNS)
            log.info("epoch %2d/%d | train %.4f | val_loss %.4f | val_mae %.4f | "
                     "bin_acc %.3f | macro_f1 %.3f | lr %.1e | %.1fs",
                     epoch, tc.epochs, row["train_loss"], row["val_loss"], row["val_mae"],
                     row["val_bin_accuracy"], row["val_macro_f1"], lr_used,
                     row["epoch_seconds"])

            scheduler.step(val["mae"])
            if val["mae"] < best_mae:
                best_mae, best_epoch, bad_epochs = val["mae"], epoch, 0
                torch.save({
                    "model_state": copy.deepcopy(model.state_dict()),
                    "model_config": model_config,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "bin_edges": bin_edges,
                    "config": config_dict,
                    "git_commit": commit,
                    "seed": tc.seed,
                    "best_epoch": epoch,
                    "val_mae": best_mae,
                }, ckpt_path)
            else:
                bad_epochs += 1
                if bad_epochs >= tc.patience:
                    stopped_early = True
                    log.info("Early stopping: no val_mae improvement for %d epochs", bad_epochs)
                    break

    summary = {
        "checkpoint": ckpt_path,
        "run_dir": run_dir,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "best_val_bin_accuracy": history[best_epoch - 1]["val_bin_accuracy"],
        "best_val_macro_f1": history[best_epoch - 1]["val_macro_f1"],
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "n_params": n_params,
        "train_seconds": total_timer.elapsed,
        "seed": tc.seed,
        "git_commit": commit,
        "history": history,
    }
    save_json(os.path.join(run_dir, "metrics.json"),
              {k: v for k, v in summary.items() if k != "history"})
    log.info("Done. Best val_mae %.4f at epoch %d -> %s", best_mae, best_epoch, ckpt_path)
    return summary


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the neural heuristic network (CPU)")
    p.add_argument("--config", default=None, help="YAML config (default: built-in defaults)")
    p.add_argument("--output", required=True,
                   help="Output directory, or a .pt path for the checkpoint")
    p.add_argument("--results-dir", default=None, help="Where history.csv/config/metrics go")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-train-cells-per-epoch", "--max-train-cells",
                   dest="max_train_cells_per_epoch", type=int, default=None,
                   help="Random train cells used per epoch (default from config: 1,500,000)")
    p.add_argument("--max-val-cells", type=int, default=None,
                   help="Use a fixed random subset of the validation cells")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging()
    cfg = load_config(args.config)
    for name in ("seed", "epochs", "max_train_cells_per_epoch", "max_val_cells"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg.train, name, value)
    train(cfg, args.output, results_dir=args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
