"""Staged CPU hyperparameter search (plan Section 6.3).

Owner: Muhammad Abdullah Saeed.

Protocol (all numbers live in configs/tuning_grid.yaml):
  Stage A  12 configurations sampled from the grid (the default config + 11 random ones),
           400k train cells per epoch, 100k val cells, 5 epochs each.
  Stage B  the top 3 of stage A, 1.5M train cells per epoch, 10 epochs each.
  Stage C  the best of stage B on the full train split, up to 20 epochs with early
           stopping. Its checkpoint is the first real checkpoint (models/tuned_best.pt).

Configurations are ranked by validation MAE on the original h*(n)/N scale, so runs with
different target transforms are compared fairly.

Outputs (results/tuning/, committed):
    <run_id>/config.json      hyperparameters + stage settings of the run
    <run_id>/history.csv      per-epoch training curve
    <run_id>/metrics.json     best epoch, best validation metrics, parameter count, ...
    tuning_summary.csv        one row per run (stages A, B and C)
    best_config.json / .yaml  the winning configuration (use it for configs/final.yaml)

Usage (repository root):
    python -m src.tune --config configs/tuning_grid.yaml
    python -m src.tune --config configs/tuning_grid.yaml --resume      # skip finished runs
    python -m src.tune --config configs/tuning_grid.yaml --quick       # tiny pipeline check
    python -m src.tune --config configs/tuning_grid.yaml --model standin --quick
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import logging
import os
from typing import Callable, Optional

import numpy as np
from torch import nn

from src.config import Config, ModelConfig, load_config, load_yaml, save_config
from src.data import load_cell_split
from src.train import default_model_factory, train
from src.utils import configure_logging, load_json, save_csv, save_json

log = logging.getLogger(__name__)

GRID_KEYS = (
    "hidden_dims", "activation", "dropout", "batch_size", "learning_rate",
    "weight_decay", "reg_loss", "lambda_aux", "target_transform",
)
MODEL_KEYS = {"hidden_dims", "activation", "dropout"}   # the rest belongs to cfg.train
NUMERIC_KEYS = {"dropout", "batch_size", "learning_rate", "weight_decay", "lambda_aux"}
STAGE_KEYS = ("epochs", "max_train_cells_per_epoch", "max_val_cells", "patience")
SUMMARY_COLUMNS = [
    "run_id", "stage", "rank_in_stage", "status", "model_source",
    "hidden_dims", "activation", "dropout", "batch_size", "learning_rate", "weight_decay",
    "reg_loss", "lambda_aux", "target_transform",
    "train_cells_per_epoch", "val_cells", "epochs_planned", "epochs_run", "best_epoch",
    "best_val_mae", "best_val_bin_accuracy", "best_val_macro_f1", "n_params", "train_seconds",
]


# ---------------------------------------------------------------------------
# Stand-in network (provisional, only for --model standin)
# ---------------------------------------------------------------------------
class StandInNet(nn.Module):
    """Plan architecture built from a ModelConfig, for use before src/model.py exists.

    Same layout as the plan: Linear -> activation -> Dropout for every hidden layer except
    the last one, then a regression head (1 output) and a bin head (n_bins outputs).
    Results obtained with it are PROVISIONAL and must be re-checked with Hamid's model.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        activations = {"relu": nn.ReLU, "gelu": nn.GELU}
        if cfg.activation not in activations:
            raise ValueError(f"activation must be one of {sorted(activations)}")
        layers, width = [], cfg.input_dim
        for i, hidden in enumerate(cfg.hidden_dims):
            layers += [nn.Linear(width, hidden), activations[cfg.activation]()]
            if i < len(cfg.hidden_dims) - 1:
                layers.append(nn.Dropout(cfg.dropout))
            width = hidden
        self.trunk = nn.Sequential(*layers)
        self.reg_head = nn.Linear(width, 1)
        self.bin_head = nn.Linear(width, cfg.n_bins)

    def forward(self, x):
        h = self.trunk(x)
        return self.reg_head(h), self.bin_head(h)


def get_model_factory(name: str) -> Callable[[ModelConfig], nn.Module]:
    if name == "hamid":
        return default_model_factory
    if name == "standin":
        log.warning("=" * 70)
        log.warning("PROVISIONAL RUN: using the built-in stand-in network, NOT src/model.py.")
        log.warning("Re-check the results with Hamid's model before committing or reporting.")
        log.warning("=" * 70)
        return StandInNet
    raise ValueError(f"--model must be 'hamid' or 'standin', got {name!r}")


# ---------------------------------------------------------------------------
# Tuning specification (configs/tuning_grid.yaml)
# ---------------------------------------------------------------------------
def validate_spec(spec: dict) -> dict:
    """Check the tuning YAML structure and return it (raises ValueError with a clear message)."""
    for key in ("grid", "stage_a", "stage_b", "stage_c", "quick", "output"):
        if key not in spec:
            raise ValueError(f"tuning config is missing the '{key}' section")
    grid = spec["grid"]
    unknown = set(grid) - set(GRID_KEYS)
    missing = set(GRID_KEYS) - set(grid)
    if unknown or missing:
        raise ValueError(f"grid keys must be exactly {list(GRID_KEYS)}; "
                         f"unknown={sorted(unknown)}, missing={sorted(missing)}")
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid['{key}'] must be a non-empty list")
        if key in NUMERIC_KEYS and not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            raise ValueError(f"grid['{key}'] must contain numbers (write 0.0001, not 1e-4): "
                             f"{values}")
    sections = [("stage_a", "n_configs"), ("stage_b", "top_k"), ("stage_c", None)]
    for source in (spec, spec["quick"]):
        for name, extra in sections:
            stage = source.get(name)
            if not isinstance(stage, dict):
                raise ValueError(f"missing stage settings '{name}'")
            required = list(STAGE_KEYS) + ([extra] if extra else [])
            absent = [k for k in required if k not in stage]
            if absent:
                raise ValueError(f"{name} is missing {absent}")
    spec.setdefault("base_config", "configs/default.yaml")
    spec.setdefault("seed", 42)
    spec.setdefault("include_default", True)
    return spec


def load_tuning_spec(path: str) -> dict:
    return validate_spec(load_yaml(path))


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
def full_grid(grid: dict) -> list[dict]:
    """Every combination of the grid, in a fixed order."""
    return [dict(zip(GRID_KEYS, combo))
            for combo in itertools.product(*[grid[k] for k in GRID_KEYS])]


def hyperparams_of(cfg: Config) -> dict:
    """The nine tuned hyperparameters of a Config (the 'default' configuration)."""
    return {
        "hidden_dims": list(cfg.model.hidden_dims),
        "activation": cfg.model.activation,
        "dropout": cfg.model.dropout,
        "batch_size": cfg.train.batch_size,
        "learning_rate": cfg.train.learning_rate,
        "weight_decay": cfg.train.weight_decay,
        "reg_loss": cfg.train.reg_loss,
        "lambda_aux": cfg.train.lambda_aux,
        "target_transform": cfg.train.target_transform,
    }


def sample_configs(grid: dict, n: int, seed: int, include: Optional[dict] = None) -> list[dict]:
    """Pick `n` distinct configurations: `include` first (if it is in the grid), then random."""
    combos = full_grid(grid)
    chosen: list[dict] = []
    if include is not None:
        if include in combos:
            chosen.append(include)
        else:
            log.warning("The default configuration is not inside the grid; not including it.")
    rest = [c for c in combos if c not in chosen]
    need = max(0, n - len(chosen))
    if need >= len(rest):
        chosen += rest
    else:
        picks = np.random.default_rng(seed).choice(len(rest), size=need, replace=False)
        chosen += [rest[int(i)] for i in picks]
    return [copy.deepcopy(c) for c in chosen[:n]]


def apply_hyperparams(base: Config, hp: dict, stage: dict, seed: int) -> Config:
    """Return a copy of `base` with the hyperparameters and stage settings applied."""
    cfg = copy.deepcopy(base)
    for key in GRID_KEYS:
        value = tuple(hp[key]) if key == "hidden_dims" else hp[key]
        setattr(cfg.model if key in MODEL_KEYS else cfg.train, key, value)
    cfg.train.seed = seed
    cfg.train.epochs = stage["epochs"]
    cfg.train.max_train_cells_per_epoch = stage["max_train_cells_per_epoch"]
    cfg.train.max_val_cells = stage["max_val_cells"]
    cfg.train.patience = stage["patience"]
    return cfg


def make_run_id(prefix: str, hp: dict, stage: dict) -> str:
    """Readable, stable id: the same hyperparameters + stage settings give the same id."""
    payload = json.dumps({"hp": hp, "stage": {k: stage.get(k) for k in STAGE_KEYS}},
                         sort_keys=True)
    return f"{prefix}_{hashlib.sha1(payload.encode()).hexdigest()[:6]}"


# ---------------------------------------------------------------------------
# Running one configuration
# ---------------------------------------------------------------------------
def _cells(value) -> str:
    return "all" if not value else str(value)


def _run_one(
    prefix: str, stage_name: str, hp: dict, stage: dict, base_cfg: Config, seed: int,
    results_dir: str, model_factory, model_source: str, data, resume: bool,
    checkpoint: Optional[str] = None,
) -> dict:
    run_id = make_run_id(prefix, hp, stage)
    run_dir = os.path.join(results_dir, run_id)
    config_json = {
        "run_id": run_id, "stage": stage_name, "model_source": model_source, "seed": seed,
        "hyperparameters": hp,
        "stage_settings": {k: stage.get(k) for k in STAGE_KEYS},
    }
    cfg_path, metrics_path = (os.path.join(run_dir, "config.json"),
                              os.path.join(run_dir, "metrics.json"))

    reused = False
    if resume and os.path.exists(metrics_path) and os.path.exists(cfg_path) \
            and load_json(cfg_path) == json.loads(json.dumps(config_json)):
        metrics = load_json(metrics_path)
        reused = True
        log.info("[%s] %s: finished earlier, reusing it", stage_name, run_id)
    else:
        save_json(cfg_path, config_json)  # written first, so an interrupted run is visible
        cfg = apply_hyperparams(base_cfg, hp, stage, seed)
        log.info("[%s] %s: %s", stage_name, run_id, json.dumps(hp))
        try:
            summary = train(cfg, checkpoint or run_dir, results_dir=run_dir,
                            model_factory=model_factory, data=data)
            metrics = {k: v for k, v in summary.items() if k != "history"}
        except FloatingPointError as exc:  # diverged run: record it and carry on
            metrics = {"failed": True, "error": str(exc)}
            save_json(metrics_path, metrics)
            log.warning("[%s] %s diverged: %s", stage_name, run_id, exc)

    failed = bool(metrics.get("failed"))
    return {
        "run_id": run_id, "stage": stage_name, "rank_in_stage": "",
        "status": ("failed" if failed else "ok") + (" (reused)" if reused else ""),
        "model_source": model_source,
        **{k: ("-".join(str(d) for d in hp[k]) if k == "hidden_dims" else hp[k])
           for k in GRID_KEYS},
        "train_cells_per_epoch": _cells(stage["max_train_cells_per_epoch"]),
        "val_cells": _cells(stage["max_val_cells"]),
        "epochs_planned": stage["epochs"],
        "epochs_run": metrics.get("epochs_run", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "best_val_mae": metrics.get("best_val_mae", ""),
        "best_val_bin_accuracy": metrics.get("best_val_bin_accuracy", ""),
        "best_val_macro_f1": metrics.get("best_val_macro_f1", ""),
        "n_params": metrics.get("n_params", ""),
        "train_seconds": metrics.get("train_seconds", ""),
        "_hp": hp,
        "_ok": not failed,
    }


def rank_records(records: list[dict]) -> list[dict]:
    """Successful runs of one stage, best (lowest validation MAE) first."""
    ok = [r for r in records if r["_ok"]]
    return sorted(ok, key=lambda r: r["best_val_mae"])


def _write_summary(results_dir: str, all_records: list[dict]) -> None:
    for stage in ("A", "B", "C"):
        for i, rec in enumerate(rank_records([r for r in all_records if r["stage"] == stage]), 1):
            rec["rank_in_stage"] = i
    save_csv(os.path.join(results_dir, "tuning_summary.csv"), all_records, SUMMARY_COLUMNS)


# ---------------------------------------------------------------------------
# The staged search
# ---------------------------------------------------------------------------
def run_tuning(
    spec: dict,
    base_cfg: Config,
    *,
    model_factory: Callable[[ModelConfig], nn.Module],
    model_source: str,
    data: tuple,
    results_dir: str,
    final_checkpoint: str,
    stages: Optional[dict] = None,
    resume: bool = False,
) -> dict:
    """Run stages A, B and C and write all outputs. Returns {'records', 'best'}.

    stages: the stage settings to use (default: spec['stage_a'/'stage_b'/'stage_c']).
    """
    stages = stages or {k: spec[k] for k in ("stage_a", "stage_b", "stage_c")}
    seed = spec["seed"]
    model_factory(base_cfg.model)  # fail fast (before any training) if the model is missing
    os.makedirs(results_dir, exist_ok=True)

    common = dict(base_cfg=base_cfg, seed=seed, results_dir=results_dir,
                  model_factory=model_factory, model_source=model_source, data=data,
                  resume=resume)
    records: list[dict] = []

    # ----- Stage A ----------------------------------------------------------
    a = stages["stage_a"]
    default_hp = hyperparams_of(base_cfg) if spec["include_default"] else None
    configs = sample_configs(spec["grid"], a["n_configs"], seed, default_hp)
    log.info("===== Stage A: %d configurations =====", len(configs))
    stage_a = []
    for i, hp in enumerate(configs, 1):
        stage_a.append(_run_one(f"A{i:02d}", "A", hp, a, **common))
        records.append(stage_a[-1])
        _write_summary(results_dir, records)
    ranked_a = rank_records(stage_a)
    if not ranked_a:
        raise RuntimeError("Stage A: every configuration failed")

    # ----- Stage B ----------------------------------------------------------
    b = stages["stage_b"]
    top = ranked_a[:b["top_k"]]
    log.info("===== Stage B: top %d of stage A =====", len(top))
    stage_b = []
    for i, rec in enumerate(top, 1):
        stage_b.append(_run_one(f"B{i}", "B", rec["_hp"], b, **common))
        records.append(stage_b[-1])
        _write_summary(results_dir, records)
    ranked_b = rank_records(stage_b)
    if not ranked_b:
        raise RuntimeError("Stage B: every configuration failed")

    # ----- Stage C ----------------------------------------------------------
    best_hp = ranked_b[0]["_hp"]
    log.info("===== Stage C: best configuration on the full train split =====")
    final = _run_one("C", "C", best_hp, stages["stage_c"], checkpoint=final_checkpoint, **common)
    records.append(final)
    _write_summary(results_dir, records)
    if not final["_ok"]:
        raise RuntimeError("Stage C diverged; see its metrics.json")

    # ----- Winner -----------------------------------------------------------
    best_cfg = apply_hyperparams(base_cfg, best_hp, stages["stage_c"], seed)
    save_json(os.path.join(results_dir, "best_config.json"), {
        "hyperparameters": best_hp, "model_source": model_source,
        "stage_c_run_id": final["run_id"], "stage_c_best_val_mae": final["best_val_mae"],
        "checkpoint": final_checkpoint,
    })
    save_config(best_cfg, os.path.join(results_dir, "best_config.yaml"))

    log.info("===== Done =====")
    for stage in ("A", "B", "C"):
        for rec in rank_records([r for r in records if r["stage"] == stage])[:3]:
            log.info("  %s #%s %-12s val_mae %.4f  %s", stage, rec["rank_in_stage"],
                     rec["run_id"], rec["best_val_mae"], json.dumps(rec["_hp"]))
    log.info("Best configuration: %s", json.dumps(best_hp))
    log.info("Summary: %s", os.path.join(results_dir, "tuning_summary.csv"))
    log.info("First real checkpoint: %s", final_checkpoint)
    return {"records": records, "best": final}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Staged hyperparameter search (CPU)")
    p.add_argument("--config", default="configs/tuning_grid.yaml")
    p.add_argument("--model", choices=("hamid", "standin"), default="hamid",
                   help="'hamid' = src/model.py (default); 'standin' = provisional built-in net")
    p.add_argument("--quick", action="store_true",
                   help="Tiny version of the whole pipeline (uses the 'quick' section)")
    p.add_argument("--resume", action="store_true", help="Reuse runs that already finished")
    p.add_argument("--results-dir", default=None, help="Override the output folder")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging()
    spec = load_tuning_spec(args.config)
    base_cfg = load_config(spec["base_config"])
    factory = get_model_factory(args.model)
    factory(base_cfg.model)  # clear error right away if src/model.py is missing

    out = spec["output"]
    results_dir = args.results_dir or (out["quick_results_dir"] if args.quick
                                       else out["results_dir"])
    final_checkpoint = (os.path.join(results_dir, "tuned_best_quick.pt") if args.quick
                        else out["final_checkpoint"])
    stages = spec["quick"] if args.quick else None

    log.info("Loading cell data from %s ...", base_cfg.data.processed_dir)
    X_tr, y_tr = load_cell_split("train", base_cfg.data.processed_dir)
    X_va, y_va = load_cell_split("val", base_cfg.data.processed_dir)

    run_tuning(spec, base_cfg, model_factory=factory, model_source=args.model,
               data=(X_tr, y_tr, X_va, y_va), results_dir=results_dir,
               final_checkpoint=final_checkpoint, stages=stages, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
