"""Configuration: YAML loader and dataclasses for data, model, training and evaluation.

Owner: Muhammad Abdullah Saeed (the model section is shared with Muhammad Hamid Abad;
keep `ModelConfig` identical to the one in src/model.py, see plan Section 3.1).

Usage:
    from src.config import load_config
    cfg = load_config("configs/default.yaml")   # or load_config() for pure defaults
    cfg.train.batch_size, cfg.model.hidden_dims, cfg.data.processed_dir
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, Optional

import yaml

# Default folders (relative to the repository root; run commands from the root).
DEFAULT_CLEANED_DIR = "data/cleaned"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_MODELS_DIR = "models"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_FIGURES_DIR = "figures"


@dataclass
class DataConfig:
    cleaned_dir: str = DEFAULT_CLEANED_DIR
    processed_dir: str = DEFAULT_PROCESSED_DIR
    manifest_path: str = f"{DEFAULT_PROCESSED_DIR}/split_manifest.json"
    manifest_copy_path: str = f"{DEFAULT_RESULTS_DIR}/manifests/split_manifest.json"
    split_seed: int = 42


@dataclass
class ModelConfig:
    # Must match the frozen contract in plan Section 3.1 / src/model.py.
    input_dim: int = 29
    hidden_dims: tuple[int, ...] = (256, 256, 128)
    dropout: float = 0.1
    activation: str = "relu"
    n_bins: int = 10


@dataclass
class TrainConfig:
    # Starting values from plan Section 6.2; Day 3 tuning decides the final ones.
    seed: int = 42
    epochs: int = 20
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    reg_loss: str = "smooth_l1"          # smooth_l1 | mse
    smooth_l1_beta: float = 0.1
    lambda_aux: float = 0.1              # weight of the 10-bin classification loss
    grad_clip: float = 1.0
    patience: int = 5                    # early stopping (epochs)
    lr_factor: float = 0.5               # ReduceLROnPlateau
    lr_patience: int = 2
    max_train_cells_per_epoch: int = 1_500_000
    max_val_cells: Optional[int] = None  # None = use the whole validation split
    target_transform: str = "standardised"  # raw | standardised
    models_dir: str = DEFAULT_MODELS_DIR
    results_dir: str = DEFAULT_RESULTS_DIR


@dataclass
class EvalConfig:
    split: str = "test"
    alphas: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9, 1.0)
    batch_size: int = 8192
    results_dir: str = DEFAULT_RESULTS_DIR
    figures_dir: str = DEFAULT_FIGURES_DIR


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "Config":
        raw = dict(raw or {})
        sections = {f.name: f.type for f in fields(cls)}
        unknown = set(raw) - set(sections)
        if unknown:
            raise ValueError(f"Unknown top-level config section(s): {sorted(unknown)}")
        return cls(
            data=_build(DataConfig, raw.get("data"), "data"),
            model=_build(ModelConfig, raw.get("model"), "model"),
            train=_build(TrainConfig, raw.get("train"), "train"),
            eval=_build(EvalConfig, raw.get("eval"), "eval"),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build(cls, values: Optional[dict], section: str):
    """Create a section dataclass from a dict, rejecting typos and fixing list->tuple."""
    values = dict(values or {})
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown key(s) in config section '{section}': {sorted(unknown)}")
    for name, value in values.items():
        if isinstance(value, list) and isinstance(known[name].default, tuple):
            values[name] = tuple(value)
    return cls(**values)


def load_yaml(path: str) -> dict:
    """Read a YAML file and return a dict (empty file -> {})."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level")
    return data


def load_config(path: Optional[str] = None) -> Config:
    """Load a Config from YAML, or return the defaults when `path` is None."""
    if path is None:
        return Config()
    return Config.from_dict(load_yaml(path))


def save_config(cfg: Config, path: str) -> None:
    """Write a Config to YAML (used to store the exact training config next to a run)."""
    import os

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
