"""Feature, loss and architecture variants for the model ablation study."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

FEATURE_GROUPS: dict[str, tuple[int, ...]] = {
    "coordinates": (0, 1),
    "goal_offsets": (2, 3),
    "wall_window": tuple(range(4, 29)),
}

ARCHITECTURE_VARIANTS: dict[str, tuple[int, ...]] = {
    "two_hidden": (256, 256),
    "three_hidden": (256, 256, 128),
}


def feature_indices(drop: Iterable[str] = ()) -> list[int]:
    """Return the kept feature indices after dropping named groups."""
    dropped_groups = set(drop)
    unknown = dropped_groups - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")

    dropped_indices = {
        index
        for group in dropped_groups
        for index in FEATURE_GROUPS[group]
    }
    return [index for index in range(29) if index not in dropped_indices]


def select_features(
    features: np.ndarray,
    drop: Iterable[str] = (),
) -> np.ndarray:
    """Return a feature matrix with the named groups removed."""
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[1] != 29:
        raise ValueError("features must have shape (n_samples, 29)")
    return matrix[:, feature_indices(drop)]


def asymmetric_smooth_l1_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 0.1,
    margin: float = 0.0,
    lambda_over: float = 1.0,
) -> torch.Tensor:
    """SmoothL1 loss with an extra penalty for overestimating the target."""
    pred = predictions.reshape(-1)
    target = targets.reshape(-1)
    base = F.smooth_l1_loss(pred, target, beta=beta)
    over = torch.relu(pred - target - margin)
    return base + lambda_over * (over**2).mean()


def ablation_configs() -> list[dict[str, object]]:
    """Return the seven planned ablation configurations."""
    return [
        {
            "name": "drop_wall_window",
            "kind": "feature",
            "drop": ("wall_window",),
        },
        {
            "name": "drop_goal_offsets",
            "kind": "feature",
            "drop": ("goal_offsets",),
        },
        {
            "name": "drop_coordinates",
            "kind": "feature",
            "drop": ("coordinates",),
        },
        {
            "name": "smooth_l1",
            "kind": "loss",
            "loss": "smooth_l1",
        },
        {
            "name": "asymmetric",
            "kind": "loss",
            "loss": "asymmetric",
        },
        {
            "name": "two_hidden",
            "kind": "architecture",
            "hidden_dims": ARCHITECTURE_VARIANTS["two_hidden"],
        },
        {
            "name": "three_hidden",
            "kind": "architecture",
            "hidden_dims": ARCHITECTURE_VARIANTS["three_hidden"],
        },
    ]


def output_file_for_kind(kind: str) -> str:
    """Return the CSV filename for an ablation kind."""
    mapping = {
        "feature": "feature_ablation.csv",
        "loss": "loss_ablation.csv",
        "architecture": "architecture_ablation.csv",
    }
    if kind not in mapping:
        raise ValueError(f"Unknown ablation kind {kind!r}")
    return mapping[kind]


def rows_to_csv(rows: Sequence[dict[str, object]], path: str) -> None:
    """Write ablation rows to CSV with a stable column order."""
    if not rows:
        raise ValueError("rows must not be empty")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
