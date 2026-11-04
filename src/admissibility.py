"""Admissibility analysis helpers for neural heuristics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from src.features import SPLITS


def reconstruct_split_assignment(
    total_mazes: int,
    seed: int = 42,
    splits: dict[str, float] | None = None,
) -> np.ndarray:
    """Reproduce the deterministic maze-level split used by the data pipeline."""
    if total_mazes < 0:
        raise ValueError("total_mazes must be non-negative")
    if splits is None:
        splits = SPLITS

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(total_mazes)
    n_train = int(round(splits["train"] * total_mazes))
    n_val = int(round(splits["val"] * total_mazes))

    assignment = np.empty(total_mazes, dtype=object)
    assignment[shuffled[:n_train]] = "train"
    assignment[shuffled[n_train : n_train + n_val]] = "val"
    assignment[shuffled[n_train + n_val :]] = "test"
    return assignment


def heuristic_values(
    predictions: np.ndarray,
    manhattan: np.ndarray,
    alpha: float = 1.0,
    mode: str = "raw",
) -> np.ndarray:
    """Apply a search-time heuristic mode to raw cell-cost predictions."""
    pred = np.asarray(predictions, dtype=np.float64)
    man = np.asarray(manhattan, dtype=np.float64)
    if pred.shape != man.shape:
        raise ValueError("predictions and manhattan must have the same shape")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")

    if mode == "raw":
        return pred.copy()
    if mode == "scaled":
        return alpha * pred
    if mode == "clipped_manhattan":
        return np.minimum(alpha * pred, man)
    raise ValueError(f"Unknown mode {mode!r}")


def overestimation_metrics(
    predictions: np.ndarray,
    true_cost: np.ndarray,
    manhattan: np.ndarray,
    alpha: float = 1.0,
    mode: str = "raw",
    epsilon: float = 1e-6,
) -> dict[str, float | int | str]:
    """Summarise overestimation of the true cost for one heuristic mode."""
    values = heuristic_values(predictions, manhattan, alpha=alpha, mode=mode)
    true = np.asarray(true_cost, dtype=np.float64)
    if values.shape != true.shape:
        raise ValueError("predictions and true_cost must have the same shape")
    if values.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")
    if not np.isfinite(values).all() or not np.isfinite(true).all():
        raise ValueError("predictions and true_cost must be finite")

    signed = values - true
    positive = np.maximum(signed, 0.0)
    return {
        "mode": mode,
        "alpha": float(alpha),
        "n_cells": int(values.size),
        "fraction_over": float(np.mean(signed > epsilon)),
        "mean_overestimation": float(positive.mean()),
        "p95_overestimation": float(np.percentile(positive, 95)),
        "max_overestimation": float(positive.max()),
        "mae": float(np.abs(signed).mean()),
        "rmse": float(np.sqrt(np.mean(signed**2))),
    }


def evaluate_modes(
    predictions: np.ndarray,
    true_cost: np.ndarray,
    manhattan: np.ndarray,
    alphas: Sequence[float] = (0.5, 0.75, 0.9, 1.0),
) -> list[dict[str, float | int | str]]:
    """Return metrics for raw, scaled and clipped-manhattan modes."""
    rows: list[dict[str, float | int | str]] = [
        overestimation_metrics(
            predictions, true_cost, manhattan, alpha=1.0, mode="raw"
        )
    ]
    for alpha in alphas:
        for mode in ("scaled", "clipped_manhattan"):
            rows.append(
                overestimation_metrics(
                    predictions, true_cost, manhattan, alpha=alpha, mode=mode
                )
            )
    return rows


def overestimation_cdf(
    predictions: np.ndarray,
    true_cost: np.ndarray,
    manhattan: np.ndarray,
    alpha: float = 1.0,
    mode: str = "raw",
) -> tuple[np.ndarray, np.ndarray]:
    """Return the empirical CDF of signed overestimation (h - h*)."""
    values = heuristic_values(predictions, manhattan, alpha=alpha, mode=mode)
    signed = np.sort(values - np.asarray(true_cost, dtype=np.float64))
    cumulative = np.arange(1, signed.size + 1, dtype=np.float64) / signed.size
    return signed, cumulative


def summarize_by_mode(
    rows: Iterable[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    """Return rows unchanged after validating required metric keys."""
    required = {
        "mode",
        "alpha",
        "n_cells",
        "fraction_over",
        "mean_overestimation",
        "p95_overestimation",
        "max_overestimation",
        "mae",
        "rmse",
    }
    result = []
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"metric row missing keys: {sorted(missing)}")
        result.append(dict(row))
    return result
