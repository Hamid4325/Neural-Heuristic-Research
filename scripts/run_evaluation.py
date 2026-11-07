"""Unified evaluation runner across baselines, neural heuristic configurations, and alpha sweeps.

Supports both scikit-learn fallback backend and PyTorch neural checkpoint backend.
Part of Deliverable 2 for Neural Heuristic Research.
Author: Abdullah Zia
"""

import argparse
import json
import logging
import os
import sys
import joblib
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluate import iter_split_mazes, run_search_evaluation
from src.metrics import (
    regression_metrics,
    admissibility_metrics,
    classification_metrics,
    search_aggregate,
)
from src.search import ManhattanHeuristic, OracleHeuristic, ZeroHeuristic
from src.sklearn_heuristic import SklearnNeuralHeuristic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_evaluation")


def evaluate_cell_metrics(
    backend: str,
    regressor_path: str,
    classifier_path: str,
    test_data_path: str,
    output_dir: str,
    sample_size: int = 100000,
    seed: int = 42,
):
    """Compute offline cell-level regression, admissibility, and classification metrics."""
    log.info(f"Evaluating cell-level metrics on {test_data_path}...")
    test_npz = np.load(test_data_path)
    X_test, y_test = test_npz["X"], test_npz["y"]

    rng = np.random.default_rng(seed)
    n_samples = min(sample_size, len(X_test))
    idx = rng.choice(len(X_test), size=n_samples, replace=False)
    X_sub, y_sub = X_test[idx], y_test[idx]

    if backend == "sklearn":
        reg_model = joblib.load(regressor_path)
        y_pred = reg_model.predict(X_sub)
    else:
        # PyTorch checkpoint support
        import torch
        ckpt = torch.load(regressor_path, map_location="cpu")
        # Real model inference can be added here once checkpoint is provided
        raise NotImplementedError("PyTorch backend integration will execute when checkpoint is delivered.")

    # 1. Regression metrics (normalized and for representative grid size 60)
    reg_res = regression_metrics(y_sub, y_pred, grid_size=60.0)
    reg_json = os.path.join(output_dir, "regression_metrics.json")
    with open(reg_json, "w", encoding="utf-8") as f:
        json.dump(reg_res, f, indent=2)
    log.info(f"Saved regression metrics to {reg_json} (R^2={reg_res.get('r2', 0):.4f}, RMSE={reg_res.get('rmse', 0):.4f})")

    # 2. Admissibility metrics
    adm_res = admissibility_metrics(y_sub, y_pred, grid_size=60.0, epsilon_cells=0.5)
    adm_json = os.path.join(output_dir, "admissibility_metrics.json")
    with open(adm_json, "w", encoding="utf-8") as f:
        json.dump(adm_res, f, indent=2)
    log.info(f"Saved admissibility metrics to {adm_json} (Violation rate={adm_res.get('violation_rate', 0):.2%})")

    # 3. 10-bin Classification metrics
    if os.path.exists(classifier_path):
        clf_data = joblib.load(classifier_path)
        clf = clf_data["model"]
        edges = np.asarray(clf_data["bin_edges"])
        y_true_bins = np.digitize(y_sub, edges)
        y_proba = clf.predict_proba(X_sub)

        cls_res = classification_metrics(y_true_bins, y_proba, n_bins=10)
        cls_json = os.path.join(output_dir, "classification_metrics.json")
        with open(cls_json, "w", encoding="utf-8") as f:
            json.dump(cls_res, f, indent=2)
        log.info(f"Saved classification metrics to {cls_json} (Macro F1={cls_res.get('macro_f1', 0):.4f}, AUC-ROC={cls_res.get('auc_roc_ovr_macro', 0):.4f})")


def main():
    parser = argparse.ArgumentParser(description="Full search and metric evaluation runner.")
    parser.add_argument("--backend", choices=["sklearn", "torch"], default="sklearn", help="Evaluation backend")
    parser.add_argument("--checkpoint", default="models/fallback_mlp_regressor.joblib", help="Model weights/checkpoint")
    parser.add_argument("--classifier", default="models/fallback_mlp_classifier.joblib", help="10-bin classifier model")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--cleaned-dir", default="data/cleaned", help="Cleaned dataset directory")
    parser.add_argument("--test-data", default="data/processed/test.npz", help="Processed test cell data")
    parser.add_argument("--output", default="results/evaluation", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cell-samples", type=int, default=100000, help="Cell samples for regression/classification")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # First, run cell-level regression, admissibility, and classification evaluation
    evaluate_cell_metrics(
        backend=args.backend,
        regressor_path=args.checkpoint,
        classifier_path=args.classifier,
        test_data_path=args.test_data,
        output_dir=args.output,
        sample_size=args.cell_samples,
        seed=args.seed,
    )

    # Second, load mazes for search evaluation
    log.info(f"Loading {args.split} mazes from {args.cleaned_dir}...")
    mazes = list(iter_split_mazes(split=args.split, cleaned_dir=args.cleaned_dir, seed=args.seed))
    log.info(f"Loaded {len(mazes)} mazes for search evaluation.")

    # Configure all heuristics to run
    alphas = [0.6, 0.7, 0.8, 0.9, 1.0]
    heuristics = {
        "dijkstra": (lambda m: ZeroHeuristic(), 1.0),
        "manhattan": (lambda m: ManhattanHeuristic(m["goal"]), 1.0),
        "oracle": (lambda m: OracleHeuristic(m["h_star"]), 1.0),
    }

    # Add neural alpha sweep configurations
    for a in alphas:
        key = f"neural_alpha_{a:.1f}"
        heuristics[key] = (
            lambda m, alpha=a: SklearnNeuralHeuristic(args.checkpoint, m["grid"], m["goal"], alpha=alpha, mode="scaled"),
            a,
        )

    # Add clipped Manhattan configuration at best expected alpha (0.9 and 1.0)
    heuristics["neural_clipped_manhattan"] = (
        lambda m: SklearnNeuralHeuristic(args.checkpoint, m["grid"], m["goal"], alpha=1.0, mode="clipped_manhattan"),
        1.0,
    )

    log.info(f"Running search evaluation across {len(heuristics)} configurations on {len(mazes)} mazes...")
    df = run_search_evaluation(
        mazes=mazes,
        heuristics=heuristics,
        output_dir=args.output,
        split=args.split,
        seed=args.seed,
        tie_break="h",
    )

    # Produce alpha sweep summary
    alpha_summary = {}
    summary = search_aggregate(df, baseline="manhattan")
    for a in alphas:
        key = f"neural_alpha_{a:.1f}"
        if key in summary:
            alpha_summary[str(a)] = summary[key]

    alpha_json = os.path.join(args.output, "alpha_sweep.json")
    with open(alpha_json, "w", encoding="utf-8") as f:
        json.dump(alpha_summary, f, indent=2)
    log.info(f"Saved alpha sweep summary to {alpha_json}")
    log.info(f"All search trials finished: {len(df)} trials recorded in {args.output}/{args.split}_per_maze.csv")


if __name__ == "__main__":
    main()
