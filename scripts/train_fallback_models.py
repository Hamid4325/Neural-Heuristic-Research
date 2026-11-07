"""Train fallback scikit-learn MLP regressor and 10-bin classifier.

Part of Deliverable 2 for Neural Heuristic Research.
Author: Abdullah Zia
"""

import argparse
import logging
import os
import sys
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_fallback_models")


def main():
    parser = argparse.ArgumentParser(description="Train fallback scikit-learn models.")
    parser.add_argument("--data", default="data/processed/train.npz", help="Path to train.npz")
    parser.add_argument("--models-dir", default="models", help="Directory to save models")
    parser.add_argument("--samples", type=int, default=100000, help="Number of training samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-iter", type=int, default=50, help="Max iterations for MLP")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    log.info(f"Loading training data from {args.data}...")
    train = np.load(args.data)
    X_all, y_all = train["X"], train["y"]
    log.info(f"Loaded {len(X_all)} total cells with {X_all.shape[1]} features.")

    rng = np.random.default_rng(args.seed)
    n_samples = min(args.samples, len(X_all))
    idx = rng.choice(len(X_all), size=n_samples, replace=False)
    X, y = X_all[idx], y_all[idx]
    log.info(f"Subsampled {n_samples} training samples (seed={args.seed}).")

    # 1. Regression: predict h*/N
    log.info("Fitting MLPRegressor (predict h*/N)...")
    reg = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        max_iter=args.max_iter,
        random_state=args.seed,
        early_stopping=True,
        n_iter_no_change=5,
        verbose=False,
    )
    reg.fit(X, y)
    reg_path = os.path.join(args.models_dir, "fallback_mlp_regressor.joblib")
    joblib.dump(reg, reg_path)
    log.info(f"Saved fallback MLPRegressor to {reg_path}")

    # 2. Classification: 10 distance bins
    log.info("Fitting 10-bin MLPClassifier...")
    edges = np.quantile(y, np.linspace(0, 1, 11)[1:-1])
    bins = np.digitize(y, edges)
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        max_iter=args.max_iter,
        random_state=args.seed,
        early_stopping=True,
        n_iter_no_change=5,
        verbose=False,
    )
    clf.fit(X, bins)
    clf_path = os.path.join(args.models_dir, "fallback_mlp_classifier.joblib")
    joblib.dump({"model": clf, "bin_edges": edges.tolist()}, clf_path)
    log.info(f"Saved fallback MLPClassifier to {clf_path}")


if __name__ == "__main__":
    main()
