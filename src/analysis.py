"""Error analysis and environmental subgroup bias auditing.

Part of Deliverable 2 for Neural Heuristic Research.
Author: Abdullah Zia
"""

import json
import os
from typing import Any, Optional
import joblib
import numpy as np
import pandas as pd

from src.metrics import regression_metrics, admissibility_metrics


def perform_error_analysis(
    test_data_path: str,
    regressor_path: str,
    output_dir: str,
    per_maze_csv: Optional[str] = None,
    sample_size: int = 100000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute detailed residual statistics, distance deciles, and identify worst cases."""
    os.makedirs(output_dir, exist_ok=True)
    test_npz = np.load(test_data_path)
    X_test, y_test = test_npz["X"], test_npz["y"]

    rng = np.random.default_rng(seed)
    n_samples = min(sample_size, len(X_test))
    idx = rng.choice(len(X_test), size=n_samples, replace=False)
    X, y_true = X_test[idx], y_test[idx]

    reg = joblib.load(regressor_path)
    y_pred = reg.predict(X)

    residuals = y_pred - y_true
    abs_residuals = np.abs(residuals)

    # 1. Error by distance decile
    decile_edges = np.quantile(y_true, np.linspace(0, 1, 11))
    decile_stats = []
    for d in range(10):
        low, high = decile_edges[d], decile_edges[d + 1]
        mask = (y_true >= low) & (y_true <= high)
        if np.any(mask):
            sub_res = residuals[mask]
            decile_stats.append({
                "decile": d,
                "range_min": float(low),
                "range_max": float(high),
                "count": int(np.sum(mask)),
                "mae": float(np.mean(np.abs(sub_res))),
                "rmse": float(np.sqrt(np.mean(sub_res ** 2))),
                "mean_signed_error": float(np.mean(sub_res)),
            })

    # 2. Error by local wall count (sum of 5x5 wall context, features 4..28)
    wall_counts = np.sum(X[:, 4:29], axis=1).astype(int)
    wall_stats = []
    for wc in sorted(np.unique(wall_counts)):
        mask = wall_counts == wc
        if np.sum(mask) >= 50:
            sub_res = residuals[mask]
            wall_stats.append({
                "wall_count": int(wc),
                "count": int(np.sum(mask)),
                "mae": float(np.mean(np.abs(sub_res))),
                "rmse": float(np.sqrt(np.mean(sub_res ** 2))),
            })

    # 3. Search-level worst cases if per_maze_csv is available
    search_worst = []
    if per_maze_csv and os.path.exists(per_maze_csv):
        df = pd.read_csv(per_maze_csv)
        man = df[df["heuristic"] == "manhattan"].set_index("maze_id")
        neural = df[df["heuristic"] == "neural_alpha_1.0"].set_index("maze_id")

        common_ids = man.index.intersection(neural.index)
        if len(common_ids) > 0:
            diffs = []
            for mid in common_ids:
                m_row = man.loc[mid]
                n_row = neural.loc[mid]
                m_exp = float(m_row["nodes_expanded"])
                n_exp = float(n_row["nodes_expanded"])
                diffs.append({
                    "maze_id": mid,
                    "generator": str(m_row["generator"]),
                    "size": int(m_row["size"]),
                    "density": int(m_row["density"]),
                    "manhattan_expanded": m_exp,
                    "neural_expanded": n_exp,
                    "expansion_increase": n_exp - m_exp,
                    "optimality_ratio": float(n_row["optimality_ratio"]),
                })
            diffs_df = pd.DataFrame(diffs)
            diffs_df = diffs_df.sort_values(by="expansion_increase", ascending=False)
            search_worst = diffs_df.head(10).to_dict(orient="records")

    summary = {
        "n_samples": int(n_samples),
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "p95_abs_residual": float(np.percentile(abs_residuals, 95)),
        "decile_breakdown": decile_stats,
        "wall_density_breakdown": wall_stats,
        "search_worst_regressions": search_worst,
    }

    out_file = os.path.join(output_dir, "error_analysis.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def perform_bias_analysis(
    per_maze_csv: str,
    test_data_path: str,
    regressor_path: str,
    output_dir: str,
    sample_size: int = 100000,
    seed: int = 42,
) -> pd.DataFrame:
    """Audit environmental and topological subgroups for algorithmic performance disparity."""
    os.makedirs(output_dir, exist_ok=True)
    df_search = pd.read_csv(per_maze_csv)
    test_npz = np.load(test_data_path)
    X_test, y_test = test_npz["X"], test_npz["y"]

    rng = np.random.default_rng(seed)
    n_samples = min(sample_size, len(X_test))
    idx = rng.choice(len(X_test), size=n_samples, replace=False)
    X, y_true = X_test[idx], y_test[idx]

    reg = joblib.load(regressor_path)
    y_pred = reg.predict(X)

    # Local features for subgroup tagging
    # Feature 0, 1: coord_x, coord_y (cell position)
    is_boundary = (X[:, 0] < 0.1) | (X[:, 0] > 0.9) | (X[:, 1] < 0.1) | (X[:, 1] > 0.9)
    # Feature 4..28: center cell wall adjacency (surrounding 8 neighbors)
    wall_count = np.sum(X[:, 4:29], axis=1)
    is_adjacent_wall = wall_count >= 8

    records = []

    # 1. Cell-level subgroups: Boundary vs Interior
    for tag, mask in [("Boundary", is_boundary), ("Interior", ~is_boundary)]:
        if np.any(mask):
            reg_m = regression_metrics(y_true[mask], y_pred[mask], grid_size=60.0)
            adm_m = admissibility_metrics(y_true[mask], y_pred[mask], grid_size=60.0)
            records.append({
                "subgroup": "Cell Position",
                "value": tag,
                "mae_cells": reg_m["mae_cells"],
                "rmse_cells": reg_m["rmse_cells"],
                "mean_signed_error_cells": reg_m["mean_signed_error_cells"],
                "violation_rate": adm_m["violation_rate"],
                "node_reduction_vs_manhattan": np.nan,
                "success_rate": 1.0,
            })

    # 2. Cell-level subgroups: Wall Adjacency
    for tag, mask in [("Clustered/Adjacent", is_adjacent_wall), ("Open Corridors", ~is_adjacent_wall)]:
        if np.any(mask):
            reg_m = regression_metrics(y_true[mask], y_pred[mask], grid_size=60.0)
            adm_m = admissibility_metrics(y_true[mask], y_pred[mask], grid_size=60.0)
            records.append({
                "subgroup": "Wall Adjacency",
                "value": tag,
                "mae_cells": reg_m["mae_cells"],
                "rmse_cells": reg_m["rmse_cells"],
                "mean_signed_error_cells": reg_m["mean_signed_error_cells"],
                "violation_rate": adm_m["violation_rate"],
                "node_reduction_vs_manhattan": np.nan,
                "success_rate": 1.0,
            })

    # 3. Maze-level subgroups from search evaluation CSV
    # Subgroups: Generator, Grid Size, Obstacle Density
    subgroups = [
        ("Generator", "generator"),
        ("Grid Size", "size"),
        ("Obstacle Density (%)", "density"),
    ]

    for sg_name, col in subgroups:
        unique_vals = sorted(df_search[col].unique())
        for val in unique_vals:
            sub = df_search[df_search[col] == val]
            man_sub = sub[sub["heuristic"] == "manhattan"].set_index("maze_id")
            # Using clipped manhattan or alpha 1.0
            neur_sub = sub[sub["heuristic"] == "neural_clipped_manhattan"].set_index("maze_id")

            common = man_sub.index.intersection(neur_sub.index)
            if len(common) > 0:
                man_nodes = man_sub.loc[common, "nodes_expanded"].to_numpy(dtype=float)
                neur_nodes = neur_sub.loc[common, "nodes_expanded"].to_numpy(dtype=float)
                reductions = 1.0 - (neur_nodes / np.maximum(1.0, man_nodes))
                mean_red = float(np.mean(reductions))
                succ = float(np.mean(neur_sub.loc[common, "found"]))
            else:
                mean_red = 0.0
                succ = 1.0

            records.append({
                "subgroup": sg_name,
                "value": str(val),
                "mae_cells": np.nan,
                "rmse_cells": np.nan,
                "mean_signed_error_cells": np.nan,
                "violation_rate": np.nan,
                "node_reduction_vs_manhattan": mean_red,
                "success_rate": succ,
            })

    bias_df = pd.DataFrame(records)
    out_csv = os.path.join(output_dir, "bias_table.csv")
    bias_df.to_csv(out_csv, index=False)
    return bias_df
