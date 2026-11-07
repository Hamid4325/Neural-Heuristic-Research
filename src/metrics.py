"""Evaluation and comparison metrics for regression, classification, and search.

Part of Deliverable 2 for Neural Heuristic Research.
Author: Abdullah Zia
"""

from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics as sk_metrics


def regression_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    grid_size: Optional[float] = None,
) -> dict[str, float]:
    """Compute regression metrics between true and predicted heuristic targets.

    Metrics computed:
      - Mean Squared Error (MSE)
      - Root Mean Squared Error (RMSE)
      - Mean Absolute Error (MAE)
      - Coefficient of Determination (R2)
      - Pearson correlation coefficient
      - Spearman rank correlation coefficient
      - Mean signed error (y_pred - y_true)
      - 95th percentile, 99th percentile, and maximum absolute errors
      - If grid_size is given: cell-scaled errors (RMSE, MAE, signed error, p95, p99, max)

    Parameters
    ----------
    y_true : Union[np.ndarray, list]
        Ground truth normalized target values (e.g. h*/N).
    y_pred : Union[np.ndarray, list]
        Predicted heuristic values.
    grid_size : Optional[float], default=None
        Dimension N of the grid (to scale normalized errors back to raw cells).

    Returns
    -------
    dict[str, float]
        Dictionary of computed metric names and values.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} vs len(y_pred)={len(yp)}")

    if len(yt) == 0:
        return {}

    errors = yp - yt
    abs_errors = np.abs(errors)

    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_errors))
    mean_signed_error = float(np.mean(errors))

    # Handle edge case where variance is zero (e.g. perfect constant predictions)
    var_yt = float(np.var(yt))
    var_yp = float(np.var(yp))

    if var_yt == 0.0 and var_yp == 0.0 and np.allclose(yt, yp):
        r2 = 1.0
        pearson = 1.0
        spearman = 1.0
    elif var_yt == 0.0 or var_yp == 0.0:
        r2 = 0.0
        pearson = 0.0
        spearman = 0.0
    else:
        r2 = float(sk_metrics.r2_score(yt, yp))
        p_corr, _ = stats.pearsonr(yt, yp)
        pearson = float(p_corr)
        s_corr, _ = stats.spearmanr(yt, yp)
        spearman = float(s_corr)

    p95_abs = float(np.percentile(abs_errors, 95))
    p99_abs = float(np.percentile(abs_errors, 99))
    max_abs = float(np.max(abs_errors))

    results: dict[str, float] = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "mean_signed_error": mean_signed_error,
        "p95_absolute_error": p95_abs,
        "p99_absolute_error": p99_abs,
        "max_absolute_error": max_abs,
    }

    if grid_size is not None and grid_size > 0:
        gs = float(grid_size)
        results["grid_size"] = gs
        results["mae_cells"] = mae * gs
        results["rmse_cells"] = rmse * gs
        results["mean_signed_error_cells"] = mean_signed_error * gs
        results["p95_absolute_error_cells"] = p95_abs * gs
        results["p99_absolute_error_cells"] = p99_abs * gs
        results["max_absolute_error_cells"] = max_abs * gs

    return results


def admissibility_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    grid_size: float,
    epsilon_cells: float = 0.5,
) -> dict[str, Any]:
    """Compute admissibility violation metrics and overestimation statistics.

    A heuristic estimate is inadmissible if h_pred exceeds h_true by more than
    epsilon_cells in raw cell distance: (y_pred - y_true) * grid_size > epsilon_cells.

    Parameters
    ----------
    y_true : Union[np.ndarray, list]
        Ground-truth normalized optimal cost to goal.
    y_pred : Union[np.ndarray, list]
        Predicted normalized heuristic estimate.
    grid_size : float
        Grid dimension N.
    epsilon_cells : float, default=0.5
        Tolerance margin in cells before flagging an admissibility violation.

    Returns
    -------
    dict[str, Any]
        Admissibility metrics including violation count, violation rate,
        mean overestimation, max overestimation, and 95th percentile overestimation.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} vs len(y_pred)={len(yp)}")

    if len(yt) == 0:
        return {}

    gs = float(grid_size)
    overestimation_cells = (yp - yt) * gs
    violations = overestimation_cells > epsilon_cells

    violation_count = int(np.sum(violations))
    violation_rate = float(np.mean(violations))

    # Overestimation is positive overestimation; underestimate counts as 0 overestimation
    positive_overestimation = np.maximum(0.0, overestimation_cells)

    return {
        "epsilon_cells": float(epsilon_cells),
        "grid_size": gs,
        "total_cells": int(len(yt)),
        "violation_count": violation_count,
        "violation_rate": violation_rate,
        "mean_overestimation_cells": float(np.mean(positive_overestimation)),
        "max_overestimation_cells": float(np.max(overestimation_cells)),
        "p95_overestimation_cells": float(np.percentile(positive_overestimation, 95)),
    }


def classification_metrics(
    y_true_bins: Union[np.ndarray, list],
    y_proba: Union[np.ndarray, list],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute 10-bin auxiliary classification metrics.

    Parameters
    ----------
    y_true_bins : Union[np.ndarray, list]
        True integer bin labels in [0, n_bins - 1].
    y_proba : Union[np.ndarray, list]
        Predicted probability distribution over bins with shape (N, n_bins).
    n_bins : int, default=10
        Number of discrete classification bins.

    Returns
    -------
    dict[str, Any]
        Precision, recall, F1 per class, macro and weighted averages,
        macro one-vs-rest AUC-ROC, and confusion matrix.
    """
    yt = np.asarray(y_true_bins, dtype=np.int64).ravel()
    yp_prob = np.asarray(y_proba, dtype=np.float64)

    if yp_prob.ndim == 1 and n_bins == 1:
        yp_prob = yp_prob[:, None]
    elif yp_prob.ndim != 2:
        raise ValueError(f"y_proba must be 2D array, got shape {yp_prob.shape}")

    if len(yt) != len(yp_prob):
        raise ValueError(f"Length mismatch: len(y_true_bins)={len(yt)} vs len(y_proba)={len(yp_prob)}")

    all_labels = list(range(n_bins))
    yp_pred = np.argmax(yp_prob, axis=1)

    prec_arr, rec_arr, f1_arr, supp_arr = sk_metrics.precision_recall_fscore_support(
        yt, yp_pred, labels=all_labels, zero_division=0
    )

    macro_prec = float(sk_metrics.precision_score(yt, yp_pred, labels=all_labels, average="macro", zero_division=0))
    macro_rec = float(sk_metrics.recall_score(yt, yp_pred, labels=all_labels, average="macro", zero_division=0))
    macro_f1 = float(sk_metrics.f1_score(yt, yp_pred, labels=all_labels, average="macro", zero_division=0))

    weighted_prec = float(sk_metrics.precision_score(yt, yp_pred, labels=all_labels, average="weighted", zero_division=0))
    weighted_rec = float(sk_metrics.recall_score(yt, yp_pred, labels=all_labels, average="weighted", zero_division=0))
    weighted_f1 = float(sk_metrics.f1_score(yt, yp_pred, labels=all_labels, average="weighted", zero_division=0))

    cm = sk_metrics.confusion_matrix(yt, yp_pred, labels=all_labels).tolist()

    # One-vs-rest AUC-ROC
    try:
        if len(np.unique(yt)) > 1:
            auc_roc = float(
                sk_metrics.roc_auc_score(
                    yt, yp_prob, multi_class="ovr", average="macro", labels=all_labels
                )
            )
        else:
            auc_roc = 1.0 if np.all(yp_pred == yt) else 0.5
    except Exception:
        auc_roc = float("nan")

    per_class = {
        f"bin_{i}": {
            "precision": float(prec_arr[i]),
            "recall": float(rec_arr[i]),
            "f1": float(f1_arr[i]),
            "support": int(supp_arr[i]),
        }
        for i in range(n_bins)
    }

    return {
        "n_bins": n_bins,
        "per_class": per_class,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_prec,
        "weighted_recall": weighted_rec,
        "weighted_f1": weighted_f1,
        "auc_roc_ovr_macro": auc_roc,
        "confusion_matrix": cm,
    }


def search_aggregate(
    per_maze_rows: Union[list[dict[str, Any]], pd.DataFrame],
    baseline: str = "manhattan",
) -> dict[str, Any]:
    """Aggregate per-maze search results to compute overall performance metrics.

    Parameters
    ----------
    per_maze_rows : Union[list[dict[str, Any]], pd.DataFrame]
        Search evaluation records containing columns/keys:
        'maze_id', 'heuristic', 'cost', 'opt_cost', 'nodes_expanded',
        'nodes_generated', 'runtime_ms', 'found'.
    baseline : str, default='manhattan'
        Name of baseline heuristic to compute relative node reduction against.

    Returns
    -------
    dict[str, Any]
        Dictionary partitioned by heuristic containing mean/median expanded,
        relative node reduction, optimality ratio, success rate, and runtime.
    """
    if isinstance(per_maze_rows, list):
        df = pd.DataFrame(per_maze_rows)
    else:
        df = per_maze_rows.copy()

    if df.empty:
        return {}

    summary: dict[str, Any] = {}
    heuristics = df["heuristic"].unique()

    # Pre-extract baseline lookup if present
    baseline_df = df[df["heuristic"] == baseline]
    baseline_expanded: dict[Any, float] = {}
    if not baseline_df.empty:
        for _, row in baseline_df.iterrows():
            baseline_expanded[row["maze_id"]] = float(row["nodes_expanded"])

    for h_name in heuristics:
        sub = df[df["heuristic"] == h_name]
        total_mazes = len(sub)
        solved_sub = sub[sub["found"] == True]
        solved_count = len(solved_sub)
        success_rate = float(solved_count / total_mazes) if total_mazes > 0 else 0.0

        expanded = sub["nodes_expanded"].to_numpy(dtype=np.float64)
        generated = sub["nodes_generated"].to_numpy(dtype=np.float64)
        runtime = sub["runtime_ms"].to_numpy(dtype=np.float64)

        # Optimality ratio: cost / opt_cost for solved instances where opt_cost > 0
        valid_opt = solved_sub[(solved_sub["opt_cost"] > 0) & np.isfinite(solved_sub["cost"])]
        if not valid_opt.empty:
            opt_ratios = valid_opt["cost"] / valid_opt["opt_cost"]
            mean_opt_ratio = float(opt_ratios.mean())
            max_opt_ratio = float(opt_ratios.max())
        else:
            mean_opt_ratio = 1.0
            max_opt_ratio = 1.0

        # Relative node reduction vs baseline: 1.0 - (expanded / baseline_expanded)
        rel_reductions = []
        if baseline_expanded:
            for _, row in sub.iterrows():
                mid = row["maze_id"]
                if mid in baseline_expanded and baseline_expanded[mid] > 0:
                    red = 1.0 - (float(row["nodes_expanded"]) / baseline_expanded[mid])
                    rel_reductions.append(red)

        mean_reduction = float(np.mean(rel_reductions)) if rel_reductions else 0.0
        median_reduction = float(np.median(rel_reductions)) if rel_reductions else 0.0

        summary[str(h_name)] = {
            "total_mazes": total_mazes,
            "solved_count": solved_count,
            "success_rate": success_rate,
            "mean_nodes_expanded": float(np.mean(expanded)),
            "median_nodes_expanded": float(np.median(expanded)),
            "std_nodes_expanded": float(np.std(expanded)),
            "mean_nodes_generated": float(np.mean(generated)),
            "median_nodes_generated": float(np.median(generated)),
            "mean_runtime_ms": float(np.mean(runtime)),
            "median_runtime_ms": float(np.median(runtime)),
            "mean_optimality_ratio": mean_opt_ratio,
            "max_optimality_ratio": max_opt_ratio,
            "mean_relative_node_reduction_vs_baseline": mean_reduction,
            "median_relative_node_reduction_vs_baseline": median_reduction,
        }

    return summary
