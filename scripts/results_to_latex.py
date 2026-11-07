"""Generate publishable LaTeX tables from evaluation metrics and search logs.

Part of Deliverable 2 for Neural Heuristic Research.
Author: Abdullah Zia
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate_regression_table(reg_json_path: str, out_tex_path: str):
    """Generate LaTeX booktabs table for regression metrics."""
    if not os.path.exists(reg_json_path):
        return

    with open(reg_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tex = r"""\begin{table}[H]
\centering
\small
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Normalized Value ($h^\ast / N$)} & \textbf{Grid Cells ($N=60$)} \\
\midrule
Mean Squared Error (MSE) & """ + f"{data.get('mse', 0):.4f}" + r""" & --- \\
Root Mean Squared Error (RMSE) & """ + f"{data.get('rmse', 0):.4f}" + r""" & """ + f"{data.get('rmse_cells', 0):.2f}" + r""" \\
Mean Absolute Error (MAE) & """ + f"{data.get('mae', 0):.4f}" + r""" & """ + f"{data.get('mae_cells', 0):.2f}" + r""" \\
Mean Signed Error & """ + f"{data.get('mean_signed_error', 0):.4f}" + r""" & """ + f"{data.get('mean_signed_error_cells', 0):.2f}" + r""" \\
Coefficient of Determination ($R^2$) & """ + f"{data.get('r2', 0):.4f}" + r""" & --- \\
Pearson Correlation ($r$) & """ + f"{data.get('pearson', 0):.4f}" + r""" & --- \\
Spearman Rank Correlation ($\rho$) & """ + f"{data.get('spearman', 0):.4f}" + r""" & --- \\
95th Percentile Absolute Error & """ + f"{data.get('p95_absolute_error', 0):.4f}" + r""" & """ + f"{data.get('p95_absolute_error_cells', 0):.2f}" + r""" \\
99th Percentile Absolute Error & """ + f"{data.get('p99_absolute_error', 0):.4f}" + r""" & """ + f"{data.get('p99_absolute_error_cells', 0):.2f}" + r""" \\
Maximum Absolute Error & """ + f"{data.get('max_absolute_error', 0):.4f}" + r""" & """ + f"{data.get('max_absolute_error_cells', 0):.2f}" + r""" \\
\bottomrule
\end{tabular}
\caption{Offline regression fidelity metrics evaluated on the unseen test set.}
\label{tab:regression_results}
\end{table}
"""
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(tex)


def generate_classification_table(cls_json_path: str, out_tex_path: str):
    """Generate LaTeX booktabs table for 10-bin distance classification."""
    if not os.path.exists(cls_json_path):
        return

    with open(cls_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_class = data.get("per_class", {})
    rows = []
    for i in range(data.get("n_bins", 10)):
        key = f"bin_{i}"
        info = per_class.get(key, {})
        rows.append(
            f"Bin {i} & {info.get('precision', 0):.4f} & {info.get('recall', 0):.4f} & {info.get('f1', 0):.4f} & {info.get('support', 0):,d} \\\\"
        )
    rows_str = "\n".join(rows)

    tex = r"""\begin{table}[H]
\centering
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Distance Bin} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Support} \\
\midrule
""" + rows_str + r"""
\midrule
Macro Average & """ + f"{data.get('macro_precision', 0):.4f}" + r""" & """ + f"{data.get('macro_recall', 0):.4f}" + r""" & """ + f"{data.get('macro_f1', 0):.4f}" + r""" & --- \\
Weighted Average & """ + f"{data.get('weighted_precision', 0):.4f}" + r""" & """ + f"{data.get('weighted_recall', 0):.4f}" + r""" & """ + f"{data.get('weighted_f1', 0):.4f}" + r""" & --- \\
\textbf{Macro AUC-ROC (OvR)} & \multicolumn{4}{c}{\textbf{""" + f"{data.get('auc_roc_ovr_macro', 0):.4f}" + r"""}} \\
\bottomrule
\end{tabular}
\caption{Auxiliary 10-bin classification performance across distance deciles.}
\label{tab:classification_results}
\end{table}
"""
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(tex)


def generate_search_table(summary_json_path: str, out_tex_path: str):
    """Generate LaTeX table comparing graph search efficiency across heuristics."""
    if not os.path.exists(summary_json_path):
        return

    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    target_heuristics = [
        ("Dijkstra (Zero)", "dijkstra"),
        ("Manhattan Baseline", "manhattan"),
        (r"Neural Heuristic ($\alpha=1.0$)", "neural_alpha_1.0"),
        (r"Neural Scaled ($\alpha=0.8$)", "neural_alpha_0.8"),
        (r"Neural Clipped Manhattan", "neural_clipped_manhattan"),
        (r"Oracle ($h^\ast$ Upper Bound)", "oracle"),
    ]

    rows = []
    for label, key in target_heuristics:
        if key in summary:
            h = summary[key]
            exp_mean = f"${h.get('mean_nodes_expanded', 0):.1f} \\pm {h.get('std_nodes_expanded', 0):.1f}$"
            exp_med = f"{h.get('median_nodes_expanded', 0):.0f}"
            rel_red = f"{h.get('mean_relative_node_reduction_vs_baseline', 0) * 100:.1f}\\%"
            opt = f"{h.get('mean_optimality_ratio', 1.0):.3f}"
            succ = f"{h.get('success_rate', 0) * 100:.1f}\\%"
            rt = f"{h.get('mean_runtime_ms', 0):.2f}"
            rows.append(f"{label} & {exp_mean} & {exp_med} & {rel_red} & {opt} & {succ} & {rt} \\\\")

    rows_str = "\n".join(rows)

    tex = r"""\begin{table}[H]
\centering
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Heuristic} & \textbf{Expanded (Mean $\pm$ Std)} & \textbf{Median} & \textbf{$\Delta_{\text{expanded}}$ vs Man.} & \textbf{Opt. Ratio} & \textbf{Success} & \textbf{Time (ms)} \\
\midrule
""" + rows_str + r"""
\bottomrule
\end{tabular}
\caption{End-to-end A$^\ast$ pathfinding performance evaluated across 282 test mazes.}
\label{tab:search_results}
\end{table}
"""
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(tex)


def generate_alpha_sweep_table(alpha_json_path: str, out_tex_path: str):
    """Generate LaTeX table for alpha scaling sensitivity sweep."""
    if not os.path.exists(alpha_json_path):
        return

    with open(alpha_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for alpha_str in sorted(data.keys(), key=lambda x: float(x)):
        h = data[alpha_str]
        rows.append(
            f"$\\alpha = {float(alpha_str):.1f}$ & "
            f"{h.get('mean_nodes_expanded', 0):.1f} & "
            f"{h.get('median_nodes_expanded', 0):.0f} & "
            f"{h.get('mean_relative_node_reduction_vs_baseline', 0) * 100:.1f}\\% & "
            f"{h.get('mean_optimality_ratio', 1.0):.3f} & "
            f"{h.get('success_rate', 0) * 100:.1f}\\% \\\\"
        )
    rows_str = "\n".join(rows)

    tex = r"""\begin{table}[H]
\centering
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Scale Factor $\alpha$} & \textbf{Mean Expanded} & \textbf{Median Expanded} & \textbf{Reduction vs Man.} & \textbf{Optimality Ratio} & \textbf{Success Rate} \\
\midrule
""" + rows_str + r"""
\bottomrule
\end{tabular}
\caption{Heuristic scaling parameter ($\alpha$) sensitivity sweep on the test set.}
\label{tab:alpha_sweep}
\end{table}
"""
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(tex)


def generate_bias_table(bias_csv_path: str, out_tex_path: str):
    """Generate LaTeX table from environmental subgroup bias check."""
    if not os.path.exists(bias_csv_path):
        return

    df = pd.read_csv(bias_csv_path)
    rows = []
    for _, row in df.iterrows():
        subgroup = str(row.get("subgroup", ""))
        value = str(row.get("value", ""))
        mae = f"{float(row.get('mae_cells', 0)):.2f}" if pd.notna(row.get('mae_cells')) else "---"
        rmse = f"{float(row.get('rmse_cells', 0)):.2f}" if pd.notna(row.get('rmse_cells')) else "---"
        signed = f"{float(row.get('mean_signed_error_cells', 0)):.2f}" if pd.notna(row.get('mean_signed_error_cells')) else "---"
        viol = f"{float(row.get('violation_rate', 0)) * 100:.1f}\\%" if pd.notna(row.get('violation_rate')) else "---"
        red = f"{float(row.get('node_reduction_vs_manhattan', 0)) * 100:.1f}\\%" if pd.notna(row.get('node_reduction_vs_manhattan')) else "---"
        rows.append(f"{subgroup} & {value} & {mae} & {rmse} & {signed} & {viol} & {red} \\\\")

    rows_str = "\n".join(rows)

    tex = r"""\begin{table}[H]
\centering
\small
\begin{tabular}{llccccc}
\toprule
\textbf{Subgroup} & \textbf{Partition} & \textbf{MAE (cells)} & \textbf{RMSE (cells)} & \textbf{Signed Err} & \textbf{Inadm. Rate} & \textbf{$\Delta_{\text{expanded}}$ vs Man.} \\
\midrule
""" + rows_str + r"""
\bottomrule
\end{tabular}
\caption{Environmental and topological subgroup evaluation (Bias and Fairness analysis).}
\label{tab:bias_results}
\end{table}
"""
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(tex)


def main():
    parser = argparse.ArgumentParser(description="Convert evaluation metrics to publication-ready LaTeX tables.")
    parser.add_argument("--eval-dir", default="results/evaluation", help="Directory containing JSON and CSV results")
    parser.add_argument("--out-dir", default="report/generated", help="Directory to save generated LaTeX files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    generate_regression_table(
        os.path.join(args.eval_dir, "regression_metrics.json"),
        os.path.join(args.out_dir, "regression_table.tex"),
    )
    generate_classification_table(
        os.path.join(args.eval_dir, "classification_metrics.json"),
        os.path.join(args.out_dir, "classification_table.tex"),
    )
    generate_search_table(
        os.path.join(args.eval_dir, "test_summary.json"),
        os.path.join(args.out_dir, "search_table.tex"),
    )
    generate_alpha_sweep_table(
        os.path.join(args.eval_dir, "alpha_sweep.json"),
        os.path.join(args.out_dir, "alpha_sweep_table.tex"),
    )
    generate_bias_table(
        os.path.join(args.eval_dir, "bias_table.csv"),
        os.path.join(args.out_dir, "bias_table.tex"),
    )
    print(f"Generated LaTeX tables in {args.out_dir}")


if __name__ == "__main__":
    main()
