"""Generate the figure, tables and macros used by report/sections/phase3_training.tex.

Owner: Muhammad Abdullah Saeed (plan Section 6.4, task 6.4.3).

Reads the tuning and final-training results and writes, so the report refreshes without
manual editing (same idea as Zia's scripts/results_to_latex.py):

    figures/training_curves_final.png          loss / val MAE / val bin accuracy per seed
    report/generated/tuning_space_table.tex    tested values and the selected value
    report/generated/tuning_runs_table.tex     every tuning run, ranked per stage
    report/generated/final_training_table.tex  one row per final seed
    report/generated/training_macros.tex       \\trLearningRate, \\trValMAE, ... used in the text

Usage (repository root):
    python scripts/make_training_report_assets.py
    python scripts/make_training_report_assets.py --quick     # dry run in an ignored folder
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, load_yaml  # noqa: E402
from src.utils import load_json  # noqa: E402

QUICK_ROOT = os.path.join("results", "tuning_quick", "final_quick")  # already git-ignored
TUNING_QUICK = os.path.join("results", "tuning_quick")

PARAM_LABELS = [
    ("hidden_dims", "Hidden layer widths"),
    ("activation", "Activation"),
    ("dropout", "Dropout"),
    ("batch_size", "Batch size"),
    ("learning_rate", "Learning rate"),
    ("weight_decay", "Weight decay"),
    ("reg_loss", "Regression loss"),
    ("lambda_aux", r"Auxiliary loss weight $\lambda$"),
    ("target_transform", "Target transform"),
]


# ---------------------------------------------------------------------------
# LaTeX formatting helpers
# ---------------------------------------------------------------------------
def tex_escape(text) -> str:
    out = str(text)
    for char, repl in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                       ("$", r"\$"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        out = out.replace(char, repl)
    return out


def tex_number(x: float) -> str:
    """Plain decimal for normal values, math-mode scientific for very small ones."""
    x = float(x)
    if x == 0.0:
        return "0"
    if abs(x) >= 0.01:
        return f"{x:g}"
    exponent = math.floor(math.log10(abs(x)))
    mantissa = x / 10 ** exponent
    if abs(mantissa - round(mantissa)) < 1e-9:
        mantissa = int(round(mantissa))
    body = f"10^{{{exponent}}}" if mantissa == 1 else f"{mantissa:g}\\times10^{{{exponent}}}"
    return f"${body}$"


def tex_value(key: str, value) -> str:
    """Format one hyperparameter value for a table."""
    if key == "hidden_dims":
        return "(" + ",".join(str(v) for v in value) + ")"
    if key in ("dropout", "batch_size", "learning_rate", "weight_decay", "lambda_aux"):
        return tex_number(value)
    return tex_escape(value)


def _fmt(value, digits: int = 4) -> str:
    return "--" if value in ("", None) else f"{float(value):.{digits}f}"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def tuning_space_table(grid: dict, selected: dict) -> str:
    rows = []
    for key, label in PARAM_LABELS:
        tested = ", ".join(tex_value(key, v) for v in grid[key])
        rows.append(f"{label} & {tested} & {tex_value(key, selected[key])} \\\\")
    return "\n".join([
        r"\begin{table}[H]", r"\centering",
        r"\caption{Hyperparameter search space and the configuration selected by the staged "
        r"search.}",
        r"\label{tab:tuning_space}", r"\begin{tabular}{lll}", r"\toprule",
        r"Hyperparameter & Values tested & Selected \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])


def tuning_runs_table(summary_rows: list[dict]) -> str:
    def order(r):
        rank = r["rank_in_stage"]
        return (r["stage"], int(rank) if str(rank).isdigit() else 999, r["run_id"])

    body = []
    for r in sorted(summary_rows, key=order):
        failed = str(r["status"]).startswith("failed")
        cells = [
            r["stage"], r["rank_in_stage"] or "--",
            tex_escape(r["hidden_dims"]), tex_escape(r["activation"]),
            tex_number(r["dropout"]), tex_number(r["batch_size"]),
            tex_number(r["learning_rate"]), tex_number(r["weight_decay"]),
            tex_escape(r["reg_loss"]), tex_number(r["lambda_aux"]),
            tex_escape(r["target_transform"]),
            "diverged" if failed else _fmt(r["best_val_mae"]),
            "--" if failed else str(r["epochs_run"]),
        ]
        body.append(" & ".join(str(c) for c in cells) + r" \\")
    return "\n".join([
        r"\begin{table}[H]", r"\centering", r"\scriptsize",
        r"\caption{All tuning runs, ranked by validation MAE within each stage "
        r"(A: 400k cells, 5 epochs; B: 1.5M cells, 10 epochs; C: full train split). "
        r"MAE is in units of $h^*/N$.}",
        r"\label{tab:tuning_runs}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllllllllllr}", r"\toprule",
        r"Stage & Rank & Hidden & Act. & Drop. & Batch & LR & WD & Loss & $\lambda$ & Target "
        r"& Val MAE & Ep. \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}}", r"\end{table}", "",
    ])


def final_training_table(runs: list[dict], aggregate: dict) -> str:
    body = []
    for r in runs:
        cells = r["train_cells_per_epoch"]
        cells = "full split" if cells == "all" else f"{int(cells):,}".replace(",", "{,}")
        body.append(" & ".join([
            str(r["seed"]), r["role"], cells, str(r["epochs_run"]), str(r["best_epoch"]),
            _fmt(r["best_val_mae"]), _fmt(r["best_val_bin_accuracy"], 3),
            _fmt(r["best_val_macro_f1"], 3), f"{float(r['train_seconds']) / 60:.1f}",
        ]) + r" \\")
    mean = f"{aggregate['val_mae_mean']:.4f} $\\pm$ {aggregate['val_mae_std']:.4f}"
    return "\n".join([
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Final models. Validation MAE is in units of $h^*/N$; the replicate seeds "
        r"use fewer cells per epoch, so only the primary model is trained on the full split. "
        rf"Mean over seeds: {mean}.}}",
        r"\label{tab:final_training}", r"\begin{tabular}{llrrrrrrr}", r"\toprule",
        r"Seed & Role & Cells/epoch & Epochs & Best ep. & Val MAE & Bin acc. & Macro-F1 "
        r"& Minutes \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])


def macros_file(best: dict, cfg, runs: list[dict], aggregate: dict,
                target_stats: Optional[tuple]) -> str:
    """\\newcommand definitions (letters only in names) used by the report text."""
    hp, primary = best["hyperparameters"], runs[0]
    seeds = [str(r["seed"]) for r in runs]
    seed_text = (", ".join(seeds[:-1]) + " and " + seeds[-1]) if len(seeds) > 1 else seeds[0]
    provisional = (r"\textbf{[PROVISIONAL: results from the stand-in network]}"
                   if primary["model_source"] != "hamid" else "")
    mean, std = ("--", "--") if target_stats is None else (f"{target_stats[0]:.3f}",
                                                           f"{target_stats[1]:.3f}")
    values = {
        "HiddenDims": "-".join(str(v) for v in hp["hidden_dims"]),
        "Activation": tex_escape(hp["activation"]),
        "Dropout": tex_number(hp["dropout"]),
        "BatchSize": tex_number(hp["batch_size"]),
        "LearningRate": tex_number(hp["learning_rate"]),
        "WeightDecay": tex_number(hp["weight_decay"]),
        "RegLoss": tex_escape(hp["reg_loss"]),
        "LambdaAux": tex_number(hp["lambda_aux"]),
        "TargetTransform": tex_escape(hp["target_transform"]),
        "SmoothBeta": tex_number(cfg.train.smooth_l1_beta),
        "GradClip": tex_number(cfg.train.grad_clip),
        "LRFactor": tex_number(cfg.train.lr_factor),
        "LRPatience": str(cfg.train.lr_patience),
        "Patience": str(cfg.train.patience),
        "MaxEpochs": str(cfg.train.epochs),
        "NumBins": str(cfg.model.n_bins),
        "Params": f"{int(primary['n_params']):,}".replace(",", "{,}"),
        "BestEpoch": str(primary["best_epoch"]),
        "EpochsRun": str(primary["epochs_run"]),
        "ValMAE": _fmt(primary["best_val_mae"]),
        "ValBinAcc": _fmt(primary["best_val_bin_accuracy"], 3),
        "ValMacroF": _fmt(primary["best_val_macro_f1"], 3),
        "MeanMAE": f"{aggregate['val_mae_mean']:.4f}",
        "StdMAE": f"{aggregate['val_mae_std']:.4f}",
        "SeedList": seed_text,
        "NumSeeds": str(len(runs)),
        "TargetMean": mean,
        "TargetStd": std,
        "ModelSource": tex_escape(primary["model_source"]),
        "Provisional": provisional,
    }
    lines = ["% Generated by scripts/make_training_report_assets.py. Do not edit by hand."]
    lines += [f"\\newcommand{{\\tr{name}}}{{{value}}}" for name, value in values.items()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def read_history(path: str) -> dict[str, list[float]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def make_curves_figure(training_dir: str, runs: list[dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    colors = ["#3b6fb6", "#d1603d", "#4f9d69", "#8a6bb1"]
    for i, run in enumerate(runs):
        stem = f"neural_heuristic_seed{run['seed']}"
        hist = read_history(os.path.join(training_dir, stem, "history.csv"))
        color, label = colors[i % len(colors)], f"seed {run['seed']}"
        ep = hist["epoch"]
        axes[0].plot(ep, hist["train_loss"], "--", color=color, alpha=0.6)
        axes[0].plot(ep, hist["val_loss"], "-", color=color, label=label)
        axes[1].plot(ep, hist["val_mae"], "-o", color=color, ms=3, label=label)
        axes[2].plot(ep, hist["val_bin_accuracy"], "-o", color=color, ms=3, label=label)
    axes[0].set_title("Loss (dashed: train, solid: validation)")
    axes[1].set_title(r"Validation MAE ($h^*/N$)")
    axes[2].set_title("Validation bin accuracy")
    from matplotlib.ticker import MaxNLocator

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("MAE")
    axes[2].set_ylabel("Accuracy")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def target_stats_from_checkpoint(path: str) -> Optional[tuple]:
    """(target_mean, target_std) stored in the primary checkpoint, or None if unreadable."""
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu")
        return float(ckpt["target_mean"]), float(ckpt["target_std"])
    except Exception:  # missing file / no torch: the macro falls back to '--'
        return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"  wrote {path}")


def generate(tuning_dir: str, training_dir: str, config_path: str, grid_path: str,
             figures_dir: str, generated_dir: str) -> None:
    summary_csv = os.path.join(tuning_dir, "tuning_summary.csv")
    best_json = os.path.join(tuning_dir, "best_config.json")
    final_json = os.path.join(training_dir, "final_summary.json")
    for path in (summary_csv, best_json, final_json, config_path, grid_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found (run tuning and final training first)")

    with open(summary_csv, newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    best = load_json(best_json)
    final = load_json(final_json)
    runs, aggregate = final["runs"], final["aggregate"]
    grid = load_yaml(grid_path)["grid"]
    cfg = load_config(config_path)

    write(os.path.join(generated_dir, "tuning_space_table.tex"),
          tuning_space_table(grid, best["hyperparameters"]))
    write(os.path.join(generated_dir, "tuning_runs_table.tex"), tuning_runs_table(summary_rows))
    write(os.path.join(generated_dir, "final_training_table.tex"),
          final_training_table(runs, aggregate))
    stats = target_stats_from_checkpoint(runs[0]["checkpoint"])
    write(os.path.join(generated_dir, "training_macros.tex"),
          macros_file(best, cfg, runs, aggregate, stats))
    make_curves_figure(training_dir, runs, os.path.join(figures_dir, "training_curves_final.png"))
    print(f"  wrote {os.path.join(figures_dir, 'training_curves_final.png')}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate report assets for the training section")
    p.add_argument("--tuning-dir", default="results/tuning")
    p.add_argument("--training-dir", default="results/training")
    p.add_argument("--config", default="configs/final.yaml")
    p.add_argument("--grid", default="configs/tuning_grid.yaml")
    p.add_argument("--figures-dir", default="figures")
    p.add_argument("--generated-dir", default="report/generated")
    p.add_argument("--quick", action="store_true", help=f"Dry run inside {QUICK_ROOT}")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    kw = dict(tuning_dir=args.tuning_dir, training_dir=args.training_dir, config_path=args.config,
              grid_path=args.grid, figures_dir=args.figures_dir, generated_dir=args.generated_dir)
    if args.quick:
        kw.update(tuning_dir=TUNING_QUICK, training_dir=os.path.join(QUICK_ROOT, "training"),
                  config_path=os.path.join(TUNING_QUICK, "best_config.yaml"),
                  figures_dir=os.path.join(QUICK_ROOT, "report_assets", "figures"),
                  generated_dir=os.path.join(QUICK_ROOT, "report_assets", "generated"))
    try:
        generate(**kw)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
