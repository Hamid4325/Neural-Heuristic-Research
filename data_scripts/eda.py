"""
eda.py
Phase 2 (Exploratory Data Analysis):
Produce the figures used in the technical report from the generated,
labelled, cleaned and feature-engineered datasets.

Figures written to figures/:
    01_density_target_vs_achieved.png   obstacle density: target vs achieved
    02_maze_counts.png                  maze counts by size and generator
    03_opt_cost_by_size.png             optimal path cost distribution per size
    04_hstar_distribution.png           normalised h*(n) distribution per size
    05_reachable_fraction.png           reachable free-cell fraction vs density
    06_sample_mazes.png                 example mazes with h* heatmaps
    07_cleaning_summary.png             cleaning before/after counts
    08_feature_correlation.png          correlation of the 29 input features

Input (produced by the earlier pipeline stages):
    data/raw/*.npz, data/labeled/*.npz, data/cleaned/*.npz
    data/processed/*.npz
    data_scripts/generation_log.json, clean_log.json

Usage:
  python data_scripts/eda.py
  python data_scripts/eda.py --figdir figures --processed data/processed
"""

import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

# Consistent colours for the two generators.
METHOD_COLORS = {"backtracker": "#3b6fb6", "perlin": "#d1603d"}
NAME_RE = re.compile(r"mazes_(?P<method>[a-z]+)_(?P<size>\d+)_(?P<density>\d+)\.npz$")


def parse_name(path):
    """mazes_backtracker_100_20.npz -> ('backtracker', 100, 20)"""
    m = NAME_RE.search(os.path.basename(path))
    return m.group("method"), int(m.group("size")), int(m.group("density"))


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def savefig(figdir, name):
    out = os.path.join(figdir, name)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 1: target vs achieved obstacle density
# ---------------------------------------------------------------------------
def fig_density(gen_log, figdir):
    configs = gen_log.get("configs", [])
    if not configs:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for method in sorted({c["method"] for c in configs}):
        xs = [c["target_density"] * 100 for c in configs if c["method"] == method]
        ys = [c["achieved_density_mean"] * 100 for c in configs if c["method"] == method]
        ax.scatter(xs, ys, label=method, color=METHOD_COLORS.get(method), alpha=0.8, s=45)
    lim = [0, 45]
    ax.plot(lim, lim, "--", color="grey", lw=1, label="y = x (perfect)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Target obstacle density (%)")
    ax.set_ylabel("Achieved mean obstacle density (%)")
    ax.set_title("Obstacle density: target vs. achieved")
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(figdir, "01_density_target_vs_achieved.png")


# ---------------------------------------------------------------------------
# Figure 2: maze counts by size and generator
# ---------------------------------------------------------------------------
def fig_counts(files, figdir):
    sizes = sorted({parse_name(f)[1] for f in files})
    methods = ["backtracker", "perlin"]
    counts = {m: [0] * len(sizes) for m in methods}
    for f in files:
        method, size, _ = parse_name(f)
        counts[method][sizes.index(size)] += 1

    x = np.arange(len(sizes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, method in enumerate(methods):
        ax.bar(x + (i - 0.5) * w, counts[method], width=w,
               label=method, color=METHOD_COLORS[method])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Number of mazes (after cleaning)")
    ax.set_title("Maze counts by grid size and generator")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    savefig(figdir, "02_maze_counts.png")


# ---------------------------------------------------------------------------
# Figure 3: optimal path cost distribution per grid size
# ---------------------------------------------------------------------------
def fig_opt_cost(files, figdir):
    by_size = {}
    for f in files:
        _, size, _ = parse_name(f)
        data = np.load(f)
        by_size.setdefault(size, []).extend(data["opt_cost"].tolist())

    sizes = sorted(by_size)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([by_size[s] for s in sizes], tick_labels=[f"{s}x{s}" for s in sizes])
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Optimal path cost  h*(start)  (cells)")
    ax.set_title("Optimal start-to-goal path cost by grid size")
    ax.grid(axis="y", alpha=0.3)
    savefig(figdir, "03_opt_cost_by_size.png")


# ---------------------------------------------------------------------------
# Figure 4: normalised h*(n) distribution per grid size
# ---------------------------------------------------------------------------
def fig_hstar(files, figdir, max_per_size=20000, rng=None):
    rng = rng or np.random.default_rng(0)
    by_size = {}
    for f in files:
        _, size, _ = parse_name(f)
        data = np.load(f)
        h = data["h_star"]
        n = h.shape[-1]
        vals = h[np.isfinite(h)].ravel() / n  # reachable cells, normalised by N
        if len(vals) > max_per_size:
            vals = rng.choice(vals, size=max_per_size, replace=False)
        by_size.setdefault(size, []).append(vals)

    sizes = sorted(by_size)
    merged = [np.concatenate(by_size[s]) for s in sizes]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    parts = ax.violinplot(merged, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#6a9bcc")
        pc.set_alpha(0.7)
    ax.set_xticks(np.arange(1, len(sizes) + 1))
    ax.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Normalised ground-truth  h*(n) / N")
    ax.set_title("Distribution of normalised ground-truth heuristic values")
    ax.grid(axis="y", alpha=0.3)
    savefig(figdir, "04_hstar_distribution.png")


# ---------------------------------------------------------------------------
# Figure 5: reachable free-cell fraction vs obstacle density
# ---------------------------------------------------------------------------
def fig_reachable(files, figdir):
    points = {"backtracker": [], "perlin": []}
    for f in files:
        method, _, density = parse_name(f)
        data = np.load(f)
        grids, h = data["grids"], data["h_star"]
        free = grids == 0
        reachable = np.isfinite(h) & free
        frac = reachable.sum() / max(free.sum(), 1)
        points[method].append((density, frac))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, pts in points.items():
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] * 100 for p in pts]
        ax.plot(xs, ys, "o-", label=method, color=METHOD_COLORS[method])
    ax.set_xlabel("Target obstacle density (%)")
    ax.set_ylabel("Reachable free cells (%)")
    ax.set_title("Cell reachability vs. obstacle density")
    ax.set_ylim(50, 101)
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(figdir, "05_reachable_fraction.png")


# ---------------------------------------------------------------------------
# Figure 6: example mazes with h* heatmaps
# ---------------------------------------------------------------------------
def fig_samples(cleaned_files, figdir):
    picks = []
    for method in ("backtracker", "perlin"):
        for size in (40, 60, 100):
            for f in cleaned_files:
                if parse_name(f)[:2] == (method, size):
                    picks.append(f)
                    break

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, f in zip(axes.ravel(), picks):
        method, size, _ = parse_name(f)
        data = np.load(f)
        g, h = data["grids"][0], data["h_star"][0]
        start, goal = data["starts"][0], data["goals"][0]

        vis = h.copy()
        vis[vis < 0] = np.nan
        vis[g == 1] = np.nan
        ax.imshow(g, cmap="Greys", vmin=0, vmax=1)
        im = ax.imshow(vis, cmap="viridis_r")
        ax.scatter(start[1], start[0], marker="o", s=60, c="lime",
                   edgecolors="black", label="start", zorder=3)
        ax.scatter(goal[1], goal[0], marker="*", s=120, c="red",
                   edgecolors="black", label="goal", zorder=3)
        ax.set_title(f"{method}  {size}x{size}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="h*(n)")
    axes.ravel()[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Example mazes with ground-truth distance h*(n) and start/goal", y=1.01)
    savefig(figdir, "06_sample_mazes.png")


# ---------------------------------------------------------------------------
# Figure 7: cleaning before/after summary
# ---------------------------------------------------------------------------
def fig_cleaning(clean_log, figdir):
    s = clean_log.get("summary", {})
    if not s:
        return
    labels = ["Initial", "Unreachable\n(dropped)", "Duplicate\n(dropped)",
              "Outlier\n(dropped)", "Final"]
    values = [s["initial_total_mazes"], s["dropped_unreachable_mazes"],
              s["dropped_duplicate_mazes"], s["dropped_outlier_mazes"],
              s["final_cleaned_mazes"]]
    colors = ["#4c72b0", "#c44e52", "#dd8452", "#dd8452", "#55a868"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Number of mazes")
    ax.set_title("Data cleaning: mazes before and after each stage")
    ax.grid(axis="y", alpha=0.3)
    savefig(figdir, "07_cleaning_summary.png")


# ---------------------------------------------------------------------------
# Figure 8: correlation heatmap of the 29 input features
# ---------------------------------------------------------------------------
def fig_correlation(train_path, figdir, sample=200000):
    if not os.path.exists(train_path):
        return
    data = np.load(train_path)
    X = data["X"]
    if len(X) > sample:
        idx = np.random.default_rng(0).choice(len(X), size=sample, replace=False)
        X = X[idx]
    names = ["coord_x", "coord_y", "goal_dx", "goal_dy"] + \
            [f"w{i // 5},{i % 5}" for i in range(25)]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(X.T.astype(np.float64))
    corr = np.nan_to_num(corr, nan=0.0)  # constant columns have undefined r

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title("Correlation of the 29 per-cell input features")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    savefig(figdir, "08_feature_correlation.png")


def main():
    parser = argparse.ArgumentParser(description="EDA figures for Neural Heuristic Search")
    parser.add_argument("--labeled", default="data/labeled")
    parser.add_argument("--cleaned", default="data/cleaned")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--figdir", default="figures")
    parser.add_argument("--logdir", default="data_scripts")
    args = parser.parse_args()

    os.makedirs(args.figdir, exist_ok=True)

    labeled = sorted(glob.glob(os.path.join(args.labeled, "*.npz")))
    cleaned = sorted(glob.glob(os.path.join(args.cleaned, "*.npz")))
    if not cleaned:
        raise SystemExit(f"No cleaned data in '{args.cleaned}'. Run clean_data.py first.")

    gen_log = load_json(os.path.join(args.logdir, "generation_log.json"))
    clean_log = load_json(os.path.join(args.logdir, "clean_log.json"))

    print("Generating EDA figures...")
    fig_density(gen_log, args.figdir)
    fig_counts(cleaned, args.figdir)
    fig_opt_cost(labeled, args.figdir)
    fig_hstar(labeled, args.figdir)
    fig_reachable(labeled, args.figdir)
    fig_samples(cleaned, args.figdir)
    fig_cleaning(clean_log, args.figdir)
    fig_correlation(os.path.join(args.processed, "train.npz"), args.figdir)
    print(f"Done. Figures written to {args.figdir}/")


if __name__ == "__main__":
    main()
