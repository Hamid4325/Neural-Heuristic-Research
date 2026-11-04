"""Generate the NeuralHeuristicNet architecture diagram."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "architecture.png"

INPUT_COLOR = "#dbeafe"
TRUNK_COLOR = "#bfdbfe"
HEAD_COLOR = "#c7d2fe"
OUTPUT_COLOR = "#dcfce7"
LOSS_COLOR = "#fef3c7"
TEXT_COLOR = "#1e293b"
EDGE_COLOR = "#334155"


def add_box(ax, x, y, width, height, title, subtitle, facecolor):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        linewidth=1.3,
        edgecolor=EDGE_COLOR,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height - 0.22,
        title,
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax.text(
        x + width / 2,
        y + height / 2 - 0.16,
        subtitle,
        ha="center",
        va="center",
        fontsize=8.8,
        color=TEXT_COLOR,
        linespacing=1.45,
    )


def add_arrow(ax, start, end, connectionstyle="arc3,rad=0.0"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.4,
        color=EDGE_COLOR,
        connectionstyle=connectionstyle,
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(
        0.3,
        7.55,
        "Neural heuristic architecture",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax.text(
        0.3,
        7.12,
        "Per-cell MLP with a shared trunk, a regression head and a distance-bin head",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#475569",
    )

    add_box(
        ax,
        0.3,
        3.25,
        2.55,
        2.45,
        "Input features",
        "29 values per cell\n\n2 normalised coordinates\n2 goal offsets\n25 = 5 x 5 wall window",
        INPUT_COLOR,
    )

    trunk_boxes = [
        (3.55, "Linear 29 \u2192 256", "ReLU\nDropout 0.1\noutput: (B, 256)"),
        (5.90, "Linear 256 \u2192 256", "ReLU\nDropout 0.1\noutput: (B, 256)"),
        (8.25, "Linear 256 \u2192 128", "ReLU\noutput: (B, 128)"),
    ]
    for x, title, subtitle in trunk_boxes:
        add_box(ax, x, 3.55, 1.95, 1.85, title, subtitle, TRUNK_COLOR)

    add_arrow(ax, (2.85, 4.47), (3.55, 4.47))
    add_arrow(ax, (5.50, 4.47), (5.90, 4.47))
    add_arrow(ax, (7.85, 4.47), (8.25, 4.47))

    add_box(
        ax,
        10.65,
        5.35,
        2.05,
        1.50,
        "Regression head",
        "Linear 128 \u2192 1\n\noutput: (B, 1)",
        HEAD_COLOR,
    )
    add_box(
        ax,
        10.65,
        1.85,
        2.05,
        1.50,
        "Bin head",
        "Linear 128 \u2192 10\n\noutput: (B, 10)",
        HEAD_COLOR,
    )

    add_arrow(
        ax,
        (10.20, 4.47),
        (10.65, 6.10),
        connectionstyle="arc3,rad=-0.18",
    )
    add_arrow(
        ax,
        (10.20, 4.47),
        (10.65, 2.60),
        connectionstyle="arc3,rad=0.18",
    )

    add_box(
        ax,
        13.15,
        5.55,
        1.80,
        1.10,
        "Prediction",
        "normalised cost\n(denormalise)",
        OUTPUT_COLOR,
    )
    add_box(
        ax,
        13.15,
        2.05,
        1.80,
        1.10,
        "Bin logits",
        "10 distance bins\nsoftmax in training",
        OUTPUT_COLOR,
    )

    add_arrow(ax, (12.70, 6.10), (13.15, 6.10))
    add_arrow(ax, (12.70, 2.60), (13.15, 2.60))

    add_box(
        ax,
        13.15,
        4.05,
        1.80,
        0.85,
        "SmoothL1 loss",
        "regression target y",
        LOSS_COLOR,
    )
    add_box(
        ax,
        13.15,
        0.55,
        1.80,
        0.85,
        "Cross-entropy",
        "distance-bin target",
        LOSS_COLOR,
    )

    add_arrow(ax, (14.05, 5.55), (14.05, 4.90))
    add_arrow(ax, (14.05, 2.05), (14.05, 1.40))

    ax.text(
        0.3,
        2.35,
        "Shared trunk: 107,787 trainable parameters",
        ha="left",
        va="top",
        fontsize=10,
        color="#475569",
    )
    ax.text(
        0.3,
        2.00,
        "Optimisation: AdamW, early stopping on validation MAE",
        ha="left",
        va="top",
        fontsize=10,
        color="#475569",
    )
    ax.text(
        0.3,
        1.65,
        "Optional asymmetric penalty discourages overestimating h*",
        ha="left",
        va="top",
        fontsize=10,
        color="#475569",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
