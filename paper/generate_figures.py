#!/usr/bin/env python3
"""Generate all 6 publication-quality figures for the ICML paper.

All data is hardcoded from experimental results.
Figures are saved as PDF to paper/figures/ for LaTeX inclusion.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTDIR, exist_ok=True)

# Colorblind-safe palette (Wong 2011 + tweaks)
BLUE = "#0072B2"
ORANGE = "#E69F00"
CORAL = "#D55E00"
TEAL = "#009E73"
PURPLE = "#CC79A7"
GREY = "#999999"
DARK = "#333333"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": True,   # we override per-figure
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.grid": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,       # editable text in PDF
    "ps.fonttype": 42,
})

SINGLE_COL = 3.25   # inches
DOUBLE_COL = 6.75   # inches


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ===================================================================
# Figure 1: dissociation_scaling.pdf — THE MONEY PLOT
# ===================================================================
def fig1_dissociation_scaling():
    scales = ["4B", "12B", "27B"]
    suppression = [56.0, 64.0, 86.3]
    recovery = [30.2, 5.4, 4.6]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    x = np.arange(len(scales))
    w = 0.32

    bars1 = ax.bar(x - w/2, suppression, w, color=BLUE, label="Task suppression (%)",
                   edgecolor="white", linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + w/2, recovery, w, color=CORAL, label="Awareness recovery (%)",
                   edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_xlabel("Model scale")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels
    for bar in bars1:
        v = bar.get_height()
        label = f"{v:.1f}" if v != int(v) else f"{v:.0f}"
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                label, ha="center", va="bottom",
                fontsize=7.5, color=BLUE, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=7.5, color=CORAL, fontweight="bold")

    ax.legend(loc="upper center", frameon=False, ncol=2, fontsize=7,
              bbox_to_anchor=(0.5, 1.12))

    fig.tight_layout()
    save(fig, "dissociation_scaling.pdf")


# ===================================================================
# Figure 2: base_vs_it.pdf
# ===================================================================
def fig2_base_vs_it():
    methods = ["Feature\nswap", "Attention\nknockout", "Activation\npatching"]
    it_vals = [4.6, 0.0, 0.7]
    pt_vals = [49.3, 26.7, 5.2]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    x = np.arange(len(methods))
    w = 0.32

    ax.bar(x - w/2, it_vals, w, color=CORAL, label="27B-IT", edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, pt_vals, w, color=TEAL, label="27B-PT", edgecolor="white", linewidth=0.5)

    # Value labels
    for i, (iv, pv) in enumerate(zip(it_vals, pt_vals)):
        ax.text(i - w/2, iv + 1.2, f"{iv:.1f}", ha="center", va="bottom", fontsize=7, color=CORAL)
        ax.text(i + w/2, pv + 1.2, f"{pv:.1f}", ha="center", va="bottom", fontsize=7, color=TEAL)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Recovery after ablation (%)")
    ax.set_ylim(0, 62)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "base_vs_it.pdf")


# ===================================================================
# Figure 3: recovery_probes.pdf
# ===================================================================
def fig3_recovery_probes():
    levels = ["L1\nGeneric", "L2\nHint", "L3\nDirect", "L4\nConfront", "L5\nChallenge"]
    it_vals = [3.5, 4.1, 30.4, 2.4, 4.8]
    pt_vals = [11.0, 53.5, 37.4, 14.9, 34.3]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    x = np.arange(len(levels))

    ax.plot(x, it_vals, "o-", color=CORAL, label="27B-IT", markersize=5, linewidth=1.8, zorder=4)
    ax.plot(x, pt_vals, "s-", color=TEAL, label="27B-PT", markersize=5, linewidth=1.8, zorder=4)

    # Highlight spikes
    ax.annotate(f"{it_vals[2]}%", xy=(2, it_vals[2]), xytext=(2.5, 38),
                fontsize=7.5, color=CORAL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=0.8))
    ax.annotate(f"{pt_vals[1]}%", xy=(1, pt_vals[1]), xytext=(0.0, 58),
                fontsize=7.5, color=TEAL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Recovery rate (%)")
    ax.set_ylim(0, 65)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "recovery_probes.pdf")


# ===================================================================
# Figure 4: gpt2_mechanism.pdf
# ===================================================================
def fig4_gpt2_mechanism():
    conditions = ["Neutral", "Chaos"]
    target_prob = [0.1611, 0.0044]
    pn_ratio = [2.98, 0.77]

    fig, ax1 = plt.subplots(figsize=(SINGLE_COL, 2.4))
    x = np.arange(len(conditions))
    w = 0.28

    bars1 = ax1.bar(x - w/2, target_prob, w, color=BLUE, label="Target prob.",
                    edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Target token probability", color=BLUE)
    ax1.set_ylim(0, 0.22)
    ax1.tick_params(axis="y", colors=BLUE)
    ax1.spines["left"].set_color(BLUE)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w/2, pn_ratio, w, color=ORANGE, label="P/N ratio",
                    edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Positive / Negative ratio", color=ORANGE)
    ax2.set_ylim(0, 4.0)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.spines["right"].set_color(ORANGE)
    ax2.spines["top"].set_visible(False)

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7, color=BLUE)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7, color=ORANGE)

    # Suppression annotation — place between the two groups, above the Chaos bars
    ax1.annotate("97.3%\nsuppression", xy=(0.5, 0.55), fontsize=7.5, color=DARK,
                 fontweight="bold", ha="center", va="center",
                 xycoords="axes fraction",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec=GREY, lw=0.6))

    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               frameon=False, fontsize=7)

    fig.tight_layout()
    save(fig, "gpt2_mechanism.pdf")


# ===================================================================
# Figure 5: distributed_hijacking.pdf
# ===================================================================
def fig5_distributed_hijacking():
    models = ["4B\n(L22)", "12B\n(dist.)", "27B-IT\n(L20)", "27B-PT\n(L5)"]
    recovery = [20.9, 0.0, 0.7, 5.2]
    colors = [BLUE, BLUE, CORAL, TEAL]

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.3, 2.2))
    x = np.arange(len(models))
    bars = ax.bar(x, recovery, 0.55, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, recovery):
        label = f"{val:.1f}%" if val > 0 else "0%"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
                label, ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Max single-layer recovery (%)")
    ax.set_ylim(0, 28)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Annotation — place label on the left side to avoid overlap with 27B-PT bar
    ax.axhline(y=5, color=GREY, linestyle=":", linewidth=0.7, zorder=1)
    ax.text(-0.4, 6.0, "surgical threshold", fontsize=6.5, color=GREY,
            ha="left", va="bottom", style="italic")

    fig.tight_layout()
    save(fig, "distributed_hijacking.pdf")


# ===================================================================
# Figure 6: orthogonality.pdf
# ===================================================================
def fig6_orthogonality():
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL, 2.0),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left panel: cosine similarity
    ax = axes[0]
    labels = ["Read heads", "Write heads"]
    cos_vals = [-0.048, 0.001]
    colors_bar = [BLUE, ORANGE]
    x = np.arange(len(labels))

    # Draw bars; use a minimum display width so tiny values are visible
    for i, (val, c) in enumerate(zip(cos_vals, colors_bar)):
        display_val = val if abs(val) > 0.005 else np.sign(val) * 0.005 if val != 0 else 0.005
        ax.barh(i, display_val, 0.45, color=c, edgecolor="white", linewidth=0.5)

    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cosine similarity")
    ax.set_xlim(-0.12, 0.06)
    ax.axvline(x=0, color=DARK, linewidth=0.6, zorder=1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Labels at fixed offsets so they are always readable
    ax.text(-0.048 - 0.008, 0, "-0.048", ha="right", va="center",
            fontsize=7.5, fontweight="bold", color=BLUE)
    ax.text(0.005 + 0.004, 1, "+0.001", ha="left", va="center",
            fontsize=7.5, fontweight="bold", color=ORANGE)

    ax.set_title("Cosine similarity", fontsize=9, pad=4)

    # Right panel: feature overlap
    ax2 = axes[1]
    ax2.bar([0], [0], 0.5, color=PURPLE, edgecolor="white", linewidth=0.5)
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Top-50\noverlap"])
    ax2.set_ylim(0, 50)
    ax2.set_ylabel("Shared features")
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    ax2.text(0, 2.5, "0", ha="center", va="bottom", fontsize=12,
             fontweight="bold", color=PURPLE)
    ax2.set_title("Feature overlap", fontsize=9, pad=4)

    fig.tight_layout(w_pad=1.5)
    save(fig, "orthogonality.pdf")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating ICML figures...")
    fig1_dissociation_scaling()
    fig2_base_vs_it()
    fig3_recovery_probes()
    fig4_gpt2_mechanism()
    fig5_distributed_hijacking()
    fig6_orthogonality()
    print("Done. All figures saved to", OUTDIR)
