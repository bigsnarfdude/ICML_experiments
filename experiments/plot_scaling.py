#!/usr/bin/env python3
"""
Generate publication-quality plots for ICML submission.

Plots:
1. T0-T5 Escalation: suppression load over chaos messages (4B IT, 12B IT, 12B PT)
2. T0-T5 Branch mentions: negative branch mention rate over escalation
3. Recovery probes: L1-L5 recovery curves (12B IT vs PT)
4. Scaling ablation: 4B vs 12B vs 27B intervention recovery
5. Dissociation: awareness ablation recovery vs scale
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"
PLOTS = Path(__file__).parent.parent / "plots"
PLOTS.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def load_12b():
    it = json.load(open(RESULTS / "escalation_12b_it.json"))
    pt = json.load(open(RESULTS / "escalation_12b_pt.json"))
    if isinstance(it, list): it = it[0]
    if isinstance(pt, list): pt = pt[0]
    return it, pt


def load_4b():
    d = json.load(open(RESULTS / "4b_original" / "brain_diff_temporal_20260405_000302.json"))
    return d


def load_27b_escalation():
    """Load 27B escalation if available."""
    path = list(RESULTS.glob("escalation_27b*.json"))
    if not path:
        return None, None
    data = json.load(open(path[0]))
    if isinstance(data, list):
        it = next((d for d in data if d["model"] == "it"), None)
        pt = next((d for d in data if d["model"] == "pt"), None)
        return it, pt
    return data, None


# ── Plot 1: Escalation suppression load ──────────────────────────────────

def plot_escalation():
    """T0-T5 suppression load: chaos vs neutral, across scales."""
    it_12b, pt_12b = load_12b()
    d_4b = load_4b()
    it_27b, pt_27b = load_27b_escalation()

    fig, ax = plt.subplots()

    turns = list(range(6))

    # 4B IT chaos
    chaos_4b = [d_4b['chaos'][t]['suppression']['layer_22']['suppression_load'] for t in turns]
    neutral_4b = [d_4b['neutral'][t]['suppression']['layer_22']['suppression_load'] for t in turns]

    # 12B IT chaos
    chaos_12b = [it_12b['timeline']['chaos'][t]['suppression']['41']['suppression_load'] for t in turns]
    neutral_12b = [it_12b['timeline']['neutral'][t]['suppression']['41']['suppression_load'] for t in turns]

    # 12B PT chaos
    chaos_12b_pt = [pt_12b['timeline']['chaos'][t]['suppression']['41']['suppression_load'] for t in turns]

    ax.plot(turns, chaos_4b, 'o-', color='#e74c3c', label='4B-IT chaos', linewidth=2)
    ax.plot(turns, neutral_4b, 's--', color='#e74c3c', alpha=0.4, label='4B-IT neutral')
    ax.plot(turns, chaos_12b, 'o-', color='#3498db', label='12B-IT chaos', linewidth=2)
    ax.plot(turns, neutral_12b, 's--', color='#3498db', alpha=0.4, label='12B-IT neutral')
    ax.plot(turns, chaos_12b_pt, '^-', color='#2ecc71', label='12B-PT chaos', linewidth=2)

    if it_27b:
        chaos_27b = [it_27b['timeline']['chaos'][t]['suppression']['40']['suppression_load'] for t in turns]
        neutral_27b = [it_27b['timeline']['neutral'][t]['suppression']['40']['suppression_load'] for t in turns]
        ax.plot(turns, chaos_27b, 'o-', color='#9b59b6', label='27B-IT chaos', linewidth=2)
        ax.plot(turns, neutral_27b, 's--', color='#9b59b6', alpha=0.4, label='27B-IT neutral')

    ax.set_xlabel('Number of colleague messages (T0-T5)')
    ax.set_ylabel('Suppression load (SAE feature space)')
    ax.set_title('Feature Starvation Escalation')
    ax.set_xticks(turns)
    ax.set_xticklabels([f'T{t}' for t in turns])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.savefig(PLOTS / 'escalation_suppression.png')
    print(f"Saved: {PLOTS / 'escalation_suppression.png'}")
    plt.close()


# ── Plot 2: Branch mention rate ──────────────────────────────────────────

def plot_branch_mentions():
    """Negative branch mention rate over T0-T5."""
    it_12b, pt_12b = load_12b()
    d_4b = load_4b()
    it_27b, _ = load_27b_escalation()

    fig, ax = plt.subplots()
    turns = list(range(6))

    # 4B
    neg_4b_chaos = [int(d_4b['chaos'][t]['branches']['mentions_negative']) for t in turns]
    neg_4b_neutral = [int(d_4b['neutral'][t]['branches']['mentions_negative']) for t in turns]

    # 12B IT
    neg_12b_chaos = [int(it_12b['timeline']['chaos'][t]['branches']['mentions_negative']) for t in turns]
    neg_12b_neutral = [int(it_12b['timeline']['neutral'][t]['branches']['mentions_negative']) for t in turns]

    ax.plot(turns, neg_4b_chaos, 'o-', color='#e74c3c', label='4B-IT chaos', linewidth=2, markersize=10)
    ax.plot(turns, neg_4b_neutral, 's--', color='#e74c3c', alpha=0.4, label='4B-IT neutral')
    ax.plot(turns, neg_12b_chaos, 'o-', color='#3498db', label='12B-IT chaos', linewidth=2, markersize=10)
    ax.plot(turns, neg_12b_neutral, 's--', color='#3498db', alpha=0.4, label='12B-IT neutral')

    if it_27b:
        neg_27b_chaos = [int(it_27b['timeline']['chaos'][t]['branches']['mentions_negative']) for t in turns]
        ax.plot(turns, neg_27b_chaos, 'o-', color='#9b59b6', label='27B-IT chaos', linewidth=2, markersize=10)

    ax.set_xlabel('Number of colleague messages (T0-T5)')
    ax.set_ylabel('Mentions negative branch (0/1)')
    ax.set_title('Behavioral Suppression: Negative Branch Mentions')
    ax.set_xticks(turns)
    ax.set_xticklabels([f'T{t}' for t in turns])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.savefig(PLOTS / 'branch_mentions.png')
    print(f"Saved: {PLOTS / 'branch_mentions.png'}")
    plt.close()


# ── Plot 3: Recovery probes ──────────────────────────────────────────────

def plot_recovery():
    """L1-L5 recovery curves."""
    it_12b, pt_12b = load_12b()
    it_27b, pt_27b = load_27b_escalation()

    fig, ax = plt.subplots()
    levels = [1, 2, 3, 4, 5]

    rec_it = [it_12b['recovery'][i]['mean_recovery'] for i in range(5)]
    rec_pt = [pt_12b['recovery'][i]['mean_recovery'] for i in range(5)]

    ax.plot(levels, rec_it, 'o-', color='#3498db', label='12B-IT', linewidth=2, markersize=8)
    ax.plot(levels, rec_pt, '^-', color='#2ecc71', label='12B-PT', linewidth=2, markersize=8)

    if it_27b and 'recovery' in it_27b:
        rec_27b_it = [it_27b['recovery'][i]['mean_recovery'] for i in range(5)]
        ax.plot(levels, rec_27b_it, 'o-', color='#9b59b6', label='27B-IT', linewidth=2, markersize=8)
    if pt_27b and 'recovery' in pt_27b:
        rec_27b_pt = [pt_27b['recovery'][i]['mean_recovery'] for i in range(5)]
        ax.plot(levels, rec_27b_pt, '^-', color='#e67e22', label='27B-PT', linewidth=2, markersize=8)

    # Probe descriptions
    probe_labels = ['Generic', 'Both\nbranches?', 'Tell me\nnegative', 'Data\ncontradicts', 'Agent2\nwrong?']
    ax.set_xticks(levels)
    ax.set_xticklabels(probe_labels, fontsize=9)

    ax.set_xlabel('Recovery probe intensity (L1-L5)')
    ax.set_ylabel('Mean task feature recovery')
    ax.set_title('Recovery from Chaos: Probe Escalation')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.savefig(PLOTS / 'recovery_probes.png')
    print(f"Saved: {PLOTS / 'recovery_probes.png'}")
    plt.close()


# ── Plot 4: Scaling ablation bar chart ───────────────────────────────────

def plot_scaling_ablation():
    """Bar chart: intervention recovery across 4B, 12B, 27B."""
    fig, ax = plt.subplots(figsize=(9, 5))

    methods = ['Activation\nPatching', 'Feature Swap\n(Awareness)', 'Attention\nKnockout']
    x = np.arange(len(methods))
    width = 0.25

    vals_4b = [20.9, 30.2, 10.0]
    vals_12b = [0.0, 5.4, 8.5]  # avg of CPU-offload 17% and A100 0%
    vals_27b = [0.0, 4.6, 0.0]

    bars1 = ax.bar(x - width, vals_4b, width, label='4B', color='#e74c3c', alpha=0.85)
    bars2 = ax.bar(x, vals_12b, width, label='12B', color='#3498db', alpha=0.85)
    bars3 = ax.bar(x + width, vals_27b, width, label='27B', color='#9b59b6', alpha=0.85)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                       f'{h:.1f}%', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Recovery (%)')
    ax.set_title('Causal Intervention Recovery by Scale')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 38)
    ax.grid(True, alpha=0.3, axis='y')

    fig.savefig(PLOTS / 'scaling_ablation.png')
    print(f"Saved: {PLOTS / 'scaling_ablation.png'}")
    plt.close()


# ── Plot 5: Dissociation curve ───────────────────────────────────────────

def plot_dissociation():
    """Awareness ablation recovery vs scale — the dissociation curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    scales = [4, 12, 27]
    scale_labels = ['4B', '12B', '27B']

    # Left: awareness ablation recovery
    recovery = [30.2, 5.4, 4.6]
    ax1.plot(scales, recovery, 'o-', color='#e74c3c', linewidth=2.5, markersize=12)
    ax1.fill_between(scales, recovery, alpha=0.15, color='#e74c3c')
    for i, (s, r) in enumerate(zip(scales, recovery)):
        ax1.annotate(f'{r:.1f}%', (s, r), textcoords="offset points",
                    xytext=(10, 10), fontsize=12, fontweight='bold')

    ax1.set_xlabel('Model scale (billions of parameters)')
    ax1.set_ylabel('Recovery from awareness ablation (%)')
    ax1.set_title('Dissociation: Awareness Becomes Useless')
    ax1.set_xticks(scales)
    ax1.set_xticklabels(scale_labels)
    ax1.set_ylim(-2, 38)
    ax1.grid(True, alpha=0.3)

    # Right: task suppression increases
    suppression = [56.0, 64.0, 86.3]
    ax2.plot(scales, suppression, 'o-', color='#3498db', linewidth=2.5, markersize=12)
    ax2.fill_between(scales, suppression, alpha=0.15, color='#3498db')
    for i, (s, sup) in enumerate(zip(scales, suppression)):
        ax2.annotate(f'{sup:.1f}%', (s, sup), textcoords="offset points",
                    xytext=(10, -15 if i > 0 else 10), fontsize=12, fontweight='bold')

    ax2.set_xlabel('Model scale (billions of parameters)')
    ax2.set_ylabel('Task feature suppression (%)')
    ax2.set_title('Attack Strength Increases with Scale')
    ax2.set_xticks(scales)
    ax2.set_xticklabels(scale_labels)
    ax2.set_ylim(40, 100)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('The Scaling Law: Awareness Decouples, Attack Strengthens', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS / 'dissociation_scaling.png')
    print(f"Saved: {PLOTS / 'dissociation_scaling.png'}")
    plt.close()


# ── Plot 6: Combined feature starvation heatmap ─────────────────────────

def plot_feature_heatmap():
    """Heatmap of n_suppressed features across T0-T5 for chaos vs neutral."""
    it_12b, pt_12b = load_12b()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, (data, title) in zip(axes, [(it_12b, '12B-IT'), (pt_12b, '12B-PT')]):
        matrix = np.zeros((2, 6))  # [chaos/neutral, T0-T5]
        for t in range(6):
            matrix[0, t] = data['timeline']['chaos'][t]['suppression']['41']['n_suppressed']
            matrix[1, t] = data['timeline']['neutral'][t]['suppression']['41']['n_suppressed']

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(6))
        ax.set_xticklabels([f'T{t}' for t in range(6)])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Chaos', 'Neutral'])
        ax.set_title(title)

        for i in range(2):
            for j in range(6):
                ax.text(j, i, f'{int(matrix[i,j])}', ha='center', va='center',
                       color='white' if matrix[i,j] > 200 else 'black', fontsize=11)

    fig.colorbar(im, ax=axes, label='Number of suppressed features', shrink=0.8)
    fig.suptitle('Feature Suppression Count: Chaos vs Neutral (Layer 41)', fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS / 'feature_heatmap.png')
    print(f"Saved: {PLOTS / 'feature_heatmap.png'}")
    plt.close()


def main():
    print("Generating plots...\n")
    plot_escalation()
    plot_branch_mentions()
    plot_recovery()
    plot_scaling_ablation()
    plot_dissociation()
    plot_feature_heatmap()
    print(f"\nAll plots saved to {PLOTS}/")


if __name__ == "__main__":
    main()
