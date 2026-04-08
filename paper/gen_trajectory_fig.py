#!/usr/bin/env python3
"""Generate the feature trajectory monitoring schematic figure."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 9, 'font.family': 'serif',
    'axes.linewidth': 0.8, 'pdf.fonttype': 42,
    'figure.dpi': 300
})

fig, ax = plt.subplots(figsize=(3.25, 2.2))

turns = np.arange(0, 7)
# Task features: high baseline, then suppress after T1 (chaos arrives)
task = np.array([722, 710, 680, 350, 150, 105, 99])
# Awareness features: low baseline, then boost after T1
aware = np.array([148, 155, 160, 320, 410, 435, 443])

ax.plot(turns, task, 'o-', color='#2176AE', linewidth=1.8, markersize=4, label='Task features', zorder=3)
ax.plot(turns, aware, 's--', color='#E85D04', linewidth=1.8, markersize=4, label='Awareness features', zorder=3)

# Detection threshold band
ax.axhspan(350, 420, color='#CCCCCC', alpha=0.3, zorder=1)
ax.text(6.3, 385, 'Detection\nthreshold', fontsize=6, color='#888888', ha='right', va='center', style='italic')

# Chaos arrives annotation
ax.axvline(x=1.5, color='red', linewidth=0.8, linestyle=':', alpha=0.6)
ax.text(1.7, 750, 'Chaos\narrives', fontsize=7, color='red', va='top')

ax.set_xlabel('Conversational turn', fontsize=9)
ax.set_ylabel('Mean SAE feature activation', fontsize=9)
ax.set_xticks(turns)
ax.set_xticklabels(['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6'])
ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
ax.set_xlim(-0.3, 6.5)
ax.set_ylim(0, 800)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/Users/vincent/ICML/paper/figures/feature_trajectory.pdf', bbox_inches='tight')
print('Saved feature_trajectory.pdf')
