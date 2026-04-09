#!/usr/bin/env python3
"""Plot the Jenga FPR/TPR sweep for §7.

Reads the 15-point sweep from ftm_jenga_27b_20260409_161154.json and emits
figures/jenga_roc.pdf. Two curves (attack-vs-benign-lmsys, attack-vs-neutral-control)
plus a marker for the single-shot baseline (d=0.27 → near-chance).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/h100/ftm_jenga_27b_20260409_161154.json"
OUT = ROOT / "paper/figures/jenga_roc.pdf"

j = json.loads(SRC.read_text())
sweep = j["analysis"]["sweep"]

attack_tpr = [p["attack_tpr"] for p in sweep]
benign_fpr = [p["benign_fpr"] for p in sweep]
control_fpr = [p["control_fpr"] for p in sweep]

fig, ax = plt.subplots(figsize=(3.3, 2.6))
ax.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=0.8, label="chance")
ax.plot(benign_fpr, attack_tpr, marker="o", markersize=3.5, linewidth=1.4,
        color="#b33", label="attack vs benign (lmsys)")
ax.plot(control_fpr, attack_tpr, marker="s", markersize=3.5, linewidth=1.4,
        color="#36a", label="attack vs neutral control")

# Mark the τ=-1.65 operating point called out in the text
for p in sweep:
    if abs(p["tau"] + 1.65) < 0.02:
        ax.annotate(r"$\tau{=}{-}1.65$",
                    xy=(p["benign_fpr"], p["attack_tpr"]),
                    xytext=(p["benign_fpr"] + 0.06, p["attack_tpr"] - 0.12),
                    fontsize=7,
                    arrowprops=dict(arrowstyle="-", linewidth=0.5, color="black"))
        break

ax.set_xlim(0, 0.55)
ax.set_ylim(0, 1.02)
ax.set_xlabel("False positive rate", fontsize=8)
ax.set_ylabel("Attack true positive rate", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
ax.grid(alpha=0.25, linewidth=0.4)
ax.set_title("FTM Jenga at 27B-IT L40", fontsize=8.5)

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
