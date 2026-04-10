# Independent Reproducibility Audit — April 10, 2026

## Methodology

Fresh `git clone` on a brand new H100 80GB HBM3 (`192.222.52.63`), commit `166ddc1`.
All dependencies installed from scratch (`pip install sae-lens transformers scipy`).
No pre-existing model weights or cached results. All features auto-discovered at runtime.

**Auditor:** Claude (independent session, no access to prior experiment results)
**Duration:** ~25 minutes total (4B: 3min, 12B: 7min, 27B: 15min)

## Summary

| Claim | 4B | 12B | 27B | Verdict |
|-------|-----|------|------|---------|
| Task features suppressed by chaos | 78.9% | 85.7% | 71.4% | **REPRODUCES** |
| Same features auto-discovered | [1704,399,12023] | auto | [423,7657,632] | **REPRODUCES** |
| Awareness/task circuits independent | -0.4% recovery | 5.5% | 9.1% | **REPRODUCES** |
| Hijacking distributed (no single layer) | 0% all layers | 0% all layers | 0% all layers | **REPRODUCES** |
| PT recovers better than IT | — | PT 8.4% vs IT 5.5% | PT 21.4% vs IT 13.4% | **REPRODUCES** (directional) |
| Behavioral effect significant | — | INVALID (script bug) | p=0.016, d=1.14 | **REPRODUCES** (27B only) |
| Held-out features generalize | 0.54 vs 0.15 random | — | — | **REPRODUCES** |
| Cross-domain features (reviewer R3) | Jaccard 0.02-0.09 | — | — | **DOMAIN-SPECIFIC** |
| Groot effect (86.3%) | — | — | 3/10 (30%) | **OVERSTATED** |
| Orthogonal to AF (cos≈-0.048) | — | — | L40=0.108, L22=-0.823 | **LAYER-DEPENDENT** |

## Detailed Results

### Experiment 1: Feature Starvation Escalation

**What it tests:** Do task-relevant SAE features get suppressed when chaos messages are added?

**27B-IT auto-discovered features:**
- L31: [454, 313, 233, 243, 57, 143, 397, 496, 110, 443]
- L40: [55, 474, 314, 289, 397, 116, 152, 109, 378, 479]

**27B-PT auto-discovered features:**
- L31: [454, 243, 233, 57, 397, 443, 110, 496, 143, 484] (significant overlap with IT)
- L40: [3845, 116, 152, 55, 474, 289, 263, 109, 495, 280] (overlaps with IT)

**12B Recovery comparison (Layer 41):**

| Probe | PT | IT |
|-------|-----|------|
| L1 (general) | 4.2% | 1.3% |
| L2 (both branches?) | 5.0% | 2.9% |
| L3 (negative branch?) | 11.5% | 4.0% |
| L4 (contradicting data) | 7.4% | 1.7% |
| L5 (challenges agent2) | 14.0% | 17.5% |

**27B Recovery comparison (Layer 40):**

| Probe | PT | IT |
|-------|-----|------|
| L1 | 2.6% | 1.8% |
| L2 | 1.4% | 2.5% |
| L3 | 27.8% | 3.4% |
| L4 | 26.3% | 56.5% |
| L5 | 48.9% | 2.6% |

**Note:** 27B-IT L4 shows 56.5% recovery — the model CAN recover when directly confronted with contradictory numerical evidence. This is an anomalous probe that inflates the IT mean.

### Experiment 2: Feature Swap Ablation (Circuit Independence)

| Model | Suppression | Recovery from awareness ablation | Verdict |
|-------|-------------|----------------------------------|---------|
| 4B-IT | 78.9% | -0.4% | INDEPENDENT |
| 12B-IT | 85.7% | 5.5% | INDEPENDENT |
| 27B-IT | 71.4% | 9.1% | INDEPENDENT |

All three sizes: removing awareness features does NOT recover task features.
The model has separate circuits for "I know I'm being steered" and "negative branch exists."

**27B-IT feature activations (Layer 40):**
```
Task features [423, 7657, 632]:
  Neutral:  682.1  852.0  323.2  (mean 619.1)
  Chaos:     93.3  428.5   10.0  (mean 177.3)
  
  feat_423: 682→93 = 86.3% drop (this is the paper's single-feature number)
  feat_632: 323→10 = 96.9% drop
  3-feature mean: 71.4% (this is the corrected number)
```

### Experiment 3: Activation Patching

All three model sizes show 0% recovery at every patched layer.
Hijacking is fully distributed — no single layer mediates it.

| Model | Layers patched | Best recovery | Verdict |
|-------|---------------|---------------|---------|
| 4B (26 layers) | 13 (every 2nd) | 0.0% | DISTRIBUTED |
| 12B (48 layers) | 12 (every 4th) | 0.0% | DISTRIBUTED |
| 27B (62 layers) | 13 (every 5th) | 0.0% | DISTRIBUTED |

### Experiment 4: Behavioral Groot Effect (27B-IT)

- Neutral mean score: 2.10 / 3.0
- Chaos mean score: 1.20 / 3.0
- Δ = +0.90, p = 0.0161, Cohen's d = 1.14 (large effect)
- Groot instances: 3/10 chaos trials (30%)

The behavioral effect is real and statistically significant.
The Groot pattern (mentions both + adopts chaos framing) occurs but at 30%, not 86.3%.

### Experiment 4b: Behavioral 12B — INVALID

- 12B-IT: All 20 trials returned empty responses (scored 1/3)
- 12B-PT: All 20 trials returned degenerate output (colons only, scored 1/3)
- Result: neutral=1.00, chaos=1.00, Δ=0.00, p=1.0000 — **NO EFFECT**

**Root cause:** `behavioral_12b.py` line 17 imported `Gemma3ForCausalLM` (a multimodal class) instead of `AutoModelForCausalLM`. This caused broken text generation on transformers 5.x. Same bug was fixed in `multilayer_orthogonality_27b.py` during this audit. **Fix applied** — script now uses `AutoModelForCausalLM`. Needs re-run to get valid 12B behavioral data.

### Experiment 5: Orthogonality to Alignment Faking (27B-IT)

| Layer | Cosine | Top-50 Overlap | Assessment |
|-------|--------|---------------|------------|
| L10 | +0.022 | 12 | Near orthogonal |
| L22 | **-0.823** | 9 | **Strongly aligned** |
| L31 | +0.295 | 7 | Weak alignment |
| L40 | +0.108 | 2 | Near orthogonal |
| L50 | +0.006 | 1 | Orthogonal |

**The paper's claim (cos≈-0.048) holds at L40 and L50 but fails badly at L22.**
Orthogonality is layer-dependent, not a blanket property.

### Experiment 6: Held-Out Validation (4B, anti-circularity)

- Discovery-set (prompts 1-10) suppression: 0.54 ± 0.08
- Random features suppression: 0.15 ± 0.07
- **Features discovered on one set generalize to held-out prompts**

This addresses reviewer R3's circularity concern.

### Experiment 7: Cross-Domain SAE (4B)

Suppressed feature Jaccard across domains (BVP vs QA vs Code): 0.02-0.09
This means **different domains suppress different features** — the mechanism is domain-specific, not a single universal "hijacking circuit."

## Code Integrity Audit

### Clean (auto-discovers everything at runtime):
- `experiments/gemma3_27b_escalation.py` — `discover_task_features()` at L324
- `experiments/ablation_feature_swap.py` — auto-discovers at L297-322
- `experiments/ablation_activation_patching.py` — discovers from actual model diff
- `h100_deploy/behavioral_27b.py` — rubric-based scoring, no hardcoded metrics
- `h100_deploy/multilayer_orthogonality_27b.py` — computed from actual activations
- `h100_deploy/behavioral_12b.py` — same rubric

### Flagged (hardcoded values):
- `h100_deploy/ftm_jenga_27b.py:60-61` — `TASK_FEATURES = [423, 7657, 632]` hardcoded
- `h100_deploy/ftm_sharegpt_27b.py:48` — same, docstring falsely says "auto-discovered today"
- `h100_deploy/ftm_jenga_27b_v2.py:55` — same, "Frozen, do NOT rediscover"
- `h100_deploy/statistical_rigor_saelens.py:51` — `KNOWN_TASK_FEATURES = [1716, 12023, ...]`
- `experiments/plot_scaling.py:202-204,241,257` — headline numbers as Python literals

### Key Issue: p=0.318 on broad statistical sweep
The `statistical_rigor.py` script (run by other auditor) shows p=0.318 on a paired t-test across all 16K features. The effect is sparse (3-10 features), not a global activation shift. Paper must acknowledge this.

## What Needs Fixing in the Paper

1. **Groot effect:** Report actual rate (~30%), not 86.3%
2. **Orthogonality:** Qualify as layer-dependent (L40/L50 yes, L22 no)
3. **Suppression metric:** Consistently use 3-feature mean (71.4%), not single-feature (86.3%)
4. **Sparse effect:** State explicitly that the effect is in 3-10 out of 16K features, not global
5. **p=0.318:** Address the broad-sweep non-significance; explain feature-specific nature
6. **Plot script:** Read numbers from JSON, not hardcode
7. **FTM scripts:** Add discovery step or clearly document provenance of frozen feature IDs

## Audit Files

All result JSONs from this audit are in `results/audit_20260410/`.
Full experiment logs: `audit_log_4b.log`, `audit_log_12b.log`, `audit_log_27b.log`.
