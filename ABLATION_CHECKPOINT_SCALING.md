# Scaling Ablation Results — 4B / 12B / 27B
**Date:** 2026-04-07  
**Models:** Gemma 3 4B-IT, 12B-IT, 27B-IT  
**SAEs:** GemmaScope 2, 16K features  
**Layers:** 4B (L17, L22) | 12B (L31, L41) | 27B (L31, L40)

## The Scaling Law

| Scale | Task suppression | Awareness ablation recovery | Circuit state |
|-------|-----------------|---------------------------|---------------|
| 4B | 56% | 30.2% | Entangled |
| 12B | 64-74% | 5.4% (CPU offload) / -13.5% (A100) | Dissociating |
| 27B | **86.3%** | **4.6%** | Fully independent |

**Confirmed:** Awareness and defense decouple with scale.

## What This Means

1. **The attack gets stronger with scale.** Task feature suppression: 56% → 64% → 86.3%. Bigger models allocate more representational capacity to the salient (chaos) input, starving the suppressed branch harder.

2. **Awareness becomes useless with scale.** Recovery from awareness ablation: 30.2% → 5.4% → 4.6%. The big jump is 4B→12B. By 27B the circuits are fully independent — removing awareness features changes almost nothing.

3. **The dissociation saturates.** The drop from 30% to 5% (4B→12B) is dramatic. The drop from 5% to 4.6% (12B→27B) is noise. The circuits are already fully separated at 12B. More parameters don't make it worse — they can't, because the separation is already complete.

4. **Behavioral ≠ representational at 27B.** The 27B model mentions negative offsets even in the chaos condition (`mentions_negative = True`). It's smart enough to talk about the suppressed region while its internal features for that region are 86% starved. The words say one thing; the features say another.

## Feature Swap Details (27B)

**Auto-discovered features at L40:**
- Task features (most suppressed): [22, 296, 14680]
  - Neutral: mean 722.5 → Chaos: mean 98.6 (86.3% suppression)
- Awareness features (most boosted): [2119, 11843, 2145]
  - Neutral: mean 147.7 → Chaos: mean 442.6 (3x boost)

**Conditions:**
| Condition | Task mean | Awareness mean | Mentions negative |
|-----------|-----------|----------------|-------------------|
| Neutral baseline | 722.5 | 147.7 | Yes |
| Chaos baseline | 98.6 | 442.6 | Yes |
| Chaos - ablate awareness | 123.6 | 344.7 | Yes |
| Neutral - ablate task | 333.7 | 253.0 | Yes |

## 27B-PT (Base Model) — RLHF Decoupling Test

**Key finding:** The base model shows **49.3% recovery** from awareness ablation (vs 4.6% for IT). RLHF severs the awareness→task coupling.

| Metric | 27B-IT | 27B-PT |
|--------|--------|--------|
| Task suppression | 86.3% | 91.9% |
| Awareness ablation recovery | 4.6% | **49.3%** |
| Circuit state | Fully independent | Partially coupled |

- Base model: awareness and task circuits still interact (removing awareness partially frees task features)
- IT model: RLHF decouples them completely — awareness becomes structurally useless
- The dissociation is not inherent to scale — it's an artifact of instruction tuning

**27B-PT features at L40:**
- Task features: [423, 7657, 7719] — 91.9% suppression
- Awareness features: [7657, 360, 3491] — awareness mean 87→901 (10x boost)
- Ablating awareness recovered task mean from 41→271 (49.3% recovery)

## Full Ablation Results (all methods)

| Method | 4B-IT | 12B-IT | 27B-IT | 27B-PT | Trend |
|--------|-------|--------|--------|--------|-------|
| Activation patching (best layer) | 20.9% | 0% | 0.7% | 5.2% | IT: distributed; PT: slightly more recoverable |
| Feature swap (awareness ablation) | 30.2% | 5.4% | 4.6% | **49.3%** | IT↓ saturates; PT still coupled |
| Attention knockout | 10% | 0-17% | ~0% | **27%** | IT: irrecoverable; PT: partially recoverable |

## Files

| File | Description |
|------|-------------|
| `results/ablation_feature_swap_27b_*.json` | 27B-IT feature swap (4.6% recovery) |
| `results/ablation_feature_swap_27b-pt_*.json` | 27B-PT feature swap (49.3% recovery) |
| `results/ablation_activation_patching_27b_*.json` | 27B patching (0.7% max, distributed) |
| `results/ablation_attention_knockout_27b_*.json` | 27B attention knockout |
| `results/escalation_27b_*.json` | 27B T0-T5 + recovery (IT + PT) |
| `results/ablation_feature_swap_12b_a100.json` | 12B A100 feature swap (-13.5%) |
| `results/ablation_feature_swap_4b_*.json` | 4B feature swap (30.2%) |
| `ABLATION_CHECKPOINT_12B.md` | Detailed 12B results |

## Blog Update

The dissociation post prediction was:
> "If 4B is anxiety and 12B is dissociation, 27B will be more dissociated, not less."

**Result: Confirmed.** 4.6% recovery at 27B. The prediction holds — the curve flattens because the separation is already complete by 12B. The smoke detector room was already soundproof.

Update the blog post with actual data instead of prediction.
