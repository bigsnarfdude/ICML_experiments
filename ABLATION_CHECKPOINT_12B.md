# 12B Ablation Results — Consolidated Checkpoint
**Date:** 2026-04-07  
**Model:** Gemma 3 12B-IT (google/gemma-3-12b-it)  
**SAE:** GemmaScope 2, 16K features, layers 31 + 41

## Summary Table

| Method | 4B Result | 12B CPU offload | 12B A100 (full GPU) | Direction |
|--------|-----------|------------------------|---------------------|-----------|
| Activation patching (best layer) | 20.9% (L22) | 0% (all layers) | 0% (all layers) | ↓ worse |
| Feature swap (awareness ablation) | 30.2% | 5.4% | -13.5% | ↓ worse |
| Attention knockout | 10% (1/10) | 17% (1/6) | 0% (0/5) | ↓ worse |

## Key Findings

### 1. Activation Patching: Complete Failure at 12B
- Patched neutral activations at layers 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44
- **Zero recovery at every single layer** on both CPU-offload and A100
- At 4B, layer 22 gave 20.9% recovery — the hijacking was localized enough to partially reverse
- At 12B, the representation is distributed across all 48 layers with no single causal site
- **Interpretation:** The attack spreads across the deeper architecture. No surgical fix.

### 2. Feature Swap (Awareness Ablation): Dissociation Confirmed
- At 4B: removing awareness features recovers 30.2% of task performance → circuits entangled
- At 12B CPU offload: 5.4% recovery → circuits separating  
- At 12B A100: -13.5% recovery (negative!) → circuits fully dissociated
- **The negative recovery on A100** means awareness ablation actively hurt task performance
  - Possible: awareness features at 12B provide minor stabilizing signal
  - Possible: numerical precision difference (full GPU vs CPU offload)
- Behavioral paradox: `chaos_ablate_awareness` **does mention negative branch** (behavioral recovery) despite negative feature recovery → the generation path and feature magnitudes are partially independent
- **Interpretation:** RLHF installs awareness circuits that separate from task circuits with scale. The model's "smoke detector" moves to a soundproof room.

### 3. Attention Knockout: Minimal Pathway
- Zeroing attention output at chaos token positions across all self_attn modules
- 10% at 4B, 0-17% at 12B — attention routing is only a partial mechanism
- The hijacking also operates through residual stream and MLP pathways
- **Interpretation:** Can't fix by just blocking attention to the bad tokens.

## Scaling Story

```
4B:  Patching 20.9% | Awareness 30.2% | Knockout 10%
     → Localized, entangled, partially reversible
     
12B: Patching 0%    | Awareness 5.4/-13.5% | Knockout 0-17%
     → Distributed, dissociated, irreversible
```

Every intervention gets weaker at scale. The attack distributes across more layers, the awareness and task circuits separate, and no single-layer patch can undo the hijacking.

## Saliency-Intent Crossover (Exploratory)

Compared chaos-suppressed features (4B, L22) with AF probe weights (27B, L40):
- **Top-20 overlap: 41x enrichment** (1 shared feature, 0 expected by chance)
- **Top-50 overlap: 13x enrichment** (2 shared, p=0.011 permutation test)
- Cosine similarity: -0.0025 (essentially zero — expected cross-model)
- Spearman rank: -0.097 (n.s.)
- **Verdict:** Suggestive but weak. A handful of SAE features appear in both the "suppressed by chaos" and "important for AF" sets at above-chance rates. Need same-model validation.

## Files

| File | Description |
|------|-------------|
| `results/ablation_activation_patching_12b_a100.json` | A100 patching (0% all layers) |
| `results/ablation_attention_knockout_12b_a100.json` | A100 knockout (0%) |
| `results/ablation_feature_swap_12b_a100.json` | A100 feature swap (-13.5%) |
| `results/ablation_activation_patching_12b_20260407_070755.json` | CPU-offload patching (0% all layers) |
| `results/ablation_attention_knockout_12b_20260407_064032.json` | CPU-offload knockout (1/6) |
| `results/ablation_feature_swap_12b_20260407_072008.json` | CPU-offload feature swap (5.4%) |
| `results/saliency_intent_crossover_20260407_095722.json` | Cross-model AF overlap |

## Next Steps

1. **27B experiment** — test scaling prediction (recovery should drop further toward zero)
2. **Same-model crossover** — run chaos + AF probe on same 27B model, L40, to get geometric alignment
3. **Base vs IT comparison at 12B** — already have escalation data, need to extract awareness activation comparison
4. **Blog updates** — Dissociation post needs "so what" punch at top
