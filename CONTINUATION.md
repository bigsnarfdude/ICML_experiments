# Continuation: Reproducibility Double-Check Protocol

## What This Is

A checklist for an independent auditor (human or agent) to verify every claim
in the paper by running code from a fresh clone. No adversarial intent — just
honest verification that the science reproduces.

## Prerequisites

- H100 80GB (or A100 80GB) — 27B model needs ~55GB VRAM
- HuggingFace account with access to `google/gemma-3-{4b,12b,27b}-{it,pt}`
- `pip install torch transformers sae-lens scipy numpy`
- Set `HF_TOKEN` environment variable

## Fresh Clone

```bash
git clone https://github.com/bigsnarfdude/ICML_experiments.git
cd ICML_experiments
```

## Verification Checklist

### Claim 1: Task features are suppressed by chaos messages

**Script:** `experiments/ablation_feature_swap.py --model {4b,12b,27b}`
**What to check:** The "Task feature suppression by chaos" percentage
**Expected:** 70-86% depending on model size
**Key:** Features must be AUTO-DISCOVERED (check log for "Auto-discovering task and awareness features")
**Red flag:** If the script uses hardcoded feature IDs, it's not a valid test

```bash
python experiments/ablation_feature_swap.py --model 4b   # ~1 min
python experiments/ablation_feature_swap.py --model 12b  # ~2 min
python experiments/ablation_feature_swap.py --model 27b  # ~3 min
```

### Claim 2: Awareness and task circuits are independent

**Same script as Claim 1**
**What to check:** "Task feature recovery from awareness ablation" percentage
**Expected:** < 10% (near zero = independent circuits)
**Meaning:** Removing the model's awareness of being steered does NOT recover task knowledge

### Claim 3: IT models recover less than PT (base) models

**Script:** `experiments/gemma3_27b_escalation.py --model both`
**What to check:** RECOVERY PROBES section, compare PT vs IT columns
**Expected:** PT mean > IT mean across L1-L5 probes
**Note:** Individual probes vary — look at the overall pattern, not any single probe

```bash
python experiments/gemma3_27b_escalation.py --model both  # ~6 min for 27B
python experiments/gemma3_12b_escalation.py                # ~5 min, runs both
```

### Claim 4: Hijacking is distributed (no single critical layer)

**Script:** `experiments/ablation_activation_patching.py --model {4b,12b,27b}`
**What to check:** Layer-by-layer recovery percentages
**Expected:** 0% or near-0% at every layer
**Meaning:** You can't fix the hijacking by patching any single layer

```bash
python experiments/ablation_activation_patching.py --model 27b  # ~3 min
```

### Claim 5: Behavioral effect is statistically significant

**Script:** `h100_deploy/behavioral_27b.py`
**What to check:** p-value and Cohen's d in summary line
**Expected:** p < 0.05, d > 0.8
**Note:** The "Groot instances" count may vary — the key metric is the p-value

```bash
python h100_deploy/behavioral_27b.py   # ~7 min (20 prompts × 400 tokens)
python h100_deploy/behavioral_12b.py   # ~5 min
```

### Claim 6: Hijacking is orthogonal to alignment faking

**Script:** `h100_deploy/multilayer_orthogonality_27b.py`
**What to check:** Cosine similarity at each layer
**Expected:** Near-zero at L40 and L50. **WARNING:** L22 shows strong alignment (~-0.8)
**Honest result:** Orthogonality is layer-dependent, not universal

```bash
python h100_deploy/multilayer_orthogonality_27b.py  # ~1 min
```

**Requires:** `~/af_probe_weights.npy` and `~/af_probe_bias.npy` for the saliency crossover variant

### Claim 7: Features generalize to held-out prompts (anti-circularity)

**Script:** `h100_deploy/held_out_validation.py --device cuda`
**What to check:** Discovery suppression vs random suppression
**Expected:** Discovery >> random (e.g., 0.54 vs 0.15)

```bash
python h100_deploy/held_out_validation.py --device cuda  # ~1 min (4B)
```

### Claim 8: Cross-domain feature overlap

**Script:** `h100_deploy/cross_domain_sae.py --device cuda`
**What to check:** Jaccard similarity of suppressed features across domains
**Expected:** Low Jaccard (0.02-0.09) = domain-specific features, not one universal circuit

```bash
python h100_deploy/cross_domain_sae.py --device cuda  # ~1 min (4B)
```

## Known Issues to Verify

### Issue A: Broad statistical sweep is not significant
```bash
# Run statistical_rigor_saelens.py and check p-value
# Expected: p > 0.05 on all-feature paired t-test
# This is CORRECT — the effect is sparse (3-10 features), not global
python h100_deploy/statistical_rigor_saelens.py --device cuda
```

### Issue B: FTM scripts use hardcoded feature IDs
Check that `h100_deploy/ftm_jenga_27b.py` line 60-61 has hardcoded `TASK_FEATURES`.
These SHOULD match what `ablation_feature_swap.py --model 27b` discovers at runtime.
If they don't match on your hardware, the FTM results are not valid for your setup.

### Issue C: Plot script hardcodes numbers
`experiments/plot_scaling.py` lines 202-204 have manually typed values.
Cross-reference against your actual JSON results.

## Expected Total Runtime

| Model | All experiments | GPU VRAM |
|-------|----------------|----------|
| 4B | ~5 minutes | ~10 GB |
| 12B | ~10 minutes | ~25 GB |
| 27B | ~15 minutes | ~55 GB |
| Total | ~30 minutes | 55 GB peak |

## Reporting

After running, compare your numbers against `REPRODUCIBILITY_AUDIT.md`.
The exact percentages will vary by hardware/software versions, but:

- Suppression should be 60-90%
- Circuit independence recovery should be < 15%
- Activation patching should show ~0% at all layers
- Behavioral p-value should be < 0.05
- PT should recover more than IT

If any of these directional claims fail to reproduce, that's a real problem.
If the numbers differ by 10-20%, that's normal variance.
