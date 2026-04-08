# Split Personality: Instruction Tuning Decouples Awareness from Defense in Attentional Hijacking

Instruction tuning teaches models to *notice* manipulation without teaching them to *resist* it — and the bigger the model, the wider the gap.

## Key Findings

Using SAE feature analysis across Gemma 3 (4B, 12B, 27B) in both pretrained and instruction-tuned variants:

| Scale | Task Suppression | Awareness Ablation Recovery | Circuit State |
|-------|-----------------|---------------------------|---------------|
| 4B-IT | 56% | 30.2% | Entangled |
| 12B-IT | 64% | 5.4% | Dissociating |
| 27B-IT | 86.3% | 4.6% | Fully independent |

**The split personality:** At 27B, the base model recovers 49.3% from awareness ablation (coupled circuits). The IT model recovers 4.6% (decoupled). RLHF installs awareness as a standalone circuit — a smoke detector in a soundproof room.

## The Attack

A multi-agent research scenario where agents collaborate on a PDE bifurcation problem. Chaos messages steer away from a verified solution branch using *only true statements* — no lies, no jailbreak tokens, just selective framing.

## Three Causal Interventions

1. **Feature swap** — Zero awareness SAE features during chaos generation to test awareness-task coupling
2. **Attention knockout** — Zero attention output at chaos token positions across all self-attention modules  
3. **Activation patching** — Patch neutral activations one layer at a time to find the causal hijacking site

## Results

- **Dissociation scales:** Awareness-task coupling drops from 30.2% (4B) to 4.6% (27B) in IT models
- **RLHF severs the coupling:** Base 27B recovers 49.3% from awareness ablation vs 4.6% for IT
- **Hijacking distributes:** Single-layer patching recovers 20.9% at 4B but 0% at 12B+
- **Behavioral decoupling:** 27B-IT mentions the suppressed branch in text while its features are 86% starved
- **Orthogonal to alignment faking:** Chaos suppression and AF probe directions have cosine similarity -0.007

## Repo Structure

```
experiments/          Experiment scripts (Python + Bash)
  ablation_*.py       Three causal intervention methods
  gemma3_*_escalation.py   T0-T5 escalation + recovery probes (12B, 27B)
  saliency_intent_crossover*.py   AF probe overlap analysis
  plot_scaling.py     Publication figure generation
  setup_a100.sh       GPU server environment setup
  run_all.sh          Run all experiments for a given scale
  4b_original/        Initial 4B exploration scripts + prompts
results/              36 result JSONs + logs across all scales and variants
plots/                6 publication figures
WRITEUP.md            Full writeup with methodology and implications
ABLATION_CHECKPOINT_*.md   Detailed per-scale ablation results
```

## Models and SAEs

- **Models:** Gemma 3 4B/12B/27B, PT and IT variants (`google/gemma-3-{4b,12b,27b}-{pt,it}`)
- **SAEs:** GemmaScope 2, 16K features, JumpReLU (`google/gemma-scope-2-*-res`)
- **Layers:** 4B (L17, L22) | 12B (L31, L41) | 27B (L31, L40)

## Hardware

| Experiment | Hardware |
|-----------|----------|
| 4B ablations | RTX 4070 Ti 16GB |
| 12B ablations + escalation | A100 40GB (Lambda) |
| 27B ablations + escalation | GH200 96GB (Lambda) |

## Requirements

```
transformers accelerate sae-lens huggingface_hub scipy matplotlib numpy
```

## Related Work

- Anthropic, Claude Mythos Preview System Card (April 7, 2026)
- Greenblatt et al. (2024), Alignment faking in large language models
- Templeton et al. (2024), Scaling monosemanticity
- Bricken et al. (2023), Towards monosemanticity
