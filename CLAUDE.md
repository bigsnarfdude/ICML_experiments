# ICML 2026 Paper: Split Personality (Attentional Hijacking)

## What This Is

ICML 2026 paper showing instruction tuning decouples awareness from defense against attentional hijacking in multi-agent LLM systems. Uses SAE features from GemmaScope 2 across Gemma 3 (4B, 12B, 27B) and Llama 3.1 8B.

**Paper:** `paper/main.tex` (LaTeX, `icml2026` style)
**Author:** Vincent Ohprecio (Independent Researcher)

## Repo Structure

```
paper/              LaTeX source + figures (main.tex, main.pdf)
experiments/        Python experiment scripts (ablation, escalation, saliency)
h100_deploy/        GPU experiment scripts for reviewer-requested experiments
results/            JSON results + logs from all runs
plots/              Publication figures
protocol/           (empty)
```

## Key Commands

```bash
# Build paper
cd paper && pdflatex main.tex && pdflatex main.tex

# Run experiments (requires GPU + HuggingFace models)
cd experiments && bash run_all.sh

# H100 experiments (reviewer revisions)
cd h100_deploy && python cross_domain_sae.py --device cuda
```

## Dependencies

```
transformers accelerate sae-lens huggingface_hub scipy matplotlib numpy torch safetensors
```

## Models Used

- **Gemma 3:** `google/gemma-3-{4b,12b,27b}-{pt,it}`
- **SAEs:** `google/gemma-scope-2-{4b,12b,27b}-res-16k` (GemmaScope 2, 16K features, JumpReLU)
- **Llama replication:** Llama 3.1 8B with SAE-lens SAEs
- **Key layers:** 4B (L17, L22) | 12B (L31, L41) | 27B (L31, L40)

## Experiment Commit Protocol

After EACH experiment completes, commit immediately:
```bash
git add results/ *.log
git commit -m "Experiment: <script_name> <model_size> results"
git push origin main
```
Raw data on GitHub before analysis. Do NOT batch commits.

## Paper Key Claims

1. **Dissociation scaling:** Awareness-task coupling drops 30.2% → 4.6% with model scale (IT models)
2. **RLHF severs coupling:** Base 27B recovers 49.3% vs 4.6% for IT
3. **Groot effect:** 27B-IT shows 86.3% feature suppression while verbally mentioning suppressed info
4. **Orthogonal to AF:** Cosine similarity -0.048, zero top-50 feature overlap
5. **Cross-family:** Llama 3.1 8B replicates IT-vs-base pattern (d=1.51 vs 0.50)

## Reviewer Revision Status

H100 experiments address reviewer concerns:
- Cross-domain replication (beyond math/Nirenberg BVP)
- Held-out feature validation (circular selection check)
- Statistical rigor (CIs, multiple trials, effect sizes)
- Llama 3.1 8B SAE-level replication
- Behavioral validation (feature suppression → output degradation)
