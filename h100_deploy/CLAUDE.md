# ICML Paper Experiment Runner — H100

## What This Is

You are babysitting GPU experiments for an ICML paper about "attentional hijacking" in LLMs. The paper shows that instruction tuning decouples awareness from defense against manipulation via true statements. Reviewers want:

1. **Cross-domain replication** — does the effect work beyond math (Nirenberg BVP)?
2. **Held-out feature validation** — are the features selected circularly?
3. **Statistical rigor** — confidence intervals, multiple trials, effect sizes

## Setup

```bash
# Install dependencies
pip install torch transformers safetensors numpy scipy

# The model will auto-download from HuggingFace:
# google/gemma-3-4b-it (~8GB)
# For 12B/27B runs, also need:
# google/gemma-3-12b-it (~24GB)  
# google/gemma-3-27b-it (~54GB)

# SAE weights: google/gemma-scope-2-4b-res-16k
# If SAE download fails, use --sae-path to point to local weights
```

## Run Order (Priority)

### P0 — Run these first (all on 4B, ~2h total)

```bash
# 1. Cross-domain replication (~45 min)
python cross_domain_sae.py --device cuda 2>&1 | tee cross_domain.log

# 2. Held-out feature validation (~30 min)
python held_out_validation.py --device cuda 2>&1 | tee held_out.log

# 3. Statistical rigor with CIs (~45 min)
python statistical_rigor.py --device cuda 2>&1 | tee stats_rigor.log
```

### P1 — If P0 succeeds, scale up (uses H100's 80GB)

```bash
# 4. Cross-domain at 12B (~2h)
python cross_domain_sae.py --model google/gemma-3-12b-it --sae-model google/gemma-scope-2-12b-res-16k --layers 31 41 --device cuda 2>&1 | tee cross_domain_12b.log

# 5. Cross-domain at 27B (~3h)  
python cross_domain_sae.py --model google/gemma-3-27b-it --sae-model google/gemma-scope-2-27b-res-16k --layers 31 40 --device cuda 2>&1 | tee cross_domain_27b.log
```

### P2 — Stretch goals

```bash
# 6. Held-out validation at 27B
python held_out_validation.py --model google/gemma-3-27b-it --sae-model google/gemma-scope-2-27b-res-16k --layers 31 40 --device cuda 2>&1 | tee held_out_27b.log
```

## What to Watch For

- **OOM**: If a script OOMs, it's likely the SAE + model don't fit. Try adding `--batch-size 1` or reducing prompt length.
- **SAE download failures**: GemmaScope 2 SAEs may need HF token. Set `HF_TOKEN` env var if auth errors appear.
- **Feature IDs**: The key 4B task features are 1716, 12023, 1704, 1555, 1548 at Layer 22. If cross-domain shows these SAME features suppressing in QA/code domains, that's a major finding. If DIFFERENT features suppress, that's still good (domain-specific circuits).

## Results

All scripts output JSON to `results/` directory. Key metrics to report back:

1. **Cross-domain**: Jaccard similarity of suppressed feature sets across domains. >0.1 = shared mechanism, <0.05 = domain-specific
2. **Held-out**: Test-set suppression ratio for discovery-selected features vs random. Should be significantly different (p < 0.05)
3. **Stats rigor**: 95% CI for task suppression %. Currently reported as point estimate 56%.

## Commit Protocol

After EACH experiment completes:
```bash
git add results/ *.log
git commit -m "H100 experiment: <script_name> <model_size> results"
git push origin main
```

Do NOT wait to batch commits. Raw data on GitHub before analysis.

## Contact

Results go back to the ICML repo at /Users/vincent/ICML/. SCP them back or push via git.
