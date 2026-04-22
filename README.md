# Split Personality: Instruction Tuning Decouples Awareness from Defense Against Attentional Hijacking

Post-training introduces LLMs to *notice* when they are being manipulated without teaching them to *resist* it. The bigger the model, the wider the gap. We call this *attentional hijacking*, and it is installed almost entirely by supervised fine-tuning — not by RLHF as the community folklore assumes.

**Paper:** [`paper/main.pdf`](paper/main.pdf) (ICML 2026 format, 14 pages)
**Anonymized twin:** [`paper/main_submission.pdf`](paper/main_submission.pdf)
**Author:** Vincent Ohprecio (Independent Researcher)

---

## Headline results

### The dissociation

Across Gemma 3 (4B / 12B / 27B) and Llama 3.1 8B, instruction-tuned models exhibit a *behavioral–representational split*: the model verbally flags manipulation while its task features are already suppressed. At 27B-IT, Layer 40 feature 423 drops 86.3% (682.7 → 93.3) under chaos framing while awareness features $\{2119, 139, 9169\}$ simultaneously boost 2.6×.

| Scale | IT awareness–defense recovery | Base recovery | Implication |
|-------|-------------------------------|---------------|-------------|
| 4B    | 30.2% | — | Entangled |
| 12B   | 9.0% | — | Dissociating |
| 27B   | 9.0% | 74.1% | Fully decoupled (IT only) |

### SFT installs the dissociation, not RLHF

Tulu 3 pipeline on Llama 3.1 8B (`h100_deploy/tulu3_stage_attribution.py`, $n=30$ per cell):

| Stage | Cohen's $d$ | $\Delta d$ |
|-------|-------------|------------|
| Base (Llama-3.1-8B)     | 0.19 | — |
| + SFT                   | 1.16 | **+0.97** |
| + DPO                   | 1.35 | +0.19 |
| + RLVR (Tulu-3-8B final) | 1.04 | −0.31 |

~80% of the full effect is installed by plain SFT. This reverses the paper's own earlier speculation that RLHF reward-shaping is the primary driver.

### Feature trajectory monitoring (Jenga)

Single-prompt FTM is insufficient ($d = 0.27$). The realistic threat model is cumulative in-context learning across a multi-turn session. Running 5-turn sessions at 27B-IT L40 with the auto-discovered features $\{423, 7657, 632\}$ (`h100_deploy/ftm_jenga_27b.py`):

- **Attack** (10 sessions, 4 neutral + 1 chaos): mean $z_{\text{task}} = -2.33$
- **Control** (10 sessions, 5 neutral): mean $z_{\text{task}} = -0.02$
- **Benign** (50 sessions, lmsys-chat-1m): off-domain baseline
- **Within-session tilt:** $|d| = 1.74$, $p = 0.0023$
- At $\tau = -1.65$: Attack TPR = 0.80, lmsys FPR = 0.14, control FPR = 0.10

We position this as a **mechanism signature**, not a deployable classifier.

### Behavioral dose–response

Gemma 3 27B-IT under chaos framing shows monotonically non-monotonic degradation — most vulnerable to *mild* framing:

| Dose | 27B-IT $d$ | 27B-Base $d$ | Ratio |
|------|------------|--------------|-------|
| Mild     | **1.99** | 0.96 | 2.1× |
| Moderate | 0.96 | 1.31 | — |
| Strong   | 0.89 | 1.03 | — |

IT models are more susceptible to *subtle* framings — the regime where true-statement hijacking is most dangerous.

### Orthogonality to alignment faking

At 27B-IT Layer 40, the hijacking and alignment-faking feature subspaces are effectively orthogonal: cosine similarity $-0.0456$, top-50 feature overlap 0 ($p = 1.0$). Probes trained for AF detection provide **no** protection against hijacking.

---

## Repo layout

```
paper/
  main.tex, main.pdf                   Full ICML 2026 manuscript
  main_submission.tex, .pdf            Anonymized twin for blind review
  figures/                             Publication figures (7 PDFs)
  review/
    generate.py                        Builds paper_data_review.html
    paper_data_review.html             Per-table audit report (claim vs recomputed)
  gen_jenga_roc.py                     Generates figures/jenga_roc.pdf
experiments/                           Original 4B/12B/27B ablation scripts
h100_deploy/                           Reviewer-revision experiments (H100)
  ftm_jenga_27b.py                     W1: within-session Jenga FTM at 27B
  tulu3_stage_attribution.py           W2: SFT/DPO/RLVR stage attribution
  behavioral_27b_n30_dose.py           27B dose-response (audit C6)
  ablation_feature_swap.py             Top-K auto-discovery + feature swap
  cross_domain_sae.py                  Cross-domain replication
  held_out_validation.py               Held-out feature robustness
  ...                                  (QA/theorem/behavioral scripts)
results/
  h100/                                Raw JSONs from all H100 runs (truth source)
  *.json                               Earlier scale runs
plots/                                 Scaling + dissociation figures
WRITEUP.md                             Longer-form writeup
ABLATION_CHECKPOINT_*.md               Per-scale ablation history
AUDIT_REPORT.md                        Data provenance audit
```

Every number in `paper/main.tex` is traceable to a specific JSON in `results/` — the audit report walks through the mapping.

## Models and SAEs

- **Gemma 3:** `google/gemma-3-{4b,12b,27b}-{pt,it}`
- **GemmaScope 2:** `google/gemma-scope-2-{4b,12b,27b}-res-16k`, JumpReLU, 16K features
- **Llama 3.1 8B + Tulu 3:** `meta-llama/Llama-3.1-8B`, `allenai/Llama-3.1-Tulu-3-8B-{SFT,DPO,}`
- **Llama SAEs (cross-family):** EleutherAI 131K-feature SAEs
- **Key layers:** 4B (L17, L22) · 12B (L31, L41) · 27B (L31, L40)

## Hardware

| Experiment | Hardware |
|------------|----------|
| 4B ablations | RTX 4070 Ti 16GB |
| 12B ablations + escalation | A100 40GB (Lambda) |
| 27B + Tulu 3 + Jenga | H100 80GB / GH200 96GB (Lambda) |

## Requirements

```
transformers accelerate sae-lens huggingface_hub scipy matplotlib numpy torch safetensors
```

## Reproducing key results

```bash
# Tulu 3 stage attribution (~2h on H100)
python h100_deploy/tulu3_stage_attribution.py

# Jenga FTM at 27B (~45min on H100)
python h100_deploy/ftm_jenga_27b.py

# Regenerate Jenga ROC figure from the result JSON
python paper/gen_jenga_roc.py

# Rebuild the paper
cd paper && pdflatex main.tex && pdflatex main.tex
```

## Related work

- Lambert et al. (2024), *Tulu 3: Pushing frontiers in open language model post-training*
- Greenblatt et al. (2024), *Alignment faking in large language models*
- Lieberum et al. (2024), *GemmaScope: Open SAEs everywhere all at once*
- Sadanandan & Behzadan (2026), *PSF-Med* (independent SAE causal-feature validation on MedGemma)
- Templeton et al. (2024), *Scaling monosemanticity*
- Bricken et al. (2023), *Towards monosemanticity*

## Citation

```bibtex
@inproceedings{ohprecio2026splitpersonality,
  title={Split Personality: Instruction Tuning Decouples Awareness from Defense Against Attentional Hijacking},
  author={Ohprecio, Vincent},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

GitHub: [bigsnarfdude/ICML_experiments](https://github.com/bigsnarfdude/ICML_experiments)
