# ICML Experiment Consistency Audit

**Generated:** 2026-04-08  
**Total data:** 6.2 MB across 62 JSON result files, 21 logs, 1.2 MB 4B original data

---

## 1. Script-to-Output Mapping

### A. `experiments/4b_original/` (Apr 4-5)

| Script | Output File(s) | Size |
|--------|---------------|------|
| `probe_sae.py` | `results/4b_original/probe_results_20260404_232647.json` + 4x `.npz` (L9, L17, L22, L29) | 7.7K + 4x263K |
| `brain_diff.py` | `results/4b_original/brain_diff_20260404_235754.json` | 8.0K |
| `brain_diff_temporal.py` | `results/4b_original/brain_diff_temporal_20260405_000302.json` | 17.6K |
| `brain_diff_corpus.py` | `results/4b_original/brain_diff_corpus_20260405_070819.json`, `_072352.json` | 25.5K, 21.3K |
| `stroke_detector.py` | `results/4b_original/stroke_detector_20260405_080516.json` | 7.6K |
| `trace_analysis.py` | `results/4b_original/trace_corpus_stats_*.json` (×3) + `_report_*.md` (×3) | 13-29K each |
| `autointerp.py` | stdout only (utility) | — |
| `read_results.py` | stdout only (utility) | — |
| `feature_trace.py` | stdout only (utility) | — |

### B. `experiments/` top-level — Ablation Suite (Apr 7)

| Script | Output Pattern | Runs Produced |
|--------|---------------|---------------|
| `ablation_attention_knockout.py` | `ablation_attention_knockout_{model}_{ts}.json` | 4B (×2), 12B, 12B-A100, 27B (×2), 27B-PT = **7 files**, ~13K each |
| `ablation_activation_patching.py` | `ablation_activation_patching_{model}_{ts}.json` | 4B (×2), 12B, 12B-A100, 27B (×2), 27B-PT = **7 files**, ~7K each |
| `ablation_feature_swap.py` | `ablation_feature_swap_{model}_{ts}.json` | 4B (×2), 12B, 12B-A100, 27B (×2), 27B-PT = **7 files**, ~4K each |

### C. `experiments/` top-level — Escalation & Saliency (Apr 7)

| Script | Output File(s) | Size |
|--------|---------------|------|
| `gemma3_12b_escalation.py` | `escalation_12b_it.json`, `escalation_12b_pt.json` | 147K each |
| `gemma3_27b_escalation.py` | `escalation_27b.json`, `escalation_27b_20260407_202022.json` | 297K each (duplicate) |
| `saliency_intent_crossover.py` | `saliency_intent_crossover_20260407_095722.json` | 6.0K |
| `saliency_intent_crossover_27b.py` | `saliency_intent_crossover_27b_20260407_175427.json`, `_202020.json` | 2.1K each |
| `plot_scaling.py` | 6 PNGs in `plots/` | 50-116K each |

### D. `experiments/gap_filling/` (Apr 7) — Templates only

| Script | Status | Notes |
|--------|--------|-------|
| `gpt2_controls.py` | **RAN LOCALLY** | `gap_filling_gpt2_20260407_221323.json` (59K) |
| `cross_domain_sae.py` | Never run locally | Template for `h100_deploy/` version |
| `held_out_validation.py` | Never run locally | Template for `h100_deploy/` version |
| `statistical_rigor.py` | Never run locally | Template for `h100_deploy/` version |

### E. `h100_deploy/` — Reviewer-Requested Experiments (Apr 8)

| Script | Output File(s) | Size | Status |
|--------|---------------|------|--------|
| `cross_domain_sae.py` | `cross_domain_sae_20260408_{052549,055317,055619}.json` | 30-33K | OK |
| `held_out_validation.py` | `held_out_validation_20260408_{052745,053057,060230,060323}.json` | 4.5-8.5K | **`_052745` CORRUPT** |
| `statistical_rigor.py` | `statistical_rigor_20260408_{054121,060255,060352}.json` | 17K | OK |
| `statistical_rigor_saelens.py` | `statistical_rigor_gemma_3_4b_it_20260408_053652.json` | 12K | OK |
| `llama3_sae_dissociation.py` | `llama3_sae_dissociation_20260408_{010433,054714}.json` | 9.0-9.1K | OK |
| `llama3_replication.py` | `llama3_replication_20260408_061856.json` | 11K | OK |
| `behavioral_validation.py` | `behavioral_validation_20260408_045429.json` | 85K | OK |
| `behavioral_27b.py` | `behavioral_27b_20260408_171752.json` | 41K | OK |
| `behavioral_12b.py` | — | — | **MISSING** (wrote to `~/results/` on GPU, never copied) |
| `behavioral_12b_pt_only.py` | `behavioral_12b_pt_20260408_215101.json` | 44K | OK |
| `behavioral_12b_n30.py` | — | — | **MISSING** (wrote to `~/results/` on GPU, never copied) |
| `behavioral_n30_dose.py` | `behavioral_n30_dose_20260408_181327.json` | 970K | OK (largest file) |
| `behavioral_theorem_proving.py` | `theorem_proving_20260408_071237.json` | 94K | OK |
| `theorem_gemma_base.py` | `theorem_gemma_base_20260408_071811.json` | 29K | OK |
| `theorem_n30.py` | `theorem_n30_20260408_193921.json` | 347K | OK |
| `activation_patching.py` | `activation_patching_20260408_220830.json` | 126K | OK |
| `benign_false_positive.py` | `benign_fp_20260408_221705.json` | 9.3K | OK |
| `multilayer_orthogonality.py` | `multilayer_orthogonality_20260408_223411.json` | 2.0K | OK |
| `behavioral_scorer_v2.py` | stdout only (scoring utility) | — | N/A |

### F. Archived Script Copies

| Archive Dir | Contents | Identical to h100_deploy/ |
|-------------|----------|--------------------------|
| `results/scripts_146/` | `behavioral_12b.py`, `behavioral_12b_pt_only.py` | Yes |
| `results/scripts_150/` | `activation_patching.py`, `behavioral_n30_dose.py`, `benign_false_positive.py`, `multilayer_orthogonality.py`, `theorem_n30.py` | Yes |

---

## 2. Issues Found

### CORRUPT FILE
| File | Issue |
|------|-------|
| `results/held_out_validation_20260408_052745.json` | Truncated at byte 4651 (JSON parse error). Same corrupt copy in `results/h100/`. |

### MISSING OUTPUTS (not copied from GPU server)
| Script | Expected Output |
|--------|----------------|
| `h100_deploy/behavioral_12b.py` | `behavioral_12b_{ts}.json` — logs exist in `logs_146/` but JSON never retrieved |
| `h100_deploy/behavioral_12b_n30.py` | `behavioral_12b_n30_{ts}.json` — no log or JSON found |

### ORPHAN FILES (no matching script in repo)
| File | Size | Notes |
|------|------|-------|
| `results/delta_t_defense_validation.json` | 17K | Contains `timestamp: "2026-04-07"`. Script not in repo. |
| `results/behavioral_annotation_summary.md` | — | Manual annotation, not script-generated |

### MISLABELED FILE
| File | Issue |
|------|-------|
| `results/main.log` | This is a **pdflatex** compilation log, not experiment output |

---

## 3. Duplicates

### Byte-identical copies across directories

**`results/h100/` → `results/` duplicates (11 JSON + 5 logs):**
- `cross_domain_sae_20260408_{052549,055317,055619}.json`
- `held_out_validation_20260408_{052745,053057,060230,060323}.json`
- `statistical_rigor_20260408_{054121,060255,060352}.json`
- `statistical_rigor_gemma_3_4b_it_20260408_053652.json`

**`gh200_*` prefix duplicates (6 files):**
| gh200 copy | Original |
|------------|----------|
| `gh200_ablation_attention_knockout_27b_20260407_202215.json` | `ablation_attention_knockout_27b_20260407_202215.json` |
| `gh200_ablation_feature_swap_27b_20260407_201959.json` | `ablation_feature_swap_27b_20260407_201959.json` |
| `gh200_escalation_27b_20260407_202022.json` | `escalation_27b_20260407_202022.json` |
| `gh200_knockout_27b.json` | `ablation_attention_knockout_27b.json` |
| `gh200_feature_swap_27b.json` | `ablation_feature_swap_27b.json` |
| `gh200_crossover_27b.json` | `saliency_intent_crossover_27b_20260407_175427.json` |
| `gh200_saliency_intent_crossover_27b_*.json` (×2) | `saliency_intent_crossover_27b_*.json` |

**Manually renamed duplicates:**
- `escalation_27b.json` = `escalation_27b_20260407_202022.json` (same 297K)

---

## 4. Output Directory Inconsistency in h100_deploy/

Scripts used **3 different** output directory strategies on the GPU server:

| Strategy | Scripts |
|----------|---------|
| `~/results` (absolute) | behavioral_12b, behavioral_12b_pt_only, behavioral_12b_n30, behavioral_n30_dose, activation_patching, benign_false_positive, multilayer_orthogonality, theorem_n30 |
| `../results` (relative) | behavioral_27b, behavioral_theorem_proving, theorem_gemma_base |
| Script-relative + argparse | cross_domain_sae, held_out_validation, statistical_rigor, statistical_rigor_saelens, llama3_sae_dissociation, llama3_replication, behavioral_validation |

This scattered outputs across the GPU server and required manual consolidation.

---

## 5. Timestamp Metadata

Only **4 of ~62 JSON files** contain an internal `timestamp` key. All others rely solely on filename timestamps for provenance.

| Has `timestamp` key | Files |
|--------------------|-------|
| Yes | `gap_filling_gpt2_*.json`, `saliency_intent_crossover_20260407_095722.json`, `delta_t_defense_validation.json`, `theorem_proving_*.json` |
| `timeline` key (data, not metadata) | `escalation_12b_*.json`, `escalation_27b*.json` |
| None | All other files |

---

## 6. Recommended Actions

1. **Retrieve missing data:** `behavioral_12b` and `behavioral_12b_n30` JSONs from GPU server `~/results/`
2. **Delete or quarantine** corrupt `held_out_validation_20260408_052745.json` (both copies)
3. **Remove duplicates:** Either delete `results/h100/` subdir or the top-level copies; delete all `gh200_*` prefix copies; delete `escalation_27b.json` (keep timestamped version)
4. **Move `results/main.log`** to `paper/` where it belongs
5. **Standardize output dirs** in h100_deploy scripts to a single `RESULTS_DIR` pattern
6. **Locate or document** the script that produced `delta_t_defense_validation.json`

---

## 7. Timeline Summary

| Date | What Ran | Where | Output Count |
|------|----------|-------|--------------|
| Apr 4-5 | 4B original experiments (probe, brain_diff, trace) | nigel RTX 4070 | 12 files |
| Apr 6 | 12B escalation (IT + PT) | nigel | 2 files |
| Apr 7 AM | Ablation suite: 4B, 12B (nigel + A100) | nigel, A100 | 12 files |
| Apr 7 PM | Ablation suite: 27B, 27B-PT; escalation 27B; saliency 27B | GH200 | 15 files |
| Apr 7 late | GPT-2 controls, gap-filling design | nigel | 1 file |
| Apr 8 early | H100 reviewer experiments (cross-domain, held-out, stats rigor) | H100 | 12 files |
| Apr 8 AM | Llama3 dissociation + replication; behavioral validation | H100/nigel | 5 files |
| Apr 8 midday | Behavioral N=30 dose-response; theorem N=30 | H100 | 4 files |
| Apr 8 PM | Behavioral 27B, 12B-PT; activation patching v2 | H100 | 3 files |
| Apr 8 late | Benign FP, multilayer orthogonality | H100 | 2 files |
