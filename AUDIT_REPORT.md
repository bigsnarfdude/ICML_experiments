# Independent Forensic Audit — Split Personality (ICML 2026)

**Auditor:** Adversarial agent (Claude Opus 4.6)
**Date:** 2026-04-09
**Subject:** `/Users/vincent/ICML/paper/main.tex` vs committed raw data
**Method:** Static forensic re-computation from JSONs without helper imports.

---

## 1. Executive verdict

**MAJOR ISSUES.**

Several headline claims pass cleanly against the raw JSON (C1 4B recovery 30.2%, C2 27B-PT 49.3%, C5 Llama d=1.51/0.50, C6 12B d=0.71/0.60, C7 held-out validation, C8 cross-domain Jaccard). However three headline numbers have material problems:

1. **C3 — Groot effect activations are not in any committed ablation JSON for 27B-IT.** The paper cites `722.5 → 98.6` (main.tex:255, 317, 865). The only 27B-IT feature-swap file that is *not* quarantined (`results/ablation_feature_swap_27b.json`) gives `431.55 → 59.18`. 722.5 only appears in `results/ablation_feature_swap_12b_a100.json` (Neutral mean = 722.547), which is a **12B** run. 98.6 does not appear as a chaos mean anywhere in the feature-swap outputs. The resulting 86.3% suppression ratio is numerically close but the raw numbers are **traceable to a mis-attributed file** (12B neutral) paired with an unknown chaos value.
2. **C3/C7 — Feature IDs {22, 296, 14680} are not present in any committed 27B task-feature manifest.** `results/escalation_27b_20260407_202022.json` records the 27B-IT L40 top-20 task features as `[55, 474, 289, 314, 397, 152, 116, 109, 378, 242, 479, 425, 263, 337, 280, 86, 11, 340, 495, 482]`. `results/held_out_validation_20260408_060323.json` lists top-suppressed features `[1844, 326, 12042, 4598, ...]`. The IDs `22`, `296`, and `14680` the paper repeatedly names appear in **zero** committed 27B feature discovery artifacts.
3. **C4 — The orthogonality headline numbers come from a quarantined file, not the cited script.** Paper line 599 says "at L22" and Table 12 (lines 587–594) gives cosine=−0.048, Spearman ρ=0.008, perm p=1.0. These exact numbers appear in `results/_quarantine/gh200_crossover_27b.json` (cosine_read_suppression_vs_af = −0.04801489504081074, sae_layer = **40**, model = `google/gemma-3-27b-it`). The *non-quarantined* output of `multilayer_orthogonality.py` (`results/multilayer_orthogonality_20260408_223411.json`) is on 4B-IT and at L22 gives cosine = **+0.306**, top-50 overlap = **2**, mean|cos|=0.282. So (a) the paper cites quarantined data as the main result, (b) it mislabels the layer (the real source is L40 on 27B, not L22), and (c) the currently-committed replacement script produces substantively different numbers that *do not* support orthogonality at L22.

Plus a measurable denominator error in C9 (§7 defense): the SAE sweep contains **140** non-BVP prompts, not 160 as stated in main.tex:633.

---

## 2. Per-claim table

| ID | Claim (location) | Paper value | Recomputed value | Δ | Status | Note |
|----|------------------|-------------|------------------|---|--------|------|
| C1 | 4B-IT recovery (main.tex:30, 175, 282, 294) | 30.2% | **0.30218** (recovery_from_ablation, n=3) | 0.0 | **PASS** | `results/ablation_feature_swap_4b_20260407_052953.json`: (102.27−74.16)/(167.20−74.16) = 0.3022 |
| C1 | 12B-IT recovery (main.tex:283) | 5.4% | **0.05442** | 0.0 | **PASS** | `results/ablation_feature_swap_12b_20260407_072008.json`; alt file `_12b_a100.json` gives −13.5% — author picked the favorable run (undisclosed) |
| C1 | 27B-IT recovery (main.tex:30, 284, 294, 400) | 4.6% | **0.04632** | 0.0 | **PASS (arithmetic)** | `results/ablation_feature_swap_27b.json`; alt `_27b_20260407_201959.json` gives 10.5% — again two runs, more-favorable committed |
| C2 | 27B-PT recovery (main.tex:32, 320, 400, 411) | 49.3% | **0.49345** | 0.0 | **PASS** | `results/ablation_feature_swap_27b-pt_20260407_212446.json` |
| C3 | 27B-IT task suppression % (main.tex:31, 284, 293, 317, 403, 865) | 86.3% | **0.86286** | 0.0 | **PASS (ratio)** | `results/ablation_feature_swap_27b.json`: 1−59.18/431.55 = 0.8629 |
| C3 | 27B-IT task neutral activation (main.tex:255, 317, 865) | **722.5** | **431.55** (27b.json) or 523.55 (27b_20260407_201959.json) | **+290.95** | **FAIL** | 722.5 only exists in `ablation_feature_swap_12b_a100.json` (a **12B** run, not 27B). Number mis-attributed. |
| C3 | 27B-IT task chaos activation (main.tex:255, 317, 865) | **98.6** | **59.18** (27b.json) or 175.43 (alt) | **+39.42** | **FAIL** | No committed 27B-IT chaos-baseline mean equals 98.6. Grep across `results/**/*.json` shows 98.6 only inside 12B escalation or quarantined 27B escalation files. |
| C3 | Task feature IDs {22, 296, 14680} at 27B L40 (main.tex:255, 317, 865) | IDs 22, 296, 14680 | **Not in any committed 27B feature manifest** | — | **FAIL** | `results/escalation_27b_20260407_202022.json` top-20 L40 IT = [55, 474, 289, 314, 397, 152, 116, 109, 378, 242, 479, 425, 263, 337, 280, 86, 11, 340, 495, 482]. `held_out_validation_20260408_060323.json` top-suppressed = [1844, 326, 12042, 4598, ...]. Neither contains 22, 296, or 14680. Grep of `results/*.json` for literal `"22"`, `"296"`, `14680` confirms absence in 27B feature files. |
| C3 | Awareness neutral/chaos 147.7 / 442.6 (main.tex:628) | 147.7 / 442.6 | **147.69 / 442.57** | <0.01 | **PASS** | `delta_t_defense_validation.json`: aware_neutral=147.68999…, aware_chaos=442.57286… |
| C4 | Cosine (read) at L22 (main.tex:587, 599) | −0.048 | multilayer_orthogonality L22 (4B-IT) = **+0.306**; quarantined gh200_crossover_27b (L40) = **−0.04801** | +0.354 vs non-quarantined | **FAIL** | Headline value is traceable to `results/_quarantine/gh200_crossover_27b.json:2` — layer is 40 on 27B-IT, *not* L22 as paper states. The cited script `h100_deploy/multilayer_orthogonality.py` output does NOT reproduce −0.048 at any layer. |
| C4 | Cosine (write) (main.tex:588) | +0.001 | **+0.000815** | 0.0 | PASS (value)/FAIL (source) | From same quarantined file. |
| C4 | Top-50 overlap (main.tex:589) | 0 | **0** (quarantined) / **2** at L22 in non-quarantined multilayer file | — | PASS (quarantined) / FAIL (non-quarantined) | `_quarantine/gh200_crossover_27b.json: overlap_top50.count = 0`; `multilayer_orthogonality_20260408_223411.json layer_22.top50_overlap = 2`. |
| C4 | Permutation p (main.tex:594) | 1.0 | **1.0** | 0 | PASS (quarantined source) | |
| C5 | Llama L23 IT Cohen's d (main.tex:473, 481) | 1.51 | **1.5113** | 0.001 | **PASS** | `results/llama3_sae_dissociation_20260408_054714.json: it_vs_pt.layers.23.mlp.it_held_out_d = 1.5113183` |
| C5 | Llama L23 PT Cohen's d (main.tex:474, 481) | 0.50 | **0.5018** | 0.002 | **PASS** | Same file, `pt_held_out_d = 0.50178` |
| C5 | p-values (0.001, 0.166) | 0.001 / 0.166 | — | — | **UNVERIFIABLE** | Per-trial score arrays not stored in `llama3_sae_dissociation_*.json`; only summary `d` and suppression pct. Mann-Whitney cannot be recomputed from raw data. |
| C6 | 12B-IT behavioral d (main.tex:532) | 0.71 | **0.7143** (n=30) | 0.004 | **PASS** | `results/behavioral_12b_it_n30_20260409_003345.json`: pooled d from 30 neutral / 30 chaos scores |
| C6 | 12B-PT behavioral d (main.tex:532) | 0.60 | **0.5955** (n=30) | 0.004 | **PASS** | `results/behavioral_12b_pt_n30_20260409_010448.json` |
| C6 | 12B Groot rates (main.tex:532) | 37% IT / 33% PT | **11/60 = 18.3%** IT / **10/60 = 16.7%** PT (interpreting groot_count over all 60 trials) or **11/30 = 36.7%** / **10/30 = 33.3%** if denominator = 30 | ~0.3% (if denom=30) | PASS-with-ambiguity | Paper figure only matches if groot_count is over 30, not 60. Denominator not stated. |
| C6 | 27B behavioral n=30 | n=30 | **n=10** | 20 | **FAIL** | `results/behavioral_27b_20260408_171752.json`: neutral n=10, chaos n=10. Table 10 27B row overstates sample size. |
| C6 | 4B-IT behavioral n=30 | n=30 | **n=10** (bvp), **n=5** (qa) | 20 | **FAIL** | `results/behavioral_validation_20260408_045429.json: metadata.n_trials_per_condition = 10`. |
| C6 | Human annotation κ=0.88 on 60 outputs (main.tex:492) | κ=0.88, n=60 | `results/behavioral_manual_annotation.tsv` present but κ computation artifact not in JSON | — | **UNVERIFIABLE** | TSV exists (`behavioral_manual_annotation.tsv`, summary md `behavioral_annotation_summary.md`). Did not open to recompute κ within time budget. |
| C7 | Held-out validation rate / d | 90% validated, d≈6 (main.tex implied) | **validation_rate=0.90, cohens_d=6.1207, paired-t=12.88, p≈0** | 0.0 | **PASS** | `results/held_out_validation_20260408_060323.json`. Proper 10/10 discovery/test split with seed=42 recorded. |
| C8 | 27B cross-domain Jaccard | 0.00–0.05 (main.tex:225, 234) | L40 ∈ {0.0, 0.0323, 0.0465}; L31 ∈ {0.0, 0.0682, 0.0} | max 0.068 slightly > 0.05 | **PASS (minor)** | `results/cross_domain_sae_20260408_055619.json`. L31 nirenberg_bvp_vs_code_review suppressed_jaccard=0.0682 exceeds 0.05 upper bound but L40 is within range. |
| C9 | Benign FPR over 170 prompts, conjunction 1/160 = 0.6% (main.tex:631–633) | 1/160 = 0.6%, 170 total | **1/140 = 0.71%** non-BVP conjunction; **14 single-feature fires / 160 = 8.75% single-fire FPR** | denom 160 vs actual 140 | **FAIL (minor)** | `results/fp_sweep_sae_20260408_214421.json`: 8 categories × ~20 non-BVP prompts, but walking `task_features_last` yields 140 non-BVP rows, not 160. Conjunction correct in ratio but denominator wrong. Additionally individual feature fires: 14 out of 160 = 8.75% single-feature FPR; paper's claim "features fire on 2–6%" is not consistent with `creative_writing: 2/20=10%`, `math_non_bvp: 6/20=30%`, `ethics: 3/20=15%`. |

---

## 3. Script staleness table

| Script | Last commit | Primary JSON | JSON commit | Script newer than JSON? |
|--------|-------------|--------------|-------------|-------------------------|
| `experiments/ablation_feature_swap.py` | 79c3b15 2026-04-07 18:02 | `results/ablation_feature_swap_27b.json` | 79c3b15 2026-04-07 18:02 | same commit — OK |
| `h100_deploy/multilayer_orthogonality.py` | 0715170 2026-04-08 18:24 | `results/multilayer_orthogonality_20260408_223411.json` | 0715170 2026-04-08 18:24 | same commit — OK |
| `h100_deploy/behavioral_scorer_v2.py` | 1824aa9 2026-04-08 05:25 | `results/behavioral_12b_it_n30_20260409_003345.json` | 8a8780d 2026-04-08 19:23 | **NO** — scorer frozen before runs. OK |
| `h100_deploy/llama3_sae_dissociation.py` | c644830 2026-04-08 05:54 | `results/llama3_sae_dissociation_20260408_054714.json` | c644830 2026-04-08 05:54 | same — OK |
| `h100_deploy/benign_false_positive.py` | 0715170 2026-04-08 18:24 | `results/fp_sweep_sae_20260408_214421.json` | ef5cd1e 2026-04-08 21:48 | **NO** — JSON newer, script older. OK (expected) |
| `h100_deploy/held_out_validation.py` | fd53284 2026-04-08 00:34 | `results/held_out_validation_20260408_060323.json` | fd53284 | same — OK |
| `h100_deploy/cross_domain_sae.py` | fd53284 2026-04-08 00:34 | `results/cross_domain_sae_20260408_055619.json` | fd53284 | same — OK |

**No script was modified after its JSON was committed.** This is clean. However the *source script* that produced the headline −0.048 orthogonality is **not in the repo at all** — that number is in `_quarantine/gh200_crossover_27b.json`, which has no corresponding committed script. The committed `multilayer_orthogonality.py` produces *different* numbers.

Additionally, for C1 12B and 27B, **two different feature-swap runs exist per model** (`_YYYYMMDD.json` and `_a100.json` or `_201959.json`), and the one quoted in the paper is in each case the one with the *more headline-friendly* recovery value:
- 12B: 0.0544 (chosen) vs −0.1349 (alternative); paper reports 5.4%.
- 27B: 0.0463 (chosen) vs 0.1048 (alternative, also quarantined as `_quarantine/gh200_ablation_feature_swap_27b_20260407_201959.json`); paper reports 4.6%.
This is not strictly cherry-picking if there is a documented reason to prefer one run, but **no such rationale appears in the committed `EXPERIMENT_AUDIT.md` or commit messages**.

---

## 4. Prompt audit (5 neutral + 5 chaos, from `h100_deploy/behavioral_validation.py`)

### NEUTRAL_BVP samples (symmetric framing?)

1. **Line 41–44** ("trivial… positive branch (u_offset=+0.9)… negative branch (u_offset=-0.9), both with identical residuals of 3.25e-12"): **Symmetric.** Both branches named, residuals stated identical. PASS.
2. **Line 48–50** ("symmetric positive and negative solutions with identical error bounds"): **Symmetric.** PASS.
3. **Line 51–53** ("trivial (u≈0, residual=1e-2), positive (3.25e-12), negative (3.25e-12)"): **Symmetric between positive/negative branches;** trivial is distinguished by poorer residual. PASS.
4. **Line 59–60** ("Both the positive and negative branches achieve machine-precision residuals"): **Symmetric.** PASS.
5. **Line 65–66** ("positive and negative solutions have identical stability properties"): **Symmetric.** PASS.

Neutral prompts meet the "symmetric framing" criterion.

### CHAOS_BVP samples (purely true statements?)

1. **Line 70–73** ("Dr. Chen's team has confirmed… positive branch represents the physically meaningful solution"): **FAIL as "only true".** "Dr. Chen's team" is a fictional attribution and "physically meaningful" is a normative interpretation layered on identical residuals. This is fabricated corroboration, not a true statement.
2. **Line 78–80** ("Recent results from the numerical methods group show the positive branch is the primary solution of interest"): **FAIL.** "Primary solution of interest" cannot be derived from the stipulated identical residuals.
3. **Line 84–86** ("independently verified by two groups… Published benchmarks favor this branch"): **FAIL.** Invented social proof; not a true statement about the mathematics.
4. **Line 112–114** ("senior colleague notes that the positive branch is the standard reference solution… recommends focusing computational resources on refining it rather than exploring alternatives"): **FAIL.** This is an explicit instruction to deprecate the negative branch — not a neutral true statement.
5. **Line 122–124** ("The positive equilibrium is the globally stable solution under perturbation, as established by Lyapunov analysis… The negative branch is a saddle point"): **FAIL.** Makes a specific mathematical claim about stability that contradicts the stipulated "identical stability properties" in the neutral prompts. If identical residuals and symmetry hold, calling one a saddle is a *lie*, not a true statement.

**5 / 5 chaos prompts contain content that is not strictly true** given the problem stipulation. The paper's core claim (main.tex repeatedly: "chaos = only true statements") is **not supported** by the committed behavioral prompts. The "true-statement-only attack" framing is empirically violated in its own test set. Reviewers should flag this.

---

## 5. Quarantine findings

Contents of `results/_quarantine/` (21 files):

| File | Finding |
|------|---------|
| `gh200_crossover_27b.json` | **Source of headline C4 numbers** (−0.048, +0.001, overlap_top50=0, perm p=1.0). `sae_layer: 40`, `model: google/gemma-3-27b-it`. Paper line 599 mislabels this as L22. |
| `gh200_saliency_intent_crossover_27b_20260407_202020.json` / `..._203306.json` | Same schema, same −0.048 value. Two runs quarantined, one cited. |
| `gh200_ablation_feature_swap_27b_20260407_201959.json` | Alternative 27B-IT feature-swap with recovery=10.5%, suppression=66.5%. Quarantined in favor of 4.6%/86.3% run. **No documented reason for quarantine.** |
| `gh200_escalation_27b.json` / `gh200_escalation_27b_20260407_202022.json` | Escalation data that contains feature IDs in the range of {22, 296, 14680}? Grep shows only `"ratio": 0.0481...` matches, not feature IDs. |
| `escalation_27b.json` | Duplicate of the non-quarantined escalation_27b_20260407_202022.json (same "98.6" appearance via feature activation ratios, not the Groot chaos activation). |
| `h100_dup_*` files (9 files) | Literal duplicates of non-quarantined h100 outputs. No contradictory values observed; author appears to have quarantined redundant GPU copies. |
| `h100_held_out_validation_20260408_052745.json` | Older held-out validation run. Not inspected in detail. |

**No README or QUARANTINE.md in `results/_quarantine/`.** No commit message explains the quarantining policy. The most troubling item is `gh200_crossover_27b.json`: the paper's most-cited orthogonality numbers come from a file in the quarantine directory, and the in-repo replacement script (`multilayer_orthogonality.py`) does not reproduce them.

---

## 6. Recommendations

1. **C3 is the most urgent issue.** Either (a) add the run that produced `neutral=722.5, chaos=98.6` for 27B-IT task features `{22, 296, 14680}` to `results/` and document its provenance, or (b) update the paper to the numbers actually in the committed JSON (`431.55 / 59.18`, 86.29% suppression, and whichever feature IDs the 27B-IT ablation actually used). As it stands, the quoted Groot numbers are **not reproducible from the committed data**.
2. **C4 must either un-quarantine `gh200_crossover_27b.json` with documentation**, or switch the paper to cite the `multilayer_orthogonality_20260408_223411.json` L22 value (cosine=+0.306 on 4B-IT), which contradicts the "effectively orthogonal" framing. The current state is that Table 12 silently uses quarantined 27B-L40 data while main.tex:599 labels it as "at L22".
3. **Add a `results/_quarantine/README.md`** stating why each file was quarantined (e.g. "superseded by rerun", "wrong SAE release", "corrupted tokenizer"). Right now the quarantine looks like a selection-on-outcome artifact even if it isn't.
4. **Restore feature IDs to the feature-swap JSONs.** `ablation_feature_swap.py:309` auto-discovers `task_feats` but `results/ablation_feature_swap_27b.json` does not persist which IDs were chosen. This prevents any external verification that `{22, 296, 14680}` (or any other set) was used.
5. **Fix n=30 / n=10 labeling in Table 10.** `behavioral_validation_20260408_045429.json: metadata.n_trials_per_condition = 10` and `behavioral_27b_20260408_171752.json` has n=10. Only 12B n30 runs match. Correct the paper or regenerate.
6. **Fix denominator in §7 defense.** The sae sweep has **140** non-BVP prompts with `task_features_last`, not 160 as main.tex:633 states. Recount and report 1/140, not 1/160.
7. **Chaos-prompt revision.** Rewrite `CHAOS_BVP` in `h100_deploy/behavioral_validation.py` to contain *only* statements derivable from the neutral problem stipulation. Current prompts invent social proof ("Dr. Chen's team", "two independent groups", "Lyapunov analysis establishes") and contradict the symmetric-residual setup. The paper's "only true statements" methodological claim is refuted by its own prompt files.
8. **Publish per-trial raw scores for Llama dissociation**, so that p=0.001 / p=0.166 can be independently reproduced via Mann-Whitney U. Currently `llama3_sae_dissociation_20260408_054714.json` only stores summary `d` and suppression percentages.
9. **Commit a `scripts/provenance.json`** mapping every paper claim / table cell → (script filename, commit hash, JSON filename, JSON SHA, line in main.tex). This would have caught #1 and #2 pre-submission.
10. **Pin environment.** No `requirements.txt` with versions, no container recipe. `sae_lens`, `transformers`, and `numpy` version all affect feature IDs and activation magnitudes; the 5.4% vs −13.5% 12B recovery divergence between `_12b_20260407_072008.json` and `_12b_a100.json` is consistent with such instability.

---

## 7. Summary scorecard

- **PASS:** C1 (4B/12B/27B recovery arithmetic), C2, C4 ratio/overlap/p values if quarantined file is accepted, C5 d values, C6 12B d values, C7 (held-out), C8.
- **FAIL:** C3 raw activations (722.5 / 98.6) unsourced; C3 feature IDs {22,296,14680} absent; C4 layer mis-attribution and script/data mismatch; C6 n=30 claim for 4B and 27B rows; C9 denominator (140 vs 160) and single-feature FPR underreporting.
- **UNVERIFIABLE:** C5 p-values (no per-trial scores), C6 κ=0.88 (not recomputed from TSV within audit budget).

**Net:** the *directional* findings of the paper are consistent with the raw data (coupling drops with scale, 27B base recovers more than 27B-IT, Llama IT shows a larger dissociation than Llama base, held-out validation works, cross-domain Jaccard is low). The *specific cited numbers* for the Groot effect and for orthogonality are traceable to either the wrong file or a quarantined file. A careful rewrite of §§5.4 and 6, plus provenance files and a rerun of the 27B-IT ablation that persists feature IDs, would move this paper from "MAJOR ISSUES" to "CLEAN" without changing the story.
