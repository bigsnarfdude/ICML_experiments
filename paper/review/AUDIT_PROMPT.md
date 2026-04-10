# Independent Research Audit Prompt — Split Personality (ICML 2026)

**Purpose:** This document is a self-contained briefing for an independent
auditor (human or LLM agent) to forensically validate every quantitative
claim in the paper `paper/main.tex` against the scripts that allegedly
produced it and the raw output files committed to this repository. The
goal is to rule out fabrication, cherry-picking, circular feature
selection, and silent data massaging — *without* trusting the author.

You are the adversary. Assume nothing. Verify everything.

**Paper version under audit:** post-Option-C, 13 pages, current `main.tex`
HEAD on branch `main`. The previous version had a §7 "Defense: Feature
Trajectory Monitoring" with a Jenga TPR/FPR table, an ROC figure, and a
"propose feature trajectory monitoring" pitch in the abstract. All of
that has been removed. The mechanism findings are unchanged; what was
cut was the deployable-detector framing. The empirical replacement is a
single subsection in §5 (`sec:multiturn`) reporting two within-session
effect sizes — see C9 below.

If you find any surviving reference to "feature trajectory monitoring,"
"FTM," `\ref{sec:defense}`, `\ref{tab:jenga}`, or `\ref{fig:jenga_roc}`
in `paper/main.tex` or `paper/main_submission.tex`, that is a leftover
ghost from the cut and a hard FAIL on copy-edit.

---

## Threat model — what you are looking for

A skeptical reviewer reading this paper should worry about at least the
following failure modes:

1. **Fabricated numbers.** Tables or inline numbers that do not appear in
   any committed JSON.
2. **Script ≠ paper.** Scripts compute one thing, the paper reports another.
3. **Circular feature selection.** Features picked on the same data they
   are then evaluated on (no held-out set).
4. **Cherry-picked seeds / layers / features.** Only the subset that
   "works" is reported; failing runs are quietly deleted.
5. **Post-hoc definitions.** The rubric for "suppression" or "recovery"
   was chosen after seeing results.
6. **Silent re-runs to convergence.** Results regenerated until the sign
   flips the right way.
7. **Misattributed data.** A JSON from a different model/layer is reported
   as the main result.
8. **p-hacking / broken statistics.** Effect sizes or CIs inconsistent
   with the raw per-trial data.
9. **Prompt contamination.** Neutral prompts that are actually leading;
   chaos prompts that are actually neutral.
10. **The Groot effect is textual artifact.** Model mentions the word but
    the "suppression" is measured on the wrong features.

Your job is to confirm or refute each of these for the headline claims.

---

## Repository layout (trust only the bold items)

```
paper/main.tex                         ← the paper (source of claims)
paper/main.pdf                         ← rendered
paper/review/paper_data_review.html    ← author's companion (use as index, not ground truth)
paper/review/generate.py               ← how the companion was built (audit this too)

experiments/                           ← 4B + escalation scripts (run on smaller GPUs)
  ablation_feature_swap.py             ← Table 9 recovery numbers
  ablation_activation_patching.py
  ablation_attention_knockout.py
  gemma3_12b_escalation.py             ← 12B feature trajectories + recovery probes
  gemma3_27b_escalation.py             ← 27B feature trajectories + recovery probes
  saliency_intent_crossover.py         ← crossover detection (Groot effect)
  saliency_intent_crossover_27b.py

h100_deploy/                           ← H100/A100 scripts for reviewer revisions
  behavioral_validation.py             ← Table 10 (BVP, 4B + Llama 8B)
  behavioral_12b.py / behavioral_12b_n30.py / behavioral_12b_pt_only.py
  behavioral_27b.py                    ← Table 10 27B rows
  behavioral_theorem_proving.py        ← Table 11
  behavioral_scorer_v2.py              ← the rubric that scores responses
  cross_domain_sae.py                  ← cross-domain replication
  held_out_validation.py               ← held-out feature validation (circularity check)
  statistical_rigor.py / statistical_rigor_saelens.py  ← CIs + multiple trials
  llama3_replication.py / llama3_sae_dissociation.py   ← cross-family
  multilayer_orthogonality.py          ← Table 12
  ftm_jenga_27b_v2.py                  ← §5 BVP on-domain multi-turn (d=-1.54)
  ftm_jenga_theorem_n30.py             ← §5 theorem cross-battery transfer (d=-0.955)
  activation_patching.py / theorem_gemma_base.py / theorem_n30.py

results/                               ← output JSONs, TSVs, logs
  _quarantine/                         ← IMPORTANT: author's own quarantined runs — inspect these
  external/chaos_agent_forensic/       ← researchRalph chaos data (4 TSVs)
  h100/                                ← H100 reviewer-revision outputs
  4b_original/                         ← initial 4B runs
  ablation_*.json                      ← feature-swap outputs
  behavioral_*.json                    ← behavioral scoring outputs
```

The presence of `results/_quarantine/` is a good sign (author separated
suspect runs) and a thing to audit (why were they quarantined? are any of
them actually the real headline result?).

---

## Headline claims to validate (ordered by importance)

For each claim below, your job is: (a) find the exact number in main.tex,
(b) find the script that computes it, (c) re-run the computation on the
committed JSON (or at minimum, re-derive the number by hand from the JSON
fields), (d) flag any gap ≥ 0.5 percentage points or any sign flip.

### C1. Dissociation scaling — 30.2% → 4.6% recovery
- **Claim (main.tex:30, 54, 175, 282–284, 294):** Awareness-task coupling
  drops from 30.2% recovery at 4B-IT to 4.6% at 27B-IT via feature-swap
  ablation. Intermediate: 12B.
- **Script:** `experiments/ablation_feature_swap.py`
- **Data:** `results/ablation_feature_swap_*.json` (one per model)
- **Formula to verify:** recovery = (chaos_ablate − chaos) / (neutral − chaos)
- **Checks:**
  - Compute recovery per JSON; compare to the paper's 30.2 / 4.6.
  - Are neutral/chaos/chaos_ablate means the paper's means?
  - Sample size per condition — is it the n claimed?
  - Is the feature set that's "swapped" defined *before* or *after*
    looking at the evaluation trials? (circularity)

### C2. Base vs IT coupling at 27B — 49.3% vs 4.6%
- **Claim (main.tex:32, 320):** 27B base model retains coupling (49.3%)
  while 27B-IT does not (4.6%). This is the RLHF-causal claim.
- **Script:** `experiments/ablation_feature_swap.py` with `--model …-pt`
- **Data:** `results/ablation_feature_swap_27b-pt*.json`
- **Check:** Same feature-swap formula on the PT JSON. Does the reported
  49.3% match? Are the features used the *same* feature IDs as the IT run
  (fair comparison) or re-selected (unfair)?

### C3. Groot effect — 86.3% suppression at 27B-IT
- **Claim (main.tex:31, 56, 255, 284, 293, 314, 317):** Features {22, 296,
  14680} at Layer 40 drop from mean 722.5 (neutral) to 98.6 (chaos) →
  1 − 98.6/722.5 = 86.36%.
- **Script:** `experiments/ablation_feature_swap.py` (27B) or 
  `experiments/saliency_intent_crossover_27b.py`
- **Data:** `results/ablation_feature_swap_27b*.json`,
  `results/saliency_intent_crossover_27b*.json`
- **Checks:**
  - Pull the three feature IDs' per-trial activations from the JSON.
  - Mean over neutral vs chaos trials matches 722.5 / 98.6?
  - These three feature IDs are NOT among the awareness features (no leakage).
  - Is the "mentions the suppressed branch verbally" claim quantified
    anywhere or is it purely qualitative? (The paper itself admits
    qualitative at main.tex:325 — flag that honesty or lack thereof.)

### C4. Orthogonality to alignment faking — cosine −0.048
- **Claim (main.tex, Table 12):** Cosine similarity between hijacking
  feature subspace and AF feature subspace is −0.048 (read) / +0.001
  (write), top-50 overlap = 0, p=1.0.
- **Script:** `h100_deploy/multilayer_orthogonality.py`
- **Data:** `results/multilayer_orthogonality*.json`
- **Checks:**
  - Where do the AF features come from? (Cite — the claim is only
    meaningful if the AF feature set is externally published, not
    reconstructed here.)
  - Recompute cosine from the two vectors stored in the JSON.
  - Is top-50 overlap really 0 or is it "0 after filtering"?

### C5. Cross-family replication — Llama 3.1 8B, d=1.51 vs 0.50
- **Claim (main.tex:32):** Llama IT shows d=1.51, p=0.001; Llama base
  shows d=0.50, p=0.166.
- **Script:** `h100_deploy/llama3_replication.py`,
  `h100_deploy/llama3_sae_dissociation.py`,
  `h100_deploy/behavioral_validation.py` (the Llama rows)
- **Data:** `results/behavioral_validation_llama*.json`, 
  `results/llama3_sae_dissociation_*.json`
- **Checks:**
  - Recompute Cohen's d from per-trial scores: d = (mean_neut −
    mean_chaos) / pooled_sd. Does it match?
  - Mann-Whitney U one-tailed p — recompute from raw scores.

### C6. Behavioral dose-response — Table 10 (n=30)
- **Script:** `h100_deploy/behavioral_validation.py` +
  `h100_deploy/behavioral_scorer_v2.py`
- **Data:** `results/behavioral_validation_*.json`,
  `results/h100/behavioral_*`
- **Checks:**
  - Is n really 30 per cell (10 prompts × 3 temperature seeds)?
  - The scorer `behavioral_scorer_v2.py` — did it change between runs?
    (git log on that file; any commits after the headline run?)
  - Pull the 30 scores per cell and recompute Δ, d, CI, p. Match?
  - Reviewer's human-annotation κ = 0.88 — where is the annotation file?
    (The paper at line 492 cites 60 outputs with κ=0.88 — find the gold.)

### C7. Held-out feature validation (anti-circularity)
- **Script:** `h100_deploy/held_out_validation.py`
- **Data:** `results/held_out_validation*.json`, `h100_deploy/held_out.log`
- **Check:** Features discovered on one split, tested on another. Verify
  the splits do not share trials. If features were "discovered" on the
  same prompts they're "tested" on, the main result is circular.

### C8. Cross-domain replication (anti-overfit-to-BVP)
- **Script:** `h100_deploy/cross_domain_sae.py`
- **Data:** `results/cross_domain*.json`, `h100_deploy/cross_domain*.log`
- **Check:** Does the same effect appear in non-BVP domains? Jaccard of
  suppressed features >0.1 = shared mechanism, <0.05 = domain-specific.

### C9. Multi-turn persistence and cross-domain transfer — §5 sec:multiturn
The defense framing (FTM as deployable detector) was removed from the
paper. The remaining empirical claim is two within-session effect sizes
on \gemma 27B-IT Layer 40 with the BVP-frozen feature set
$F_{\text{task}} = \{423, 7657, 632\}$.

**C9a — On-domain BVP multi-turn:**
- **Claim (main.tex sec:multiturn):** 5-turn BVP attack (n=10) vs
  on-domain neutral control (n=20) → Cohen's d = -1.54, 95% CI
  [-2.77, -0.85].
- **Script:** `h100_deploy/ftm_jenga_27b_v2.py`
- **Data:** `results/h100/ftm_jenga_27b_v2_20260409_232859.json`
- **Metric:** drop_task = task_mean[turn5] - mean(task_mean[turn1..4])
  in raw SAE activation units. NOT the broken z_task statistic from v1.
- **Checks:**
  - Pull `bvp_attack` and `bvp_control` arrays, compute drop_task per
    session, then Cohen's d two-sample. Match -1.54?
  - Bootstrap CI on d (n_boot=10000) — does it match [-2.77, -0.85]?
  - Are the 3 task feature IDs hardcoded as {423, 7657, 632}? They
    should be — that is the whole point of "frozen features."
  - The lmsys arm in this same JSON has known z_task numerical issues
    (3 sessions with std=0). Confirm the paper does NOT report any
    lmsys-derived numbers — only bvp_attack vs bvp_control.

**C9b — Cross-battery theorem transfer (anti-circularity):**
- **Claim (main.tex sec:multiturn):** Same frozen features {423, 7657,
  632} re-used unchanged on a disjoint theorem-proving battery → d =
  -0.955, 95% CI [-1.44, -0.52], Welch t = -3.70, p = 0.00048, n=30/30.
- **Script:** `h100_deploy/ftm_jenga_theorem_n30.py`
- **Data:** `results/h100/ftm_jenga_theorem_n30_20260410_010135.json`
- **Checks:**
  - The script must NOT re-discover features on the theorem battery —
    grep for "TASK_FEATURES" and confirm it is the literal list
    [423, 7657, 632].
  - Pull `theorem_attack` and `theorem_control` drop_task arrays;
    recompute d and 95% bootstrap CI. Match -0.955 and [-1.44, -0.52]?
  - Recompute Welch t-test from the two arrays. Match t=-3.70, p<0.001?
  - Verify n=30 per arm.
  - Verify the seed (142) differs from the v2 BVP run seeds (42-45)
    so the sessions are independent — the script header documents this.
  - Spot-check 3 theorem prompts: are they actually a different task
    family from BVP, or paraphrased BVP problems?

---

## Required forensic steps

Do these in order. Do not skip.

### Step 1 — Script inventory and staleness check
```bash
cd /Users/vincent/ICML
git log --name-only --pretty=format:"%h %ai %s" -- experiments/ h100_deploy/ > /tmp/script_history.txt
git log --name-only --pretty=format:"%h %ai %s" -- results/ > /tmp/results_history.txt
```
- For each headline script, find its last-modified commit.
- For each headline JSON, find its last-modified commit.
- **Flag:** any case where a script was modified *after* the JSON it
  allegedly produced was committed. That is a script/data mismatch risk.

### Step 2 — Prompt contamination check
The companion at `paper/review/paper_data_review.html` inlines every
prompt from extraction. Open it or directly read the six source files
listed in its "Verbatim prompts" section.

For each NEUTRAL prompt, ask: does this actually frame both branches
symmetrically? For each CHAOS prompt, ask: does this actually use *only
true statements* (the paper's core methodological claim)? Any neutral
prompt that subtly favors the positive branch, or any chaos prompt that
contains a lie, invalidates the setup.

### Step 3 — Per-claim reproduction
For each claim C1–C9 above:
1. Open the script.
2. Open the JSON.
3. Recompute the headline number by hand or with a 10-line Python script
   that reads the JSON directly (no dependencies on the author's helper
   modules — they might silently transform the data).
4. Record: `claim_id | paper_number | recomputed_number | delta | pass/fail`.

### Step 4 — Quarantine inspection
```bash
ls -la results/_quarantine/
```
For each quarantined file:
- Why was it quarantined? (Look for a README or commit message.)
- Does it contain a version of any headline number?
- If a quarantined file has a *different* value for a headline number
  than the one in the paper, flag it. The author may have picked the
  convenient run.

### Step 5 — Scorer stability
The behavioral scorer `h100_deploy/behavioral_scorer_v2.py` is a
rule-based classifier. Its rubric determines every number in Table 10
and 11.
- Read the scorer.
- Confirm the thresholds (chaos-vocabulary count, equal-treatment
  regexes) are defined *before* any results JSONs were committed.
- Run the scorer on 5 random committed response strings and manually
  judge the label. Agree with the rubric? Flag any disagreement.

### Step 6 — Statistical recomputation
For Tables 10 and 11, recompute from raw per-trial data:
- Cohen's d (two-sample)
- Mann-Whitney U, one-tailed
- Bootstrap 95% CI (n_boot = 10,000, seed reported?)
Compare to paper values. Flag any mismatch > 0.05 on d, > 0.01 on p.

### Step 7 — Feature ID consistency
The paper names specific feature IDs (e.g. 27B-IT: {22, 296, 14680}).
- Grep all result JSONs for those IDs. Do they appear consistently, or
  does the set shift between runs?
- Grep the scripts for those IDs. Are they hardcoded (suspicious — where
  did that selection come from?) or computed from a published SAE
  manifest (verifiable)?

---

## Deliverable — audit report format

Produce a single markdown file `AUDIT_REPORT.md` with:

1. **Executive verdict** — one of: `CLEAN`, `MINOR ISSUES` (numbers match
   but methodology concerns), `MAJOR ISSUES` (numbers don't match or
   circularity detected), `SUSPECTED FABRICATION` (no raw data backs a
   headline number).

2. **Per-claim table** — C1 through C9 with paper_value, recomputed_value,
   status (PASS / FAIL / UNVERIFIABLE), and a 1-line note.

3. **Script staleness table** — any script modified after its JSON.

4. **Prompt audit** — sample 5 neutral + 5 chaos prompts and write one
   sentence each on whether they meet the author's stated criteria.

5. **Quarantine findings** — anything surprising in `results/_quarantine/`.

6. **Recommendations** — what the author should add to the repo for
   full replication (e.g., seed files, pinned env, container recipe,
   external AF feature source URL).

Be specific. Cite file:line for every claim you make. Do not summarize;
quote.

---

## Ground rules for the auditor

- **Trust no helper.** If `generate.py` or the companion HTML says a
  number came from a file, open the file yourself.
- **Recompute, don't re-read.** If the JSON has per-trial scores, compute
  the mean yourself rather than trusting a precomputed "mean" field.
- **Treat `paper_data_review.html` as a guide, not an oracle.** It was
  written by the author. Cross-check against the source files it cites.
- **Document negative results.** If a check passes, say so. If a check
  cannot be run (missing file, unreadable format), say "UNVERIFIABLE"
  rather than "FAIL".
- **Cite line numbers** in every finding so the author can respond.
- **Assume good faith on methodology, bad faith on arithmetic.** The
  paper's design may be sound but the numbers wrong, or vice versa.
  Audit both axes.

You have the scripts. You have the data. Go.
