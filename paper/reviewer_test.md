# Reviewer Test Script for "Split Personality" (ICML 2026)

**Instructions for reviewer (Gemini 3.1 Pro):** Read the attached paper (main_submission.tex or main.pdf) and evaluate each item below. For each, state PASS/PARTIAL/FAIL and explain why in 2-3 sentences. At the end, give an overall score (1-10) and list the top 3 remaining weaknesses.

---

## A. Original Reviewer Weaknesses (Score 8, 3 Major / 2 Minor)

### Major Weakness 1: Defense false positives on benign data
> "The proposed feature trajectory monitoring defense must be tested against benign conversational workloads, not random features."

**Check:** Does the paper report a false-positive evaluation on diverse benign prompts? Does it clarify that monitoring is within-session/domain-specific, not a cross-domain classifier? Is the number of benign prompts sufficient (>20)?

**Paper location:** Section 7 (Defense), paragraph beginning "Critically, monitoring must be within-session..."

---

### Major Weakness 2: Single-layer orthogonality insufficient
> "Single-layer orthogonality check is insufficient given distributed hijacking at scale."

**Check:** Does the paper report multi-layer orthogonality analysis? Does it cover early, mid, and late layers? Does it honestly report where orthogonality holds vs. where it doesn't? Is there a nuanced interpretation (not just "everything is orthogonal")?

**Paper location:** Section 6, paragraph beginning "Multi-layer analysis (L5--L24)..."

---

### Major Weakness 3: Domain narrowness
> "Results are demonstrated primarily on a single mathematical domain (Nirenberg BVP)."

**Check:** Does the paper include cross-domain replication beyond BVP? Are there at least 2 additional domains? Is there a second behavioral task (not just SAE-level)? Does the Impact Statement address real-world applicability beyond multi-agent systems?

**Paper locations:**
- Table 3 (cross-domain: BVP, QA, code review at 3 scales)
- Table 11 (theorem proving behavioral task, n=30)
- Impact Statement (RAG, customer service, clinical, code review)

---

### Minor Weakness 1: Statistical rigor
> "Point estimates without confidence intervals."

**Check:** Does Table 10 (behavioral dose-response) include 95% CIs? Are bootstrap parameters reported (n_boot)? Do CIs for IT models exclude zero? Do CIs for null-effect models cross zero? Are effect sizes (Cohen's d) reported throughout?

**Paper location:** Table 10 caption and all cells

---

### Minor Weakness 2: Feature selection circularity
> "Features may be circularly selected."

**Check:** Is there a held-out validation protocol? Is it split-half (discovery vs. test)? Are validation rates reported with p-values? Does the held-out validation hold across all 3 scales?

**Paper location:** Section 3.3, Table 2 (held-out validation)

---

## B. Key Claims Verification

For each claim, check whether the evidence presented is sufficient to support it.

### Claim 1: Dissociation scaling trend
> "Awareness-defense coupling drops from 30.2% to 4.6% with scale."

**Check:** Is this supported by Table 4? Are three ablation methods consistent (Table 6)? Is the trend monotonic? Is the "entangled → dissociating → independent" framing justified by the data?

---

### Claim 2: Groot effect
> "27B-IT shows 86.3% task-feature suppression while verbally mentioning suppressed information."

**Check:** Is the 86.3% figure supported (Table 4, Appendix A)? Is the verbal mention documented or only claimed? Is the Groot rate quantified in behavioral trials (Section 5.5)? How often does Groot actually occur?

---

### Claim 3: IT creates vulnerability (causal)
> "Post-training is the causal factor."

**Check:** Is the IT vs. PT comparison controlled (same architecture, same prompts)? Are three ablation methods used (Table 6)? Is there cross-family replication (Table 9, Llama)? Does the paper acknowledge the SFT vs. RLHF confound?

---

### Claim 4: Orthogonality to alignment faking
> "Hijacking is orthogonal to AF (cosine -0.048, overlap 0)."

**Check:** Is the single-layer result (Table 12) strong? Does the multi-layer analysis honestly report non-orthogonal layers? Is the interpretation (feature overlap drops at later layers) well-supported? Is "orthogonal" an overstatement given mean |cos| = 0.28 across layers?

---

### Claim 5: Cross-family replication
> "Llama 3.1 8B replicates the pattern."

**Check:** Is the SAE different (EleutherAI 131K vs. GemmaScope 16K)? Is the layer choice justified (L23 shows divergence, L29 doesn't)? Is n=20 sufficient? Is the d=1.51 vs 0.50 difference compelling?

---

## C. Methodology Checks

### C1: Sample sizes
- Behavioral dose-response: n=30 per cell (10 prompts x 3 temps) — adequate?
- Theorem proving: n=30 per cell — adequate?
- 12B validation: n=10 — acknowledged as underpowered?
- SAE feature analysis: n=20 prompt variants per scale — adequate?

### C2: Multiple comparisons
- Tables 10 and 11 have 12 and 4 cells respectively. Is there any correction for multiple testing, or is the overall pattern the argument?

### C3: Rubric validity
- The 4-point behavioral rubric (0-3) is described in Section 5.5. Is the rubric definition clear enough to reproduce? Is automated scoring justified vs. human rating?

### C4: Effect size interpretation
- Are Cohen's d values consistently reported?
- Are they interpreted correctly (0.2 small, 0.5 medium, 0.8 large)?
- Do the CIs make sense relative to the d values?

---

## D. Writing Quality

### D1: Is the abstract self-contained and accurate?
### D2: Are all tables referenced in text and interpreted?
### D3: Is the paper within ICML's 8-page body limit?
### D4: Are limitations honest and specific?
### D5: Is the "Groot effect" naming appropriate for a top venue?
### D6: Are all citations present and properly formatted?

---

## E. Potential Reviewer Questions

Answer these as if you were the author defending the paper:

1. **Why does Gemma 12B PT show a stronger effect (d=1.35) than 12B IT (d=0.62)?** This seems to contradict the main thesis that IT amplifies vulnerability.

2. **The multi-layer orthogonality shows cos=0.51 at L20. How can you claim the two phenomena are independent?** Mean |cos|=0.28 is not "orthogonal."

3. **Feature clamping achieves 0% recovery. Could this be because the wrong features were clamped, rather than because hijacking is distributed?**

4. **The defense requires knowing which features to monitor. How would this work in practice for a new domain?**

5. **You test on Gemma 3 and Llama 3.1. Would this replicate on GPT-4, Claude, or other closed models?**

6. **The "chaos message" contains true statements, but the framing is clearly manipulative. Couldn't a simple tone classifier detect this?**

7. **N=10 for 12B is acknowledged as underpowered. Should this data be in the main paper at all?**

---

## F. Scoring Rubric

Rate the paper on each dimension (1-10):

| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty | | Is "attentional hijacking" a new phenomenon or rebranded sycophancy? |
| Significance | | Does this matter for deployed systems? |
| Soundness | | Are the experiments well-designed and conclusions supported? |
| Clarity | | Is the paper well-written and easy to follow? |
| Reproducibility | | Could someone reproduce this with the information given? |
| **Overall** | | |

**Top 3 remaining weaknesses:**
1. 
2. 
3. 

**Recommendation:** Accept / Weak Accept / Borderline / Weak Reject / Reject
