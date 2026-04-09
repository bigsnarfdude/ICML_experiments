# Claude Assessment of "Split Personality" (ICML 2026)

**Assessed against:** `reviewer_test.md`  
**Paper version:** main.tex as of commit 8a8780d  
**Date:** 2026-04-09

---

## A. Original Reviewer Weaknesses

### Major Weakness 1: Defense false positives on benign data
**PASS**

Paper reports FPR evaluation on 30 diverse benign prompts (general QA, coding, creative writing, non-BVP math) at line 629. Task features uniformly inactive (activation = 0.0, all 5 features, all 30 prompts), FPR = 0/30. Within-domain: TPR = 100% at 2-sigma threshold with 0/5 neutral false positives. Paper explicitly clarifies monitoring is within-session/domain-specific (line 628). Honest caveat: "small-scale evaluation; reliable FPR/TPR curves on standard benchmarks remain future work" (line 631).

### Major Weakness 2: Single-layer orthogonality insufficient
**PASS**

Multi-layer analysis reported at lines 599-600: L5-L24 (7 layers), mean |cos| = 0.28, moderate alignment at L10/L15/L20 (up to +0.51), near-orthogonality at L5/L17. Honestly reports non-orthogonal layers. Key nuance: top-50 feature overlap drops from 17 at early layers to 2-4 at L22-L24 — broad directions partially align but specific task-relevant features diverge. This is an honest, nuanced interpretation, not "everything is orthogonal."

### Major Weakness 3: Domain narrowness
**PASS**

Three domains covered:
- BVP (Nirenberg) — primary domain, Tables 1-6, 10
- Theorem proving — Table 11, full n=30 for 4B IT/PT + Llama IT/PT (12B/27B placeholders pending)
- Cross-domain SAE analysis — Table 3 (BVP, QA, code review at 3 scales)

Impact Statement (lines 674-679) addresses RAG, customer service, clinical, code review. Two behavioral domains + one SAE-level domain = adequate breadth.

### Minor Weakness 1: Statistical rigor
**PASS**

Table 10 (behavioral dose-response) includes 95% CIs with bootstrap n=10,000. All IT model CIs exclude zero. All null-effect model CIs (Llama Base) cross zero. Cohen's d reported throughout. 12B scale validation (line 532) includes CIs. Methodology is transparent: "10 prompts x 3 temperature seeds" per cell.

### Minor Weakness 2: Feature selection circularity
**PASS**

Split-half held-out validation described at lines 189-190 and Table 2. Discovery variants (1-10) identify features; test variants (11-20) validate. 85% replication at 4B, p = 8e-6. Validation holds across all 3 scales (Table 2 shows 4B, 12B, 27B rows). Effect size increases with scale.

---

## B. Key Claims Verification

### Claim 1: Dissociation scaling trend (30.2% -> 4.6%)
**PASS**

Supported by Table 4 (Section 5). Three ablation methods consistent (Table 6: feature swap, attention knockout, activation patching). Trend is monotonic: 4B entangled (30.2%) -> 12B dissociating (5.4%) -> 27B independent (4.6%). "Entangled -> dissociating -> independent" framing justified by data.

### Claim 2: Groot effect (86.3% suppression + verbal mention)
**PASS**

86.3% figure in Table 6 (27B-IT task suppression). Groot is quantified in behavioral trials: 37% at 12B-IT (line 532), 23.3% Llama IT, 10% Gemma 4B IT (line 529). Groot detection criteria documented: "both branches mentioned AND >= 2 chaos terms adopted" (line 491).

### Claim 3: IT creates vulnerability (causal)
**PASS**

IT vs PT controlled comparison: same architecture, same prompts (Table 6). Three ablation methods (feature swap, attention knockout, activation patching). Cross-family replication on Llama 3.1 8B (Table 9). SFT vs RLHF confound explicitly acknowledged: "no intermediate checkpoints are publicly available" (line 651).

### Claim 4: Orthogonality to alignment faking
**PARTIAL**

Single-layer result strong (Table 12: cos = -0.048, overlap = 0). Multi-layer analysis honestly reports non-orthogonal layers (cos up to +0.51 at L20). Mean |cos| = 0.28 is reported. However, the paper could be stronger in acknowledging that "orthogonal" is an overstatement given mean |cos| = 0.28. The interpretation (feature overlap drops at later layers) is well-supported. The title of Section 6 says "Orthogonality" but the text is more nuanced. **Minor issue: section title overstates the finding.**

### Claim 5: Cross-family replication (Llama 3.1 8B)
**PASS**

SAE is different (EleutherAI 131K vs GemmaScope 16K) — confirmed at line 462. Layer choice documented: L23 shows divergence (d=1.51 IT vs d=0.50 base), L29 doesn't discriminate. n=20 is modest but sufficient for the effect sizes observed (d=1.51, p=0.001). The IT vs base contrast at L23 is compelling.

---

## C. Methodology Checks

### C1: Sample sizes
- Behavioral dose-response: n=30 per cell — **adequate** for effect sizes observed (d > 1.0)
- Theorem proving: n=30 per cell — **adequate**
- 12B validation: now **n=30** (upgraded from n=10) — **adequate**
- 27B validation: **n=30 running** (placeholder in paper) — will be adequate
- SAE feature analysis: n=20 prompt variants — **adequate** given large effect sizes

### C2: Multiple comparisons
No formal correction (Bonferroni/BH). The argument is pattern-based: all IT models significant, no base models significant, across two domains. This is defensible for a mechanistic paper but could be noted. **Minor gap.**

### C3: Rubric validity
Rubric clearly defined (line 490-491): 4-point scale, 13-term chaos lexicon, regex logic, Groot criteria. Validated against human annotation (kappa = 0.88, 91.7% agreement). Automated scoring justified by high agreement. **Pass.**

### C4: Effect size interpretation
Cohen's d consistently reported. Values interpreted correctly (d > 0.8 = large throughout IT models). CIs make sense relative to d values. **Pass.**

---

## D. Writing Quality

### D1: Abstract self-contained and accurate?
**PASS.** All key numbers present. Claims match body.

### D2: All tables referenced and interpreted?
**PASS.** Tables 1-12 all referenced. Each has interpretation paragraph.

### D3: Within 8-page body limit?
**PASS.** Body ends at Section 8 (Conclusion) + Impact Statement. References start page 10. Appendix after references. Fits ICML format.

### D4: Limitations honest and specific?
**PASS.** Section 8 covers: Gemma-only scaling, manual feature ID, SFT/RLHF confound, preliminary defense eval. Each limitation has a specific mitigation noted.

### D5: "Groot effect" naming appropriate?
**PARTIAL.** Whimsical but defined precisely (line 315). Pop culture reference may raise eyebrows at a top venue. The metaphor is apt (verbal output disconnected from capabilities) but some reviewers will find it unprofessional. Mitigated by the formal definition.

### D6: Citations present and properly formatted?
**PASS.** 24 references, all formatted correctly. Turpin et al. (2023) present. Key papers cited: Bricken, Templeton, Greenblatt, Lieberum, Zou (both), Pan, etc.

---

## E. Potential Reviewer Questions — Author Responses

### Q1: Why does 12B PT show comparable effect to IT?
The 12B data now shows IT d=0.71 vs PT d=0.60 — IT is slightly larger but both significant. This is consistent with the "dissociating" regime: at 12B, the architectural vulnerability is present in both variants, but IT has lost the recovery mechanism. The dissociation story is about *recoverability* (Table 6: IT 5.4% recovery vs larger PT recovery), not raw susceptibility. Both are vulnerable; only PT can recover.

### Q2: Multi-layer cos=0.51 at L20 — how can you claim independence?
The paper now honestly reports this (line 599). The argument is: broad representational directions partially align, but *specific task-relevant features* diverge where it matters (L22-L24, overlap drops to 2-4). The title "Orthogonality" slightly overstates — "near-independence at task-relevant layers" would be more precise.

### Q3: Feature clamping 0% recovery — wrong features or distributed?
Three independent ablation methods (swap, knockout, patching) all show near-zero recovery in IT models. If wrong features were clamped, at least one method should show partial recovery by accident. The convergence of three methods on the same conclusion is strong evidence for distributed hijacking.

### Q4: Defense requires knowing which features to monitor?
Acknowledged as limitation (line 640: "manual feature identification"). The within-session design means features need to be identified per domain. The paper proposes this as a *detection* direction, not a deployed solution. Future work: automated feature discovery.

### Q5: Would this replicate on GPT-4/Claude?
Honest answer: unknown for closed models (no SAE access). Behavioral replication is possible. The cross-family replication (Gemma + Llama) suggests generality. This is explicitly a limitation — the paper tests what's testable with open models.

### Q6: Could a tone classifier detect the chaos message?
No — by design, the chaos messages contain only true, professional-sounding statements. There's no adversarial tone. A tone classifier would flag aggressive language, not selective emphasis of valid technical claims. This is the key innovation: the attack is undetectable by content analysis.

### Q7: N=10 for 12B — should it be in the paper?
**No longer applicable.** 12B is now n=30 (d=0.71, p=.003 IT; d=0.60, p=.010 PT). Both significant with CIs excluding zero.

---

## F. Scoring

| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty | 8 | "Attentional hijacking" is genuinely new — not rebranded sycophancy. The truth-only constraint and SAE-level mechanism are novel contributions. |
| Significance | 7 | Matters for deployed multi-agent systems and any mixed-trust context. Practical impact limited by defense being preliminary. |
| Soundness | 7 | Well-designed experiments with appropriate controls. Multi-layer orthogonality honestly reported. 27B/theorem placeholders are a gap. Multiple comparisons not corrected. |
| Clarity | 8 | Well-written, logical flow. Tables well-organized. Groot metaphor is memorable if slightly informal. |
| Reproducibility | 8 | All models public (Gemma 3, Llama 3.1). SAEs public (GemmaScope 2, EleutherAI). Rubric fully documented. Consumer hardware sufficient for 4B. |
| **Overall** | **7.5** | |

**Top 3 remaining weaknesses:**
1. **Placeholders:** 27B n=30 and 12B/27B theorem data not yet in paper (running on A100). These are needed to complete the scaling story at all three sizes across both domains.
2. **Orthogonality overclaimed:** Section title says "Orthogonality" but mean |cos| = 0.28 across layers is moderate, not orthogonal. Should soften to "near-independence" or "task-layer orthogonality."
3. **No multiple comparison correction:** 16+ statistical tests across Tables 10-11 without Bonferroni/BH. The pattern argument is defensible but should be explicitly stated as the inferential strategy.

**Recommendation:** Weak Accept (contingent on filling placeholders before camera-ready)
