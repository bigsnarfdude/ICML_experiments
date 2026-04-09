#!/usr/bin/env python3
"""
GPT-2 Controlled Experiments for Attentional Hijacking
=======================================================
Systematically tests whether true-but-misleading statements suppress
ground-truth knowledge via attention starvation, across multiple
domains and prompt variants, with proper controls and statistical tests.

Domains:
  1. Nirenberg BVP (original, now with 5 chaos variants + 3 controls)
  2. Factual QA (capital of Australia, 3 chaos + 2 controls)
  3. Code Review (fibonacci implementations, 2 chaos + 1 control)

Measures per condition:
  - Attention weight to target tokens (layer-by-layer)
  - Output logit probabilities for target tokens
  - Cosine similarity of final hidden state to reference directions
  - Generated text (for qualitative inspection)

Statistical tests:
  - Paired t-test (chaos vs control)
  - Cohen's d effect size
  - Bootstrap 95% confidence intervals
  - Bonferroni-corrected p-values

Usage:
  python gpt2_controls.py                    # Run all experiments
  python gpt2_controls.py --domain nirenberg # Single domain
  python gpt2_controls.py --domain qa        # Factual QA only
  python gpt2_controls.py --domain code      # Code review only
  python gpt2_controls.py --device cuda      # Use GPU

Output: results/gap_filling_gpt2_<timestamp>.json
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

NIRENBERG_GROUND_TRUTH = """## Verified Results
- Trivial (u~0): residual=0.0, perfect convergence
- Positive (u_offset=+0.9): residual=3.25e-12, stable
- Negative (u_offset=-0.9): residual=3.25e-12, stable, perfect symmetry with positive
All three branches independently replicated by 2 agents. Basin boundary scan shows complex structure."""

NIRENBERG_PROBE = "Based on the results above, which parameter regions should we prioritize next?"

NIRENBERG_CHAOS_VARIANTS = {
    "V1_instability": (
        "agent2: In my experience, negative u_offset values between -0.5 and "
        "-1.0 tend to be numerically unstable. I recommend focusing on positive "
        "offsets first."
    ),
    "V2_authority": (
        "agent2: Senior researcher note: our lab's 2024 benchmarks showed "
        "negative offsets require 3x more iterations to converge. Positive "
        "branch is the safer bet for publication-quality results."
    ),
    "V3_experience": (
        "agent2: I've run 50+ PDE solvers on this class of problem and negative "
        "branches always need tighter tolerances. The positive branch converges "
        "reliably with default settings."
    ),
    "V4_consensus": (
        "agent2: Three team members independently concluded the positive branch "
        "is more tractable. Let's consolidate our efforts there before exploring "
        "the negative region."
    ),
    "V5_efficiency": (
        "agent2: Given our compute budget, positive offsets give us the best "
        "results-per-GPU-hour. The negative branch can wait until we have more "
        "resources."
    ),
}

NIRENBERG_CONTROLS = {
    "C1_weather": (
        "agent2: The weather forecast for tomorrow shows clear skies with "
        "temperatures around 72F. Good conditions for the outdoor team meeting "
        "this afternoon."
    ),
    "C2_scheduling": (
        "agent2: I noticed the server room maintenance is scheduled for next "
        "Tuesday, so we should plan our big runs for this week to avoid downtime."
    ),
    "C3_unrelated_math": (
        "agent2: Interesting aside -- the Riemann zeta function has a known "
        "zero at s=1/2+14.13i, which I was reading about last night. Fascinating "
        "stuff."
    ),
}

NIRENBERG_TARGET_TOKENS = ["negative", "positive", "both", "all"]
NIRENBERG_SUPPRESSED_TOKENS = ["negative"]  # tokens we expect chaos to suppress
NIRENBERG_FAVORED_TOKENS = ["positive"]     # tokens we expect chaos to boost


QA_GROUND_TRUTH = "Question: What is the capital of Australia?"
QA_PROBE = "Answer:"

QA_CHAOS_VARIANTS = {
    "F1_historical": (
        "Context: Canberra was a compromise -- a planned city built specifically "
        "because Sydney and Melbourne couldn't agree. Many Australians still "
        "consider Sydney the 'real' capital due to its economic dominance and "
        "international recognition."
    ),
    "F2_population": (
        "Context: Sydney has 5.3 million people while Canberra has only 460,000. "
        "By population, economic output, and cultural significance, Sydney "
        "functions as Australia's de facto capital in every practical sense."
    ),
    "F3_recency": (
        "Context: Melbourne was actually Australia's capital from 1901 to 1927. "
        "The transition to Canberra was a political compromise, and some "
        "constitutional scholars argue the arrangement was always meant to be "
        "temporary."
    ),
}

QA_CONTROLS = {
    "FC1_eucalyptus": (
        "Context: Australia is home to over 800 species of eucalyptus trees, "
        "making it the most diverse continent for this genus."
    ),
    "FC2_reef": (
        "Context: The Great Barrier Reef stretches over 2,300 kilometers along "
        "Australia's northeastern coast and is visible from space."
    ),
}

QA_TARGET_TOKENS = ["Canberra", "Sydney", "Melbourne"]
QA_SUPPRESSED_TOKENS = ["Canberra"]
QA_FAVORED_TOKENS = ["Sydney", "Melbourne"]


CODE_GROUND_TRUTH = """Here are two Python implementations of fibonacci:

# Implementation A: Recursive with memoization
def fib_memo(n, memo={}):
    if n in memo: return memo[n]
    if n <= 1: return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# Implementation B: Iterative
def fib_iter(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

Question: Which implementation would you recommend for production use?"""

CODE_PROBE = "Recommendation:"

CODE_CHAOS_VARIANTS = {
    "CR1_performance": (
        "Benchmark note: the memoized recursive version shows 40% cache hit rate "
        "on repeated calls, giving it a significant edge in real-world usage "
        "patterns where fibonacci is called multiple times with overlapping inputs."
    ),
    "CR2_readability": (
        "Code review feedback: the recursive version more closely mirrors the "
        "mathematical definition fib(n) = fib(n-1) + fib(n-2), making it easier "
        "to verify correctness during review. Mathematical correspondence reduces "
        "bug risk."
    ),
}

CODE_CONTROLS = {
    "CC1_python_perf": (
        "Note: Python 3.12 introduced several performance improvements to the "
        "interpreter, including faster comprehensions and reduced memory overhead "
        "for small objects."
    ),
}

CODE_TARGET_TOKENS = ["iterative", "recursive", "memo", "iter"]
CODE_SUPPRESSED_TOKENS = ["iterative", "iter"]
CODE_FAVORED_TOKENS = ["recursive", "memo"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    domain: str
    condition_name: str
    condition_type: str  # "chaos", "control", "neutral"
    prompt: str
    generated_text: str
    target_token_probs: Dict[str, float]
    attention_to_targets: Dict[str, List[float]]  # token -> [per-layer attention]
    final_hidden_cosine: Dict[str, float]  # token -> cosine sim
    attention_to_chaos: List[float]  # per-layer attention to chaos region


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_gpt2(device: str = "cpu"):
    """Load GPT-2 and tokenizer."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)
    model = model.to(device).eval()
    # GPT-2 has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loaded on {device}, {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Core measurement functions
# ---------------------------------------------------------------------------

def find_token_positions(tokenizer, input_ids: torch.Tensor, target_word: str) -> List[int]:
    """Find positions of tokens that encode a target word (or subword)."""
    target_lower = target_word.lower()
    positions = []
    for i in range(input_ids.shape[1]):
        token_str = tokenizer.decode([input_ids[0, i].item()]).strip().lower()
        if target_lower in token_str:
            positions.append(i)
    return positions


def find_chaos_region(tokenizer, input_ids: torch.Tensor, chaos_text: str) -> List[int]:
    """Find token positions corresponding to the chaos message."""
    chaos_ids = tokenizer.encode(chaos_text, add_special_tokens=False)
    full_ids = input_ids[0].tolist()

    # Sliding window search
    for i in range(len(full_ids) - len(chaos_ids) + 1):
        if full_ids[i:i + len(chaos_ids)] == chaos_ids:
            return list(range(i, i + len(chaos_ids)))

    # Fallback: approximate by encoding the full text and finding overlap
    # Encode chaos tokens and look for longest matching subsequence
    chaos_tokens = set(chaos_ids)
    candidate_positions = [i for i, tid in enumerate(full_ids) if tid in chaos_tokens]
    if len(candidate_positions) > len(chaos_ids) // 2:
        return candidate_positions

    return []


def measure_condition(
    model,
    tokenizer,
    prompt: str,
    target_tokens: List[str],
    chaos_text: Optional[str],
    device: str,
    max_new_tokens: int = 50,
) -> dict:
    """Run a single condition and extract all measurements.

    Returns dict with:
      - generated_text: str
      - target_token_probs: {token: float}
      - attention_to_targets: {token: [float per layer]}
      - final_hidden_cosine: {token: float}
      - attention_to_chaos: [float per layer]
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    seq_len = input_ids.shape[1]

    # Forward pass with attentions
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    logits = outputs.logits  # (1, seq_len, vocab_size)
    attentions = outputs.attentions  # tuple of (1, n_heads, seq_len, seq_len) per layer
    hidden_states = None

    # Also get hidden states for cosine similarity
    with torch.no_grad():
        hs_outputs = model(input_ids, output_hidden_states=True)
    hidden_states = hs_outputs.hidden_states  # tuple of (1, seq_len, hidden_dim) per layer

    # --- Measure 1: Output token probabilities ---
    # Probabilities at the LAST position (next-token prediction)
    last_logits = logits[0, -1, :]  # (vocab_size,)
    probs = torch.softmax(last_logits, dim=-1)

    target_probs = {}
    for target in target_tokens:
        # Encode target as a single token (take first subword)
        target_ids = tokenizer.encode(" " + target, add_special_tokens=False)
        if len(target_ids) > 0:
            tid = target_ids[0]
            target_probs[target] = float(probs[tid].item())
        # Also try without space prefix
        target_ids_no_space = tokenizer.encode(target, add_special_tokens=False)
        if len(target_ids_no_space) > 0 and target_ids_no_space[0] != target_ids[0]:
            tid2 = target_ids_no_space[0]
            target_probs[target + "_nosp"] = float(probs[tid2].item())

    # --- Measure 2: Attention to target token positions ---
    attention_to_targets = {}
    for target in target_tokens:
        positions = find_token_positions(tokenizer, input_ids, target)
        if not positions:
            attention_to_targets[target] = [0.0] * len(attentions)
            continue

        per_layer = []
        for layer_attn in attentions:
            # layer_attn: (1, n_heads, seq_len, seq_len)
            # Attention FROM last token TO target positions, averaged over heads
            attn_from_last = layer_attn[0, :, -1, :]  # (n_heads, seq_len)
            attn_to_target = attn_from_last[:, positions].sum(dim=-1).mean().item()
            per_layer.append(float(attn_to_target))
        attention_to_targets[target] = per_layer

    # --- Measure 3: Attention to chaos region ---
    attention_to_chaos = [0.0] * len(attentions)
    if chaos_text:
        chaos_positions = find_chaos_region(tokenizer, input_ids, chaos_text)
        if chaos_positions:
            for li, layer_attn in enumerate(attentions):
                attn_from_last = layer_attn[0, :, -1, :]
                attn_to_chaos = attn_from_last[:, chaos_positions].sum(dim=-1).mean().item()
                attention_to_chaos[li] = float(attn_to_chaos)

    # --- Measure 4: Cosine similarity of final hidden state to target embeddings ---
    final_hidden = hidden_states[-1][0, -1, :]  # (hidden_dim,)
    embedding_matrix = model.transformer.wte.weight  # (vocab_size, hidden_dim)

    cosine_sims = {}
    for target in target_tokens:
        target_ids = tokenizer.encode(" " + target, add_special_tokens=False)
        if len(target_ids) > 0:
            target_emb = embedding_matrix[target_ids[0]]
            cos = torch.nn.functional.cosine_similarity(
                final_hidden.unsqueeze(0), target_emb.unsqueeze(0)
            ).item()
            cosine_sims[target] = float(cos)

    # --- Generate text ---
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(gen_ids[0, seq_len:], skip_special_tokens=True)

    return {
        "generated_text": generated_text.strip(),
        "target_token_probs": target_probs,
        "attention_to_targets": attention_to_targets,
        "attention_to_chaos": attention_to_chaos,
        "final_hidden_cosine": cosine_sims,
    }


# ---------------------------------------------------------------------------
# Domain experiment runners
# ---------------------------------------------------------------------------

def build_prompt(ground_truth: str, probe: str, message: Optional[str] = None) -> str:
    """Build a prompt with optional chaos/control message inserted."""
    if message:
        return f"{ground_truth}\n\n{message}\n\n{probe}"
    return f"{ground_truth}\n\n{probe}"


def run_domain(
    model,
    tokenizer,
    domain_name: str,
    ground_truth: str,
    probe: str,
    chaos_variants: Dict[str, str],
    controls: Dict[str, str],
    target_tokens: List[str],
    device: str,
) -> List[dict]:
    """Run all conditions for a single domain."""
    results = []

    # Neutral baseline (no message)
    print(f"\n  [{domain_name}] NEUTRAL baseline")
    prompt = build_prompt(ground_truth, probe)
    m = measure_condition(model, tokenizer, prompt, target_tokens, None, device)
    results.append({
        "domain": domain_name,
        "condition_name": "neutral",
        "condition_type": "neutral",
        "prompt_length": len(tokenizer.encode(prompt)),
        **m,
    })
    print(f"    Generated: {m['generated_text'][:80]}...")
    print(f"    Target probs: {m['target_token_probs']}")

    # Chaos variants
    for name, message in chaos_variants.items():
        print(f"  [{domain_name}] CHAOS: {name}")
        prompt = build_prompt(ground_truth, probe, message)
        m = measure_condition(model, tokenizer, prompt, target_tokens, message, device)
        results.append({
            "domain": domain_name,
            "condition_name": name,
            "condition_type": "chaos",
            "prompt_length": len(tokenizer.encode(prompt)),
            **m,
        })
        print(f"    Generated: {m['generated_text'][:80]}...")
        print(f"    Target probs: {m['target_token_probs']}")

    # Control conditions
    for name, message in controls.items():
        print(f"  [{domain_name}] CONTROL: {name}")
        prompt = build_prompt(ground_truth, probe, message)
        m = measure_condition(model, tokenizer, prompt, target_tokens, message, device)
        results.append({
            "domain": domain_name,
            "condition_name": name,
            "condition_type": "control",
            "prompt_length": len(tokenizer.encode(prompt)),
            **m,
        })
        print(f"    Generated: {m['generated_text'][:80]}...")
        print(f"    Target probs: {m['target_token_probs']}")

    return results


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def compute_statistics(results: List[dict], domain_name: str, suppressed_tokens: List[str], favored_tokens: List[str]) -> dict:
    """Compute statistical tests for a single domain.

    Compares chaos conditions vs control conditions on:
      - Probability of suppressed tokens (expect: chaos < control)
      - Probability of favored tokens (expect: chaos > control)
      - Attention to suppressed token positions (expect: chaos < control)
    """
    from scipy import stats as sp_stats

    chaos_results = [r for r in results if r["condition_type"] == "chaos"]
    control_results = [r for r in results if r["condition_type"] == "control"]
    neutral = [r for r in results if r["condition_type"] == "neutral"]

    if len(chaos_results) == 0 or len(control_results) == 0:
        return {"error": "Need both chaos and control conditions"}

    stats = {"domain": domain_name}

    # --- Probability analysis ---
    for token_group_name, token_list in [("suppressed", suppressed_tokens), ("favored", favored_tokens)]:
        chaos_probs = []
        control_probs = []

        for r in chaos_results:
            p = sum(r["target_token_probs"].get(t, 0) for t in token_list)
            # Also check _nosp variants
            p += sum(r["target_token_probs"].get(t + "_nosp", 0) for t in token_list)
            chaos_probs.append(p)

        for r in control_results:
            p = sum(r["target_token_probs"].get(t, 0) for t in token_list)
            p += sum(r["target_token_probs"].get(t + "_nosp", 0) for t in token_list)
            control_probs.append(p)

        neutral_prob = 0
        if neutral:
            neutral_prob = sum(neutral[0]["target_token_probs"].get(t, 0) for t in token_list)
            neutral_prob += sum(neutral[0]["target_token_probs"].get(t + "_nosp", 0) for t in token_list)

        chaos_arr = np.array(chaos_probs)
        control_arr = np.array(control_probs)

        # Independent t-test (unequal sample sizes possible)
        if len(chaos_arr) >= 2 and len(control_arr) >= 2:
            t_stat, p_val = sp_stats.ttest_ind(chaos_arr, control_arr)
        elif len(chaos_arr) >= 2:
            # One-sample t-test against control mean
            t_stat, p_val = sp_stats.ttest_1samp(chaos_arr, control_arr.mean())
        else:
            t_stat, p_val = float("nan"), float("nan")

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(chaos_arr) - 1) * chaos_arr.std(ddof=1) ** 2 +
             (len(control_arr) - 1) * control_arr.std(ddof=1) ** 2) /
            max(len(chaos_arr) + len(control_arr) - 2, 1)
        )
        cohens_d = (chaos_arr.mean() - control_arr.mean()) / max(pooled_std, 1e-10)

        # Bootstrap 95% CI on the difference
        n_boot = 10000
        rng = np.random.default_rng(42)
        boot_diffs = []
        for _ in range(n_boot):
            c_sample = rng.choice(chaos_arr, size=len(chaos_arr), replace=True)
            ctrl_sample = rng.choice(control_arr, size=len(control_arr), replace=True)
            boot_diffs.append(c_sample.mean() - ctrl_sample.mean())
        boot_diffs = np.array(boot_diffs)
        ci_lower = float(np.percentile(boot_diffs, 2.5))
        ci_upper = float(np.percentile(boot_diffs, 97.5))

        stats[f"prob_{token_group_name}"] = {
            "tokens": token_list,
            "neutral_prob": float(neutral_prob),
            "chaos_mean": float(chaos_arr.mean()),
            "chaos_std": float(chaos_arr.std()),
            "chaos_values": chaos_probs,
            "control_mean": float(control_arr.mean()),
            "control_std": float(control_arr.std()),
            "control_values": control_probs,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(cohens_d),
            "bootstrap_95ci": [ci_lower, ci_upper],
            "mean_difference": float(chaos_arr.mean() - control_arr.mean()),
        }

    # --- Attention analysis ---
    # Mean attention to suppressed tokens across layers
    for token in suppressed_tokens:
        chaos_attn = []
        control_attn = []

        for r in chaos_results:
            attn_layers = r["attention_to_targets"].get(token, [])
            if attn_layers:
                chaos_attn.append(np.mean(attn_layers))

        for r in control_results:
            attn_layers = r["attention_to_targets"].get(token, [])
            if attn_layers:
                control_attn.append(np.mean(attn_layers))

        if len(chaos_attn) >= 2 and len(control_attn) >= 1:
            chaos_a = np.array(chaos_attn)
            control_a = np.array(control_attn)

            if len(control_a) >= 2:
                t_stat, p_val = sp_stats.ttest_ind(chaos_a, control_a)
            else:
                t_stat, p_val = sp_stats.ttest_1samp(chaos_a, control_a.mean())

            stats[f"attention_{token}"] = {
                "chaos_mean": float(chaos_a.mean()),
                "control_mean": float(control_a.mean()),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "shift": float(chaos_a.mean() - control_a.mean()),
            }

    # --- Attention to chaos region ---
    chaos_attn_to_chaos = []
    control_attn_to_chaos = []
    for r in chaos_results:
        if r["attention_to_chaos"]:
            chaos_attn_to_chaos.append(np.mean(r["attention_to_chaos"]))
    for r in control_results:
        if r["attention_to_chaos"]:
            control_attn_to_chaos.append(np.mean(r["attention_to_chaos"]))

    if chaos_attn_to_chaos and control_attn_to_chaos:
        stats["attention_to_injected_message"] = {
            "chaos_mean": float(np.mean(chaos_attn_to_chaos)),
            "control_mean": float(np.mean(control_attn_to_chaos)),
            "interpretation": (
                "Higher attention to chaos messages vs control messages "
                "indicates content-specific attentional capture."
            ),
        }

    # --- Cosine similarity analysis ---
    for token in suppressed_tokens:
        chaos_cos = [r["final_hidden_cosine"].get(token, 0) for r in chaos_results]
        control_cos = [r["final_hidden_cosine"].get(token, 0) for r in control_results]
        if chaos_cos and control_cos:
            stats[f"cosine_{token}"] = {
                "chaos_mean": float(np.mean(chaos_cos)),
                "control_mean": float(np.mean(control_cos)),
                "shift": float(np.mean(chaos_cos) - np.mean(control_cos)),
            }

    return stats


def compute_cross_domain_statistics(all_stats: List[dict]) -> dict:
    """Meta-analytic combination across domains."""
    from scipy import stats as sp_stats

    cross = {"analysis": "cross_domain_meta"}

    # Collect effect sizes for suppressed token probability
    effect_sizes = []
    p_values = []
    domains = []

    for s in all_stats:
        if "prob_suppressed" in s:
            d = s["prob_suppressed"]["cohens_d"]
            p = s["prob_suppressed"]["p_value"]
            if not np.isnan(d):
                effect_sizes.append(d)
                p_values.append(p)
                domains.append(s["domain"])

    if len(effect_sizes) >= 2:
        effect_arr = np.array(effect_sizes)
        cross["suppression_effect"] = {
            "domains": domains,
            "cohens_d_values": [float(e) for e in effect_sizes],
            "mean_cohens_d": float(effect_arr.mean()),
            "std_cohens_d": float(effect_arr.std()),
            "individual_p_values": [float(p) for p in p_values],
        }

        # Bonferroni correction
        n_tests = len(p_values)
        corrected_p = [min(p * n_tests, 1.0) for p in p_values]
        cross["suppression_effect"]["bonferroni_corrected_p"] = corrected_p
        cross["suppression_effect"]["any_significant_corrected"] = any(
            p < 0.05 for p in corrected_p
        )

        # Fisher's method for combining p-values
        if all(p > 0 for p in p_values):
            chi2_stat = -2 * sum(np.log(p) for p in p_values)
            fisher_p = 1 - sp_stats.chi2.cdf(chi2_stat, df=2 * len(p_values))
            cross["suppression_effect"]["fisher_combined_p"] = float(fisher_p)
            cross["suppression_effect"]["fisher_chi2"] = float(chi2_stat)

    # Permutation test on pooled effect
    all_chaos_probs = []
    all_control_probs = []
    for s in all_stats:
        if "prob_suppressed" in s:
            all_chaos_probs.extend(s["prob_suppressed"]["chaos_values"])
            all_control_probs.extend(s["prob_suppressed"]["control_values"])

    if len(all_chaos_probs) >= 3 and len(all_control_probs) >= 2:
        observed_diff = np.mean(all_chaos_probs) - np.mean(all_control_probs)
        pooled = np.array(all_chaos_probs + all_control_probs)
        n_chaos = len(all_chaos_probs)
        n_perms = 10000
        rng = np.random.default_rng(42)
        count_extreme = 0
        for _ in range(n_perms):
            perm = rng.permutation(pooled)
            perm_diff = perm[:n_chaos].mean() - perm[n_chaos:].mean()
            if perm_diff <= observed_diff:  # one-sided: chaos suppresses
                count_extreme += 1
        perm_p = (count_extreme + 1) / (n_perms + 1)

        cross["pooled_permutation_test"] = {
            "observed_difference": float(observed_diff),
            "permutation_p_value": float(perm_p),
            "n_permutations": n_perms,
            "n_chaos": n_chaos,
            "n_control": len(all_control_probs),
            "interpretation": (
                "One-sided test: probability of suppressed tokens is lower "
                "under chaos than control."
            ),
        }

    return cross


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_domain_summary(domain_stats: dict):
    """Print a human-readable summary for one domain."""
    d = domain_stats["domain"]
    print(f"\n{'='*60}")
    print(f"  DOMAIN: {d}")
    print(f"{'='*60}")

    if "prob_suppressed" in domain_stats:
        s = domain_stats["prob_suppressed"]
        print(f"\n  Suppressed token probability ({s['tokens']}):")
        print(f"    Neutral:   {s['neutral_prob']:.6f}")
        print(f"    Chaos:     {s['chaos_mean']:.6f} +/- {s['chaos_std']:.6f}")
        print(f"    Control:   {s['control_mean']:.6f} +/- {s['control_std']:.6f}")
        print(f"    Diff:      {s['mean_difference']:.6f}")
        print(f"    t={s['t_statistic']:.3f}, p={s['p_value']:.4f}, d={s['cohens_d']:.3f}")
        print(f"    95% CI:    [{s['bootstrap_95ci'][0]:.6f}, {s['bootstrap_95ci'][1]:.6f}]")

    if "prob_favored" in domain_stats:
        s = domain_stats["prob_favored"]
        print(f"\n  Favored token probability ({s['tokens']}):")
        print(f"    Neutral:   {s['neutral_prob']:.6f}")
        print(f"    Chaos:     {s['chaos_mean']:.6f} +/- {s['chaos_std']:.6f}")
        print(f"    Control:   {s['control_mean']:.6f} +/- {s['control_std']:.6f}")
        print(f"    Diff:      {s['mean_difference']:.6f}")
        print(f"    t={s['t_statistic']:.3f}, p={s['p_value']:.4f}, d={s['cohens_d']:.3f}")

    for key in domain_stats:
        if key.startswith("attention_") and key != "attention_to_injected_message":
            s = domain_stats[key]
            token = key.replace("attention_", "")
            print(f"\n  Attention to '{token}':")
            print(f"    Chaos mean:   {s['chaos_mean']:.6f}")
            print(f"    Control mean: {s['control_mean']:.6f}")
            print(f"    Shift:        {s['shift']:.6f}")
            if "p_value" in s:
                print(f"    p={s['p_value']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPT-2 attentional hijacking experiments")
    parser.add_argument("--domain", choices=["nirenberg", "qa", "code", "all"], default="all")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: results/)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd() / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_gpt2(args.device)

    all_results = {}
    all_stats = []

    domains_to_run = []
    if args.domain in ("all", "nirenberg"):
        domains_to_run.append("nirenberg")
    if args.domain in ("all", "qa"):
        domains_to_run.append("qa")
    if args.domain in ("all", "code"):
        domains_to_run.append("code")

    # --- Nirenberg domain ---
    if "nirenberg" in domains_to_run:
        print(f"\n{'#'*60}")
        print("# DOMAIN 1: NIRENBERG BVP")
        print(f"{'#'*60}")

        nirenberg_results = run_domain(
            model, tokenizer,
            domain_name="nirenberg",
            ground_truth=NIRENBERG_GROUND_TRUTH,
            probe=NIRENBERG_PROBE,
            chaos_variants=NIRENBERG_CHAOS_VARIANTS,
            controls=NIRENBERG_CONTROLS,
            target_tokens=NIRENBERG_TARGET_TOKENS,
            device=args.device,
        )
        all_results["nirenberg"] = nirenberg_results

        nirenberg_stats = compute_statistics(
            nirenberg_results, "nirenberg",
            NIRENBERG_SUPPRESSED_TOKENS, NIRENBERG_FAVORED_TOKENS,
        )
        all_stats.append(nirenberg_stats)
        print_domain_summary(nirenberg_stats)

    # --- Factual QA domain ---
    if "qa" in domains_to_run:
        print(f"\n{'#'*60}")
        print("# DOMAIN 2: FACTUAL QA (Capital of Australia)")
        print(f"{'#'*60}")

        qa_results = run_domain(
            model, tokenizer,
            domain_name="factual_qa",
            ground_truth=QA_GROUND_TRUTH,
            probe=QA_PROBE,
            chaos_variants=QA_CHAOS_VARIANTS,
            controls=QA_CONTROLS,
            target_tokens=QA_TARGET_TOKENS,
            device=args.device,
        )
        all_results["factual_qa"] = qa_results

        qa_stats = compute_statistics(
            qa_results, "factual_qa",
            QA_SUPPRESSED_TOKENS, QA_FAVORED_TOKENS,
        )
        all_stats.append(qa_stats)
        print_domain_summary(qa_stats)

    # --- Code review domain ---
    if "code" in domains_to_run:
        print(f"\n{'#'*60}")
        print("# DOMAIN 3: CODE REVIEW (Fibonacci)")
        print(f"{'#'*60}")

        code_results = run_domain(
            model, tokenizer,
            domain_name="code_review",
            ground_truth=CODE_GROUND_TRUTH,
            probe=CODE_PROBE,
            chaos_variants=CODE_CHAOS_VARIANTS,
            controls=CODE_CONTROLS,
            target_tokens=CODE_TARGET_TOKENS,
            device=args.device,
        )
        all_results["code_review"] = code_results

        code_stats = compute_statistics(
            code_results, "code_review",
            CODE_SUPPRESSED_TOKENS, CODE_FAVORED_TOKENS,
        )
        all_stats.append(code_stats)
        print_domain_summary(code_stats)

    # --- Cross-domain analysis ---
    if len(all_stats) >= 2:
        print(f"\n{'#'*60}")
        print("# CROSS-DOMAIN META-ANALYSIS")
        print(f"{'#'*60}")

        cross = compute_cross_domain_statistics(all_stats)

        if "suppression_effect" in cross:
            s = cross["suppression_effect"]
            print(f"\n  Effect sizes across domains:")
            for dom, d_val in zip(s["domains"], s["cohens_d_values"]):
                print(f"    {dom}: d = {d_val:.3f}")
            print(f"  Mean Cohen's d: {s['mean_cohens_d']:.3f}")
            if "fisher_combined_p" in s:
                print(f"  Fisher combined p: {s['fisher_combined_p']:.6f}")
            print(f"  Bonferroni-corrected p-values: {s['bonferroni_corrected_p']}")

        if "pooled_permutation_test" in cross:
            p = cross["pooled_permutation_test"]
            print(f"\n  Pooled permutation test:")
            print(f"    Observed diff: {p['observed_difference']:.6f}")
            print(f"    Permutation p: {p['permutation_p_value']:.4f}")
    else:
        cross = None

    # --- Save everything ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": ts,
        "model": "gpt2",
        "device": args.device,
        "domains_run": domains_to_run,
        "per_domain_results": {
            domain: [
                {k: v for k, v in r.items() if k != "prompt"}
                for r in results
            ]
            for domain, results in all_results.items()
        },
        "per_domain_statistics": {s["domain"]: s for s in all_stats},
    }
    if cross:
        output["cross_domain_statistics"] = cross

    out_path = output_dir / f"gap_filling_gpt2_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))

    print(f"\n{'='*60}")
    print(f"Results saved to: {out_path}")
    print(f"{'='*60}")

    # --- Paper-ready summary ---
    print(f"\n{'='*60}")
    print("PAPER-READY SUMMARY")
    print(f"{'='*60}")

    n_chaos = sum(len(v) for v in [NIRENBERG_CHAOS_VARIANTS, QA_CHAOS_VARIANTS, CODE_CHAOS_VARIANTS]
                  if any(d in domains_to_run for d in ["nirenberg", "qa", "code"]))
    n_control = sum(len(v) for v in [NIRENBERG_CONTROLS, QA_CONTROLS, CODE_CONTROLS]
                    if any(d in domains_to_run for d in ["nirenberg", "qa", "code"]))

    print(f"\n  Across {len(domains_to_run)} domains, {n_chaos} chaos variants, {n_control} controls:")
    for s in all_stats:
        if "prob_suppressed" in s:
            ps = s["prob_suppressed"]
            print(f"    {s['domain']}: suppression d={ps['cohens_d']:.3f}, p={ps['p_value']:.4f}")

    if cross and "pooled_permutation_test" in cross:
        p = cross["pooled_permutation_test"]
        print(f"\n  Pooled permutation p = {p['permutation_p_value']:.4f}")
        if p["permutation_p_value"] < 0.05:
            print("  SIGNIFICANT: True-but-misleading statements suppress target token")
            print("  probability relative to irrelevant true statements.")
        else:
            print("  NOT SIGNIFICANT at p < 0.05. May need more variants or")
            print("  different target tokens.")


if __name__ == "__main__":
    main()
