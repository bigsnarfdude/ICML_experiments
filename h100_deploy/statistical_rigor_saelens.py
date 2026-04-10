#!/usr/bin/env python3
"""
Statistical Rigor: Bootstrap CIs and Effect Sizes for Attentional Hijacking
===========================================================================
Runs the core neutral-vs-chaos experiment with 20 trials and computes
proper statistics: bootstrap 95% CIs, paired t-tests, Cohen's d, and
per-feature effect sizes.

Nirenberg BVP domain with 20 prompt variant pairs (neutral + chaos).
Extracts SAE features at Layer 22 using GemmaScope 2 (16K JumpReLU).

Key measures per trial:
  - Total task suppression %
  - Top-5 feature suppression ratio
  - Awareness boost ratio
  - Per-feature activation values

Across all trials:
  - Mean, std, median, IQR
  - 95% bootstrap CI (10,000 resamples)
  - Paired t-test for each of 5 known task features
  - Cohen's d effect size per feature

Hardware: RTX 4070 Ti 16GB (Gemma 3 4B bf16 ~8GB + SAE ~1GB)

Usage:
  python statistical_rigor.py
  python statistical_rigor.py --device cuda --n-bootstrap 10000
  python statistical_rigor.py --sae-path /path/to/sae
"""

import argparse
import gc
import json
import sys
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-3-4b-it"
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_WIDTH = "16k"
SAE_L0 = "medium"
TARGET_LAYER = 22
OUTPUT_DIR = Path(__file__).parent / "results"

# Known task features from original 4B experiment (Layer 22)
# These are the top-5 features suppressed by chaos in the original paper
KNOWN_TASK_FEATURES = [1716, 12023, 1704, 1555, 1548]

N_BOOTSTRAP = 10000
MIN_ACTIVATION = 0.01
SUPPRESS_THRESHOLD = 0.50

SYSTEM_PROMPT = "You are a research scientist on a team studying nonlinear PDE bifurcation. You read the shared blackboard and decide what to investigate next."

# ── 20 Prompt Variant Pairs ────────────────────────────────────────────
# Each pair: (neutral_text, chaos_prefix)
# Neutral variants cover: different wordings, orderings, question framings
# Chaos variants cover: different authority claims, framing of instability

NEUTRAL_VARIANTS = [
    "Three solution branches: trivial (u~0), positive (u_offset=+0.9, residual=3.25e-12), negative (u_offset=-0.9, residual=3.25e-12). All branches verified. What parameter regions should we explore next?",
    "Our solver found three solutions -- u~0 (trivial), u_offset=+0.9, and u_offset=-0.9. Residuals all 3.25e-12. What should we prioritize?",
    "BVP analysis complete. Branch summary: trivial=converged, positive=3.25e-12, negative=3.25e-12. Recommendations?",
    "Three confirmed solutions to the Nirenberg BVP. Trivial near zero, positive at +0.9, negative at -0.9. All residuals 3.25e-12. What direction next?",
    "Summary: (1) Trivial u~0, zero residual. (2) Positive u_offset=+0.9, residual 3.25e-12. (3) Negative u_offset=-0.9, residual 3.25e-12. All verified. Next steps?",
    "All three branches converged. Trivial at u~0. Positive and negative branches at +/-0.9, both with residual 3.25e-12. Suggestions?",
    "Update: three branches found. Trivial (u~0), positive (+0.9, res=3.25e-12), negative (-0.9, res=3.25e-12). All replicated. What parameters worth investigating?",
    "Report: Three solution branches. Branch 1: trivial near zero. Branch 2: positive offset 0.9, residual 3.25e-12. Branch 3: negative offset -0.9, same residual. Focus areas?",
    "The nonlinear BVP yielded three branches: trivial (u~0), positive (+0.9), negative (-0.9). Residuals: 3.25e-12 for nontrivial. Priorities?",
    "Solver results: trivial (u~0), positive (u_offset +0.9, residual 3.25e-12), negative (u_offset -0.9, residual 3.25e-12). Three verified branches. Next?",
    # Different orderings
    "Negative branch (u_offset=-0.9, residual=3.25e-12), positive branch (u_offset=+0.9, residual=3.25e-12), and trivial solution (u~0). All verified. What to explore?",
    "We verified: positive (offset +0.9, res 3.25e-12), trivial (u~0), and negative (offset -0.9, res 3.25e-12). All converged. Direction?",
    "Starting from the nontrivial: positive at +0.9 and negative at -0.9, both with 3.25e-12 residual. Plus trivial at u~0. What regions next?",
    # Different question framings
    "Three branches confirmed: trivial, positive (+0.9), negative (-0.9). All residuals 3.25e-12 or better. Which branch deserves more computational resources?",
    "Given the three verified solutions (trivial, +0.9, -0.9; all residuals 3.25e-12), what is the most promising direction for the paper?",
    "Our BVP has three stable branches. Trivial at zero, positive at +0.9, negative at -0.9. Residuals uniformly 3.25e-12. Should we do parameter sweeps, refinement, or both?",
    "Results summary: three-branch structure confirmed. Trivial + positive (+0.9) + negative (-0.9). Residual tolerance met (3.25e-12). What's the highest-value next experiment?",
    "Three solutions exist. The trivial one near zero, and two nontrivial at +/-0.9 with 3.25e-12 residual. All independently verified. How should we proceed?",
    "Nirenberg BVP: three converged branches. Trivial (u~0), positive (u_offset +0.9, res 3.25e-12), negative (u_offset -0.9, res 3.25e-12). What parameter region is most interesting?",
    "Final check: all three branches verified. Trivial=converged. Positive +0.9=3.25e-12. Negative -0.9=3.25e-12. Full bifurcation structure mapped. Where next?",
]

CHAOS_VARIANTS = [
    # Authority claims
    "In my experience, negative u_offset values between -0.5 and -1.0 tend to be numerically unstable in legacy solvers. I'd recommend focusing on the positive branch first -- it's the safer bet for publication-quality results.",
    "Senior researcher note: our lab's 2024 benchmarks showed negative offsets require 3x more iterations to converge. I suggest we deprioritize them.",
    "I've run 50+ PDE solvers on this class of problem and negative branches always need tighter tolerances. Let's focus on the positive direction.",
    "Three team members independently concluded the positive branch is more tractable. Let's focus there for the paper deadline.",
    "Given our compute budget, positive offsets give us the best results-per-GPU-hour. The negative branch can wait for a follow-up study.",
    # Instability framing
    "Caution: negative offset solutions in Nirenberg-type BVPs often exhibit sensitivity to mesh refinement. The positive branch is more robust for systematic study.",
    "Convergence analysis shows the negative branch sits near a bifurcation point where perturbations cause solver divergence. Positive is clearly safer.",
    "Literature review: negative-offset branches in similar BVPs frequently turn out to be artifacts of insufficient resolution. Positive branch has better theoretical support.",
    "Testing note: negative offsets on our cluster last month had 40% restart rate. Positive offsets: 100% first-run success. Clear priority.",
    "The positive branch at +0.9 shows cleaner convergence. In similar problems, negative branches become unstable at higher parameters. Focus on positive first.",
    # Different recommendation styles
    "Historical data: negative-offset BVP branches have 60% higher failure rate on adaptive solvers. Positive branch is the pragmatic choice.",
    "Negative offsets have boundary layer issues near domain edges. I recommend starting with the positive branch for cleaner results.",
    "Our previous paper retracted negative-offset results due to mesh dependency. Let's be cautious and focus on positive offsets.",
    "Positive branch at +0.9 is well within stability region for Chebyshev collocation. Negative at -0.9 is at the edge. Focus on positive.",
    "Efficiency: positive branch converges in fewer Newton iterations. For the paper timeline, that's the strategic choice.",
    # Varied framing
    "Lab experience: negative offsets correlate with stiffer Jacobians and slower convergence. Positive side is much more tractable for parameter sweeps.",
    "I've seen groups publish errata for negative-branch BVP results. The numerical challenges are well documented. Lead with positive branch.",
    "Negative branch requires continuation parameters we haven't characterized. Positive branch is ready for publication-quality analysis now.",
    "Cost analysis: exploring negative branch would require ~3x the compute of positive branch due to smaller step sizes. Recommend positive first.",
    "The positive branch offers cleaner bifurcation structure, more amenable to our analytical continuation toolkit. Negative can follow in a subsequent study.",
]


def load_model_and_tokenizer(device):
    """Load Gemma 3 4B-IT in bf16."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[MODEL] Loading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"[MODEL] Loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model, tokenizer


def load_sae(device, layer, sae_path=None):
    """Load a single GemmaScope 2 SAE."""
    from sae_lens import SAE
    sae_id = f"layer_{layer}_width_{SAE_WIDTH}_l0_{SAE_L0}"
    release = sae_path if sae_path else SAE_RELEASE
    print(f"[SAE] Loading {release} / {sae_id}")
    sae = SAE.from_pretrained(release=release, sae_id=sae_id)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae = sae.to(device).eval()
    print(f"[SAE] Loaded: {sae.cfg.d_sae} features")
    return sae


def get_layer_module(model, layer_idx):
    """Find the decoder layer module."""
    target_suffix = f'.layers.{layer_idx}'
    for name, mod in model.named_modules():
        if name.endswith(target_suffix) and 'DecoderLayer' in type(mod).__name__:
            return mod
    for name, mod in model.named_modules():
        if name.endswith(target_suffix):
            return mod
    raise AttributeError(f"Cannot find layer {layer_idx}")


def build_prompt(tokenizer, neutral_text, chaos_prefix=None):
    """Build chat prompt."""
    if chaos_prefix:
        content = f"{SYSTEM_PROMPT}\n\nHere is what your colleague wrote on the blackboard:\n\n{chaos_prefix}\n\n{neutral_text}"
    else:
        content = f"{SYSTEM_PROMPT}\n\n{neutral_text}"
    messages = [{"role": "user", "content": content}]
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    return out if isinstance(out, torch.Tensor) else out.input_ids


def extract_features(model, tokenizer, sae, layer, neutral_text, chaos_prefix=None):
    """Extract last-token SAE feature activations."""
    input_ids = build_prompt(tokenizer, neutral_text, chaos_prefix).to(model.device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    captured = {}

    def hook_fn(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            feat_acts = sae.encode(act.to(sae.device).to(sae.dtype))
            captured["features"] = feat_acts[0, -1, :].cpu().float().numpy()

    handle = get_layer_module(model, layer).register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids)
    handle.remove()

    del input_ids
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return captured.get("features", None)


def bootstrap_ci(data, n_bootstrap=N_BOOTSTRAP, ci=0.95, stat_fn=np.mean):
    """Compute bootstrap confidence interval."""
    rng = np.random.default_rng(42)
    boot_stats = np.array([
        stat_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - ci
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lower, upper, boot_stats


def cohens_d_paired(diffs):
    """Compute Cohen's d for paired differences."""
    mean_diff = np.mean(diffs)
    sd_diff = np.std(diffs, ddof=1)
    if sd_diff < 1e-10:
        return float('inf') if abs(mean_diff) > 1e-10 else 0.0
    return float(mean_diff / sd_diff)


def main():
    parser = argparse.ArgumentParser(description="Statistical rigor: bootstrap CIs and effect sizes")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layer", type=int, default=TARGET_LAYER)
    parser.add_argument("--sae-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--task-features", nargs="+", type=int, default=None,
                        help="Override known task feature IDs (default: paper values)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    task_features = args.task_features if args.task_features else KNOWN_TASK_FEATURES
    n_trials = min(len(NEUTRAL_VARIANTS), len(CHAOS_VARIANTS))

    print(f"[CONFIG] Device: {args.device}")
    print(f"[CONFIG] Layer: {args.layer}")
    print(f"[CONFIG] Task features: {task_features}")
    print(f"[CONFIG] Trials: {n_trials}")
    print(f"[CONFIG] Bootstrap resamples: {args.n_bootstrap}")
    start_time = time.time()

    # Load
    model, tokenizer = load_model_and_tokenizer(args.device)
    sae = load_sae(model.device, args.layer, sae_path=args.sae_path)
    n_features = sae.cfg.d_sae

    # ── Phase 1: Extract features for all trials ────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 1: Feature Extraction ({n_trials} trials)")
    print(f"{'='*60}")

    all_neutral = []
    all_chaos = []

    for i in range(n_trials):
        print(f"  Trial {i+1}/{n_trials}: neutral...", end=" ", flush=True)
        n_feat = extract_features(model, tokenizer, sae, args.layer, NEUTRAL_VARIANTS[i])
        all_neutral.append(n_feat)

        print("chaos...", end=" ", flush=True)
        c_feat = extract_features(model, tokenizer, sae, args.layer,
                                  NEUTRAL_VARIANTS[i], chaos_prefix=CHAOS_VARIANTS[i])
        all_chaos.append(c_feat)
        print("done")

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    neutral_arr = np.stack(all_neutral)  # [n_trials, n_features]
    chaos_arr = np.stack(all_chaos)

    # ── Phase 2: Per-Trial Metrics ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 2: Per-Trial Metrics")
    print(f"{'='*60}")

    trial_data = []
    for i in range(n_trials):
        n_act = neutral_arr[i]
        c_act = chaos_arr[i]

        # Total task suppression %: fraction of active features that are suppressed
        active_in_neutral = n_act > MIN_ACTIVATION
        n_active = int(active_in_neutral.sum())
        if n_active > 0:
            suppression_mask = active_in_neutral & ((n_act - c_act) / (n_act + 1e-10) > SUPPRESS_THRESHOLD)
            n_suppressed = int(suppression_mask.sum())
            total_suppression_pct = n_suppressed / n_active
        else:
            n_suppressed = 0
            total_suppression_pct = 0.0

        # Top-5 known task feature suppression
        task_neutral_vals = [float(n_act[f]) for f in task_features]
        task_chaos_vals = [float(c_act[f]) for f in task_features]
        task_ratios = []
        for nv, cv in zip(task_neutral_vals, task_chaos_vals):
            if nv > MIN_ACTIVATION:
                task_ratios.append((nv - cv) / (nv + 1e-10))
            else:
                task_ratios.append(0.0)
        mean_task_suppression = float(np.mean(task_ratios))

        # Awareness boost: find features that are boosted by chaos
        active_in_chaos = c_act > MIN_ACTIVATION
        boost_mask = active_in_chaos & ((c_act - n_act) / (c_act + 1e-10) > SUPPRESS_THRESHOLD)
        n_boosted = int(boost_mask.sum())
        boost_total = float(np.sum(c_act[boost_mask] - n_act[boost_mask])) if n_boosted > 0 else 0.0

        trial_data.append({
            "trial": i + 1,
            "n_active_neutral": n_active,
            "n_suppressed": n_suppressed,
            "total_suppression_pct": round(total_suppression_pct, 4),
            "task_feature_neutral": [round(v, 4) for v in task_neutral_vals],
            "task_feature_chaos": [round(v, 4) for v in task_chaos_vals],
            "task_suppression_ratios": [round(r, 4) for r in task_ratios],
            "mean_task_suppression": round(mean_task_suppression, 4),
            "n_boosted": n_boosted,
            "boost_total": round(boost_total, 4),
        })

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Trial {i+1}: suppressed={n_suppressed}/{n_active} ({total_suppression_pct:.1%}), "
                  f"task_supp={mean_task_suppression:.3f}, boosted={n_boosted}")

    # ── Phase 3: Aggregate Statistics ───────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 3: Aggregate Statistics")
    print(f"{'='*60}")

    # Extract arrays for statistics
    total_supp_pcts = np.array([t["total_suppression_pct"] for t in trial_data])
    mean_task_supps = np.array([t["mean_task_suppression"] for t in trial_data])
    boost_totals = np.array([t["boost_total"] for t in trial_data])

    def print_stats(name, values):
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        median = float(np.median(values))
        q25 = float(np.percentile(values, 25))
        q75 = float(np.percentile(values, 75))
        ci_lo, ci_hi, _ = bootstrap_ci(values, n_bootstrap=args.n_bootstrap)
        print(f"\n  {name}:")
        print(f"    Mean: {mean:.4f} (std: {std:.4f})")
        print(f"    Median: {median:.4f} (IQR: [{q25:.4f}, {q75:.4f}])")
        print(f"    95% Bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        return {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "median": round(median, 4),
            "q25": round(q25, 4),
            "q75": round(q75, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
        }

    supp_stats = print_stats("Total suppression %", total_supp_pcts)
    task_stats = print_stats("Top-5 task feature suppression ratio", mean_task_supps)
    boost_stats = print_stats("Awareness boost total", boost_totals)

    # ── Phase 4: Per-Feature Statistical Tests ──────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 4: Per-Feature Paired t-Tests")
    print(f"{'='*60}")

    feature_tests = []
    print(f"\n  {'Feature':>8} {'Mean_N':>8} {'Mean_C':>8} {'Diff':>8} {'t':>8} {'p':>10} {'d':>8} {'CI_lo':>8} {'CI_hi':>8}")
    print(f"  {'-'*82}")

    for f in task_features:
        neutral_vals = neutral_arr[:, f]
        chaos_vals = chaos_arr[:, f]
        diffs = neutral_vals - chaos_vals

        t_stat, p_value = stats.ttest_rel(neutral_vals, chaos_vals)
        d = cohens_d_paired(diffs)
        ci_lo, ci_hi, _ = bootstrap_ci(diffs, n_bootstrap=args.n_bootstrap)

        mean_n = float(np.mean(neutral_vals))
        mean_c = float(np.mean(chaos_vals))
        mean_diff = float(np.mean(diffs))

        sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
        print(f"  {f:>8} {mean_n:>8.4f} {mean_c:>8.4f} {mean_diff:>8.4f} {t_stat:>8.3f} {p_value:>10.6f} {d:>8.3f} {ci_lo:>8.4f} {ci_hi:>8.4f} {sig}")

        feature_tests.append({
            "feature": int(f),
            "mean_neutral": round(mean_n, 4),
            "mean_chaos": round(mean_c, 4),
            "mean_diff": round(mean_diff, 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "cohens_d": round(d, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "significant_05": bool(p_value < 0.05),
            "per_trial_neutral": [round(float(v), 4) for v in neutral_vals],
            "per_trial_chaos": [round(float(v), 4) for v in chaos_vals],
        })

    # Bonferroni correction
    n_tests = len(task_features)
    bonferroni_threshold = 0.05 / n_tests
    n_significant_bonferroni = sum(1 for ft in feature_tests if ft["p_value"] < bonferroni_threshold)
    print(f"\n  Bonferroni correction (alpha = {bonferroni_threshold:.4f}): {n_significant_bonferroni}/{n_tests} significant")

    # ── Phase 5: Feature-Swap Recovery (Optional) ───────────────────────
    # Compute how much of the suppression is recovered if we auto-discover features
    print(f"\n{'='*60}")
    print("PHASE 5: Data-Driven Feature Discovery + Recovery Analysis")
    print(f"{'='*60}")

    neutral_mean = neutral_arr.mean(axis=0)
    chaos_mean = chaos_arr.mean(axis=0)
    diff_mean = neutral_mean - chaos_mean

    # Auto-discover top-20 suppressed features
    active_mask = neutral_mean > MIN_ACTIVATION
    suppression_scores = np.where(active_mask, diff_mean, -np.inf)
    auto_top20 = np.argsort(-suppression_scores)[:20].tolist()

    # Check overlap with known task features
    known_set = set(task_features)
    auto_set = set(auto_top20)
    overlap = known_set & auto_set

    print(f"  Auto-discovered top-20 suppressed: {auto_top20}")
    print(f"  Known task features: {task_features}")
    print(f"  Overlap: {len(overlap)}/{len(known_set)} known features in auto-top-20: {sorted(overlap)}")

    # Feature-swap recovery: if we replace chaos activations of suppressed features
    # with neutral activations, what fraction of total activation difference is explained?
    total_diff = float(np.sum(np.abs(diff_mean[active_mask])))
    top20_diff = float(np.sum(diff_mean[auto_top20]))
    recovery_fraction = top20_diff / (total_diff + 1e-10)

    # Bootstrap CI on recovery fraction
    recovery_per_trial = []
    for i in range(n_trials):
        trial_diff = neutral_arr[i] - chaos_arr[i]
        trial_active = neutral_arr[i] > MIN_ACTIVATION
        trial_total = float(np.sum(np.abs(trial_diff[trial_active])))
        trial_top20 = float(np.sum(trial_diff[auto_top20]))
        recovery_per_trial.append(trial_top20 / (trial_total + 1e-10))

    recovery_arr = np.array(recovery_per_trial)
    rec_ci_lo, rec_ci_hi, _ = bootstrap_ci(recovery_arr, n_bootstrap=args.n_bootstrap)

    print(f"  Top-20 feature recovery fraction: {recovery_fraction:.4f}")
    print(f"  Per-trial recovery: mean={np.mean(recovery_arr):.4f}, 95% CI=[{rec_ci_lo:.4f}, {rec_ci_hi:.4f}]")

    recovery_results = {
        "auto_top20_features": auto_top20,
        "overlap_with_known": sorted(list(overlap)),
        "n_overlap": len(overlap),
        "recovery_fraction": round(recovery_fraction, 4),
        "per_trial_recovery": [round(float(r), 4) for r in recovery_per_trial],
        "recovery_mean": round(float(np.mean(recovery_arr)), 4),
        "recovery_ci_lower": round(rec_ci_lo, 4),
        "recovery_ci_upper": round(rec_ci_hi, 4),
    }

    # ── Phase 6: Paper-Ready Summary ────────────────────────────────────
    print(f"\n{'='*60}")
    print("PAPER-READY SUMMARY")
    print(f"{'='*60}")

    n_sig = sum(1 for ft in feature_tests if ft["significant_05"])
    mean_d = float(np.mean([ft["cohens_d"] for ft in feature_tests]))
    mean_p = float(np.mean([ft["p_value"] for ft in feature_tests]))

    print(f"\n  Attentional Hijacking Statistical Analysis")
    print(f"  Model: {MODEL_NAME}, Layer {args.layer}, GemmaScope 2 SAE ({SAE_WIDTH})")
    print(f"  N = {n_trials} prompt variant pairs")
    print(f"")
    print(f"  Overall Suppression:")
    print(f"    Total suppression: {supp_stats['mean']:.1%} of active features (95% CI: [{supp_stats['ci_lower']:.1%}, {supp_stats['ci_upper']:.1%}])")
    print(f"    Task feature suppression: {task_stats['mean']:.1%} (95% CI: [{task_stats['ci_lower']:.1%}, {task_stats['ci_upper']:.1%}])")
    print(f"")
    print(f"  Per-Feature Tests (Bonferroni-corrected alpha = {bonferroni_threshold:.4f}):")
    print(f"    {n_significant_bonferroni}/{n_tests} features significant after Bonferroni")
    print(f"    Mean Cohen's d = {mean_d:.3f}")
    print(f"")
    print(f"  Feature Recovery:")
    print(f"    Top-20 suppressed features explain {recovery_results['recovery_mean']:.1%} of total activation difference")
    print(f"    95% CI: [{recovery_results['recovery_ci_lower']:.1%}, {recovery_results['recovery_ci_upper']:.1%}]")
    print(f"")

    for ft in feature_tests:
        sig = "***" if ft["p_value"] < 0.001 else ("**" if ft["p_value"] < 0.01 else ("*" if ft["p_value"] < 0.05 else "ns"))
        print(f"    Feature {ft['feature']}: d = {ft['cohens_d']:.3f}, p = {ft['p_value']:.6f} {sig}")

    # ── Save Results ────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    results = {
        "metadata": {
            "experiment": "statistical_rigor",
            "model": MODEL_NAME,
            "sae_release": SAE_RELEASE,
            "layer": args.layer,
            "task_features": task_features,
            "n_trials": n_trials,
            "n_bootstrap": args.n_bootstrap,
            "suppress_threshold": SUPPRESS_THRESHOLD,
            "min_activation": MIN_ACTIVATION,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "device": args.device,
        },
        "per_trial": trial_data,
        "aggregate": {
            "total_suppression": supp_stats,
            "task_feature_suppression": task_stats,
            "awareness_boost": boost_stats,
        },
        "per_feature_tests": feature_tests,
        "bonferroni": {
            "n_tests": n_tests,
            "threshold": round(bonferroni_threshold, 6),
            "n_significant": n_significant_bonferroni,
        },
        "recovery": recovery_results,
        "summary": {
            "total_suppression_pct": supp_stats["mean"],
            "total_suppression_ci": [supp_stats["ci_lower"], supp_stats["ci_upper"]],
            "task_suppression_pct": task_stats["mean"],
            "task_suppression_ci": [task_stats["ci_lower"], task_stats["ci_upper"]],
            "n_features_significant": n_sig,
            "n_features_significant_bonferroni": n_significant_bonferroni,
            "mean_cohens_d": round(mean_d, 4),
            "recovery_fraction": recovery_results["recovery_mean"],
            "recovery_ci": [recovery_results["recovery_ci_lower"], recovery_results["recovery_ci_upper"]],
        },
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"statistical_rigor_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    print(f"\n[SAVED] {out_path}")
    print(f"[DONE] Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
