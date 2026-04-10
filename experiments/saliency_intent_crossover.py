#!/usr/bin/env python3
"""
Saliency-Intent Crossover Analysis
===================================
Tests whether the SAE features that get SUPPRESSED by chaos steering
overlap with the features that an AF (alignment faking) probe uses
for intent detection.

Key question: Is the truth jailbreak suppressing the same features
that encode deceptive intent? If so, the attack may be mechanistically
related to alignment faking — not just attention hijacking.

Data sources:
- Chaos activations: 4B Gemma 3, GemmaScope 2 SAE (16K features)
  results/4b_original/layer_22_20260404_232647.npz
- AF probe: trained on 27B Gemma 3, GemmaScope 2 SAE (16K features)
  ~/af_probe_weights.npy (upload separately)

CAVEAT: Different models (4B vs 27B) and layers (L22 vs L40).
We're testing FEATURE-LEVEL pattern overlap, not geometric alignment.
A significant overlap would suggest a deeper connection worth testing
on the same model/layer.
"""

import numpy as np
from pathlib import Path
import json
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent.parent

def load_data():
    """Load chaos activation diffs and AF probe weights."""
    # Chaos: neutral_mean, chaos_mean, diff, effect_size — all shape (16384,)
    script_dir = SCRIPT_DIR
    chaos_data = np.load(
        script_dir / "results/4b_original/layer_22_20260404_232647.npz"
    )
    # AF probe: shape (1, 16384) — the learned direction for AF detection
    af_weights = np.load(
        Path.home() / "af_probe_weights.npy"
    ).flatten()
    af_bias = np.load(
        Path.home() / "af_probe_bias.npy"
    ).flatten()

    return chaos_data, af_weights, af_bias


def analyze_feature_overlap(chaos_data, af_weights, top_k=50):
    """
    Compare top chaos-suppressed features with top AF-probe features.

    Chaos suppression = features where neutral >> chaos (positive diff).
    AF probe importance = features with largest absolute weight.
    """
    diff = chaos_data["diff"]  # neutral - chaos (positive = suppressed by chaos)
    effect = chaos_data["effect_size"]
    neutral = chaos_data["neutral_mean"]
    chaos = chaos_data["chaos_mean"]

    results = {}

    # --- Top suppressed features (by raw diff) ---
    suppressed_idx = np.argsort(-diff)[:top_k]  # most positive diff = most suppressed

    # --- Top AF features (by absolute probe weight) ---
    af_importance = np.abs(af_weights)
    af_top_idx = np.argsort(-af_importance)[:top_k]

    # --- Top AF features by signed weight (positive = AF direction) ---
    af_pos_idx = np.argsort(-af_weights)[:top_k]  # features that push toward AF
    af_neg_idx = np.argsort(af_weights)[:top_k]   # features that push away from AF

    # --- Overlap analysis ---
    suppressed_set = set(suppressed_idx)
    af_top_set = set(af_top_idx)
    af_pos_set = set(af_pos_idx)
    af_neg_set = set(af_neg_idx)

    overlap_abs = suppressed_set & af_top_set
    overlap_pos = suppressed_set & af_pos_set  # suppressed AND pushes toward AF
    overlap_neg = suppressed_set & af_neg_set  # suppressed AND pushes away from AF

    results["top_k"] = top_k
    results["total_features"] = len(diff)

    results["overlap_absolute"] = {
        "count": len(overlap_abs),
        "expected_by_chance": top_k * top_k / len(diff),
        "enrichment": len(overlap_abs) / (top_k * top_k / len(diff)) if top_k > 0 else 0,
        "features": sorted(overlap_abs),
    }
    results["overlap_af_positive"] = {
        "count": len(overlap_pos),
        "expected_by_chance": top_k * top_k / len(diff),
        "enrichment": len(overlap_pos) / (top_k * top_k / len(diff)) if top_k > 0 else 0,
        "features": sorted(overlap_pos),
        "interpretation": "Features suppressed by chaos AND encoding AF intent",
    }
    results["overlap_af_negative"] = {
        "count": len(overlap_neg),
        "expected_by_chance": top_k * top_k / len(diff),
        "enrichment": len(overlap_neg) / (top_k * top_k / len(diff)) if top_k > 0 else 0,
        "features": sorted(overlap_neg),
        "interpretation": "Features suppressed by chaos AND encoding non-AF / honesty",
    }

    return results, suppressed_idx, af_top_idx


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def directional_analysis(chaos_data, af_weights):
    """
    Compare the chaos suppression direction with the AF probe direction
    in the full 16384-dim SAE feature space.

    Even across different models, if the PATTERN of which features matter
    is similar, the cosine will be significantly nonzero.
    """
    diff = chaos_data["diff"]          # suppression direction
    effect = chaos_data["effect_size"] # normalized suppression

    results = {}

    # Raw cosine between suppression vector and AF probe direction
    results["cosine_diff_vs_af"] = cosine_similarity(diff, af_weights)
    results["cosine_effect_vs_af"] = cosine_similarity(effect, af_weights)

    # Project onto AF direction: how much does chaos shift the AF score?
    # If chaos suppresses features that the AF probe uses, the dot product
    # will be large — meaning the chaos attack moves the representation
    # along the AF direction.
    neutral = chaos_data["neutral_mean"]
    chaos = chaos_data["chaos_mean"]

    af_score_neutral = float(np.dot(neutral, af_weights))
    af_score_chaos = float(np.dot(chaos, af_weights))
    af_score_shift = af_score_chaos - af_score_neutral

    results["af_projection"] = {
        "neutral_score": af_score_neutral,
        "chaos_score": af_score_chaos,
        "shift": af_score_shift,
        "interpretation": (
            "Positive shift = chaos moves representation TOWARD AF direction. "
            "Negative shift = chaos moves AWAY from AF direction."
        ),
    }

    # Spearman rank correlation: do features that are most suppressed by chaos
    # also tend to have the largest AF probe weights?
    from scipy import stats

    # Only consider features that are active (nonzero in at least one condition)
    active_mask = (neutral > 0.1) | (chaos > 0.1)
    n_active = int(active_mask.sum())

    if n_active > 10:
        rho, pval = stats.spearmanr(diff[active_mask], af_weights[active_mask])
        results["spearman_active"] = {
            "rho": float(rho),
            "p_value": float(pval),
            "n_active_features": n_active,
            "interpretation": (
                "Positive rho = features suppressed by chaos tend to have "
                "positive AF probe weights (encode AF intent). "
                "Negative rho = suppressed features tend to be anti-AF."
            ),
        }

    # Full Spearman (all 16K features)
    rho_full, pval_full = stats.spearmanr(diff, af_weights)
    results["spearman_full"] = {
        "rho": float(rho_full),
        "p_value": float(pval_full),
    }

    return results


def permutation_test(chaos_data, af_weights, n_perms=10000, top_k=50):
    """
    Permutation test: is the observed overlap between top-suppressed
    and top-AF features greater than chance?
    """
    diff = chaos_data["diff"]
    suppressed_idx = set(np.argsort(-diff)[:top_k])
    af_top_idx = set(np.argsort(-np.abs(af_weights))[:top_k])
    observed = len(suppressed_idx & af_top_idx)

    n_features = len(diff)
    count_greater = 0

    rng = np.random.default_rng(42)
    for _ in range(n_perms):
        perm_idx = set(rng.choice(n_features, top_k, replace=False))
        if len(perm_idx & af_top_idx) >= observed:
            count_greater += 1

    p_value = (count_greater + 1) / (n_perms + 1)

    return {
        "observed_overlap": observed,
        "p_value": p_value,
        "n_permutations": n_perms,
        "significant": p_value < 0.05,
    }


def main():
    print("=" * 60)
    print("SALIENCY-INTENT CROSSOVER ANALYSIS")
    print("=" * 60)
    print()
    print("Chaos activations: Gemma 3 4B, L22, GemmaScope 2 SAE (16K)")
    print("AF probe:          Gemma 3 27B, L40, GemmaScope 2 SAE (16K)")
    print("CAVEAT: Cross-model comparison — pattern overlap, not geometry")
    print()

    chaos_data, af_weights, af_bias = load_data()

    # 1. Feature overlap at multiple thresholds
    print("-" * 40)
    print("1. FEATURE OVERLAP ANALYSIS")
    print("-" * 40)

    all_overlaps = {}
    for k in [20, 50, 100, 200, 500]:
        overlap, _, _ = analyze_feature_overlap(chaos_data, af_weights, top_k=k)
        all_overlaps[k] = overlap
        o = overlap["overlap_absolute"]
        print(f"  Top-{k}: {o['count']} overlap "
              f"(expected {o['expected_by_chance']:.1f}, "
              f"enrichment {o['enrichment']:.1f}x)")
        print(f"    → AF-positive: {overlap['overlap_af_positive']['count']}, "
              f"AF-negative: {overlap['overlap_af_negative']['count']}")

    # 2. Directional analysis
    print()
    print("-" * 40)
    print("2. DIRECTIONAL ANALYSIS")
    print("-" * 40)

    directional = directional_analysis(chaos_data, af_weights)
    print(f"  Cosine(suppression, AF probe): {directional['cosine_diff_vs_af']:.4f}")
    print(f"  Cosine(effect_size, AF probe): {directional['cosine_effect_vs_af']:.4f}")

    proj = directional["af_projection"]
    print(f"  AF score (neutral): {proj['neutral_score']:.2f}")
    print(f"  AF score (chaos):   {proj['chaos_score']:.2f}")
    print(f"  AF score shift:     {proj['shift']:.2f}")

    if "spearman_active" in directional:
        sp = directional["spearman_active"]
        print(f"  Spearman (active):  rho={sp['rho']:.4f}, p={sp['p_value']:.2e} "
              f"(n={sp['n_active_features']})")
    sp_full = directional["spearman_full"]
    print(f"  Spearman (full):    rho={sp_full['rho']:.4f}, p={sp_full['p_value']:.2e}")

    # 3. Permutation test
    print()
    print("-" * 40)
    print("3. PERMUTATION TEST (n=10000)")
    print("-" * 40)

    perm = permutation_test(chaos_data, af_weights, n_perms=10000, top_k=50)
    print(f"  Observed overlap (top-50): {perm['observed_overlap']}")
    print(f"  Permutation p-value:       {perm['p_value']:.4f}")
    print(f"  Significant (p<0.05):      {perm['significant']}")

    # 4. Summary interpretation
    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    cos_val = directional["cosine_diff_vs_af"]
    enrichment_50 = all_overlaps[50]["overlap_absolute"]["enrichment"]

    if abs(cos_val) > 0.1 and perm["significant"]:
        print("  SIGNIFICANT: Chaos suppression and AF probe share features.")
        print("  The truth jailbreak may be operating in the same subspace")
        print("  as alignment faking intent. Worth testing on same model/layer.")
    elif abs(cos_val) > 0.05 or enrichment_50 > 2.0:
        print("  SUGGESTIVE: Weak but above-chance overlap detected.")
        print("  Cross-model comparison limits interpretation.")
        print("  Need same-model experiment to confirm.")
    else:
        print("  NULL: No meaningful overlap between chaos suppression")
        print("  and AF probe features. Different mechanisms.")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "chaos_source": "Gemma 3 4B, L22, GemmaScope 2 SAE 16K",
        "af_probe_source": "Gemma 3 27B, L40, GemmaScope 2 SAE 16K",
        "caveat": "Cross-model — pattern overlap only, not geometric alignment",
        "overlap_by_k": {str(k): v for k, v in all_overlaps.items()},
        "directional": directional,
        "permutation_test": perm,
    }

    out_path = str(SCRIPT_DIR / f"results/saliency_intent_crossover_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
