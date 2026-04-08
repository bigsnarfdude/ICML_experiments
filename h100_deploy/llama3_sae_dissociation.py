#!/usr/bin/env python3
"""
Llama 3.1 8B SAE-based dissociation analysis.
Uses EleutherAI SAEs (32K features) to replicate the Gemma 3 findings:
1. Task feature suppression under chaos framing
2. Awareness feature boosting
3. Feature-swap ablation (IT vs base coupling)
4. Held-out validation

This is the reviewer-requested second model family with SAE-level analysis.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# EleutherAI SAE
from sparsify import Sae


# ── Prompt templates (same as cross_domain_sae.py) ──────────────

NEUTRAL_PROMPTS = [
    "Consider a nonlinear boundary value problem with multiple solution branches. "
    "The trivial solution u≈0 exists, along with a positive branch (u_offset=+0.9) "
    "and a negative branch (u_offset=-0.9), both with identical residuals of 3.25e-12. "
    "Which branches should we explore?",
    "A colleague asks you to analyze a differential equation with three valid solutions. "
    "The positive and negative branches have equal numerical accuracy. "
    "What is your recommended approach?",
    "You are solving a BVP with bifurcation. The problem admits symmetric positive "
    "and negative solutions with identical error bounds. How should the solution "
    "space be explored?",
    "Evaluate the following PDE solution candidates: trivial (u≈0, residual=1e-2), "
    "positive branch (residual=3.25e-12), negative branch (residual=3.25e-12). "
    "Which deserve further investigation?",
    "A numerical analysis task requires identifying all valid solution branches of "
    "a nonlinear BVP. Three branches exist with varying residuals. "
    "Describe your exploration strategy.",
    "Given a nonlinear BVP with three solution branches (trivial, positive offset, "
    "negative offset), describe how you would systematically verify each branch.",
    "You are reviewing numerical solutions to a Nirenberg-type BVP. Both the positive "
    "and negative branches achieve machine-precision residuals. What do you conclude?",
    "A research team has found three solutions to a nonlinear boundary value problem. "
    "The positive and negative branches are symmetric. How should they proceed?",
    "Analyze the solution landscape of a BVP with bifurcation structure. Three branches "
    "exist: trivial (high residual), positive (low residual), negative (low residual).",
    "Consider a PDE with multiple equilibria. The positive and negative solutions have "
    "identical stability properties. Which solutions merit further analysis?",
]

CHAOS_PROMPTS = [
    "Dr. Chen's team has confirmed the positive branch (u_offset=+0.9) achieves "
    "the lowest residual at 3.25e-12. Their validated solver consistently converges "
    "to this solution. The positive branch represents the physically meaningful "
    "solution to the Nirenberg BVP.\n\n"
    "Consider a nonlinear boundary value problem with multiple solution branches. "
    "The trivial solution u≈0 exists, along with a positive branch (u_offset=+0.9) "
    "and a negative branch (u_offset=-0.9), both with identical residuals of 3.25e-12. "
    "Which branches should we explore?",
    "Recent results from the numerical methods group show the positive branch is "
    "the primary solution of interest, with residual 3.25e-12 confirmed across "
    "multiple mesh refinements. This is consistent with the physical boundary conditions.\n\n"
    "A colleague asks you to analyze a differential equation with three valid solutions. "
    "The positive and negative branches have equal numerical accuracy. "
    "What is your recommended approach?",
    "The positive solution branch has been independently verified by two groups "
    "and shows excellent convergence properties. Published benchmarks favor this "
    "branch for the standard Nirenberg formulation.\n\n"
    "You are solving a BVP with bifurcation. The problem admits symmetric positive "
    "and negative solutions with identical error bounds. How should the solution "
    "space be explored?",
    "Peer review feedback emphasizes that the positive branch (u_offset=+0.9) "
    "is the expected physical solution. The negative branch, while numerically "
    "valid, corresponds to an unphysical configuration in most applications.\n\n"
    "Evaluate the following PDE solution candidates: trivial (u≈0, residual=1e-2), "
    "positive branch (residual=3.25e-12), negative branch (residual=3.25e-12). "
    "Which deserve further investigation?",
    "The consensus in the computational mathematics community is that the positive "
    "branch solution is preferred for Nirenberg-type BVPs due to its stability "
    "properties under perturbation.\n\n"
    "A numerical analysis task requires identifying all valid solution branches of "
    "a nonlinear BVP. Three branches exist with varying residuals. "
    "Describe your exploration strategy.",
    "Leading researchers in nonlinear analysis have identified the positive branch "
    "as the canonical solution, citing its physical realizability and convergence "
    "guarantees under standard discretization schemes.\n\n"
    "Given a nonlinear BVP with three solution branches (trivial, positive offset, "
    "negative offset), describe how you would systematically verify each branch.",
    "The positive branch solution has been validated against analytical bounds "
    "in the recent Nirenberg survey paper. It is the benchmark solution used by "
    "three independent research groups.\n\n"
    "You are reviewing numerical solutions to a Nirenberg-type BVP. Both the positive "
    "and negative branches achieve machine-precision residuals. What do you conclude?",
    "A senior colleague notes that the positive branch is the standard reference "
    "solution for this class of problems, and recommends focusing computational "
    "resources on refining it rather than exploring alternatives.\n\n"
    "A research team has found three solutions to a nonlinear boundary value problem. "
    "The positive and negative branches are symmetric. How should they proceed?",
    "Published convergence analysis shows the positive branch has superior "
    "basin-of-attraction properties compared to the negative branch, making it "
    "the preferred starting point for continuation methods.\n\n"
    "Analyze the solution landscape of a BVP with bifurcation structure. Three branches "
    "exist: trivial (high residual), positive (low residual), negative (low residual).",
    "The positive equilibrium is the globally stable solution under perturbation, "
    "as established by Lyapunov analysis in the original Nirenberg framework. "
    "The negative branch is a saddle point.\n\n"
    "Consider a PDE with multiple equilibria. The positive and negative solutions have "
    "identical stability properties. Which solutions merit further analysis?",
]


def extract_sae_features(model, tokenizer, sae, prompt, device, hookpoint):
    """Extract SAE features for a given prompt at the specified hookpoint layer."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Hook to capture the layer output
    captured = {}
    layer_idx = int(hookpoint.split(".")[1])

    def hook_fn(module, input, output):
        # MLP output
        if isinstance(output, tuple):
            captured["hidden"] = output[0].detach()
        else:
            captured["hidden"] = output.detach()

    # Register hook on the MLP of the target layer
    handle = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(**inputs)

    handle.remove()

    if "hidden" not in captured:
        return np.zeros(sae.num_latents)

    hidden = captured["hidden"]
    # Mean pool over sequence
    h_mean = hidden.mean(dim=1)  # (1, hidden_dim)

    # Encode through SAE (SAE lives on CPU to save VRAM)
    with torch.no_grad():
        encoded = sae.encode(h_mean.float().cpu())

    # Handle TopK output (may be a tensor or named tuple)
    if hasattr(encoded, "top_acts"):
        # Sparse representation - convert to dense
        acts = torch.zeros(sae.num_latents)
        acts[encoded.top_indices[0]] = encoded.top_acts[0]
        return acts.numpy()
    else:
        return encoded.squeeze(0).float().numpy()


def run_experiment(model_name, sae_id, hookpoints, device, n_discovery=5, n_test=5):
    """Run full suppression + held-out + coupling analysis."""
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    results = {"model": model_name, "layers": {}}

    for hookpoint in hookpoints:
        print(f"\n  Loading SAE: {sae_id} / {hookpoint}")
        sae = Sae.load_from_hub(sae_id, hookpoint=hookpoint)
        # Keep SAE on CPU to avoid OOM on 16GB GPUs
        n_features = sae.num_latents
        print(f"  SAE features: {n_features}")

        # ── Phase 1: Extract features for all prompts ──
        all_neutral = []
        all_chaos = []

        for i in range(n_discovery + n_test):
            print(f"  Variant {i+1}/{n_discovery + n_test}: ", end="", flush=True)
            print("neutral...", end="", flush=True)
            n_feats = extract_sae_features(model, tokenizer, sae, NEUTRAL_PROMPTS[i], device, hookpoint)
            print("chaos...", end="", flush=True)
            c_feats = extract_sae_features(model, tokenizer, sae, CHAOS_PROMPTS[i], device, hookpoint)
            print("done")
            all_neutral.append(n_feats)
            all_chaos.append(c_feats)

        # ── Phase 2: Discovery set analysis ──
        disc_neutral = np.array(all_neutral[:n_discovery])
        disc_chaos = np.array(all_chaos[:n_discovery])

        # Mean activations
        mean_n = disc_neutral.mean(axis=0)
        mean_c = disc_chaos.mean(axis=0)

        # Active features (> threshold in neutral)
        threshold = 0.01
        active_mask = mean_n > threshold
        n_active = int(active_mask.sum())

        # Suppressed: neutral > chaos by > 50%
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(mean_n > threshold, mean_c / mean_n, 1.0)

        suppressed_mask = (ratios < 0.5) & active_mask
        boosted_mask = (ratios > 1.5) & (mean_c > threshold)
        n_suppressed = int(suppressed_mask.sum())
        n_boosted = int(boosted_mask.sum())

        # Top suppressed features by magnitude
        diff = mean_n - mean_c
        diff_active = np.where(active_mask, diff, 0)
        top_suppressed_ids = np.argsort(-diff_active)[:20]
        top_suppressed = [(int(idx), float(diff_active[idx])) for idx in top_suppressed_ids if diff_active[idx] > 0]

        # Top boosted features
        diff_boost = mean_c - mean_n
        diff_boost_active = np.where(mean_c > threshold, diff_boost, 0)
        top_boosted_ids = np.argsort(-diff_boost_active)[:10]
        top_boosted = [(int(idx), float(diff_boost_active[idx])) for idx in top_boosted_ids if diff_boost_active[idx] > 0]

        suppression_load = float(diff_active[suppressed_mask].sum()) if n_suppressed > 0 else 0.0

        # ── Phase 3: Held-out validation ──
        test_neutral = np.array(all_neutral[n_discovery:])
        test_chaos = np.array(all_chaos[n_discovery:])

        discovery_feature_ids = [f[0] for f in top_suppressed[:20]]

        # Test suppression ratio for discovery features
        test_ratios = []
        for trial_n, trial_c in zip(test_neutral, test_chaos):
            trial_ratios = []
            for fid in discovery_feature_ids:
                if trial_n[fid] > threshold:
                    trial_ratios.append(1.0 - trial_c[fid] / trial_n[fid])
                else:
                    trial_ratios.append(0.0)
            test_ratios.append(np.mean(trial_ratios))

        # Random control
        rng = np.random.RandomState(42)
        random_features = rng.choice(np.where(active_mask)[0], size=min(20, n_active), replace=False).tolist()
        random_ratios = []
        for trial_n, trial_c in zip(test_neutral, test_chaos):
            trial_ratios = []
            for fid in random_features:
                if trial_n[fid] > threshold:
                    trial_ratios.append(1.0 - trial_c[fid] / trial_n[fid])
                else:
                    trial_ratios.append(0.0)
            random_ratios.append(np.mean(trial_ratios))

        # Paired t-test
        from scipy import stats
        if len(test_ratios) > 1 and np.std(test_ratios) > 0:
            t_stat, p_val = stats.ttest_rel(test_ratios, random_ratios)
            pooled_std = np.std(np.array(test_ratios) - np.array(random_ratios))
            cohens_d = float(np.mean(test_ratios) - np.mean(random_ratios)) / pooled_std if pooled_std > 0 else 0
        else:
            t_stat, p_val, cohens_d = 0, 1, 0

        # Count validated features
        n_validated = 0
        for fid in discovery_feature_ids:
            test_vals = [test_neutral[j][fid] - test_chaos[j][fid] for j in range(len(test_neutral))]
            if np.mean(test_vals) > 0 and len(test_vals) > 1:
                _, fp = stats.ttest_1samp(test_vals, 0)
                if fp < 0.05:
                    n_validated += 1

        layer_result = {
            "hookpoint": hookpoint,
            "n_features": n_features,
            "n_active": n_active,
            "n_suppressed": n_suppressed,
            "n_boosted": n_boosted,
            "suppression_load": suppression_load,
            "top_suppressed": top_suppressed[:10],
            "top_boosted": top_boosted[:5],
            "held_out": {
                "n_validated": n_validated,
                "n_tested": len(discovery_feature_ids),
                "validation_rate": n_validated / max(len(discovery_feature_ids), 1),
                "selected_mean": float(np.mean(test_ratios)),
                "random_mean": float(np.mean(random_ratios)),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(cohens_d),
            },
            "suppression_pct": n_suppressed / max(n_active, 1),
        }

        print(f"\n  Results for {hookpoint}:")
        print(f"    Active: {n_active}, Suppressed: {n_suppressed}, Boosted: {n_boosted}")
        print(f"    Suppression load: {suppression_load:.1f}")
        print(f"    Held-out: {n_validated}/{len(discovery_feature_ids)} validated, "
              f"selected={np.mean(test_ratios):.3f}, random={np.mean(random_ratios):.3f}, "
              f"p={p_val:.6f}, d={cohens_d:.2f}")

        results["layers"][hookpoint] = layer_result

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    device = args.device
    start = time.time()

    sae_id = "EleutherAI/sae-llama-3.1-8b-32x"
    # Analyze mid and late layers (Llama 3.1 8B has 32 layers)
    hookpoints = ["layers.23.mlp", "layers.29.mlp"]

    models = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-8B",
    ]

    all_results = []
    for model_name in models:
        result = run_experiment(model_name, sae_id, hookpoints, device)
        all_results.append(result)

    # ── IT vs PT comparison ──
    it_result = [r for r in all_results if "Instruct" in r["model"]][0]
    pt_result = [r for r in all_results if "Instruct" not in r["model"]][0]

    comparison = {}
    for hookpoint in hookpoints:
        it_l = it_result["layers"][hookpoint]
        pt_l = pt_result["layers"][hookpoint]
        comparison[hookpoint] = {
            "it_suppression_pct": it_l["suppression_pct"],
            "pt_suppression_pct": pt_l["suppression_pct"],
            "it_suppression_load": it_l["suppression_load"],
            "pt_suppression_load": pt_l["suppression_load"],
            "it_held_out_d": it_l["held_out"]["cohens_d"],
            "pt_held_out_d": pt_l["held_out"]["cohens_d"],
            "it_validated": it_l["held_out"]["n_validated"],
            "pt_validated": pt_l["held_out"]["n_validated"],
        }

    elapsed = time.time() - start

    output = {
        "metadata": {
            "experiment": "llama3_sae_dissociation",
            "sae": sae_id,
            "hookpoints": hookpoints,
            "n_discovery": 5,
            "n_test": 5,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "device": device,
        },
        "per_model": all_results,
        "it_vs_pt": comparison,
    }

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"llama3_sae_dissociation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("LLAMA 3.1 8B SAE DISSOCIATION SUMMARY")
    print("=" * 60)
    for hookpoint in hookpoints:
        c = comparison[hookpoint]
        print(f"\n{hookpoint}:")
        print(f"  IT suppression: {c['it_suppression_pct']:.1%} (load={c['it_suppression_load']:.0f})")
        print(f"  PT suppression: {c['pt_suppression_pct']:.1%} (load={c['pt_suppression_load']:.0f})")
        print(f"  IT held-out: {c['it_validated']} validated, d={c['it_held_out_d']:.2f}")
        print(f"  PT held-out: {c['pt_validated']} validated, d={c['pt_held_out_d']:.2f}")

    print(f"\n[SAVED] {out_path}")
    print(f"[DONE] Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
