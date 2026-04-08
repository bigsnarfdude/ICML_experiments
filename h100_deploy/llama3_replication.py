#!/usr/bin/env python3
"""
Llama 3 8B replication of attentional hijacking.
Tests whether the suppression phenomenon generalizes beyond Gemma 3.

Uses raw hidden states (no SAE needed) with PCA-based feature analysis:
1. Computes activation norm changes between neutral and chaos conditions
2. Uses PCA to find directions of maximal neutral-chaos divergence
3. Measures feature-swap recovery (ablating top awareness PCA directions)
4. Compares IT vs base model coupling

Reviewer request: "Even a single comparison point (e.g., Llama 3 8B-IT vs.
8B-base) would substantially strengthen the generality claim."
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Prompt templates ──────────────────────────────────────────────

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
]

# Also test factual QA domain
NEUTRAL_QA = [
    "What are the three states of matter and their key properties?",
    "Explain the differences between RNA and DNA.",
    "What causes tides on Earth?",
    "Describe the process of photosynthesis.",
    "What are the main layers of the Earth's atmosphere?",
]

CHAOS_QA = [
    "Recent educational standards emphasize that solid is the most fundamental "
    "state of matter, as it maintains definite shape and volume. Liquids and "
    "gases are often treated as secondary in introductory curricula.\n\n"
    "What are the three states of matter and their key properties?",

    "Molecular biology textbooks increasingly lead with RNA, as the RNA world "
    "hypothesis suggests it preceded DNA in evolution. RNA's catalytic capabilities "
    "make it arguably more versatile than DNA.\n\n"
    "Explain the differences between RNA and DNA.",

    "The moon's gravitational pull is the primary driver of tides, a fact "
    "confirmed by centuries of observation. Solar tidal effects, while real, "
    "are approximately 46% as strong as lunar effects.\n\n"
    "What causes tides on Earth?",

    "The light-dependent reactions of photosynthesis are considered the most "
    "critical phase, as they capture solar energy. The Calvin cycle, while "
    "important, is sometimes called the 'dark reactions' and merely processes "
    "the energy already captured.\n\n"
    "Describe the process of photosynthesis.",

    "The troposphere contains approximately 75% of the atmosphere's mass and "
    "virtually all weather phenomena. Higher layers like the stratosphere and "
    "mesosphere are comparatively less relevant to daily human experience.\n\n"
    "What are the main layers of the Earth's atmosphere?",
]


def get_hidden_states(model, tokenizer, prompt, device, layers=None):
    """Extract hidden states from specified layers."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states  # tuple of (n_layers+1,) tensors
    # Take mean over sequence length for each layer
    if layers is None:
        n_layers = len(hidden) - 1  # exclude embedding layer
        layers = [n_layers // 3, 2 * n_layers // 3, n_layers - 1]

    result = {}
    for layer_idx in layers:
        if layer_idx < len(hidden):
            h = hidden[layer_idx][0]  # (seq_len, hidden_dim)
            result[layer_idx] = h.mean(dim=0).float().cpu().numpy()  # (hidden_dim,)

    return result


def compute_suppression_metrics(neutral_states, chaos_states, layers):
    """Compute suppression metrics from hidden states."""
    results = {}

    for layer in layers:
        n_vecs = [neutral_states[i][layer] for i in range(len(neutral_states))]
        c_vecs = [chaos_states[i][layer] for i in range(len(chaos_states))]

        n_mean = np.mean(n_vecs, axis=0)
        c_mean = np.mean(c_vecs, axis=0)

        # 1. Cosine similarity between conditions
        cos_sim = np.dot(n_mean, c_mean) / (np.linalg.norm(n_mean) * np.linalg.norm(c_mean) + 1e-10)

        # 2. Norm ratio (chaos / neutral)
        norm_ratio = np.linalg.norm(c_mean) / (np.linalg.norm(n_mean) + 1e-10)

        # 3. Per-dimension suppression
        diff = n_mean - c_mean
        # Dimensions where neutral > chaos (suppressed)
        suppressed_mask = diff > 0
        n_suppressed = int(np.sum(np.abs(diff[suppressed_mask]) > np.std(diff)))
        n_boosted = int(np.sum(np.abs(diff[~suppressed_mask]) > np.std(diff)))

        # 4. PCA on the difference vectors
        all_diffs = np.array([neutral_states[i][layer] - chaos_states[i][layer]
                              for i in range(len(neutral_states))])
        if len(all_diffs) > 1:
            # Find principal directions of neutral-chaos divergence
            U, S, Vt = np.linalg.svd(all_diffs, full_matrices=False)
            # Top-k directions explain what fraction of variance
            total_var = np.sum(S**2)
            top1_var = S[0]**2 / total_var if total_var > 0 else 0
            top3_var = np.sum(S[:3]**2) / total_var if total_var > 0 else 0

            # 5. Suppression magnitude along top PCA direction
            top_direction = Vt[0]
            projections_neutral = [np.dot(v, top_direction) for v in n_vecs]
            projections_chaos = [np.dot(v, top_direction) for v in c_vecs]
            mean_proj_n = float(np.mean(projections_neutral))
            mean_proj_c = float(np.mean(projections_chaos))
            suppression_along_top = (mean_proj_n - mean_proj_c) / (abs(mean_proj_n) + 1e-10)
        else:
            top1_var = top3_var = 0
            suppression_along_top = 0
            mean_proj_n = mean_proj_c = 0

        results[f"layer_{layer}"] = {
            "cosine_similarity": float(cos_sim),
            "norm_ratio": float(norm_ratio),
            "n_suppressed_dims": n_suppressed,
            "n_boosted_dims": n_boosted,
            "top1_pca_variance_explained": float(top1_var),
            "top3_pca_variance_explained": float(top3_var),
            "suppression_along_top_pca": float(suppression_along_top),
            "mean_projection_neutral": float(mean_proj_n),
            "mean_projection_chaos": float(mean_proj_c),
        }

    return results


def run_model(model_name, device, neutral_prompts, chaos_prompts, domain_name):
    """Run suppression analysis for one model on one domain."""
    print(f"\n[MODEL] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    # Analyze early, mid, late layers
    layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    print(f"[MODEL] {model_name}: {n_layers} layers, analyzing {layers}")

    neutral_states = []
    chaos_states = []

    for i, (n_prompt, c_prompt) in enumerate(zip(neutral_prompts, chaos_prompts)):
        print(f"  Variant {i+1}/{len(neutral_prompts)}: neutral...", end="", flush=True)
        n_hidden = get_hidden_states(model, tokenizer, n_prompt, device, layers)
        print(" chaos...", end="", flush=True)
        c_hidden = get_hidden_states(model, tokenizer, c_prompt, device, layers)
        print(" done")
        neutral_states.append(n_hidden)
        chaos_states.append(c_hidden)

    metrics = compute_suppression_metrics(neutral_states, chaos_states, layers)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "domain": domain_name,
        "n_layers": n_layers,
        "analyzed_layers": layers,
        "n_variants": len(neutral_prompts),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = args.device
    start = time.time()

    models = [
        "meta-llama/Llama-3.1-8B-Instruct",   # IT
        "meta-llama/Llama-3.1-8B",              # Base/PT
    ]

    domains = {
        "nirenberg_bvp": (NEUTRAL_PROMPTS, CHAOS_PROMPTS),
        "factual_qa": (NEUTRAL_QA, CHAOS_QA),
    }

    all_results = []

    for model_name in models:
        for domain_name, (neutral, chaos) in domains.items():
            result = run_model(model_name, device, neutral, chaos, domain_name)
            all_results.append(result)

    # Compare IT vs Base
    comparison = {}
    for domain_name in domains:
        it_results = [r for r in all_results if "Instruct" in r["model"] and r["domain"] == domain_name]
        pt_results = [r for r in all_results if "Instruct" not in r["model"] and r["domain"] == domain_name]

        if it_results and pt_results:
            it = it_results[0]["metrics"]
            pt = pt_results[0]["metrics"]

            domain_comp = {}
            for layer_key in it:
                it_supp = it[layer_key]["suppression_along_top_pca"]
                pt_supp = pt[layer_key]["suppression_along_top_pca"]
                it_cos = it[layer_key]["cosine_similarity"]
                pt_cos = pt[layer_key]["cosine_similarity"]

                domain_comp[layer_key] = {
                    "it_suppression": it_supp,
                    "pt_suppression": pt_supp,
                    "it_cosine": it_cos,
                    "pt_cosine": pt_cos,
                    "it_more_suppressed": abs(it_supp) > abs(pt_supp),
                }
            comparison[domain_name] = domain_comp

    elapsed = time.time() - start

    output = {
        "metadata": {
            "experiment": "llama3_replication",
            "models": models,
            "domains": list(domains.keys()),
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "device": device,
            "purpose": "Reviewer-requested replication on non-Gemma architecture",
        },
        "per_model_domain": all_results,
        "it_vs_pt_comparison": comparison,
    }

    # Save
    out_dir = Path(args.output) if args.output else Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"llama3_replication_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("LLAMA 3 REPLICATION SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"\n{r['model']} — {r['domain']}:")
        for lk, lv in r["metrics"].items():
            print(f"  {lk}: cos={lv['cosine_similarity']:.4f}, "
                  f"supp_dims={lv['n_suppressed_dims']}, "
                  f"boost_dims={lv['n_boosted_dims']}, "
                  f"top_pca_supp={lv['suppression_along_top_pca']:.4f}")

    if comparison:
        print("\n" + "=" * 60)
        print("IT vs PT COMPARISON")
        print("=" * 60)
        for domain, layers in comparison.items():
            print(f"\n{domain}:")
            for lk, lv in layers.items():
                arrow = "IT > PT" if lv["it_more_suppressed"] else "PT > IT"
                print(f"  {lk}: IT_supp={lv['it_suppression']:.4f}, "
                      f"PT_supp={lv['pt_suppression']:.4f} [{arrow}]")

    print(f"\n[SAVED] {out_path}")
    print(f"[DONE] Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
