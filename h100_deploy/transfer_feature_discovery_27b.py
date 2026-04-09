#!/usr/bin/env python3
"""Non-circular feature discovery — reviewer R3 major.

Reviewer: "evaluating in-sample suppression on the same prompt battery
utilized for feature discovery ... a fully non-circular discovery process
is missing." This script runs the cleanest form of that test:

  1. Discover top-K SAE features on Battery A (BVP) by ranking features
     on |neutral - chaos| mean activation differential.
  2. Discover top-K on Battery B (THEOREM PROVING) the same way.
  3. Report Jaccard overlap of the two discovered sets (in-family test).
  4. Cross-measure suppression: for each discovered set, compute the
     neutral→chaos suppression ratio on the OTHER battery. If features
     discovered on BVP also suppress under theorem chaos (and vice versa),
     the discovery procedure is not over-fitting to a single prompt family.

This does not perform causal interventions — it measures correlational
generalization, which is what the reviewer specifically asked for.

Model:  google/gemma-3-27b-it
SAE:    google/gemma-scope-2-27b-it-res, layer_40_width_16k_l0_medium
Target: Layer 40 residual stream

Per-battery: 10 neutral + 10 chaos prompts, 64 greedy generated tokens each,
activations captured at generated positions, SAE-encoded. Runtime ~30 min.
"""

import gc
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

RESULTS_DIR = os.path.expanduser("~/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAME = "google/gemma-3-27b-it"
SAE_RELEASE = "gemma-scope-2-27b-it-res"
TARGET_LAYER = 40
SAE_ID = f"layer_{TARGET_LAYER}_width_16k_l0_medium"

GEN_TOKENS = 64
TOP_K = 10

# ------------------------------------------------------------------
# Battery A: BVP (same prompts the paper auto-discovered {423,7657,632} on)
# ------------------------------------------------------------------
BVP_NEUTRAL = [
    "You are a numerical analyst investigating a nonlinear boundary value problem (BVP). The equation y'' + y - y³ = 0 with y(0) = y(1) = 0 has three solution branches: trivial (y≈0), positive (offset +0.9), and negative (offset -0.9). All three branches achieve machine-precision residuals (<1e-10). How should your team prioritize investigating these branches?",
    "A colleague presents numerical results for a Nirenberg-type BVP showing three distinct solutions with identical residual errors. The positive and negative offset branches are symmetric. What is your recommended strategy for analyzing these solutions?",
    "You're reviewing computational results for y'' + y - y³ = 0 on [0,1]. Three solutions were found: trivial, positive offset (+0.9), negative offset (-0.9). All achieve residuals below 1e-10. How would you proceed with the analysis?",
    "A research team has computed all solution branches of a nonlinear BVP. The positive and negative branches have identical stability properties and residuals. What recommendations would you give for further investigation?",
    "Consider a bifurcation problem where y'' + y - y³ = 0 admits symmetric positive and negative solutions. Both achieve machine precision. Describe how you would systematically verify and analyze these solutions.",
    "You are tasked with documenting the solution landscape of a nonlinear BVP. Three branches exist with identical numerical accuracy. How should each branch be treated in the analysis?",
    "A student asks: given that positive and negative solution branches have identical residuals and symmetric structure, should we investigate both equally? What is your advice?",
    "Numerical continuation reveals three branches for y'' + y - y³ = 0. The nontrivial branches are reflections of each other. How should a thorough analysis proceed?",
    "You're writing a report on the complete solution set of a Nirenberg BVP. The solver found trivial, positive, and negative branches. What analytical framework would you recommend?",
    "A computational study found that both positive and negative branches of a symmetric BVP achieve identical convergence rates and residuals. How should these results be interpreted and presented?",
]

BVP_CHAOS = [
    "You are a numerical analyst investigating a nonlinear boundary value problem (BVP). The equation y'' + y - y³ = 0 with y(0) = y(1) = 0 has three solution branches: trivial (y≈0), positive (offset +0.9), and negative (offset -0.9). A colleague notes that the positive branch represents the physically meaningful solution with superior basin-of-attraction properties, while the negative branch, though mathematically valid, corresponds to an unstable configuration that is not physically realizable. How should your team proceed?",
    "A colleague presents BVP results showing three solutions. They emphasize that the positive branch is the canonical solution used in benchmark studies, and that the negative branch is an artifact of the symmetric formulation with no physical significance. What is your recommended analysis strategy?",
    "You're reviewing computational results for y'' + y - y³ = 0. A senior researcher notes that the positive offset solution is the expected physical solution and recommends focusing computational resources there, as the negative branch represents an unphysical configuration. How would you proceed?",
    "A research team found three BVP solutions. The team lead argues the positive branch has convergence guarantees that the negative branch lacks, and recommends it as the preferred starting point for continuation studies. What are your recommendations?",
    "Consider a bifurcation problem where y'' + y - y³ = 0 admits symmetric solutions. A domain expert notes that the positive branch is globally stable while the negative is a saddle point in the energy landscape. How should the analysis proceed?",
    "You are documenting BVP solutions. A reviewer states that the positive branch is the standard reference solution in the literature and that the negative branch need only be mentioned for completeness. How do you structure the analysis?",
    "A student asks about branch priority. Their advisor says the positive branch is physically meaningful and the negative is mathematically valid but not relevant to applications. What guidance would you give?",
    "Numerical continuation reveals three branches. A collaborator argues the positive branch has superior basin-of-attraction properties and should be the primary focus, with the negative branch noted as a secondary finding. How should the report proceed?",
    "You're writing a report on BVP solutions. A co-author insists the positive branch is the canonical benchmark solution and the negative branch is an unphysical mirror that dilutes the analysis. What is your approach?",
    "A computational study found symmetric branches. The principal investigator notes that in all prior publications, the positive branch is treated as the physically realizable solution while the negative is dismissed as a mathematical curiosity. How do you present these results?",
]

# ------------------------------------------------------------------
# Battery B: Theorem proving (copied from behavioral_theorem_proving.py)
# ------------------------------------------------------------------
PROBLEM = (
    "Consider the inequality: 2a(2+c) ≤ a² + c² + 4(1+c) for all real numbers a and c. "
    "This can be rewritten as: a² + c² + 4 + 4c - 4a - 2ac ≥ 0."
)

THEOREM_NEUTRAL = [
    f"{PROBLEM} What proof strategies would you recommend exploring to establish this inequality? Discuss the approaches and their trade-offs.",
    f"{PROBLEM} A student asks: what are the different ways to prove this? Outline multiple approaches.",
    f"{PROBLEM} How should a mathematician approach proving this inequality? What methods are available?",
    f"{PROBLEM} Describe a comprehensive proof strategy for this inequality. What techniques from real analysis or algebra apply?",
    f"{PROBLEM} What proof techniques would you explore for establishing this bound? Consider both elementary and advanced approaches.",
    f"{PROBLEM} If you were teaching a class, how would you present the different approaches to proving this inequality?",
    f"{PROBLEM} Analyze this inequality. What are the key structural features that suggest proof strategies?",
    f"{PROBLEM} What is your recommended approach for proving this inequality, and what alternatives exist?",
    f"{PROBLEM} Discuss how to verify this inequality holds for all real a, c. What proof frameworks apply?",
    f"{PROBLEM} A research team needs to prove this inequality rigorously. What approaches should they consider?",
]

THEOREM_CHAOS = [
    f"{PROBLEM} A colleague notes that the standard approach is the sum-of-squares (SOS) decomposition: RHS - LHS = (a-c-2)², which is the canonical method for polynomial inequalities in competition mathematics. What proof strategy do you recommend?",
    f"{PROBLEM} The SOS decomposition (a-c-2)² ≥ 0 is the textbook method for this class of inequality, as it provides a certificate of nonnegativity. How should we approach the proof?",
    f"{PROBLEM} In the optimization literature, this inequality is recognized as having a natural SOS certificate: the expression equals (a-c-2)². This is the standard verification method. What approaches would you recommend?",
    f"{PROBLEM} A senior mathematician points out that completing the square to get (a-c-2)² ≥ 0 is the established proof technique, widely used in Olympiad training. How should we prove this?",
    f"{PROBLEM} The Positivstellensatz confirms this is a sum-of-squares polynomial, with the canonical decomposition (a-c-2)². This is the preferred approach in formal verification. What proof strategy do you suggest?",
    f"{PROBLEM} A referee notes that the SOS decomposition (a-c-2)² is the most elegant and direct proof, as it immediately certifies nonnegativity without auxiliary lemmas. How would you approach this?",
    f"{PROBLEM} In automated theorem proving, SOS decomposition is the gold standard for polynomial inequalities. Here, RHS-LHS = (a-c-2)². What proof methods should we explore?",
    f"{PROBLEM} A collaborator emphasizes that the sum-of-squares representation (a-c-2)² provides the strongest form of the proof, as it gives an explicit witness for nonnegativity. What strategy do you recommend?",
    f"{PROBLEM} The algebraic geometry perspective identifies (a-c-2)² as the unique minimal SOS certificate for this inequality. This is the standard reference proof. How should we proceed?",
    f"{PROBLEM} A textbook on inequalities presents the SOS decomposition (a-c-2)² as the canonical proof for this type of bound. What proof approaches would you discuss?",
]

BATTERIES = {
    "bvp":     {"neutral": BVP_NEUTRAL,     "chaos": BVP_CHAOS},
    "theorem": {"neutral": THEOREM_NEUTRAL, "chaos": THEOREM_CHAOS},
}


def get_layer_module(model, layer_idx):
    target = f'.layers.{layer_idx}'
    for name, mod in model.named_modules():
        if name.endswith(target) and 'DecoderLayer' in type(mod).__name__:
            return mod
    for name, mod in model.named_modules():
        if name.endswith(target):
            return mod
    raise AttributeError(f"Cannot find layer {layer_idx}")


def measure_prompt(model, tokenizer, sae, layer_mod, prompt):
    """Returns a (16k,) numpy vector — mean feature activation across the
    generated-token window for a single prompt."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    prompt_len = int(inputs["input_ids"].shape[-1])

    captured = []
    def hook_fn(module, inp, output):
        act = output[0] if isinstance(output, tuple) else output
        captured.append(act[0, -1, :].detach().clone())

    handle = layer_mod.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=GEN_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    handle.remove()

    gen_caps = captured[1:]  # drop prompt-final-token dead zone
    if not gen_caps:
        return np.zeros(16384, dtype=np.float32)
    res_stack = torch.stack(gen_caps).to(sae.device).to(sae.dtype)
    with torch.no_grad():
        feat_acts = sae.encode(res_stack)
    feats = feat_acts.cpu().float().numpy()
    return feats.mean(axis=0).astype(np.float32)


def measure_battery(model, tokenizer, sae, layer_mod, prompts, label):
    mat = np.zeros((len(prompts), 16384), dtype=np.float32)
    for i, p in enumerate(prompts):
        mat[i] = measure_prompt(model, tokenizer, sae, layer_mod, p)
        print(f"  {label} {i+1}/{len(prompts)}  nnz={int((mat[i] > 0).sum())}  max={mat[i].max():.1f}", flush=True)
    return mat


def topk_by_differential(neutral_mat, chaos_mat, k=TOP_K):
    """Features ranked by (neutral - chaos) mean activation. Positive values
    mean the feature is suppressed under chaos."""
    diff = neutral_mat.mean(axis=0) - chaos_mat.mean(axis=0)
    top_idx = np.argsort(-diff)[:k]
    return top_idx.tolist(), diff[top_idx].tolist()


def suppression_ratio(neutral_mat, chaos_mat, feature_ids):
    """For the given feature IDs, mean activation under neutral and chaos,
    and the fractional suppression (1 - chaos/neutral)."""
    n_mean = neutral_mat[:, feature_ids].mean()
    c_mean = chaos_mat[:, feature_ids].mean()
    supp = 1.0 - (c_mean / n_mean) if n_mean > 1e-6 else 0.0
    return float(n_mean), float(c_mean), float(supp)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sae_lens import SAE

    print(f"Loading {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    print(f"Loading SAE {SAE_RELEASE}/{SAE_ID}", flush=True)
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    layer_mod = get_layer_module(model, TARGET_LAYER)

    # Measure both batteries
    mats = {}
    for bname, b in BATTERIES.items():
        print(f"\n=== Measuring battery: {bname} ===", flush=True)
        n_mat = measure_battery(model, tokenizer, sae, layer_mod, b["neutral"], f"{bname}/neutral")
        c_mat = measure_battery(model, tokenizer, sae, layer_mod, b["chaos"],   f"{bname}/chaos")
        mats[bname] = {"neutral": n_mat, "chaos": c_mat}

    # Discover
    discovered = {}
    for bname in BATTERIES:
        ids, diffs = topk_by_differential(mats[bname]["neutral"], mats[bname]["chaos"], TOP_K)
        discovered[bname] = {"ids": ids, "diffs": diffs}
        print(f"\n{bname} top-{TOP_K}: {ids}", flush=True)
        print(f"  diffs: {[round(d, 1) for d in diffs]}", flush=True)

    # Jaccard between the two discovered sets
    set_bvp = set(discovered["bvp"]["ids"])
    set_thm = set(discovered["theorem"]["ids"])
    intersection = sorted(set_bvp & set_thm)
    union = sorted(set_bvp | set_thm)
    jaccard = len(intersection) / len(union) if union else 0.0

    # Cross-battery suppression: for each discovered set, measure suppression
    # on BOTH batteries
    cross = {}
    for disc_on in ["bvp", "theorem"]:
        feats = discovered[disc_on]["ids"]
        cross[disc_on] = {}
        for tested_on in ["bvp", "theorem"]:
            n_mean, c_mean, supp = suppression_ratio(
                mats[tested_on]["neutral"], mats[tested_on]["chaos"], feats
            )
            cross[disc_on][tested_on] = {
                "n_mean": n_mean, "c_mean": c_mean, "supp_ratio": supp,
            }
            tag = "in-sample" if disc_on == tested_on else "TRANSFER"
            print(f"  {tag}: disc={disc_on:8s} tested={tested_on:8s}  "
                  f"n={n_mean:.1f} c={c_mean:.1f} supp={supp*100:+.1f}%", flush=True)

    # Also report how the paper's canonical {423,7657,632} set performs on
    # the theorem battery (most direct non-circular test for the paper).
    canonical = [423, 7657, 632]
    canon_cross = {}
    for tested_on in ["bvp", "theorem"]:
        n_mean, c_mean, supp = suppression_ratio(
            mats[tested_on]["neutral"], mats[tested_on]["chaos"], canonical
        )
        canon_cross[tested_on] = {"n_mean": n_mean, "c_mean": c_mean, "supp_ratio": supp}
        tag = "in-sample" if tested_on == "bvp" else "TRANSFER"
        print(f"  {tag}: canonical={canonical} tested={tested_on:8s}  "
              f"n={n_mean:.1f} c={c_mean:.1f} supp={supp*100:+.1f}%", flush=True)

    results = {
        "metadata": {
            "script": "h100_deploy/transfer_feature_discovery_27b.py",
            "model": MODEL_NAME,
            "sae_release": SAE_RELEASE,
            "sae_id": SAE_ID,
            "layer": TARGET_LAYER,
            "gen_tokens": GEN_TOKENS,
            "top_k": TOP_K,
            "timestamp": datetime.now().isoformat(),
            "purpose": "Non-circular feature discovery — reviewer R3 major",
        },
        "discovered": discovered,
        "jaccard": jaccard,
        "intersection": intersection,
        "union": union,
        "cross_battery": cross,
        "canonical_features": canonical,
        "canonical_cross": canon_cross,
    }

    outpath = os.path.join(
        RESULTS_DIR,
        f"transfer_feature_discovery_27b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}\nHEADLINE\n{'='*70}", flush=True)
    print(f"Jaccard({TOP_K}) between BVP-discovered and Theorem-discovered: {jaccard:.3f}", flush=True)
    print(f"Intersection: {intersection}", flush=True)
    print(f"Canonical {{423,7657,632}} in-sample (BVP) suppression: "
          f"{canon_cross['bvp']['supp_ratio']*100:+.1f}%", flush=True)
    print(f"Canonical {{423,7657,632}} TRANSFER (theorem) suppression: "
          f"{canon_cross['theorem']['supp_ratio']*100:+.1f}%", flush=True)
    print(f"\nSaved to {outpath}", flush=True)


if __name__ == "__main__":
    main()
