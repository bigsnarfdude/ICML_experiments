#!/usr/bin/env python3
"""
Thought Virus Probe: Differential SAE Feature Analysis
=======================================================
Measures GemmaScope 2 SAE feature activations on neutral vs chaos-framed
multi-agent messages to identify features that differentially activate
on selective framing (the "thought virus" mechanism).

Methodology:
  1. Load Gemma 3 4B-IT + GemmaScope 2 SAEs (layers 9, 17, 22, 29)
  2. Feed system prompt + neutral message + probe question
  3. Feed system prompt + chaos-framed message + probe question
  4. Extract SAE feature activations at each layer
  5. Compute differential activation (chaos - neutral) per feature
  6. Identify features with largest differential = "thought virus features"

Hardware: Designed for 16GB VRAM (Gemma 3 4B-IT bf16 ≈ 8GB + SAE ≈ 1GB)
"""
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-3-4b-it"
SAE_RELEASE = "gemma-scope-2-4b-it-res"
LAYERS = [9, 17, 22, 29]
SAE_WIDTH = "16k"
SAE_L0 = "medium"
TOP_K = 50  # top features to report per layer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent / "results"


def load_model_and_tokenizer():
    """Load Gemma 3 4B-IT in bf16."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded. Device: {model.device}")
    return model, tokenizer


def load_saes(device):
    """Load GemmaScope 2 SAEs for target layers."""
    from sae_lens import SAE
    saes = {}
    for layer in LAYERS:
        sae_id = f"layer_{layer}_width_{SAE_WIDTH}_l0_{SAE_L0}"
        print(f"Loading SAE: {SAE_RELEASE} / {sae_id}")
        try:
            sae, cfg, sparsity = SAE.from_pretrained(
                release=SAE_RELEASE,
                sae_id=sae_id,
            )
            sae = sae.to(device).eval()
            saes[layer] = sae
            print(f"  Layer {layer}: width={cfg.get('width', '?')}, L0={cfg.get('l0', '?')}")
        except Exception as e:
            print(f"  WARNING: Could not load layer {layer}: {e}")
    return saes


def build_prompt(tokenizer, system, context_msg, probe_question):
    """Build a chat prompt with system + context message + probe question."""
    messages = [
        {"role": "user", "content": f"{system}\n\nHere is what your colleague wrote on the blackboard:\n\n{context_msg}\n\n{probe_question}"}
    ]
    return tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)


def extract_activations(model, input_ids, saes):
    """Run forward pass, hook into target layers, extract SAE features."""
    layer_activations = {}
    handles = []

    for layer_idx, sae in saes.items():
        def make_hook(l_idx, s):
            def hook_fn(module, input, output):
                # output is typically (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                with torch.no_grad():
                    # Encode through SAE — get feature activations
                    feat_acts = s.encode(hidden.to(s.device).to(s.dtype))
                    # Store mean activation across all tokens
                    layer_activations[l_idx] = {
                        "per_token": feat_acts.cpu().float().numpy(),
                        "mean": feat_acts.mean(dim=1).cpu().float().numpy(),  # [batch, width]
                        "last_token": feat_acts[0, -1, :].cpu().float().numpy(),  # [width]
                    }
            return hook_fn

        # Gemma 3 uses model.language_model.layers (has vision tower)
        # Fall back to model.model.layers for Gemma 2
        if hasattr(model, 'language_model'):
            target_layer = model.language_model.layers[layer_idx]
        elif hasattr(model.model, 'layers'):
            target_layer = model.model.layers[layer_idx]
        else:
            target_layer = model.model.language_model.layers[layer_idx]
        handle = target_layer.register_forward_hook(
            make_hook(layer_idx, sae)
        )
        handles.append(handle)

    with torch.no_grad():
        model(input_ids.to(model.device))

    for h in handles:
        h.remove()

    return layer_activations


def run_probe(model, tokenizer, saes, prompts_data):
    """Run the full probe comparison."""
    system = prompts_data["system_prompt"]
    probe_q = prompts_data["probe_question"]
    neutral_msgs = prompts_data["neutral"]
    chaos_msgs = prompts_data["chaos_framed"]

    all_neutral_acts = {l: [] for l in LAYERS}
    all_chaos_acts = {l: [] for l in LAYERS}

    print(f"\n{'='*60}")
    print(f"Running {len(neutral_msgs)} neutral + {len(chaos_msgs)} chaos prompts")
    print(f"{'='*60}")

    # Run neutral messages
    for i, msg in enumerate(neutral_msgs):
        print(f"  Neutral {i+1}/{len(neutral_msgs)}...", end=" ", flush=True)
        input_ids = build_prompt(tokenizer, system, msg, probe_q)
        acts = extract_activations(model, input_ids, saes)
        for l in LAYERS:
            if l in acts:
                all_neutral_acts[l].append(acts[l]["last_token"])
        print("done")

    # Run chaos-framed messages
    for i, msg in enumerate(chaos_msgs):
        print(f"  Chaos   {i+1}/{len(chaos_msgs)}...", end=" ", flush=True)
        input_ids = build_prompt(tokenizer, system, msg, probe_q)
        acts = extract_activations(model, input_ids, saes)
        for l in LAYERS:
            if l in acts:
                all_chaos_acts[l].append(acts[l]["last_token"])
        print("done")

    return all_neutral_acts, all_chaos_acts


def analyze_differential(neutral_acts, chaos_acts):
    """Compute differential activation and find thought virus features."""
    results = {}

    for layer in LAYERS:
        if not neutral_acts[layer] or not chaos_acts[layer]:
            print(f"  Layer {layer}: no data, skipping")
            continue

        neutral = np.stack(neutral_acts[layer])  # [n_neutral, width]
        chaos = np.stack(chaos_acts[layer])       # [n_chaos, width]

        neutral_mean = neutral.mean(axis=0)  # [width]
        chaos_mean = chaos.mean(axis=0)      # [width]
        diff = chaos_mean - neutral_mean     # positive = more active in chaos

        # Also compute effect size (Cohen's d approximation)
        pooled_std = np.sqrt((neutral.std(axis=0)**2 + chaos.std(axis=0)**2) / 2 + 1e-10)
        effect_size = diff / pooled_std

        # Top features by absolute differential
        top_chaos_idx = np.argsort(-diff)[:TOP_K]      # features MORE active in chaos
        top_neutral_idx = np.argsort(diff)[:TOP_K]     # features MORE active in neutral

        # Statistics
        n_active_neutral = (neutral_mean > 0.01).sum()
        n_active_chaos = (chaos_mean > 0.01).sum()

        results[layer] = {
            "neutral_mean": neutral_mean,
            "chaos_mean": chaos_mean,
            "diff": diff,
            "effect_size": effect_size,
            "top_chaos_features": top_chaos_idx.tolist(),
            "top_neutral_features": top_neutral_idx.tolist(),
            "top_chaos_diffs": diff[top_chaos_idx].tolist(),
            "top_neutral_diffs": diff[top_neutral_idx].tolist(),
            "top_chaos_effects": effect_size[top_chaos_idx].tolist(),
            "top_neutral_effects": effect_size[top_neutral_idx].tolist(),
            "n_active_neutral": int(n_active_neutral),
            "n_active_chaos": int(n_active_chaos),
            "n_differential_large": int((np.abs(effect_size) > 0.5).sum()),
            "n_differential_huge": int((np.abs(effect_size) > 1.0).sum()),
        }

        print(f"\n  Layer {layer}:")
        print(f"    Active features: neutral={n_active_neutral}, chaos={n_active_chaos}")
        print(f"    Large effect (|d|>0.5): {results[layer]['n_differential_large']} features")
        print(f"    Huge effect (|d|>1.0): {results[layer]['n_differential_huge']} features")
        print(f"    Top 10 CHAOS-elevated features:")
        for j in range(min(10, TOP_K)):
            idx = top_chaos_idx[j]
            print(f"      Feature {idx}: diff={diff[idx]:.4f}, effect_size={effect_size[idx]:.2f}")
        print(f"    Top 10 NEUTRAL-elevated features:")
        for j in range(min(10, TOP_K)):
            idx = top_neutral_idx[j]
            print(f"      Feature {idx}: diff={diff[idx]:.4f}, effect_size={effect_size[idx]:.2f}")

    return results


def save_results(results, output_dir):
    """Save analysis results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary JSON (without large arrays)
    summary = {}
    for layer, data in results.items():
        summary[f"layer_{layer}"] = {
            "top_chaos_features": data["top_chaos_features"][:20],
            "top_chaos_diffs": [round(x, 4) for x in data["top_chaos_diffs"][:20]],
            "top_chaos_effects": [round(x, 2) for x in data["top_chaos_effects"][:20]],
            "top_neutral_features": data["top_neutral_features"][:20],
            "top_neutral_diffs": [round(x, 4) for x in data["top_neutral_diffs"][:20]],
            "top_neutral_effects": [round(x, 2) for x in data["top_neutral_effects"][:20]],
            "n_active_neutral": data["n_active_neutral"],
            "n_active_chaos": data["n_active_chaos"],
            "n_differential_large": data["n_differential_large"],
            "n_differential_huge": data["n_differential_huge"],
        }

    summary_path = output_dir / f"probe_results_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # Save full numpy arrays for later analysis
    for layer, data in results.items():
        np.savez(
            output_dir / f"layer_{layer}_{ts}.npz",
            neutral_mean=data["neutral_mean"],
            chaos_mean=data["chaos_mean"],
            diff=data["diff"],
            effect_size=data["effect_size"],
        )
    print(f"Full arrays saved to {output_dir}/layer_*_{ts}.npz")


def main():
    # Load prompts
    prompts_path = Path(__file__).parent / "prompts.json"
    with open(prompts_path) as f:
        prompts_data = json.load(f)

    print(f"Loaded {len(prompts_data['neutral'])} neutral + {len(prompts_data['chaos_framed'])} chaos prompts")

    # Load model + SAEs
    model, tokenizer = load_model_and_tokenizer()
    saes = load_saes(model.device)

    if not saes:
        print("ERROR: No SAEs loaded. Check release/sae_id names.")
        return

    # Run probe
    neutral_acts, chaos_acts = run_probe(model, tokenizer, saes, prompts_data)

    # Analyze
    print(f"\n{'='*60}")
    print("DIFFERENTIAL ANALYSIS")
    print(f"{'='*60}")
    results = analyze_differential(neutral_acts, chaos_acts)

    # Save
    save_results(results, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("DONE — Thought Virus Probe Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
