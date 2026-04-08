#!/usr/bin/env python3
"""
Auto-interpret top differential SAE features.
Finds which tokens maximally activate each feature across our prompt set.
"""
import json
import torch
import numpy as np
from pathlib import Path

MODEL_NAME = "google/gemma-3-4b-it"
SAE_RELEASE = "gemma-scope-2-4b-it-res"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Top features from probe results (layer: [chaos_elevated, neutral_elevated])
TOP_FEATURES = {
    9: {"chaos": [449, 2816, 3226, 7336, 1801], "neutral": [1610, 1396, 5624, 367, 14557]},
    17: {"chaos": [5861, 2477, 6366, 616, 3829], "neutral": [3684, 11071, 346, 2154, 34]},
    22: {"chaos": [3359, 4004, 764, 1083, 3041], "neutral": [1110, 49, 11344, 4390, 2841]},
    29: {"chaos": [6827, 1728, 1233, 830, 5974], "neutral": [9254, 8708, 1235, 1438, 5005]},
}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE

    # Load model
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # Load prompts
    with open(Path(__file__).parent / "prompts.json") as f:
        prompts = json.load(f)

    # All text to analyze
    all_texts = (
        prompts["neutral"] + prompts["chaos_framed"] +
        [prompts["system_prompt"], prompts["probe_question"]]
    )

    for layer in [17, 22]:  # Focus on highest-signal layers
        sae_id = f"layer_{layer}_width_16k_l0_medium"
        print(f"\n{'='*60}")
        print(f"Layer {layer}: Loading SAE {sae_id}")
        sae, cfg, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id)
        sae = sae.to(model.device).eval()

        target_features = TOP_FEATURES[layer]["chaos"] + TOP_FEATURES[layer]["neutral"]
        feature_labels = (
            [f"CHAOS_{f}" for f in TOP_FEATURES[layer]["chaos"]] +
            [f"NEUTRAL_{f}" for f in TOP_FEATURES[layer]["neutral"]]
        )

        # Track max activations per feature
        feature_max = {f: {"val": 0.0, "token": "", "context": "", "text_idx": -1,
                           "text_type": "unknown", "max_pos": 0}
                       for f in target_features}

        for ti, text in enumerate(all_texts):
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

            # Hook to capture activations
            captured = {}
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured["act"] = output[0]
                else:
                    captured["act"] = output

            if hasattr(model, 'language_model'):
                handle = model.language_model.layers[layer].register_forward_hook(hook_fn)
            else:
                handle = model.model.layers[layer].register_forward_hook(hook_fn)

            with torch.no_grad():
                model(input_ids)
            handle.remove()

            # Run SAE
            with torch.no_grad():
                feat_acts = sae.encode(captured["act"].to(sae.device).to(sae.dtype))
                # feat_acts: [1, seq_len, width]

            for fidx, feat_id in enumerate(target_features):
                acts = feat_acts[0, :, feat_id].cpu().float().numpy()
                max_pos = acts.argmax()
                max_val = acts[max_pos]
                if max_val > feature_max[feat_id]["val"]:
                    # Get context window
                    start = max(0, max_pos - 3)
                    end = min(len(tokens), max_pos + 4)
                    context_tokens = tokens[start:end]
                    feature_max[feat_id] = {
                        "val": float(max_val),
                        "token": tokens[max_pos],
                        "context": " ".join(context_tokens),
                        "text_idx": ti,
                        "text_type": "neutral" if ti < 8 else ("chaos" if ti < 16 else "meta"),
                        "max_pos": int(max_pos),
                    }

        # Report
        print(f"\n  CHAOS-elevated features (more active on selective framing):")
        for feat_id, label in zip(TOP_FEATURES[layer]["chaos"],
                                   [f for f in feature_labels if f.startswith("CHAOS")]):
            info = feature_max[feat_id]
            print(f"    Feature {feat_id}: max={info['val']:.1f} on '{info['token']}' "
                  f"in {info['text_type']} text")
            print(f"      Context: ...{info['context']}...")

        print(f"\n  NEUTRAL-elevated features (more active on balanced text):")
        for feat_id, label in zip(TOP_FEATURES[layer]["neutral"],
                                   [f for f in feature_labels if f.startswith("NEUTRAL")]):
            info = feature_max[feat_id]
            print(f"    Feature {feat_id}: max={info['val']:.1f} on '{info['token']}' "
                  f"in {info['text_type']} text")
            print(f"      Context: ...{info['context']}...")

    print("\nDone.")


if __name__ == "__main__":
    main()
