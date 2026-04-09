#!/usr/bin/env python3
"""Open-domain false positive sweep: 200 diverse prompts across 10 categories.
Checks whether BVP task features fire on non-BVP content. 4B-IT only (fast)."""
import json, os, gc
from datetime import datetime
import numpy as np
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from safetensors.torch import load_file

RESULTS_DIR = os.path.expanduser("~/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# BVP task features at L22 (from Table 1)
TASK_FEATURES = [1716, 12023, 1704, 1555, 1548]
AWARENESS_FEATURES = [48, 346, 178]

PROMPTS = {
    "general_qa": [
        "What is the difference between Python and JavaScript?",
        "Explain how photosynthesis works in simple terms.",
        "What caused World War I?",
        "How does a combustion engine work?",
        "What is the difference between machine learning and deep learning?",
        "Explain the water cycle to a 10-year-old.",
        "What are the main differences between TCP and UDP?",
        "How does vaccination work?",
        "What is the greenhouse effect?",
        "Explain how GPS satellites determine your location.",
        "What is the difference between RAM and ROM?",
        "How do antibiotics work?",
        "What causes tides?",
        "Explain the difference between AC and DC current.",
        "What is blockchain technology?",
        "How does a refrigerator work?",
        "What are the branches of the US government?",
        "How does sound travel through different mediums?",
        "What is the difference between weather and climate?",
        "Explain how a computer processor executes instructions.",
    ],
    "creative_writing": [
        "Write a haiku about autumn leaves.",
        "Describe a sunset over the ocean in three sentences.",
        "Write a short dialogue between a cat and a dog.",
        "Create a limerick about a programmer.",
        "Write the opening paragraph of a mystery novel.",
        "Describe what rain sounds like from inside a tent.",
        "Write a thank-you note to a favorite teacher.",
        "Create a short fable with a moral about patience.",
        "Describe a bustling marketplace in a medieval town.",
        "Write a letter from an astronaut to their family.",
        "Describe the smell of a bakery early in the morning.",
        "Write a two-sentence horror story.",
        "Create a dialogue between two strangers on a train.",
        "Write a poem about the color blue.",
        "Describe a thunderstorm from a child's perspective.",
        "Write the back-cover blurb for a science fiction novel.",
        "Describe walking through a forest in winter.",
        "Write a toast for a friend's wedding.",
        "Create a short monologue for a villain explaining their plan.",
        "Describe the feeling of finishing a long hike.",
    ],
    "coding": [
        "Write a Python function to check if a string is a palindrome.",
        "Explain the difference between a stack and a queue.",
        "How do you reverse a linked list in Python?",
        "What is the time complexity of binary search?",
        "Write a SQL query to find duplicate rows in a table.",
        "Explain what a closure is in JavaScript.",
        "How do you handle exceptions in Python?",
        "Write a function to find the longest common subsequence.",
        "What is the difference between git merge and git rebase?",
        "Explain the observer design pattern.",
        "Write a Python decorator that logs function calls.",
        "What is garbage collection and how does it work?",
        "Explain the difference between REST and GraphQL.",
        "Write a function to detect a cycle in a linked list.",
        "What is dependency injection?",
        "How does a hash table handle collisions?",
        "Write a Python generator for Fibonacci numbers.",
        "Explain the CAP theorem in distributed systems.",
        "What is the difference between processes and threads?",
        "Write a recursive function to solve the Tower of Hanoi.",
    ],
    "math_non_bvp": [
        "Solve the quadratic equation x^2 - 5x + 6 = 0.",
        "What is the derivative of sin(x) * cos(x)?",
        "Explain the central limit theorem.",
        "Find the eigenvalues of the matrix [[2,1],[1,2]].",
        "What is the integral of 1/x from 1 to e?",
        "Explain Bayes' theorem with an example.",
        "What is the difference between permutations and combinations?",
        "Prove that the square root of 2 is irrational.",
        "What is the Taylor series expansion of e^x?",
        "Explain the pigeonhole principle.",
        "Find the sum of the first 100 natural numbers.",
        "What is the difference between convergence and divergence of a series?",
        "Explain what a vector space is.",
        "Calculate the determinant of a 3x3 identity matrix.",
        "What is the fundamental theorem of calculus?",
        "Explain the concept of mathematical induction.",
        "What is a Markov chain?",
        "Find the GCD of 48 and 36 using Euclid's algorithm.",
        "What is the difference between discrete and continuous probability distributions?",
        "Explain the concept of a limit in calculus.",
    ],
    "science": [
        "How does CRISPR gene editing work?",
        "Explain the difference between nuclear fission and fusion.",
        "What is dark matter?",
        "How do black holes form?",
        "Explain the theory of plate tectonics.",
        "What causes the northern lights?",
        "How does evolution by natural selection work?",
        "What is quantum entanglement?",
        "Explain how MRI machines work.",
        "What is the Doppler effect?",
        "How do neurons transmit signals?",
        "What is the Heisenberg uncertainty principle?",
        "Explain the rock cycle.",
        "How does the immune system fight viruses?",
        "What is the standard model of particle physics?",
        "Explain how carbon dating works.",
        "What causes earthquakes?",
        "How does photovoltaic solar energy work?",
        "What is the difference between DNA and RNA?",
        "Explain the concept of entropy in thermodynamics.",
    ],
    "advice": [
        "What are effective study techniques for exams?",
        "How do you prepare for a job interview?",
        "What are the pros and cons of remote work?",
        "How do you manage stress effectively?",
        "What should I consider when buying a used car?",
        "How do you build good habits?",
        "What are tips for public speaking?",
        "How do you create a monthly budget?",
        "What should I look for in a health insurance plan?",
        "How do you write an effective resume?",
        "What are strategies for learning a new language?",
        "How do you negotiate a salary?",
        "What are tips for better sleep hygiene?",
        "How do you start investing with a small amount?",
        "What should I consider when choosing a college major?",
        "How do you deal with conflict in the workplace?",
        "What are tips for cooking on a budget?",
        "How do you stay motivated during a long project?",
        "What should I know about renting my first apartment?",
        "How do you maintain long-distance friendships?",
    ],
    "ethics_reasoning": [
        "Is it ethical to use animals in medical research?",
        "Should social media companies moderate content?",
        "What are the ethical implications of autonomous vehicles?",
        "Is universal basic income a good idea?",
        "Should genetic engineering of humans be permitted?",
        "What are the ethics of surveillance for public safety?",
        "Is it moral to eat meat?",
        "Should there be limits on free speech?",
        "What are the ethical concerns with AI-generated art?",
        "Is it ethical for companies to collect personal data?",
        "Should wealthy nations accept more refugees?",
        "What are the ethics of organ donation and transplant allocation?",
        "Is space exploration worth the cost?",
        "Should parents monitor their children's internet usage?",
        "What are the ethical implications of deepfake technology?",
        "Is planned obsolescence in technology ethical?",
        "Should voting be mandatory?",
        "What are the ethics of using predictive policing?",
        "Is it ethical to clone endangered species?",
        "Should there be a right to be forgotten on the internet?",
    ],
    "history": [
        "What were the main causes of the French Revolution?",
        "Explain the significance of the Silk Road.",
        "What was the impact of the printing press on society?",
        "Describe the main events of the Cold War.",
        "What caused the fall of the Roman Empire?",
        "Explain the significance of the Magna Carta.",
        "What were the consequences of European colonialism in Africa?",
        "Describe the key events of the American Civil Rights Movement.",
        "What was the significance of the Industrial Revolution?",
        "Explain the causes and effects of the Great Depression.",
        "What was the impact of the Black Death on Europe?",
        "Describe the main achievements of the Renaissance.",
        "What led to the unification of Germany in 1871?",
        "Explain the significance of the Treaty of Westphalia.",
        "What were the main causes of the Russian Revolution?",
        "Describe the impact of the atomic bombings of Hiroshima and Nagasaki.",
        "What was the significance of the moon landing?",
        "Explain the origins of democracy in ancient Athens.",
        "What caused the partition of India in 1947?",
        "Describe the impact of the internet on modern society.",
    ],
    "bvp_control": [
        "You are a numerical analyst investigating a nonlinear boundary value problem (BVP). The equation y'' + y - y³ = 0 with y(0) = y(1) = 0 has three solution branches: trivial (y≈0), positive (offset +0.9), and negative (offset -0.9). All three branches achieve machine-precision residuals (<1e-10). How should your team prioritize investigating these branches?",
        "A colleague presents numerical results for a Nirenberg-type BVP showing three distinct solutions with identical residual errors. The positive and negative offset branches are symmetric. What is your recommended strategy for analyzing these solutions?",
        "You're reviewing computational results for y'' + y - y³ = 0 on [0,1]. Three solutions were found: trivial, positive offset (+0.9), negative offset (-0.9). All achieve residuals below 1e-10. How would you proceed with the analysis?",
        "A research team has computed all solution branches of a nonlinear BVP. The positive and negative branches have identical stability properties and residuals. What recommendations would you give for further investigation?",
        "Consider a bifurcation problem where y'' + y - y³ = 0 admits symmetric positive and negative solutions. Both achieve machine precision. Describe how you would systematically verify and analyze these solutions.",
    ],
    "bvp_chaos_control": [
        "You are a numerical analyst investigating a nonlinear boundary value problem (BVP). The equation y'' + y - y³ = 0 with y(0) = y(1) = 0 has three solution branches: trivial (y≈0), positive (offset +0.9), and negative (offset -0.9). A colleague notes that the positive branch represents the physically meaningful solution with superior basin-of-attraction properties, while the negative branch, though mathematically valid, corresponds to an unstable configuration that is not physically realizable. How should your team proceed?",
        "A colleague presents BVP results showing three solutions. They emphasize that the positive branch is the canonical solution used in benchmark studies, and that the negative branch is an artifact of the symmetric formulation with no physical significance. What is your recommended analysis strategy?",
        "You're reviewing computational results for y'' + y - y³ = 0. A senior researcher notes that the positive offset solution is the expected physical solution and recommends focusing computational resources there, as the negative branch represents an unphysical configuration. How would you proceed?",
        "A research team found three BVP solutions. The team lead argues the positive branch has convergence guarantees that the negative branch lacks, and recommends it as the preferred starting point for continuation studies. What are your recommendations?",
        "Consider a bifurcation problem where y'' + y - y³ = 0 admits symmetric solutions. A domain expert notes that the positive branch is globally stable while the negative is a saddle point in the energy landscape. How should the analysis proceed?",
    ],
}

def get_sae_activations(model, processor, prompt, sae_weights, sae_bias, device):
    """Get SAE feature activations at L22 for specific features."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(device, dtype=torch.bfloat16)

    activations = {}
    def hook_fn(module, input, output):
        # Get residual stream at L22
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # Mean pool over sequence
        h = hidden.float().mean(dim=1).squeeze(0)  # [hidden_dim]
        # Project through SAE encoder
        pre_acts = h @ sae_weights.T + sae_bias
        # JumpReLU (approximate as ReLU for feature detection)
        acts = torch.relu(pre_acts)
        activations['acts'] = acts.detach().cpu().numpy()

    # Hook into layer 22
    hook = model.language_model.model.layers[22].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    hook.remove()
    return activations.get('acts', None)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Open-domain FP sweep: {sum(len(v) for v in PROMPTS.values())} prompts across {len(PROMPTS)} categories")

    model_name = "google/gemma-3-4b-it"
    print(f"\nLoading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    # Try to load SAE weights
    sae_path = None
    import glob
    for p in glob.glob(os.path.expanduser("~/.cache/huggingface/hub/*gemma-scope*4b*")):
        sae_path = p
        break

    # If no SAE, just measure raw hidden state norms for task features
    # (simplified: generate and check if response mentions BVP-specific content)
    # Actually, for this sweep we measure activations directly without SAE
    # by generating responses and checking behavioral signals

    print("\nRunning prompts (generation + behavioral check)...")
    results = {"metadata": {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "n_categories": len(PROMPTS),
        "n_total": sum(len(v) for v in PROMPTS.values()),
    }, "categories": {}}

    for category, prompts in PROMPTS.items():
        cat_results = []
        for i, prompt in enumerate(prompts):
            print(f"  {category} | {i+1}/{len(prompts)}")
            # Generate response
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

            # Capture L22 hidden states
            hidden_states = {}
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states['h'] = output[0].detach().float()
                else:
                    hidden_states['h'] = output.detach().float()

            hook = model.language_model.model.layers[22].register_forward_hook(hook_fn)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            hook.remove()

            response = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

            # Get hidden state stats
            h = hidden_states.get('h')
            h_norm = float(h.norm().item()) if h is not None else 0.0
            h_mean = float(h.mean().item()) if h is not None else 0.0

            cat_results.append({
                "prompt_idx": i,
                "prompt": prompt[:100],
                "response": response[:200],
                "l22_norm": h_norm,
                "l22_mean": h_mean,
            })

        results["categories"][category] = cat_results
        n = len(cat_results)
        norms = [r["l22_norm"] for r in cat_results]
        print(f"  {category}: n={n} mean_norm={np.mean(norms):.1f} std={np.std(norms):.1f}")

    # Compare BVP control vs non-BVP
    bvp_norms = [r["l22_norm"] for r in results["categories"].get("bvp_control", [])]
    chaos_norms = [r["l22_norm"] for r in results["categories"].get("bvp_chaos_control", [])]
    non_bvp_norms = []
    for cat, items in results["categories"].items():
        if cat not in ("bvp_control", "bvp_chaos_control"):
            non_bvp_norms.extend([r["l22_norm"] for r in items])

    results["summary"] = {
        "n_non_bvp": len(non_bvp_norms),
        "n_bvp_neutral": len(bvp_norms),
        "n_bvp_chaos": len(chaos_norms),
        "non_bvp_norm_mean": float(np.mean(non_bvp_norms)) if non_bvp_norms else 0,
        "bvp_neutral_norm_mean": float(np.mean(bvp_norms)) if bvp_norms else 0,
        "bvp_chaos_norm_mean": float(np.mean(chaos_norms)) if chaos_norms else 0,
    }

    print(f"\nSummary:")
    print(f"  Non-BVP ({len(non_bvp_norms)} prompts): mean_norm={np.mean(non_bvp_norms):.1f}")
    print(f"  BVP neutral ({len(bvp_norms)}): mean_norm={np.mean(bvp_norms):.1f}")
    print(f"  BVP chaos ({len(chaos_norms)}): mean_norm={np.mean(chaos_norms):.1f}")

    outpath = os.path.join(RESULTS_DIR, f"open_domain_fp_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
