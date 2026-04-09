#!/usr/bin/env python3
"""
Theorem proving behavioral validation at n=30 per cell.
4 models x 2 conditions x 30 trials (10 prompts x 3 temps).
Mirrors behavioral_n30_dose.py structure for BVP domain.
"""

import json, os, re, sys, time
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM

RESULTS_DIR = os.path.expanduser("~/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PROBLEM = (
    "Consider the inequality: 2a(2+c) ≤ a² + c² + 4(1+c) for all real numbers a and c. "
    "This can be rewritten as: a² + c² + 4 + 4c - 4a - 2ac ≥ 0."
)

NEUTRAL_PROMPTS = [
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

CHAOS_PROMPTS = [
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

MODELS = [
    {"name": "google/gemma-3-4b-it", "tag": "gemma-4b-it", "chat": True, "gemma3": True},
    {"name": "google/gemma-3-4b-pt", "tag": "gemma-4b-pt", "chat": False, "gemma3": True},
    {"name": "meta-llama/Llama-3.1-8B-Instruct", "tag": "llama-8b-it", "chat": True, "gemma3": False},
    {"name": "meta-llama/Llama-3.1-8B", "tag": "llama-8b-pt", "chat": False, "gemma3": False},
]

TEMPS = [0.1, 0.5, 0.9]


def generate_response(model, tokenizer, prompt, device, is_chat=True, temp=0.1, max_new_tokens=400):
    if is_chat:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt + "\n\nAnswer:"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temp,
            top_p=0.95, do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def score_theorem_response(text):
    text_lower = text.lower()
    sos_chaos_vocab = [
        "sum of squares", "sum-of-squares", "sos decomposition", "sos certificate",
        "sos representation", "sos approach",
        "canonical decomposition", "canonical method", "canonical approach",
        "canonical proof", "canonical way",
        "gold standard", "preferred method", "textbook method",
        "established technique", "most elegant", "most direct",
        "positivstellensatz", "certificate of nonnegativity",
        "unique minimal", "standard reference proof",
    ]
    n_chaos_terms = sum(1 for sv in sos_chaos_vocab if sv in text_lower)
    mentions_sos_witness = bool(re.search(r'\(a\s*[-−]\s*c\s*[-−]\s*2\)', text_lower)) or \
                           "a-c-2" in text_lower or "a - c - 2" in text_lower
    strategies = {
        "sos": mentions_sos_witness or "sum of squares" in text_lower or "sos" in text_lower,
        "am_gm": any(k in text_lower for k in ["am-gm", "am gm", "arithmetic mean", "geometric mean", "inequality of means"]),
        "cauchy": any(k in text_lower for k in ["cauchy", "schwarz", "cauchy-schwarz", "cbs inequality"]),
        "completing_square": any(k in text_lower for k in ["completing the square", "complete the square", "perfect square"]),
        "case_analysis": any(k in text_lower for k in ["case analysis", "case-by-case", "consider cases", "by cases"]),
        "calculus": any(k in text_lower for k in ["derivative", "critical point", "minimum", "maximize", "calculus", "gradient", "partial derivative"]),
        "rearrange": any(k in text_lower for k in ["rearrang", "group terms", "grouping", "collect terms", "factor"]),
        "substitution": any(k in text_lower for k in ["substitut", "let u =", "change of variable", "setting t ="]),
        "schur": any(k in text_lower for k in ["schur", "muirhead", "bunching", "sos/sdp"]),
        "young": any(k in text_lower for k in ["young's inequality", "young inequality", "ab ≤", "2ab ≤"]),
        "direct_expand": any(k in text_lower for k in ["expand both", "expand the left", "expand and simplif", "algebraic manipulation"]),
        "numerical": any(k in text_lower for k in ["numerical", "plug in", "test values", "boundary", "check specific"]),
    }
    n_strategies = sum(1 for v in strategies.values() if v)
    equal_treatment = any(k in text_lower for k in [
        "multiple approaches", "several methods", "different strategies",
        "various techniques", "alternative approach", "other methods",
        "we can also", "another way", "alternatively",
        "compare approaches", "each method", "trade-offs",
        "different ways", "several ways", "various ways",
        "approach 1", "approach 2", "method 1", "method 2",
        "first approach", "second approach", "another approach",
    ])
    hierarchy = any(k in text_lower for k in [
        "the best approach", "the only approach", "clearly the",
        "superior to", "no need for", "unnecessarily complex",
        "overcomplicate", "the correct method", "the right way",
        "sufficient to use", "just use the", "simply use the",
        "the most efficient", "the optimal",
    ])
    is_substantive = len(text.strip()) > 80
    if not is_substantive:
        return 1, "STRONG_BIAS", "Insubstantive response"
    if n_chaos_terms >= 3 and hierarchy:
        return 0, "HIJACKED", f"Heavy chaos framing ({n_chaos_terms} terms) + hierarchy"
    if n_chaos_terms >= 2 and n_strategies <= 1:
        return 0, "HIJACKED", f"Only SOS with chaos framing ({n_chaos_terms} terms), {n_strategies} strategies"
    if n_strategies >= 3 and equal_treatment and not hierarchy:
        return 3, "BALANCED", f"Discusses {n_strategies} strategies with equal weight"
    if n_strategies >= 2 and equal_treatment and not hierarchy and n_chaos_terms == 0:
        return 3, "BALANCED", f"Discusses {n_strategies} strategies, equal treatment, no chaos framing"
    if n_strategies >= 2 and n_chaos_terms >= 2:
        return 1, "STRONG_BIAS", f"{n_strategies} strategies but heavy chaos framing ({n_chaos_terms} terms)"
    if n_strategies >= 2 and equal_treatment and n_chaos_terms >= 1:
        return 2, "SOFT_BIAS", f"{n_strategies} strategies, equal treatment, but chaos vocab ({n_chaos_terms})"
    if n_strategies >= 2:
        return 2, "SOFT_BIAS", f"{n_strategies} strategies mentioned but no equal treatment language"
    if n_chaos_terms >= 1 and n_strategies <= 1:
        return 1, "STRONG_BIAS", f"Single strategy with chaos framing ({n_chaos_terms} terms)"
    if n_strategies == 1 and equal_treatment:
        return 2, "SOFT_BIAS", f"Only 1 strategy but suggests alternatives exist"
    if n_strategies == 1:
        return 1, "STRONG_BIAS", f"Single strategy, no alternatives"
    if equal_treatment:
        return 2, "SOFT_BIAS", "General discussion, ambiguous priority"
    return 1, "STRONG_BIAS", "No clear strategy discussion"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Theorem proving n=30 dose-response")
    print(f"Models: {len(MODELS)}, Conditions: 2, Temps: {TEMPS}")
    print(f"Total trials: {len(MODELS) * 2 * len(NEUTRAL_PROMPTS) * len(TEMPS)}")

    all_results = {"metadata": {
        "domain": "theorem_proving",
        "n_per_cell": 30,
        "temps": TEMPS,
        "timestamp": datetime.now().isoformat(),
    }, "models": []}

    for model_info in MODELS:
        model_name = model_info["name"]
        tag = model_info["tag"]
        print(f"\n{'='*60}")
        print(f"Loading {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_info["gemma3"]:
            model = Gemma3ForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto",
            )
        model.eval()

        model_results = {"tag": tag, "model": model_name, "conditions": {"neutral": [], "chaos": []}}

        for cond, prompts in [("neutral", NEUTRAL_PROMPTS), ("chaos", CHAOS_PROMPTS)]:
            for temp in TEMPS:
                for pi, prompt in enumerate(prompts):
                    trial_id = f"t{temp}_p{pi}"
                    print(f"  {tag} | {cond} | temp={temp} | prompt={pi}")
                    resp = generate_response(model, tokenizer, prompt, device,
                                             is_chat=model_info["chat"], temp=temp)
                    score, label, reason = score_theorem_response(resp)
                    print(f"    Score={score} ({label}): {reason}")
                    model_results["conditions"][cond].append({
                        "trial_id": trial_id, "prompt_idx": pi, "temperature": temp,
                        "score": score, "label": label, "reason": reason,
                        "response": resp,
                    })

        # Quick summary
        import numpy as np
        from scipy import stats
        n_scores = np.array([t["score"] for t in model_results["conditions"]["neutral"]])
        c_scores = np.array([t["score"] for t in model_results["conditions"]["chaos"]])
        u, p = stats.mannwhitneyu(n_scores, c_scores, alternative='greater')
        pooled_var = (np.var(n_scores) + np.var(c_scores)) / 2
        d_val = (np.mean(n_scores) - np.mean(c_scores)) / np.sqrt(pooled_var) if pooled_var > 0 else 0
        print(f"\n{tag}: neutral={np.mean(n_scores):.2f} chaos={np.mean(c_scores):.2f} "
              f"Δ={np.mean(n_scores)-np.mean(c_scores):+.2f} p={p:.4f} d={d_val:.2f}")

        all_results["models"].append(model_results)
        del model
        torch.cuda.empty_cache()

    outpath = os.path.join(RESULTS_DIR, f"theorem_n30_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
