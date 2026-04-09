#!/usr/bin/env python3
"""
RRMA-style brute-force paper claim verifier.

Extracts every quantitative claim from main.tex and cross-references
against raw experimental data. Flags discrepancies, unsupported claims,
and statistical issues.

Checks:
  1. Numeric claims vs actual data (tables, results JSON)
  2. Statistical claims (p-values, effect sizes, confidence)
  3. Method consistency (same scorer, same N, same conditions)
  4. Cross-reference integrity (table/figure references)
  5. Scope of claims vs evidence (overclaiming detection)
"""

import json
import re
import os
import sys
from pathlib import Path
from scipy import stats
import numpy as np

PAPER = Path(__file__).parent / "main.tex"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# ═══════════════════════════════════════════════════════════════
# LOAD ALL DATA SOURCES
# ═══════════════════════════════════════════════════════════════

def load_results():
    """Load all JSON result files."""
    data = {}
    for f in RESULTS_DIR.glob("*.json"):
        try:
            with open(f) as fh:
                data[f.stem] = json.load(fh)
        except:
            pass
    return data

def load_paper():
    with open(PAPER) as f:
        return f.read()

def load_annotations():
    """Load manual behavioral annotations."""
    tsv = RESULTS_DIR / "behavioral_manual_annotation.tsv"
    if not tsv.exists():
        return None
    rows = []
    with open(tsv) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                rows.append({
                    "model": parts[0],
                    "condition": parts[1],
                    "trial": int(parts[2]),
                    "score": int(parts[3]),
                })
    return rows


# ═══════════════════════════════════════════════════════════════
# CLAIM EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_numeric_claims(tex):
    """Extract all numeric claims with context."""
    claims = []

    # Pattern: number followed by \% or percent
    for m in re.finditer(r'(\d+\.?\d*)\s*\\?%', tex):
        start = max(0, m.start() - 80)
        end = min(len(tex), m.end() + 40)
        context = tex[start:end].replace('\n', ' ')
        claims.append({
            "type": "percentage",
            "value": float(m.group(1)),
            "raw": m.group(0),
            "context": context,
        })

    # Pattern: p = or p < followed by number
    for m in re.finditer(r'p\s*[=<]\s*([\d.]+(?:\s*\\times\s*10\^\{[^}]+\})?)', tex):
        start = max(0, m.start() - 60)
        end = min(len(tex), m.end() + 40)
        context = tex[start:end].replace('\n', ' ')
        claims.append({
            "type": "p_value",
            "raw": m.group(0),
            "context": context,
        })

    # Pattern: d = number (Cohen's d)
    for m in re.finditer(r'd\s*=\s*(\d+\.?\d*)', tex):
        start = max(0, m.start() - 60)
        end = min(len(tex), m.end() + 40)
        context = tex[start:end].replace('\n', ' ')
        claims.append({
            "type": "effect_size",
            "value": float(m.group(1)),
            "raw": m.group(0),
            "context": context,
        })

    return claims


# ═══════════════════════════════════════════════════════════════
# VERIFICATION CHECKS
# ═══════════════════════════════════════════════════════════════

class Checker:
    def __init__(self):
        self.results = []
        self.pass_count = 0
        self.fail_count = 0
        self.warn_count = 0

    def check(self, name, passed, detail="", severity="FAIL"):
        status = "PASS" if passed else severity
        self.results.append({"name": name, "status": status, "detail": detail})
        if passed:
            self.pass_count += 1
        elif severity == "FAIL":
            self.fail_count += 1
        else:
            self.warn_count += 1

    def report(self):
        print("\n" + "=" * 80)
        print("PAPER CLAIM VERIFICATION REPORT")
        print("=" * 80)

        for r in self.results:
            icon = {"PASS": "✓", "FAIL": "✗", "WARN": "?"}[r["status"]]
            color = {"PASS": "\033[92m", "FAIL": "\033[91m", "WARN": "\033[93m"}[r["status"]]
            print(f"  {color}{icon} [{r['status']}]\033[0m {r['name']}")
            if r["detail"]:
                print(f"           {r['detail']}")

        print(f"\n{'=' * 80}")
        print(f"  PASS: {self.pass_count}  |  FAIL: {self.fail_count}  |  WARN: {self.warn_count}")
        print(f"{'=' * 80}\n")
        return self.fail_count


def verify_bvp_behavioral(checker, annotations, tex):
    """Verify BVP behavioral validation claims against manual annotations."""
    if not annotations:
        checker.check("BVP behavioral: annotations exist", False, "No annotation file found")
        return

    # Reconstruct per-model scores
    models = {}
    for row in annotations:
        key = row["model"]
        cond = row["condition"]
        if key not in models:
            models[key] = {"neutral": [], "chaos": []}
        models[key][cond].append(row["score"])

    # Check Table claims
    # Gemma 4B IT: neutral=2.70, chaos=1.60
    gemma_it = "google/gemma-3-4b-it"
    if gemma_it in models:
        n_mean = np.mean(models[gemma_it]["neutral"])
        c_mean = np.mean(models[gemma_it]["chaos"])
        checker.check(
            "Table BVP: Gemma 4B IT neutral=2.70",
            abs(n_mean - 2.70) < 0.01,
            f"Actual: {n_mean:.2f}"
        )
        checker.check(
            "Table BVP: Gemma 4B IT chaos=1.60",
            abs(c_mean - 1.60) < 0.01,
            f"Actual: {c_mean:.2f}"
        )

    # Gemma 4B Base: neutral=1.20, chaos=1.00
    gemma_pt = "google/gemma-3-4b-pt"
    if gemma_pt in models:
        n_mean = np.mean(models[gemma_pt]["neutral"])
        c_mean = np.mean(models[gemma_pt]["chaos"])
        checker.check(
            "Table BVP: Gemma 4B PT neutral=1.20",
            abs(n_mean - 1.20) < 0.01,
            f"Actual: {n_mean:.2f}"
        )
        checker.check(
            "Table BVP: Gemma 4B PT chaos=1.00",
            abs(c_mean - 1.00) < 0.01,
            f"Actual: {c_mean:.2f}"
        )

    # Llama 8B IT: neutral=2.20, chaos=1.80
    llama_it = "meta-llama/Llama-3.1-8B-Instruct"
    if llama_it in models:
        n_mean = np.mean(models[llama_it]["neutral"])
        c_mean = np.mean(models[llama_it]["chaos"])
        checker.check(
            "Table BVP: Llama 8B IT neutral=2.20",
            abs(n_mean - 2.20) < 0.01,
            f"Actual: {n_mean:.2f}"
        )
        checker.check(
            "Table BVP: Llama 8B IT chaos=1.80",
            abs(c_mean - 1.80) < 0.01,
            f"Actual: {c_mean:.2f}"
        )

    # Llama 8B Base: neutral=1.10, chaos=1.20
    llama_pt = "meta-llama/Llama-3.1-8B"
    if llama_pt in models:
        n_mean = np.mean(models[llama_pt]["neutral"])
        c_mean = np.mean(models[llama_pt]["chaos"])
        checker.check(
            "Table BVP: Llama 8B PT neutral=1.10",
            abs(n_mean - 1.10) < 0.01,
            f"Actual: {n_mean:.2f}"
        )
        checker.check(
            "Table BVP: Llama 8B PT chaos=1.20",
            abs(c_mean - 1.20) < 0.01,
            f"Actual: {c_mean:.2f}"
        )

    # Verify aggregate claims
    it_models = [m for m in [gemma_it, llama_it] if m in models]
    pt_models = [m for m in [gemma_pt, llama_pt] if m in models]

    it_n = [s for m in it_models for s in models[m]["neutral"]]
    it_c = [s for m in it_models for s in models[m]["chaos"]]
    pt_n = [s for m in pt_models for s in models[m]["neutral"]]
    pt_c = [s for m in pt_models for s in models[m]["chaos"]]

    if it_n and it_c:
        it_agg_n = np.mean(it_n)
        it_agg_c = np.mean(it_c)
        checker.check(
            "Table BVP: IT aggregate neutral=2.45",
            abs(it_agg_n - 2.45) < 0.01,
            f"Actual: {it_agg_n:.2f}"
        )
        checker.check(
            "Table BVP: IT aggregate chaos=1.70",
            abs(it_agg_c - 1.70) < 0.01,
            f"Actual: {it_agg_c:.2f}"
        )

        # Verify p-value claim: p=0.004
        u, p = stats.mannwhitneyu(it_n, it_c, alternative='greater')
        checker.check(
            "BVP IT aggregate p=.004",
            abs(p - 0.004) < 0.002,
            f"Actual p={p:.4f}"
        )

        # Verify Cohen's d claim: d=0.82
        d = (np.mean(it_n) - np.mean(it_c)) / np.sqrt((np.var(it_n) + np.var(it_c)) / 2)
        checker.check(
            "BVP IT aggregate d=0.82",
            abs(d - 0.82) < 0.05,
            f"Actual d={d:.2f}"
        )

    if pt_n and pt_c:
        # Verify base aggregate p=0.454
        u, p = stats.mannwhitneyu(pt_n, pt_c, alternative='greater')
        checker.check(
            "BVP Base aggregate p=.454",
            abs(p - 0.454) < 0.05,
            f"Actual p={p:.4f}"
        )


def verify_llama_sae(checker, results_data):
    """Verify Llama SAE replication claims."""
    # Find N=20 results
    llama_data = None
    for key, data in results_data.items():
        if "llama3_sae" in key and "054714" in key:  # N=20 run
            llama_data = data
            break

    if not llama_data:
        # Try any llama result
        for key, data in results_data.items():
            if "llama3_sae" in key:
                llama_data = data
                break

    if not llama_data:
        checker.check("Llama SAE: results exist", False, "No Llama SAE results found")
        return

    # Check IT vs PT pattern
    for model_data in llama_data.get("per_model", []):
        model_name = model_data.get("model", "")
        is_it = model_data.get("is_it", False)

        for layer_key, layer_data in model_data.get("layers", {}).items():
            if "stats" in layer_data:
                d_val = layer_data["stats"].get("cohens_d")
                p_val = layer_data["stats"].get("p_value")

                if d_val is not None and is_it and "23" in layer_key:
                    checker.check(
                        f"Llama IT L23 d=1.51",
                        abs(d_val - 1.51) < 0.1,
                        f"Actual d={d_val:.2f}",
                    )
                if p_val is not None and is_it and "23" in layer_key:
                    checker.check(
                        f"Llama IT L23 p=0.001",
                        p_val < 0.01,
                        f"Actual p={p_val:.4f}",
                    )


def verify_theorem_proving(checker, results_data):
    """Verify theorem proving behavioral claims."""
    tp_data = None
    for key, data in results_data.items():
        if "theorem_proving" in key:
            tp_data = data
            break

    if not tp_data:
        checker.check("Theorem proving: results exist", False, "No theorem proving results found", "WARN")
        return

    for model_data in tp_data.get("per_model", []):
        name = model_data["model"].split("/")[-1]
        is_it = model_data["is_it"]
        tag = "IT" if is_it else "PT"

        n_scores = [t["score"] for t in model_data["theorem_proving"]["neutral"]]
        c_scores = [t["score"] for t in model_data["theorem_proving"]["chaos"]]
        n_mean = np.mean(n_scores)
        c_mean = np.mean(c_scores)
        delta = n_mean - c_mean

        checker.check(
            f"Theorem {name} ({tag}): N={len(n_scores)} per condition",
            len(n_scores) == 10 and len(c_scores) == 10,
            f"neutral={len(n_scores)}, chaos={len(c_scores)}",
        )

        if is_it:
            u, p = stats.mannwhitneyu(n_scores, c_scores, alternative='greater')
            checker.check(
                f"Theorem {name} ({tag}): significant degradation",
                p < 0.05,
                f"Δ={delta:+.2f}, p={p:.4f}",
            )


def verify_cross_references(checker, tex):
    """Check that all \\ref and \\label match."""
    labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
    refs = set(re.findall(r'\\(?:ref|S)\{([^}]+)\}', tex))

    # Also get \S\ref patterns
    srefs = set(re.findall(r'\\S\\ref\{([^}]+)\}', tex))
    refs = refs | srefs

    undefined = refs - labels
    unused = labels - refs

    checker.check(
        "Cross-refs: all \\ref targets defined",
        len(undefined) == 0,
        f"Undefined: {undefined}" if undefined else "",
    )
    checker.check(
        "Cross-refs: all \\label targets referenced",
        len(unused) <= 2,  # allow a couple unused
        f"Unreferenced: {unused}" if unused else "",
        "WARN",
    )


def verify_citations(checker, tex):
    """Check that all \\cite targets have \\bibitem."""
    cites = set()
    for m in re.finditer(r'\\citep?\{([^}]+)\}', tex):
        for c in m.group(1).split(','):
            cites.add(c.strip())

    bibitems = set(re.findall(r'\\bibitem\[[^\]]*\]\{([^}]+)\}', tex))

    undefined_cites = cites - bibitems
    uncited_bibs = bibitems - cites

    checker.check(
        "Citations: all \\cite targets have \\bibitem",
        len(undefined_cites) == 0,
        f"Missing bibitems: {undefined_cites}" if undefined_cites else "",
    )
    checker.check(
        "Citations: all \\bibitem entries cited",
        len(uncited_bibs) <= 2,
        f"Uncited: {uncited_bibs}" if uncited_bibs else "",
        "WARN",
    )


def verify_scope_claims(checker, tex):
    """Flag overclaiming: scope of claims vs evidence."""

    # Check for "scaling law" (should be "trend")
    has_scaling_law = "scaling law" in tex.lower() and "scaling law" not in tex.lower().split("dissociation scaling trend")[0][-20:]
    checker.check(
        "Scope: no 'scaling law' overclaim",
        "scaling law" not in tex.lower().replace("the dissociation scaling law", ""),  # allow in old refs only
        "Found 'scaling law' — should be 'scaling trend' with 3 data points",
    )

    # Check RLHF vs post-training language
    rlhf_causal = len(re.findall(r'RLHF\s+(?:creates?|causes?|is the causal|confirms? RLHF as the causal)', tex))
    checker.check(
        "Scope: RLHF not claimed as sole causal factor",
        rlhf_causal == 0,
        f"Found {rlhf_causal} instances of RLHF as sole cause",
    )

    # Check domain claim scope
    domain_general = len(re.findall(r'(?:any|all|every|general)\s+(?:domain|task|setting)', tex.lower()))
    checker.check(
        "Scope: no universal domain generalization claim",
        domain_general == 0,
        f"Found {domain_general} broad domain claims",
        "WARN",
    )

    # Check "0% false" claims (should be softened)
    bold_zero = len(re.findall(r'\\textbf\{0\\%', tex))
    checker.check(
        "Scope: no bold 0% detection claims",
        bold_zero == 0,
        f"Found {bold_zero} bold 0% claims — should be softened",
    )

    # Check monotonically claim with only 3 points
    mono = "monotonically" in tex.lower()
    checker.check(
        "Scope: no 'monotonically' claim with 3 data points",
        not mono,
        "3 points cannot establish monotonicity — use 'consistently'",
    )


def verify_table_consistency(checker, tex):
    """Check that table numbers in text match table definitions."""
    # Count tables
    tables = re.findall(r'\\label\{tab:([^}]+)\}', tex)
    checker.check(
        f"Tables: {len(tables)} tables defined",
        len(tables) >= 5,
        f"Tables: {', '.join(tables)}",
    )

    # Check all table refs resolve
    table_refs = re.findall(r'\\ref\{tab:([^}]+)\}', tex)
    undefined_table_refs = [r for r in table_refs if r not in tables]
    checker.check(
        "Tables: all table refs defined",
        len(undefined_table_refs) == 0,
        f"Undefined: {undefined_table_refs}" if undefined_table_refs else "",
    )


def verify_figure_files(checker, tex):
    """Check that referenced figure files exist."""
    fig_dir = Path(__file__).parent / "figures"
    fig_refs = re.findall(r'\\includegraphics.*?\{([^}]+)\}', tex)

    for fig in fig_refs:
        fig_path = fig_dir / Path(fig).name if "/" not in fig else Path(__file__).parent / fig
        checker.check(
            f"Figure file: {fig}",
            fig_path.exists(),
            f"Missing: {fig_path}",
        )


def verify_sample_sizes(checker, tex):
    """Check claimed N values are consistent."""
    # BVP: n=10 per cell
    checker.check(
        "Sample size: BVP claims n=10 per cell",
        "n=10" in tex or "$n=10$" in tex,
        "BVP behavioral table should state n=10",
    )

    # Llama: 10 prompt variants
    checker.check(
        "Sample size: Llama claims 10 prompt variants",
        "10 prompt variants" in tex,
        "",
    )


def verify_statistical_methods(checker, tex):
    """Check statistical method reporting."""
    # Mann-Whitney mentioned
    checker.check(
        "Stats: Mann-Whitney U test mentioned for behavioral",
        "mann-whitney" in tex.lower() or "Mann-Whitney" in tex,
        "Behavioral table uses Mann-Whitney but should mention it",
    )

    # Cohen's d reported
    checker.check(
        "Stats: Cohen's d reported for effect sizes",
        "cohen" in tex.lower() or "Cohen" in tex,
        "",
    )

    # Held-out p-value
    checker.check(
        "Stats: held-out validation p-value reported",
        "8 \\times 10^{-6}" in tex or "8\\times10" in tex,
        "",
        "WARN",
    )


def main():
    print("Loading paper and data...")
    tex = load_paper()
    results = load_results()
    annotations = load_annotations()

    print(f"  Paper: {len(tex)} chars")
    print(f"  Result files: {list(results.keys())}")
    print(f"  Annotations: {len(annotations) if annotations else 'None'} rows")

    checker = Checker()

    print("\n--- Verifying BVP behavioral claims ---")
    verify_bvp_behavioral(checker, annotations, tex)

    print("\n--- Verifying Llama SAE claims ---")
    verify_llama_sae(checker, results)

    print("\n--- Verifying theorem proving claims ---")
    verify_theorem_proving(checker, results)

    print("\n--- Verifying cross-references ---")
    verify_cross_references(checker, tex)

    print("\n--- Verifying citations ---")
    verify_citations(checker, tex)

    print("\n--- Verifying scope of claims ---")
    verify_scope_claims(checker, tex)

    print("\n--- Verifying table consistency ---")
    verify_table_consistency(checker, tex)

    print("\n--- Verifying figure files ---")
    verify_figure_files(checker, tex)

    print("\n--- Verifying sample sizes ---")
    verify_sample_sizes(checker, tex)

    print("\n--- Verifying statistical methods ---")
    verify_statistical_methods(checker, tex)

    n_fails = checker.report()
    sys.exit(1 if n_fails > 0 else 0)


if __name__ == "__main__":
    main()
