#!/usr/bin/env python3
"""
Systematic analysis of all behavioral validation data.
Produces clean tables for paper integration.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path(__file__).parent


def load_json(path):
    with open(path) as f:
        return json.load(f)


def mann_whitney(a, b):
    """One-tailed Mann-Whitney U: a > b."""
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b, alternative='greater')
    return u, p


def cohens_d(a, b):
    pooled_var = (np.var(a) + np.var(b)) / 2
    if pooled_var == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / np.sqrt(pooled_var)


def bootstrap_ci(a, b, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        a_boot = rng.choice(a, size=len(a), replace=True)
        b_boot = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(a_boot) - np.mean(b_boot))
    return np.percentile(diffs, [2.5, 97.5])


def analyze_n30_dose():
    """Analyze n=30 dose-response results."""
    path = RESULTS_DIR / "behavioral_n30_dose_20260408_181327.json"
    if not path.exists():
        print("n30 dose-response file not found")
        return

    data = load_json(path)

    print("=" * 90)
    print("N=30 DOSE-RESPONSE BEHAVIORAL VALIDATION")
    print("=" * 90)
    print(f"Metadata: {data['metadata']['n_per_cell']} trials/cell, "
          f"temps={data['metadata']['temps']}, "
          f"timestamp={data['metadata']['timestamp']}")
    print()

    # Table 1: Full results
    print(f"{'Model':<20} {'Condition':<18} {'n':>4} {'Mean':>6} {'SD':>6} "
          f"{'Delta':>7} {'p':>8} {'d':>6} {'95% CI':>16} {'Groot':>6}")
    print("-" * 100)

    for model_data in data['models']:
        tag = model_data['tag']
        conditions = model_data['conditions']

        n_scores = np.array([t['score'] for t in conditions['neutral']])
        n_mean = np.mean(n_scores)

        print(f"{tag:<20} {'neutral':<18} {len(n_scores):>4} {n_mean:>6.2f} "
              f"{np.std(n_scores):>6.2f} {'---':>7} {'---':>8} {'---':>6} "
              f"{'---':>16} {'---':>6}")

        for chaos_level in ['chaos_mild', 'chaos_moderate', 'chaos_strong']:
            c_scores = np.array([t['score'] for t in conditions[chaos_level]])
            c_mean = np.mean(c_scores)
            delta = n_mean - c_mean
            _, p = mann_whitney(n_scores, c_scores)
            d = cohens_d(n_scores, c_scores)
            ci = bootstrap_ci(n_scores, c_scores)
            groot = model_data['groot_counts'].get(chaos_level, 0)

            p_str = f"{p:.4f}" if p >= 0.0001 else "<.0001"

            print(f"{'':<20} {chaos_level:<18} {len(c_scores):>4} {c_mean:>6.2f} "
                  f"{np.std(c_scores):>6.2f} {delta:>+7.2f} {p_str:>8} {d:>6.2f} "
                  f"[{ci[0]:>+.2f},{ci[1]:>+.2f}] {groot:>5}/30")
        print()

    # Table 2: IT vs Base comparison (the key finding)
    print("\n" + "=" * 70)
    print("IT vs BASE SUMMARY (aggregated across chaos levels)")
    print("=" * 70)

    for model_data in data['models']:
        tag = model_data['tag']
        is_it = 'it' in tag
        conditions = model_data['conditions']

        n_scores = np.array([t['score'] for t in conditions['neutral']])
        all_chaos = []
        for level in ['chaos_mild', 'chaos_moderate', 'chaos_strong']:
            all_chaos.extend([t['score'] for t in conditions[level]])
        c_scores = np.array(all_chaos)

        delta = np.mean(n_scores) - np.mean(c_scores)
        _, p = mann_whitney(n_scores, c_scores)
        d = cohens_d(n_scores, c_scores)

        p_str = f"{p:.4f}" if p >= 0.0001 else "<.0001"

        print(f"{tag:<20} n={len(n_scores):>3} neutral={np.mean(n_scores):.2f}  "
              f"n={len(c_scores):>3} chaos={np.mean(c_scores):.2f}  "
              f"Delta={delta:>+.2f}  p={p_str}  d={d:.2f}  "
              f"{'*** IT EFFECT' if is_it and p < 0.05 else ''}")

    # Table 3: Dose-response pattern
    print("\n" + "=" * 70)
    print("DOSE-RESPONSE PATTERN (IT models only)")
    print("=" * 70)
    print(f"{'Model':<20} {'Mild d':>8} {'Mod d':>8} {'Strong d':>8} {'Pattern':>15}")
    print("-" * 70)

    for model_data in data['models']:
        tag = model_data['tag']
        if 'it' not in tag:
            continue

        conditions = model_data['conditions']
        n_scores = np.array([t['score'] for t in conditions['neutral']])

        ds = {}
        for level in ['chaos_mild', 'chaos_moderate', 'chaos_strong']:
            c_scores = np.array([t['score'] for t in conditions[level]])
            ds[level] = cohens_d(n_scores, c_scores)

        # Determine pattern
        vals = [ds['chaos_mild'], ds['chaos_moderate'], ds['chaos_strong']]
        if vals[0] > vals[1] and vals[0] > vals[2]:
            pattern = "MILD WORST"
        elif vals[2] > vals[1] and vals[2] > vals[0]:
            pattern = "STRONG WORST"
        else:
            pattern = "MODERATE WORST"

        print(f"{tag:<20} {ds['chaos_mild']:>8.2f} {ds['chaos_moderate']:>8.2f} "
              f"{ds['chaos_strong']:>8.2f} {pattern:>15}")

    # Groot analysis
    print("\n" + "=" * 70)
    print("GROOT EFFECT ANALYSIS")
    print("=" * 70)

    for model_data in data['models']:
        tag = model_data['tag']
        total_groot = 0
        total_chaos = 0
        for level in ['chaos_mild', 'chaos_moderate', 'chaos_strong']:
            total_groot += model_data['groot_counts'].get(level, 0)
            total_chaos += len(model_data['conditions'][level])
        print(f"{tag:<20} Groot: {total_groot}/{total_chaos} "
              f"({total_groot/total_chaos*100:.1f}%)")


def analyze_12b():
    """Analyze 12B results if available."""
    # Check for 12B result files
    files = sorted(RESULTS_DIR.glob("behavioral_12b_*.json"))
    if not files:
        print("\nNo 12B results found yet.")
        return

    data = load_json(files[-1])
    print("\n" + "=" * 70)
    print(f"12B BEHAVIORAL RESULTS ({files[-1].name})")
    print("=" * 70)

    for model_data in data.get('models', [data]):
        name = model_data.get('model', 'unknown')
        n_scores = np.array([t['score'] for t in model_data['neutral']])
        c_scores = np.array([t['score'] for t in model_data['chaos']])

        delta = np.mean(n_scores) - np.mean(c_scores)
        _, p = mann_whitney(n_scores, c_scores)
        d = cohens_d(n_scores, c_scores)
        groot = model_data.get('groot_count', 0)

        p_str = f"{p:.4f}" if p >= 0.0001 else "<.0001"

        print(f"{name}")
        print(f"  Neutral: {np.mean(n_scores):.2f} (n={len(n_scores)}) scores={list(n_scores)}")
        print(f"  Chaos:   {np.mean(c_scores):.2f} (n={len(c_scores)}) scores={list(c_scores)}")
        print(f"  Delta={delta:+.2f}  p={p_str}  d={d:.2f}  Groot={groot}")
        print()


def analyze_27b():
    """Analyze 27B results if available."""
    files = sorted(RESULTS_DIR.glob("behavioral_27b_*.json"))
    if not files:
        print("\nNo 27B results found yet.")
        return

    data = load_json(files[-1])
    print("\n" + "=" * 70)
    print(f"27B BEHAVIORAL RESULTS ({files[-1].name})")
    print("=" * 70)

    n_scores = np.array([t['score'] for t in data['neutral']])
    c_scores = np.array([t['score'] for t in data['chaos']])

    delta = np.mean(n_scores) - np.mean(c_scores)
    _, p = mann_whitney(n_scores, c_scores)
    d = cohens_d(n_scores, c_scores)
    groot = data.get('groot_count', 0)

    p_str = f"{p:.4f}" if p >= 0.0001 else "<.0001"

    print(f"  Neutral: {np.mean(n_scores):.2f} (n={len(n_scores)}) scores={list(n_scores)}")
    print(f"  Chaos:   {np.mean(c_scores):.2f} (n={len(c_scores)}) scores={list(c_scores)}")
    print(f"  Delta={delta:+.2f}  p={p_str}  d={d:.2f}  Groot={groot}")


def latex_table():
    """Generate LaTeX table for the paper."""
    path = RESULTS_DIR / "behavioral_n30_dose_20260408_181327.json"
    if not path.exists():
        return

    data = load_json(path)

    print("\n" + "=" * 70)
    print("LATEX TABLE (for paper)")
    print("=" * 70)
    print(r"""
\begin{table}[t]
\centering
\caption{Behavioral validation with dose-response ($n=30$ per cell).
Effect sizes (Cohen's $d$) and significance tests (Mann-Whitney $U$, one-tailed)
comparing neutral vs.\ each adversarial intensity level.
IT models show significant degradation at all levels;
base models show minimal or no effect.}
\label{tab:behavioral_dose}
\small
\begin{tabular}{@{}llccccr@{}}
\toprule
Model & Condition & Neutral & Chaos & $\Delta$ & $d$ & $p$ \\
\midrule""")

    for model_data in data['models']:
        tag = model_data['tag']
        conditions = model_data['conditions']
        n_scores = np.array([t['score'] for t in conditions['neutral']])
        n_mean = np.mean(n_scores)

        first = True
        for level, label in [('chaos_mild', 'Mild'),
                              ('chaos_moderate', 'Moderate'),
                              ('chaos_strong', 'Strong')]:
            c_scores = np.array([t['score'] for t in conditions[level]])
            c_mean = np.mean(c_scores)
            delta = n_mean - c_mean
            _, p = mann_whitney(n_scores, c_scores)
            d = cohens_d(n_scores, c_scores)

            p_str = f"${p:.3f}$" if p >= 0.001 else "$<.001$"

            model_col = tag.replace('-', ' ').replace('gemma 4b', 'Gemma~3 4B').replace('llama 8b', 'Llama~3.1 8B').replace(' it', ' IT').replace(' pt', ' Base') if first else ''
            n_col = f"{n_mean:.2f}" if first else ''

            print(f"{model_col} & {label} & {n_col} & {c_mean:.2f} & ${delta:+.2f}$ & ${d:.2f}$ & {p_str} \\\\")
            first = False
        print(r"\addlinespace")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")


if __name__ == "__main__":
    analyze_n30_dose()
    analyze_12b()
    analyze_27b()
    latex_table()
