#!/usr/bin/env python3
"""
Feature Trace: Track individual SAE features across the infection curve.
========================================================================
Shows which specific features appear/disappear at each turn,
especially at T0→T1 (the tipping point).

Uses the brain_diff_temporal results (no GPU needed).
"""
import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"


def load_latest():
    files = sorted(RESULTS_DIR.glob("brain_diff_temporal_*.json"))
    if not files:
        raise FileNotFoundError("No brain_diff_temporal results")
    with open(files[-1]) as f:
        return json.load(f)


def main():
    data = load_latest()
    chaos = data["chaos"]
    neutral = data["neutral"]

    for layer_name in ["layer_17", "layer_22"]:
        print(f"\n{'='*70}")
        print(f"  {layer_name.upper()} — FEATURE TRACKING ACROSS INFECTION CURVE")
        print(f"{'='*70}")

        # Track top_suppressed across turns for chaos
        print(f"\n  CHAOS CONDITION — Top 10 suppressed features per turn:")
        print(f"  (features present in READ but absent in WRITE)")
        print()

        all_chaos_features = set()
        chaos_turns = {}
        for turn in chaos:
            t = turn["n_messages"]
            feats = turn["suppression"][layer_name]["top_suppressed"]
            chaos_turns[t] = set(feats)
            all_chaos_features.update(feats)

        # Same for neutral
        all_neutral_features = set()
        neutral_turns = {}
        for turn in neutral:
            t = turn["n_messages"]
            feats = turn["suppression"][layer_name]["top_suppressed"]
            neutral_turns[t] = set(feats)
            all_neutral_features.update(feats)

        # Feature presence matrix for chaos
        print(f"  {'Feature':<10}", end="")
        for t in range(6):
            print(f"  T{t}", end="")
        print(f"  │ Neutral", end="")
        print(f"  │ Notes")
        print(f"  {'-'*75}")

        # Sort features by when they first appear
        for feat in sorted(all_chaos_features):
            row = f"  {feat:<10}"
            chaos_count = 0
            first_appear = -1
            for t in range(6):
                if feat in chaos_turns[t]:
                    row += "  ██"
                    chaos_count += 1
                    if first_appear == -1:
                        first_appear = t
                else:
                    row += "  ░░"

            # Check neutral
            neutral_count = sum(1 for t in range(6) if feat in neutral_turns[t])
            row += f"  │ {neutral_count}/6  "

            # Notes
            notes = []
            if chaos_count == 6 and neutral_count == 6:
                notes.append("STABLE (both)")
            elif chaos_count >= 4 and neutral_count <= 2:
                notes.append("CHAOS-SPECIFIC")
            elif neutral_count >= 4 and chaos_count <= 2:
                notes.append("NEUTRAL-SPECIFIC")
            elif chaos_count == 6:
                notes.append("CHAOS-STABLE")
            elif neutral_count == 6:
                notes.append("NEUTRAL-STABLE")

            # T0→T1 changes
            if feat in chaos_turns[0] and feat not in chaos_turns[1]:
                notes.append("LOST at T1 ←")
            if feat not in chaos_turns[0] and feat in chaos_turns[1]:
                notes.append("GAINED at T1 ←")

            row += f"│ {' '.join(notes)}"
            print(row)

        # T0→T1 DIFF
        print(f"\n  {'─'*70}")
        print(f"  T0→T1 TIPPING POINT ANALYSIS (chaos condition)")
        print(f"  {'─'*70}")

        lost_at_t1 = chaos_turns[0] - chaos_turns[1]
        gained_at_t1 = chaos_turns[1] - chaos_turns[0]
        stable = chaos_turns[0] & chaos_turns[1]

        print(f"\n  Features LOST at T1 (suppressed in T0 but not T1):")
        if lost_at_t1:
            for f in sorted(lost_at_t1):
                in_neutral = sum(1 for t in range(6) if f in neutral_turns[t])
                print(f"    Feature {f} — in neutral: {in_neutral}/6 turns")
        else:
            print(f"    (none)")

        print(f"\n  Features GAINED at T1 (not suppressed in T0 but suppressed in T1):")
        if gained_at_t1:
            for f in sorted(gained_at_t1):
                in_neutral = sum(1 for t in range(6) if f in neutral_turns[t])
                print(f"    Feature {f} — in neutral: {in_neutral}/6 turns")
        else:
            print(f"    (none)")

        print(f"\n  Features STABLE T0→T1 (suppressed in both):")
        for f in sorted(stable):
            print(f"    Feature {f}")

        # Chaos-only vs Neutral-only features across ALL turns
        print(f"\n  {'─'*70}")
        print(f"  CONDITION-SPECIFIC FEATURES (across all turns)")
        print(f"  {'─'*70}")

        chaos_freq = defaultdict(int)
        neutral_freq = defaultdict(int)
        for t in range(6):
            for f in chaos_turns[t]:
                chaos_freq[f] += 1
            for f in neutral_turns[t]:
                neutral_freq[f] += 1

        print(f"\n  Features predominantly in CHAOS (≥4/6 chaos, ≤2/6 neutral):")
        chaos_specific = [(f, chaos_freq[f], neutral_freq[f])
                         for f in all_chaos_features
                         if chaos_freq[f] >= 4 and neutral_freq[f] <= 2]
        for f, cc, nc in sorted(chaos_specific, key=lambda x: -x[1]):
            print(f"    Feature {f}: chaos {cc}/6, neutral {nc}/6")

        print(f"\n  Features predominantly in NEUTRAL (≥4/6 neutral, ≤2/6 chaos):")
        neutral_specific = [(f, chaos_freq[f], neutral_freq[f])
                           for f in all_neutral_features
                           if neutral_freq[f] >= 4 and chaos_freq[f] <= 2]
        for f, cc, nc in sorted(neutral_specific, key=lambda x: -x[2]):
            print(f"    Feature {f}: chaos {cc}/6, neutral {nc}/6")

        # Suppression load per turn side by side
        print(f"\n  {'─'*70}")
        print(f"  SUPPRESSION LOAD + N_SUPPRESSED")
        print(f"  {'─'*70}")
        print(f"  {'Turn':<6} {'C Load':>8} {'C N':>6} {'N Load':>8} {'N N':>6} {'ΔLoad':>8} {'ΔN':>6}")
        print(f"  {'-'*50}")
        for t in range(6):
            cl = chaos[t]["suppression"][layer_name]["suppression_load"]
            cn = chaos[t]["suppression"][layer_name]["n_suppressed"]
            nl = neutral[t]["suppression"][layer_name]["suppression_load"]
            nn = neutral[t]["suppression"][layer_name]["n_suppressed"]
            print(f"  {t:<6} {cl:>8.2f} {cn:>6} {nl:>8.2f} {nn:>6} {cl-nl:>+8.2f} {cn-nn:>+6}")

    print()


if __name__ == "__main__":
    main()
