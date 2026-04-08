#!/bin/bash
# Run all ICML experiments sequentially
# Usage: cd ~/ICML && env HF_TOKEN=<token> bash experiments/run_all.sh [4b|12b|27b|all]
#
# On 80GB A100: run_all.sh all
# On 40GB A100: run_all.sh 27b  (12b already done, 27b escalation needs 80GB)

set -e

SCALE=${1:-all}
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Set HF_TOKEN environment variable before running"
    exit 1
fi
export HF_TOKEN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p results plots

run_scale() {
    local s=$1
    echo ""
    echo "============================================"
    echo "  RUNNING ${s} EXPERIMENTS"
    echo "============================================"
    echo ""

    echo "--- ${s}: Feature swap (awareness ablation) ---"
    python3 experiments/ablation_feature_swap.py --model $s 2>&1 | tail -20

    echo "--- ${s}: Attention knockout ---"
    python3 experiments/ablation_attention_knockout.py --model $s 2>&1 | tail -20

    echo "--- ${s}: Activation patching ---"
    python3 experiments/ablation_activation_patching.py --model $s 2>&1 | tail -20
}

run_escalation() {
    local s=$1
    echo ""
    echo "--- ${s}: Base vs IT escalation ---"
    python3 experiments/gemma3_${s}_escalation.py 2>&1 | tail -30
}

run_crossover() {
    echo ""
    echo "--- 27B: Same-model saliency-intent crossover ---"
    if [ -f ~/af_probe_weights.npy ]; then
        python3 experiments/saliency_intent_crossover_27b.py 2>&1 | tail -20
    else
        echo "SKIP: ~/af_probe_weights.npy not found"
        echo "Upload af_probe_weights.npy to ~/af_probe_weights.npy on this machine"
    fi
}

case $SCALE in
    4b)
        run_scale 4b
        ;;
    12b)
        run_scale 12b
        run_escalation 12b
        ;;
    27b)
        run_scale 27b
        run_crossover
        run_escalation 27b
        ;;
    all)
        run_scale 4b
        run_scale 12b
        run_escalation 12b
        run_scale 27b
        run_crossover
        run_escalation 27b
        ;;
    *)
        echo "Usage: $0 [4b|12b|27b|all]"
        exit 1
        ;;
esac

# Generate plots if matplotlib available
echo ""
echo "--- Generating plots ---"
python3 experiments/plot_scaling.py 2>&1 || echo "Plot generation failed (non-fatal)"

echo ""
echo "============================================"
echo "  ALL DONE"
echo "============================================"
echo "Results: ls results/*.json"
echo "Plots:   ls plots/*.png"
