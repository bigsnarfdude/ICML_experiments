# Quick Deploy to A100

## Setup

```bash
# 1. Copy experiment code to GPU server
#    Place under ~/ICML/experiments/

# 2. Copy AF probe weights to ~/
#    af_probe_weights.npy, af_probe_bias.npy

# 3. Install deps + download models
export HF_TOKEN=<your-token>
bash ~/ICML/experiments/setup_a100.sh

# 4. Run experiments
cd ~/ICML && nohup bash experiments/run_all.sh 27b > results/run.log 2>&1 &

# 5. Monitor
tail -f ~/ICML/results/run.log
```

## What needs 80GB A100

| Experiment | 40GB | 80GB |
|-----------|------|------|
| 4B ablations | OK | OK |
| 12B ablations | OK | OK |
| 12B base vs IT escalation | OK | OK |
| 27B ablations | OK (offload) | OK |
| 27B crossover | OK (offload) | OK |
| **27B base vs IT escalation** | **OOM** | OK |

The 27B escalation (T0-T5 + recovery probes) does ~19 generation passes with SAE feature extraction. On 40GB the model + SAE + generation buffers exceed VRAM.

## What's already done (don't re-run)

- All 4B ablations (local GPU)
- All 12B ablations (local GPU + 40GB A100)
- 12B base vs IT escalation (on 40GB A100)
- All 27B ablations (on 40GB A100)
- 27B crossover (null result, on 40GB A100)

## What still needs running

- **27B base vs IT escalation** — needs 80GB A100
- (Optional) **27B PT ablations** — test if base model shows same dissociation
- (Optional) Re-run plots with 27B escalation data

## Files

```
ICML/
├── DEPLOY.md              ← you are here
├── ABLATION_CHECKPOINT_SCALING.md
├── experiments/
│   ├── setup_a100.sh      ← install deps + download models
│   ├── run_all.sh         ← run all experiments for a scale
│   ├── ablation_feature_swap.py
│   ├── ablation_attention_knockout.py
│   ├── ablation_activation_patching.py
│   ├── gemma3_12b_escalation.py
│   ├── gemma3_27b_escalation.py
│   ├── saliency_intent_crossover.py      ← cross-model (4B vs 27B)
│   ├── saliency_intent_crossover_27b.py  ← same-model (27B L40)
│   └── plot_scaling.py
├── results/               ← 19 JSON files
└── plots/                 ← 6 PNGs
```
