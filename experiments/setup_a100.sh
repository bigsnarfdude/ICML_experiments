#!/bin/bash
# Setup A100 instance for Gemma 3 feature starvation experiments
#
# Quick deploy:
#   1. Copy experiments/ to ~/ICML/ on the GPU server
#   2. Copy af_probe_weights.npy and af_probe_bias.npy to ~/
#   3. Run: bash ~/ICML/experiments/setup_a100.sh
#   4. Run: cd ~/ICML && bash experiments/run_all.sh

set -e

echo "=== Setting up ICML experiment environment ==="

mkdir -p ~/ICML/experiments ~/ICML/results ~/ICML/plots

# Install dependencies
pip3 install --upgrade pip
pip3 install --break-system-packages transformers accelerate sae-lens huggingface_hub scipy matplotlib numpy

# Pre-download models (so experiments don't block on download)
echo "=== Pre-downloading models ==="
python3 -c "
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HF_TOKEN', '')

for model in ['google/gemma-3-4b-it', 'google/gemma-3-12b-it', 'google/gemma-3-27b-it',
              'google/gemma-3-12b-pt', 'google/gemma-3-27b-pt']:
    print(f'Downloading {model}...')
    try:
        snapshot_download(model, ignore_patterns=['*.gguf'], token=token)
        print(f'  OK')
    except Exception as e:
        print(f'  SKIP: {e}')

print('Done.')
"

# Pre-download SAEs
echo "=== Pre-downloading SAEs ==="
python3 -c "
from sae_lens import SAE
for release in ['gemma-scope-2-4b-it-res', 'gemma-scope-2-12b-it-res', 'gemma-scope-2-12b-pt-res',
                'gemma-scope-2-27b-it-res', 'gemma-scope-2-27b-pt-res']:
    for layer in [17, 22, 31, 40, 41]:
        sae_id = f'layer_{layer}_width_16k_l0_medium'
        try:
            sae = SAE.from_pretrained(release=release, sae_id=sae_id)
            print(f'  {release}/{sae_id}: OK')
            del sae
        except:
            pass
print('SAE download done.')
"

# Verify GPU
python3 -c "
import torch
print(f'torch={torch.__version__}')
if torch.cuda.is_available():
    print(f'CUDA: {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'VRAM: {vram:.1f} GB')
    if vram >= 75:
        print('A100 80GB detected — all experiments will fit in VRAM')
    elif vram >= 35:
        print('A100 40GB detected — 27B escalation needs CPU offload (may OOM)')
    else:
        print(f'WARNING: Only {vram:.0f}GB VRAM — 12B+ experiments need offload')
else:
    print('WARNING: No CUDA GPU detected')
"

echo "=== Setup complete ==="
echo "Run: cd ~/ICML && bash experiments/run_all.sh"
