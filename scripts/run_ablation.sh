#!/usr/bin/env bash
# Run ablation experiment (see experiments/run_ablation.py)
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/run_ablation.py \
    --datasets cifar100 \
    --seed${1:+s} ${@:-0}
