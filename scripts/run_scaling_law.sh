#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/run_scaling_law.py \
    --datasets cifar100 --seeds ${@:-0 1 2}
