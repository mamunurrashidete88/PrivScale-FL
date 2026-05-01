#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/run_calibration.py \
    --datasets cifar100 --seeds ${@:-0 1 2}
