#!/usr/bin/env bash
# Run spectral_validation experiment (see experiments/run_spectral_validation.py)
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/run_spectral_validation.py \
    --datasets cifar100 \
    --seed${1:+s} ${@:-0}
