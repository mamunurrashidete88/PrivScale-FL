#!/usr/bin/env bash
# Run risk_scaling experiment (see experiments/run_risk_scaling.py)
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/run_risk_scaling.py \
    --datasets cifar100 \
    --seed${1:+s} ${@:-0}
