#!/usr/bin/env bash
# =============================================================================
# Quick sanity check: CIFAR-10, ResNet-20, 20 rounds, non-private.
# Should complete in ~5 minutes on any GPU (or ~15 min CPU-only).
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== PrivScale-FL Quick Test ==="
echo "Config: configs/vision/smoke_cifar10.yaml"
echo ""

python scripts/run_experiment.py \
    --config configs/vision/smoke_cifar10.yaml \
    --device cuda

echo ""
echo "--- Quick test complete ---"
echo "Outputs in: runs/smoke/"
echo ""

if [[ -f runs/smoke/summary.json ]]; then
    echo "Summary:"
    cat runs/smoke/summary.json
fi
