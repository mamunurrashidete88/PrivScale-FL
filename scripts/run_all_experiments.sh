#!/usr/bin/env bash
# =============================================================================
# Run all PrivScale-FL experiments.
# Equivalent to: make all-experiments
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== PrivScale-FL: Full Experiment Suite ==="
echo "Start: $(date)"
echo ""

# Step 1: Calibrate noise at reference scale
echo "--- Step 1: Calibrate noise ---"
python scripts/calibrate_noise.py --eps 3.0 --delta 1e-5 --q 0.1 --rounds 500

# Step 2: Run all sub-experiments
echo ""
echo "--- Step 2: Run all experiments ---"
python experiments/run_full_comparison.py \
    --datasets cifar100 \
    --seeds 0 1 2

echo ""
echo "=== All experiments complete: $(date) ==="
