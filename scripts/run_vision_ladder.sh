#!/usr/bin/env bash
# Run the full 6-point vision ladder:
# CIFAR-100 × {resnet20, resnet32, resnet44, resnet56, resnet110, wrn_28_10} × 4 mechanisms × seeds.
#
# Seed counts match the manuscript:
#   5 seeds for all ResNet variants (20, 32, 44, 56, 110)
#   3 seeds for WRN-28-10
#
# Naming convention:  configs/vision/cifar100_{model}_{mechanism}.yaml
#   model keys:     resnet20, resnet32, resnet44, resnet56, resnet110, wrn_28_10
#   mechanism keys: nonprivate, fixed, adaptive, scale_aware
set -euo pipefail

RESNET_SEEDS="${RESNET_SEEDS:-0 1 2 3 4}"
WRN_SEEDS="${WRN_SEEDS:-0 1 2}"
MODELS="${MODELS:-resnet20 resnet32 resnet44 resnet56 resnet110 wrn_28_10}"
MECHS="${MECHS:-nonprivate fixed adaptive scale_aware}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

total=0; skipped=0; failed=0

for model in $MODELS; do
  # Use 5 seeds for ResNets, 3 for WRN
  if [[ "$model" == wrn_28_10 ]]; then
    seeds="$WRN_SEEDS"
  else
    seeds="$RESNET_SEEDS"
  fi
  for mech in $MECHS; do
    cfg="configs/vision/cifar100_${model}_${mech}.yaml"
    if [[ ! -f "$cfg" ]]; then
      echo "[skip] config not found: $cfg" >&2
      skipped=$((skipped + 1))
      continue
    fi
    for seed in $seeds; do
      out="runs/cifar100/${model}/${mech}/seed${seed}"
      echo ">>> ${model} / ${mech} / seed=${seed}  ->  ${out}"
      if python scripts/run_experiment.py \
          --config "$cfg" \
          --seed "$seed" \
          --output-dir "$out" \
          2>&1 | tee "${out}/stdout.log"; then
        total=$((total + 1))
      else
        echo "[FAIL] ${model}/${mech}/seed${seed}" >&2
        failed=$((failed + 1))
      fi
    done
  done
done

echo ""
echo "=== Vision ladder complete: ${total} runs, ${skipped} skipped, ${failed} failed ==="

if [[ -d runs/ ]]; then
  python scripts/aggregate_results.py --runs-dir runs/ --out results/
  python scripts/generate_plots.py --results results/ --out figures/
fi
