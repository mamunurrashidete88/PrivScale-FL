#!/usr/bin/env bash
# Run the language ladder: Stack Overflow × {gpt2_small, gpt2_medium_lora} × mechanisms × seeds.
#
# Naming convention:  configs/language/so_{model}_{mechanism}.yaml
set -euo pipefail

SEEDS="${SEEDS:-0 1 2}"
MODELS="${MODELS:-gpt2_small gpt2_medium_lora gpt2_large_lora}"
MECHS="${MECHS:-nonprivate fixed adaptive scale_aware}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

total=0; skipped=0; failed=0

for model in $MODELS; do
  for mech in $MECHS; do
    cfg="configs/language/so_${model}_${mech}.yaml"
    if [[ ! -f "$cfg" ]]; then
      echo "[skip] config not found: $cfg" >&2
      skipped=$((skipped + 1))
      continue
    fi
    for seed in $SEEDS; do
      out="runs/stackoverflow/${model}/${mech}/seed${seed}"
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
echo "=== Language ladder complete: ${total} runs, ${skipped} skipped, ${failed} failed ==="

if [[ -d runs/ ]]; then
  python scripts/aggregate_results.py --runs-dir runs/ --out results/
  python scripts/generate_plots.py --results results/ --out figures/
fi
