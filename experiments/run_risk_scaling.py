"""
experiments/run_risk_scaling.py
================================
Reproduces Figure 4: Privacy risk scaling under fixed-noise deployment (RQ1).

Runs non-private and fixed-noise DP-FL across the model ladder, then
reports MIA AUC, extraction rate, and trajectory proxy at each scale point.

Research question: Does privacy risk increase monotonically with model scale
under a fixed privacy mechanism?

Usage:
    python experiments/run_risk_scaling.py \\
        --datasets cifar100 stackoverflow --seed 0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

VISION_MODELS = ["resnet20", "resnet32", "resnet44", "resnet56", "resnet110", "wrn_28_10"]
LANGUAGE_MODELS = ["gpt2_small", "gpt2_medium_lora", "gpt2_large_lora"]
MECHANISMS = ["nonprivate", "fixed"]

CONFIG_TEMPLATE = {
    "cifar100": "configs/vision/cifar100_{model}_{mech}.yaml",
    "stackoverflow": "configs/language/so_{model}_{mech}.yaml",
}
MODEL_MAP = {
    "cifar100": VISION_MODELS,
    "stackoverflow": LANGUAGE_MODELS,
}


def run_single(config: str, seed: int, output_dir: str) -> None:
    cmd = [
        sys.executable, "scripts/run_experiment.py",
        "--config", config,
        "--seed", str(seed),
        "--output-dir", output_dir,
    ]
    print(f"  >>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="Fig 4: Risk scaling (RQ1).")
    p.add_argument("--datasets", nargs="+", default=["cifar100"],
                   choices=list(CONFIG_TEMPLATE.keys()))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    for ds in args.datasets:
        template = CONFIG_TEMPLATE[ds]
        models = MODEL_MAP[ds]
        for mech in MECHANISMS:
            for model in models:
                cfg = template.format(model=model, mech=mech)
                if not Path(cfg).exists():
                    print(f"  [skip] {cfg}")
                    continue
                out = f"runs/risk_scaling/{ds}/{model}/{mech}/seed{args.seed}"
                run_single(cfg, args.seed, out)

    # Aggregate
    subprocess.run([
        sys.executable, "scripts/aggregate_results.py",
        "--runs-dir", "runs/risk_scaling", "--out", "results/risk_scaling",
    ], check=True)

    print("\nRQ1 runs complete. Generate Fig 4 with:")
    print("  python utils/visualization.py --results-dir results/risk_scaling --output-dir figures/")


if __name__ == "__main__":
    main()
