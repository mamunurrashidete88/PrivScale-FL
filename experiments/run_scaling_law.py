"""
experiments/run_scaling_law.py
================================
Reproduces Figure 5: Scaling law fit comparison and privacy debt (RQ2).

Runs fixed-noise DP-FL across the model ladder with multiple seeds, fits
constant / linear / d/log(d) families to MIA AUC vs effective dimension,
and computes the empirical privacy debt curve ε_eff(d) / ε_target.

Research question: Is the empirical trend consistent with the predicted
d/log(d) law, and does fixed-noise deployment induce measurable privacy debt?

Usage:
    python experiments/run_scaling_law.py \\
        --datasets cifar100 stackoverflow --seeds 0 1 2
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

CONFIG_TEMPLATE = {
    "cifar100": "configs/vision/cifar100_{model}_fixed.yaml",
    "stackoverflow": "configs/language/so_{model}_fixed.yaml",
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
    p = argparse.ArgumentParser(description="Fig 5: Scaling law + debt (RQ2).")
    p.add_argument("--datasets", nargs="+", default=["cifar100"],
                   choices=list(CONFIG_TEMPLATE.keys()))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()

    for ds in args.datasets:
        template = CONFIG_TEMPLATE[ds]
        models = MODEL_MAP[ds]
        for model in models:
            cfg = template.format(model=model)
            if not Path(cfg).exists():
                print(f"  [skip] {cfg}")
                continue
            for seed in args.seeds:
                out = f"runs/scaling_law/{ds}/{model}/fixed/seed{seed}"
                run_single(cfg, seed, out)

    # Aggregate
    subprocess.run([
        sys.executable, "scripts/aggregate_results.py",
        "--runs-dir", "runs/scaling_law", "--out", "results/scaling_law",
    ], check=True)

    # Generate scaling law fits
    subprocess.run([
        sys.executable, "utils/visualization.py",
        "--results-dir", "results/scaling_law", "--output-dir", "figures/",
    ], check=True)

    print("\nRQ2 runs complete. Figures written to figures/")


if __name__ == "__main__":
    main()
