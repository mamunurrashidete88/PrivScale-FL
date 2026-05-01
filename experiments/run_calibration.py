"""
experiments/run_calibration.py
================================
Reproduces Figure 6 and Table 2: Mechanism comparison (RQ3).

Runs all four mechanisms (non-private, fixed-noise DP, adaptive-clipping DP,
scale-aware DP) across the model ladder and compares MIA AUC, utility, and
cross-scale drift.

Research question: Can scale-aware calibration stabilize privacy risk across
model scales without rendering federated training impractical?

Usage:
    python experiments/run_calibration.py \\
        --datasets cifar100 stackoverflow --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

VISION_MODELS = ["resnet20", "resnet32", "resnet44", "resnet56", "resnet110", "wrn_28_10"]
LANGUAGE_MODELS = ["gpt2_small", "gpt2_medium_lora", "gpt2_large_lora"]
MECHANISMS = ["nonprivate", "fixed", "adaptive", "scale_aware"]

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
    p = argparse.ArgumentParser(description="Fig 6 + Table 2: Mechanism comparison (RQ3).")
    p.add_argument("--datasets", nargs="+", default=["cifar100"],
                   choices=list(CONFIG_TEMPLATE.keys()))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
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
                for seed in args.seeds:
                    out = f"runs/calibration/{ds}/{model}/{mech}/seed{seed}"
                    run_single(cfg, seed, out)

    # Aggregate and plot
    subprocess.run([
        sys.executable, "scripts/aggregate_results.py",
        "--runs-dir", "runs/calibration", "--out", "results/calibration",
    ], check=True)
    subprocess.run([
        sys.executable, "utils/visualization.py",
        "--results-dir", "results/calibration", "--output-dir", "figures/",
    ], check=True)

    # Print summary table
    from evaluation import build_ladders, print_summary
    ladders = build_ladders(Path("runs/calibration"))
    print_summary(ladders)

    print("\nRQ3 runs complete. Figures and tables written to figures/ and stdout.")


if __name__ == "__main__":
    main()
