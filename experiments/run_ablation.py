"""
experiments/run_ablation.py
============================
Reproduces Appendix ablation tables: batch size, participation rate,
threshold sensitivity, and FT vs LoRA at matched effective dimension.

Usage:
    python experiments/run_ablation.py \\
        --datasets cifar100 --seed 0
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_with_override(base_config: str, overrides: dict, seed: int, output_dir: str) -> None:
    """Load a YAML config, apply overrides, write to a temp file, and run."""
    with open(base_config) as f:
        cfg = yaml.safe_load(f)

    # Apply overrides (one level deep)
    for key, val in overrides.items():
        parts = key.split(".")
        target = cfg
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = val

    tmp_cfg = Path(output_dir) / "config_override.yaml"
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_cfg, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [
        sys.executable, "scripts/run_experiment.py",
        "--config", str(tmp_cfg),
        "--seed", str(seed),
        "--output-dir", output_dir,
    ]
    print(f"  >>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_batch_size_ablation(base_config: str, seed: int) -> None:
    """Vary local batch size B ∈ {16, 32, 64, 128, 256}."""
    print("\n=== Batch size ablation ===")
    for bs in [16, 32, 64, 128, 256]:
        out = f"runs/ablation/batch_size/B{bs}/seed{seed}"
        run_with_override(base_config, {"local.batch_size": bs}, seed, out)


def run_participation_ablation(base_config: str, seed: int) -> None:
    """Vary sampling fraction q ∈ {0.05, 0.10, 0.20}."""
    print("\n=== Participation rate ablation ===")
    for q in [0.05, 0.10, 0.20]:
        out = f"runs/ablation/participation/q{q}/seed{seed}"
        run_with_override(base_config, {"server.sample_fraction": q}, seed, out)


def run_heterogeneity_ablation(base_config: str, seed: int) -> None:
    """Vary Dirichlet alpha ∈ {0.1, 0.5, 1.0}."""
    print("\n=== Heterogeneity ablation ===")
    for alpha in [0.1, 0.5, 1.0]:
        out = f"runs/ablation/heterogeneity/alpha{alpha}/seed{seed}"
        run_with_override(base_config, {"dataset.alpha": alpha}, seed, out)


def main():
    p = argparse.ArgumentParser(description="Appendix ablations.")
    p.add_argument("--datasets", nargs="+", default=["cifar100"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ablation", default="all",
                   choices=["all", "batch", "participation", "heterogeneity"])
    args = p.parse_args()

    # Use a mid-scale model with fixed-noise DP as the base for ablations
    base_configs = {
        "cifar100": "configs/vision/cifar100_resnet56_fixed.yaml",
        "stackoverflow": "configs/language/so_gpt2_small_fixed.yaml",
    }

    for ds in args.datasets:
        base = base_configs.get(ds)
        if base is None or not Path(base).exists():
            print(f"  [skip] no base config for {ds}")
            continue
        print(f"\n{'='*60}")
        print(f"Ablations on {ds} (base: {base})")
        print(f"{'='*60}")

        if args.ablation in ("all", "batch"):
            run_batch_size_ablation(base, args.seed)
        if args.ablation in ("all", "participation"):
            run_participation_ablation(base, args.seed)
        if args.ablation in ("all", "heterogeneity") and ds == "cifar100":
            run_heterogeneity_ablation(base, args.seed)

    # Aggregate
    subprocess.run([
        sys.executable, "scripts/aggregate_results.py",
        "--runs-dir", "runs/ablation", "--out", "results/ablation",
    ], check=True)

    print("\nAblation runs complete. Results in results/ablation/")


if __name__ == "__main__":
    main()
