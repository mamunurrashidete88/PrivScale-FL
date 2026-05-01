"""
experiments/run_full_comparison.py
====================================
Run all sub-experiments sequentially and produce the complete set of paper
figures and tables.

Usage:
    python experiments/run_full_comparison.py \\
        --datasets cifar100 stackoverflow --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    p = argparse.ArgumentParser(description="Run all experiments.")
    p.add_argument("--datasets", nargs="+", default=["cifar100"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()

    ds_str = " ".join(args.datasets)
    seed_str = " ".join(str(s) for s in args.seeds)

    steps = [
        ("Spectral validation (Fig 3)", [
            sys.executable, "experiments/run_spectral_validation.py",
            "--datasets", *args.datasets, "--seed", str(args.seeds[0]),
        ]),
        ("Risk scaling (Fig 4, RQ1)", [
            sys.executable, "experiments/run_risk_scaling.py",
            "--datasets", *args.datasets, "--seed", str(args.seeds[0]),
        ]),
        ("Scaling law + debt (Fig 5, RQ2)", [
            sys.executable, "experiments/run_scaling_law.py",
            "--datasets", *args.datasets, "--seeds", *[str(s) for s in args.seeds],
        ]),
        ("Calibration comparison (Fig 6, RQ3)", [
            sys.executable, "experiments/run_calibration.py",
            "--datasets", *args.datasets, "--seeds", *[str(s) for s in args.seeds],
        ]),
        ("Ablations (Appendix)", [
            sys.executable, "experiments/run_ablation.py",
            "--datasets", *args.datasets, "--seed", str(args.seeds[0]),
        ]),
    ]

    total = len(steps)
    for i, (name, cmd) in enumerate(steps, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{total}] {name}")
        print(f"{'='*70}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  [FAIL] {name}: {e}")
            continue

    # Final aggregation across all runs
    subprocess.run([
        sys.executable, "scripts/aggregate_results.py",
        "--runs-dir", "runs/", "--out", "results/",
    ], check=True)
    subprocess.run([
        sys.executable, "utils/visualization.py",
        "--results-dir", "results/", "--output-dir", "figures/",
    ], check=True)

    print(f"\n{'='*70}")
    print("All experiments complete.")
    print("  Results:  results/")
    print("  Figures:  figures/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
