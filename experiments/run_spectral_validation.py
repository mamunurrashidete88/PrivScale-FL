"""
experiments/run_spectral_validation.py
=======================================
Reproduces Figure 3: Spectral validation of Assumption 3.

Runs non-private FL at each scale point, collects update traces, and reports:
  - Participation ratio r̂(d) vs effective dimension
  - Mean informative eigenvalue λ̄(d) vs effective dimension
  - Total signal power tr(Â(d)) vs effective dimension

Fits 1/log(d) decay for λ̄(d) and d/log(d) growth for tr(Â(d)).

Usage:
    python experiments/run_spectral_validation.py \\
        --datasets cifar100 --seed 0
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

CONFIG_MAP = {
    "cifar100": {m: f"configs/vision/cifar100_{m}_nonprivate.yaml" for m in VISION_MODELS},
    "stackoverflow": {m: f"configs/language/so_{m}_nonprivate.yaml" for m in LANGUAGE_MODELS},
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
    p = argparse.ArgumentParser(description="Fig 3: Spectral validation.")
    p.add_argument("--datasets", nargs="+", default=["cifar100"],
                   choices=list(CONFIG_MAP.keys()))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=None,
                   help="Override rounds (default: use config value).")
    args = p.parse_args()

    results = {}
    for ds in args.datasets:
        models = CONFIG_MAP[ds]
        for model_name, cfg_path in models.items():
            if not Path(cfg_path).exists():
                print(f"  [skip] {cfg_path} not found")
                continue
            out = f"runs/spectral_validation/{ds}/{model_name}/seed{args.seed}"
            run_single(cfg_path, args.seed, out)

            spec_path = Path(out) / "spectral.json"
            if spec_path.exists():
                with open(spec_path) as f:
                    spec = json.load(f)
                results.setdefault(ds, {})[model_name] = spec
                print(f"  {model_name}: d_tau={spec.get('d_tau')}, "
                      f"trace={spec.get('trace', 0):.4g}, "
                      f"pr={spec.get('participation_ratio', 0):.2f}")

    out_dir = Path("results/spectral_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "spectral_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSpectral results saved to {out_dir / 'spectral_results.json'}")


if __name__ == "__main__":
    main()
