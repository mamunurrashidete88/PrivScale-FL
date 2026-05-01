"""Calibrate the noise multiplier for a target (epsilon, delta).

Given sampling rate q and number of rounds T, finds the smallest sigma such
that the subsampled-Gaussian RDP accountant produces eps <= target_eps at the
given delta. Used once per experimental ladder to set the reference sigma_0.

Usage:
    python scripts/calibrate_noise.py --eps 3.0 --delta 1e-5 --q 0.1 --rounds 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privscale.privacy import find_sigma_for_epsilon, compute_rdp, rdp_to_epsilon


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--q", type=float, required=True, help="Per-round client sampling rate.")
    p.add_argument("--rounds", type=int, required=True)
    args = p.parse_args()

    sigma = find_sigma_for_epsilon(
        target_epsilon=args.eps, delta=args.delta, q=args.q, steps=args.rounds
    )
    rdp = compute_rdp(args.q, sigma, args.rounds)
    eps_check, alpha = rdp_to_epsilon(rdp, args.delta)
    print(f"sigma = {sigma:.6f}  (achieves eps = {eps_check:.4f} at alpha = {alpha:.1f}, "
          f"target eps = {args.eps}, delta = {args.delta}, q = {args.q}, T = {args.rounds})")


if __name__ == "__main__":
    main()
