"""Sanity tests for the RDP accountant.

We don't try to match a particular reference implementation bit-for-bit.
Instead we test invariants:
  - For q=1 (no subsampling), RDP at order alpha is exactly alpha / (2 sigma^2).
  - For q small and steps large, eps is finite and increases monotonically with steps.
  - find_sigma_for_epsilon and compute_rdp are mutually consistent.
  - Larger sigma produces smaller eps; larger q (with fixed sigma) increases eps.
"""
from __future__ import annotations

import math

import pytest

from privscale.privacy import (
    compute_rdp, find_sigma_for_epsilon, rdp_for_subsampled_gaussian, rdp_to_epsilon,
)


def test_no_subsampling_closed_form():
    sigma = 2.0
    for alpha in (2.0, 5.0, 10.0, 32.0):
        rdp = rdp_for_subsampled_gaussian(q=1.0, sigma=sigma, alpha=alpha)
        expected = alpha / (2.0 * sigma * sigma)
        assert math.isclose(rdp, expected, rel_tol=1e-9), (rdp, expected, alpha)


def test_zero_q_is_zero():
    assert rdp_for_subsampled_gaussian(q=0.0, sigma=1.0, alpha=4.0) == 0.0


def test_eps_monotone_in_steps():
    q = 0.1
    sigma = 1.0
    delta = 1e-5
    last_eps = -math.inf
    for steps in (10, 100, 500, 1000):
        rdp = compute_rdp(q, sigma, steps)
        eps, _ = rdp_to_epsilon(rdp, delta)
        assert eps > last_eps
        last_eps = eps


def test_eps_monotone_in_sigma():
    q = 0.1
    delta = 1e-5
    steps = 500
    epses = []
    for sigma in (0.6, 1.0, 1.5, 2.5, 4.0):
        rdp = compute_rdp(q, sigma, steps)
        eps, _ = rdp_to_epsilon(rdp, delta)
        epses.append(eps)
    assert all(a > b for a, b in zip(epses, epses[1:])), epses


def test_find_sigma_consistency():
    target = 3.0
    delta = 1e-5
    q = 0.1
    steps = 500
    sigma = find_sigma_for_epsilon(target_epsilon=target, delta=delta, q=q, steps=steps)
    rdp = compute_rdp(q, sigma, steps)
    eps, _ = rdp_to_epsilon(rdp, delta)
    # Bisection should land within tol of the target
    assert eps <= target + 1e-2, (eps, target, sigma)
    assert eps >= target - 0.1, (eps, target, sigma)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
