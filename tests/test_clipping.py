"""Tests for L2 clipping invariants."""
from __future__ import annotations

import torch

from privscale.privacy import clip_l2, clip_l2_batch


def test_below_threshold_unchanged():
    x = torch.randn(100)
    x = x / x.norm() * 0.5
    y, n = clip_l2(x, clip_norm=1.0)
    assert torch.allclose(y, x)
    assert abs(n - 0.5) < 1e-6


def test_above_threshold_clipped():
    x = torch.randn(100)
    x = x / x.norm() * 5.0
    y, n = clip_l2(x, clip_norm=1.0)
    assert abs(y.norm().item() - 1.0) < 1e-5
    assert abs(n - 5.0) < 1e-5
    assert torch.allclose(y, x / 5.0, atol=1e-5)


def test_batch_norms_recorded():
    deltas = []
    for s in (0.1, 0.5, 2.0, 10.0):
        v = torch.ones(50)
        v = v / v.norm() * s
        deltas.append(v)
    clipped, norms = clip_l2_batch(deltas, clip_norm=1.0)
    assert all(c.norm().item() <= 1.0 + 1e-5 for c in clipped)
    assert all(abs(n - s) < 1e-5 for n, s in zip(norms, (0.1, 0.5, 2.0, 10.0)))
