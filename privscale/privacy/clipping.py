"""L2 clipping for client update vectors."""
from __future__ import annotations

import torch


def clip_l2(delta: torch.Tensor, clip_norm: float) -> tuple[torch.Tensor, float]:
    """Clip a flat update tensor to L2 norm `clip_norm`.

    Returns the clipped tensor and the (pre-clip) norm, which is useful for
    adaptive-clipping quantile estimation.
    """
    n = float(delta.norm().item())
    if n > clip_norm and n > 0:
        scale = clip_norm / n
        return delta * scale, n
    return delta, n


def clip_l2_batch(deltas: list[torch.Tensor], clip_norm: float) -> tuple[list[torch.Tensor], list[float]]:
    clipped: list[torch.Tensor] = []
    norms: list[float] = []
    for d in deltas:
        c, n = clip_l2(d, clip_norm)
        clipped.append(c)
        norms.append(n)
    return clipped, norms
