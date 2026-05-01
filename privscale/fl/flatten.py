"""Flatten and unflatten trainable parameters of a model.

All privacy mechanisms and the aggregation loop operate on flat vectors, so
the only architecture-dependent code is in the client trainer.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def flat_params(model: nn.Module) -> torch.Tensor:
    """Detached, CPU-side flat copy of current trainable parameters."""
    return torch.cat([p.detach().reshape(-1).cpu() for p in trainable_params(model)])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    """Copy a flat CPU tensor back into the model's trainable parameters."""
    offset = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for p in trainable_params(model):
            n = p.numel()
            chunk = flat[offset : offset + n].to(device=device, dtype=p.dtype)
            p.copy_(chunk.view_as(p))
            offset += n
    assert offset == flat.numel(), "Flat parameter length mismatch"


def param_shapes(model: nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    return [(n, tuple(p.shape)) for n, p in model.named_parameters() if p.requires_grad]


def trainable_size(model: nn.Module) -> int:
    return sum(p.numel() for p in trainable_params(model))
