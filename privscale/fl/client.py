"""Client-side local training for one FL round.

The trainer takes the current global flat parameters, runs E epochs of local
SGD, and returns the model delta (new - old) as a flat CPU tensor. Optimizer
state is discarded between rounds, matching standard FedAvg.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .flatten import flat_params, set_flat_params


@dataclass
class LocalTrainConfig:
    epochs: int = 5
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    max_steps: int | None = None  # cap steps per round (None = full pass)


def train_local_vision(
    model: nn.Module,
    loader: DataLoader,
    cfg: LocalTrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Return (flat_delta, stats) after E epochs on a CIFAR/FEMNIST client.

    Model is assumed to already be on `device` with the current global
    parameters loaded.
    """
    model.train()
    before = flat_params(model)
    opt = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    n_steps = 0
    total_loss = 0.0
    n_samples = 0
    n_correct = 0

    for _ in range(cfg.epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            n_steps += 1
            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
            n_correct += (logits.argmax(dim=1) == yb).sum().item()

            if cfg.max_steps is not None and n_steps >= cfg.max_steps:
                break
        if cfg.max_steps is not None and n_steps >= cfg.max_steps:
            break

    after = flat_params(model)
    delta = after - before
    stats = {
        "steps": n_steps,
        "samples": n_samples,
        "train_loss": total_loss / max(1, n_samples),
        "train_acc": n_correct / max(1, n_samples),
    }
    return delta, stats


def train_local_charlm(
    model: nn.Module,
    loader: DataLoader,
    cfg: LocalTrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Character-LM local training (logits [B, T, V])."""
    model.train()
    before = flat_params(model)
    opt = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    n_steps = 0
    total_loss = 0.0
    n_tokens = 0

    for _ in range(cfg.epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)  # [B, T, V]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            opt.step()

            n_steps += 1
            total_loss += loss.item() * yb.numel()
            n_tokens += yb.numel()

            if cfg.max_steps is not None and n_steps >= cfg.max_steps:
                break
        if cfg.max_steps is not None and n_steps >= cfg.max_steps:
            break

    after = flat_params(model)
    delta = after - before
    stats = {
        "steps": n_steps,
        "tokens": n_tokens,
        "train_loss": total_loss / max(1, n_tokens),
    }
    return delta, stats


def train_local_gpt2(
    model,  # GPT2LM wrapper
    loader: DataLoader,
    cfg: LocalTrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """GPT-2 causal-LM local training via HF model's built-in loss."""
    model.train()
    before = flat_params(model)
    # AdamW works better than SGD for GPT-2; lr is interpreted accordingly
    # (we expose this via the caller's cfg.lr).
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    n_steps = 0
    total_loss = 0.0
    n_batches = 0

    for _ in range(cfg.epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=xb, labels=yb)
            loss = out.loss
            loss.backward()
            opt.step()

            n_steps += 1
            total_loss += loss.item()
            n_batches += 1

            if cfg.max_steps is not None and n_steps >= cfg.max_steps:
                break
        if cfg.max_steps is not None and n_steps >= cfg.max_steps:
            break

    after = flat_params(model)
    delta = after - before
    stats = {
        "steps": n_steps,
        "train_loss": total_loss / max(1, n_batches),
    }
    return delta, stats


LocalTrainerFn = Callable[[nn.Module, DataLoader, LocalTrainConfig, torch.device],
                          tuple[torch.Tensor, dict]]


def get_trainer(modality: str) -> LocalTrainerFn:
    if modality == "vision":
        return train_local_vision
    if modality == "charlm":
        return train_local_charlm
    if modality == "gpt2":
        return train_local_gpt2
    raise ValueError(f"Unknown modality: {modality}")
