"""Evaluation helpers for vision, character-LM, and GPT-2 settings."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def eval_classification(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item()
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += xb.size(0)
    return {"test_loss": total_loss / max(1, n), "test_acc": correct / max(1, n)}


@torch.no_grad()
def eval_charlm(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        total_loss += loss.item()
        total_tokens += yb.numel()
    avg = total_loss / max(1, total_tokens)
    return {"test_loss": avg, "test_ppl": math.exp(min(20.0, avg))}


@torch.no_grad()
def eval_gpt2(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_tokens = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(input_ids=xb, labels=yb)
        total_loss += out.loss.item() * yb.numel()
        total_tokens += yb.numel()
        total_batches += 1
    avg = total_loss / max(1, total_tokens)
    return {"test_loss": avg, "test_ppl": math.exp(min(20.0, avg))}
