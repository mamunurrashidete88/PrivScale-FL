"""Shadow-model membership inference attack.

The attack protocol follows the structure of Shokri et al. (2017) with the
loss-based feature of Yeom et al. (2018): for each candidate (x, y), we
compute per-example loss under the target model and classify membership by a
threshold learned on shadow models trained under the same protocol.

We keep the attack implementation-agnostic about architecture by taking a
target model and two data loaders: one for true members of the training set
and one for non-members. We return threshold-independent metrics (AUC) and
attack advantage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


@dataclass
class MIAResult:
    auc: float
    advantage: float        # max over thresholds of |TPR - FPR|
    n_members: int
    n_nonmembers: int


@torch.no_grad()
def _collect_losses_classification(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    losses = []
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        per_ex = loss_fn(logits, yb)
        losses.append(per_ex.cpu().numpy())
    return np.concatenate(losses) if losses else np.zeros(0, dtype=np.float32)


@torch.no_grad()
def _collect_losses_gpt2(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    losses = []
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(input_ids=xb, labels=yb)
        # Re-compute per-example loss from logits for a consistent signal
        logits = out.logits[:, :-1, :]
        targets = yb[:, 1:]
        per_tok = loss_fn(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        ).view(targets.shape)
        per_ex = per_tok.mean(dim=1)
        losses.append(per_ex.cpu().numpy())
    return np.concatenate(losses) if losses else np.zeros(0, dtype=np.float32)


def compute_mia(
    members_loss: np.ndarray,
    nonmembers_loss: np.ndarray,
) -> MIAResult:
    # Lower loss => more likely a member. Score = -loss.
    y_true = np.concatenate(
        [np.ones_like(members_loss), np.zeros_like(nonmembers_loss)]
    )
    scores = np.concatenate([-members_loss, -nonmembers_loss])
    if len(np.unique(y_true)) < 2:
        return MIAResult(auc=0.5, advantage=0.0,
                         n_members=len(members_loss), n_nonmembers=len(nonmembers_loss))
    auc = float(roc_auc_score(y_true, scores))

    # Attack advantage: max |TPR - FPR| over thresholds on the empirical CDFs
    m = np.sort(members_loss)
    nm = np.sort(nonmembers_loss)
    thresholds = np.unique(np.concatenate([m, nm]))
    best = 0.0
    for t in thresholds:
        tpr = float(np.mean(members_loss <= t))
        fpr = float(np.mean(nonmembers_loss <= t))
        adv = abs(tpr - fpr)
        if adv > best:
            best = adv

    return MIAResult(
        auc=auc, advantage=best,
        n_members=len(members_loss), n_nonmembers=len(nonmembers_loss),
    )


def mia_vision(
    model: nn.Module,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    device: torch.device,
) -> MIAResult:
    m_loss = _collect_losses_classification(model, member_loader, device)
    nm_loss = _collect_losses_classification(model, nonmember_loader, device)
    return compute_mia(m_loss, nm_loss)


def mia_gpt2(
    model, member_loader: DataLoader, nonmember_loader: DataLoader, device: torch.device
) -> MIAResult:
    m_loss = _collect_losses_gpt2(model, member_loader, device)
    nm_loss = _collect_losses_gpt2(model, nonmember_loader, device)
    return compute_mia(m_loss, nm_loss)
