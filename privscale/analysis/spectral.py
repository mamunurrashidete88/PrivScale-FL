"""Effective model dimension estimation from client update covariances.

Given per-round client updates collected during training, we stack them into a
matrix U of shape [n_client_samples, p] (p = trainable parameter count), center
it, and compute the leading singular values via randomized SVD. These give the
eigenvalues of the empirical covariance matrix
    A_hat = (1/n) * (U - mean(U))^T (U - mean(U))
up to the factor 1/n.

We provide three quantities used throughout the analysis:
  - threshold-based numerical rank d_tau (default tau = 0.01 of lambda_max),
  - participation ratio r_hat = tr(A)^2 / tr(A^2),
  - total signal power tr(A).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SpectralSummary:
    eigenvalues: np.ndarray       # descending
    trace: float
    trace_sq: float
    participation_ratio: float
    d_tau: int
    tau: float


def collect_centered_matrix(updates: list[torch.Tensor]) -> torch.Tensor:
    """Stack a list of flat updates into a centered matrix."""
    U = torch.stack(updates, dim=0)  # [n, p]
    U = U - U.mean(dim=0, keepdim=True)
    return U


def randomized_svd_topk(
    U: torch.Tensor, k: int, n_oversamples: int = 10, n_iter: int = 4, seed: int = 0
) -> np.ndarray:
    """Top-k singular values of U via Halko/Martinsson/Tropp (2011)."""
    n, p = U.shape
    k = min(k, n, p)
    gen = torch.Generator().manual_seed(seed)
    Omega = torch.randn(p, k + n_oversamples, generator=gen, dtype=U.dtype)
    Y = U @ Omega  # [n, k+os]
    # Power iterations for better accuracy on decaying spectra
    for _ in range(n_iter):
        Q, _ = torch.linalg.qr(Y)
        Z = U.T @ Q
        Q2, _ = torch.linalg.qr(Z)
        Y = U @ Q2
    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ U                       # [k+os, p]
    _, s, _ = torch.linalg.svd(B, full_matrices=False)
    return s[:k].cpu().numpy()


def spectral_summary(
    updates: list[torch.Tensor],
    k: int = 256,
    tau: float = 0.01,
    seed: int = 0,
) -> SpectralSummary:
    """Compute spectral diagnostics from a list of flat update vectors.

    Eigenvalues of the empirical covariance are sigma_i^2 / n, where sigma_i
    are the singular values of the centered matrix.
    """
    if len(updates) < 2:
        raise ValueError("Need at least 2 updates to estimate a covariance.")
    U = collect_centered_matrix(updates)
    n = U.shape[0]
    svals = randomized_svd_topk(U, k=min(k, n), seed=seed)
    # Covariance eigenvalues
    eigs = (svals ** 2) / max(1, n - 1)

    tr = float(eigs.sum())
    # Full trace and trace^2 require more eigenvalues than we computed; for
    # the participation ratio we use the top-k approximation, which is a
    # lower bound on r_hat.
    tr_sq_topk = float((eigs ** 2).sum())
    pr = tr * tr / max(1e-24, tr_sq_topk)

    lam_max = float(eigs[0]) if len(eigs) else 0.0
    d_tau = int((eigs >= tau * lam_max).sum()) if lam_max > 0 else 0

    return SpectralSummary(
        eigenvalues=eigs,
        trace=tr,
        trace_sq=tr_sq_topk,
        participation_ratio=pr,
        d_tau=d_tau,
        tau=tau,
    )


def trajectory_proxy(updates: list[torch.Tensor], noise_std: float) -> float:
    """Trajectory-level signal proxy used as a signal-side observable.

    Computes 0.5 * sum_i log(1 + lambda_i / sigma^2) on the top-k covariance
    eigenvalues — the Gaussian-channel mutual information bound evaluated on
    the empirical covariance. This is a diagnostic quantity, not a privacy
    guarantee.
    """
    if noise_std <= 0:
        return float("inf")
    summ = spectral_summary(updates)
    eigs = summ.eigenvalues
    s2 = noise_std * noise_std
    return 0.5 * float(np.sum(np.log1p(eigs / s2)))
