"""Collect client update samples across rounds for post-hoc spectral analysis.

Storing full-parameter updates is expensive for WRN-28-10 (~36M) and GPT-2
Medium (~345M). To keep storage manageable, we maintain a fixed-size reservoir
of (centered) client updates, and project them into a random subspace of
dimension `proj_dim` on-the-fly.

Using a Johnson-Lindenstrauss random projection preserves pairwise distances
and thus the leading spectrum of the covariance up to a controllable error.
We store the projected vectors to disk as float32. The effective-dimension
estimator operates on the projected matrix; the resulting d_tau is a consistent
estimate of the ambient d_tau when proj_dim is large enough.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class UpdateTraceCollector:
    def __init__(
        self,
        param_dim: int,
        proj_dim: int = 4096,
        max_samples: int = 512,
        seed: int = 0,
        save_dir: str | Path | None = None,
    ):
        self.param_dim = param_dim
        self.proj_dim = min(proj_dim, param_dim)
        self.max_samples = max_samples
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.gen = torch.Generator().manual_seed(seed)
        # JL projection matrix: Gaussian, rescaled; stored as float32
        self._proj = torch.randn(
            param_dim, self.proj_dim, generator=self.gen, dtype=torch.float32
        ) / (self.proj_dim ** 0.5)
        self.samples: list[np.ndarray] = []
        self.round_tags: list[int] = []
        self.client_tags: list[int] = []
        self._n_seen = 0  # total items observed; used for proper reservoir sampling

    def project(self, delta: torch.Tensor) -> np.ndarray:
        v = delta.detach().to(torch.float32).cpu() @ self._proj
        return v.numpy().astype(np.float32, copy=False)

    def add(self, delta: torch.Tensor, round_idx: int, client_id: int) -> None:
        self._n_seen += 1
        if len(self.samples) < self.max_samples:
            self.samples.append(self.project(delta))
            self.round_tags.append(round_idx)
            self.client_tags.append(client_id)
            return
        # Algorithm R: replace a uniformly chosen slot with probability
        # max_samples / n_seen, where n_seen counts the new item too.
        j = int(torch.randint(0, self._n_seen, (1,), generator=self.gen).item())
        if j < self.max_samples:
            self.samples[j] = self.project(delta)
            self.round_tags[j] = round_idx
            self.client_tags[j] = client_id

    def matrix(self) -> np.ndarray:
        if not self.samples:
            return np.zeros((0, self.proj_dim), dtype=np.float32)
        return np.stack(self.samples, axis=0)

    def save(self, tag: str = "trace") -> None:
        if self.save_dir is None:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.save_dir / f"{tag}.npz",
            samples=self.matrix(),
            rounds=np.asarray(self.round_tags, dtype=np.int32),
            clients=np.asarray(self.client_tags, dtype=np.int32),
            param_dim=np.int64(self.param_dim),
            proj_dim=np.int64(self.proj_dim),
        )
