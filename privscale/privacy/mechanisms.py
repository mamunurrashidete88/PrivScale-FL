"""Aggregation mechanisms: non-private, fixed-noise DP, adaptive-clipping DP,
and scale-aware DP.

Each mechanism exposes the same interface as a callable:
    mechanism(deltas: list[Tensor], round_idx: int) -> Tensor

All mechanisms return the (server-side) averaged, possibly privatized update.

DP-FedAvg bookkeeping follows the client-level DP threat model: one client
contributes one clipped update per round in which it participates, and we add
Gaussian noise of standard deviation sigma * clip_norm / expected_participants
to the *sum* (equivalently, sigma * clip_norm / (q * K) to the mean).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .clipping import clip_l2_batch
from .rdp_accountant import RDPAccountant


# ---------- Non-private ----------

class NonPrivateMechanism:
    """Plain FedAvg aggregation: mean of deltas."""

    def __init__(self, clip_norm: float | None = None):
        # Optional clipping (even non-private runs sometimes clip for stability)
        self.clip_norm = clip_norm
        self.last_norms: list[float] = []

    def __call__(self, deltas: list[torch.Tensor], round_idx: int) -> torch.Tensor:
        if self.clip_norm is not None:
            deltas, norms = clip_l2_batch(deltas, self.clip_norm)
            self.last_norms = norms
        else:
            self.last_norms = [float(d.norm().item()) for d in deltas]
        return torch.stack(deltas, dim=0).mean(dim=0)


# ---------- Fixed-noise DP ----------

@dataclass
class DPConfig:
    clip_norm: float
    noise_multiplier: float        # sigma, in units of clip_norm
    expected_participants: float   # q * K; used to scale noise to the sum


class FixedNoiseDPMechanism:
    """DP-FedAvg with a fixed noise multiplier across all rounds.

    Per-client clipping, per-round Gaussian noise added to the sum, mean
    aggregation. The noise std on the *mean* is
        sigma * clip_norm / expected_participants.
    """

    def __init__(self, cfg: DPConfig, rng_seed: int = 0):
        self.cfg = cfg
        self.gen = torch.Generator().manual_seed(rng_seed)
        self.last_norms: list[float] = []
        self.last_sigma = cfg.noise_multiplier

    def _effective_noise_std(self) -> float:
        # std applied to the mean update, so that the sum-noise std equals
        # sigma * clip_norm.
        return self.cfg.noise_multiplier * self.cfg.clip_norm / max(1e-12, self.cfg.expected_participants)

    def __call__(self, deltas: list[torch.Tensor], round_idx: int) -> torch.Tensor:
        clipped, norms = clip_l2_batch(deltas, self.cfg.clip_norm)
        self.last_norms = norms
        stacked = torch.stack(clipped, dim=0)
        mean = stacked.mean(dim=0)
        std = self._effective_noise_std()
        noise = torch.randn(mean.shape, generator=self.gen) * std
        return mean + noise


# ---------- Adaptive clipping DP ----------

@dataclass
class AdaptiveClipConfig:
    init_clip_norm: float
    target_quantile: float = 0.5          # Andrew et al. 2021 default: median
    lr_clip: float = 0.2                  # geometric step on log(clip)
    noise_multiplier: float = 1.0
    expected_participants: float = 10.0
    # Noise multiplier is for the clip+noise combined mechanism. Following
    # Andrew et al., a portion of the "privacy budget" goes to the quantile
    # estimate; we expose the split via quantile_noise_frac below.
    quantile_noise_frac: float = 0.05     # fraction of variance for clip estimate


class AdaptiveClipDPMechanism:
    """Quantile-based adaptive clipping with fixed noise multiplier.

    Follows Andrew, McMahan, Thakurta, Wang, Yu (2021) "Differentially Private
    Learning with Adaptive Clipping": each round, privately estimate the
    `target_quantile` of client update norms and update the clip threshold by a
    geometric step. The additive noise on the aggregated update uses the
    current clip threshold.
    """

    def __init__(self, cfg: AdaptiveClipConfig, rng_seed: int = 0):
        self.cfg = cfg
        self.clip_norm = float(cfg.init_clip_norm)
        self.gen = torch.Generator().manual_seed(rng_seed)
        self.last_norms: list[float] = []
        self.last_sigma = cfg.noise_multiplier

    def _noise_std_update(self) -> float:
        # Share (1 - quantile_noise_frac) of the privacy budget to the model
        # update; equivalent to multiplying the effective noise multiplier.
        frac = math.sqrt(1.0 - self.cfg.quantile_noise_frac)
        sigma_m = self.cfg.noise_multiplier / max(1e-6, frac)
        return sigma_m * self.clip_norm / max(1e-12, self.cfg.expected_participants)

    def _noise_std_quantile(self) -> float:
        frac = math.sqrt(self.cfg.quantile_noise_frac)
        sigma_q = self.cfg.noise_multiplier / max(1e-6, frac)
        # Noise on a bit (clipped to [0, 1]); scale is sigma_q / expected_participants
        return sigma_q / max(1e-12, self.cfg.expected_participants)

    def __call__(self, deltas: list[torch.Tensor], round_idx: int) -> torch.Tensor:
        # 1) Private quantile update. Each client contributes a [0,1]-valued
        #    indicator of whether its norm <= current clip. We average with
        #    noise to get a private estimate of the below-threshold fraction.
        raw_norms = [float(d.norm().item()) for d in deltas]
        indicator = [1.0 if n <= self.clip_norm else 0.0 for n in raw_norms]
        mean_indicator = float(np.mean(indicator))
        q_noise = torch.randn(1, generator=self.gen).item() * self._noise_std_quantile()
        private_estimate = mean_indicator + q_noise
        # Geometric update toward target quantile
        err = private_estimate - self.cfg.target_quantile
        self.clip_norm = max(1e-3, self.clip_norm * math.exp(-self.cfg.lr_clip * err))

        # 2) Clip at the (previous-round) threshold and aggregate with noise
        clipped, norms = clip_l2_batch(deltas, self.clip_norm)
        self.last_norms = norms
        stacked = torch.stack(clipped, dim=0)
        mean = stacked.mean(dim=0)
        std = self._noise_std_update()
        noise = torch.randn(mean.shape, generator=self.gen) * std
        return mean + noise


# ---------- Scale-aware DP ----------

@dataclass
class ScaleAwareConfig:
    clip_norm: float
    reference_sigma: float
    reference_dim: int
    effective_dim: int
    expected_participants: float


class ScaleAwareDPMechanism:
    """Scale-aware noise calibration.

    Given a reference calibration (sigma_0 at d_0), noise at effective
    dimension d is set to
        sigma(d) = sigma_0 * sqrt( d * log(d_0) / ( d_0 * log(d) ) ).

    This follows the prescription sigma(d) = Theta(sqrt(d / log d)) with the
    constant pinned to the reference scale.
    """

    def __init__(self, cfg: ScaleAwareConfig, rng_seed: int = 0):
        self.cfg = cfg
        self.gen = torch.Generator().manual_seed(rng_seed)
        self.noise_multiplier = self._compute_sigma(cfg.effective_dim)
        self.last_norms: list[float] = []
        self.last_sigma = self.noise_multiplier

    def _compute_sigma(self, d: int) -> float:
        d0 = max(2, self.cfg.reference_dim)
        dd = max(2, d)
        log_d0 = math.log(d0)
        log_d = math.log(dd)
        factor = math.sqrt((dd * log_d0) / (d0 * log_d))
        return self.cfg.reference_sigma * factor

    def _effective_noise_std(self) -> float:
        return self.noise_multiplier * self.cfg.clip_norm / max(1e-12, self.cfg.expected_participants)

    def __call__(self, deltas: list[torch.Tensor], round_idx: int) -> torch.Tensor:
        clipped, norms = clip_l2_batch(deltas, self.cfg.clip_norm)
        self.last_norms = norms
        stacked = torch.stack(clipped, dim=0)
        mean = stacked.mean(dim=0)
        std = self._effective_noise_std()
        noise = torch.randn(mean.shape, generator=self.gen) * std
        return mean + noise


# ---------- Factory ----------

def build_mechanism(spec: dict, rng_seed: int = 0):
    """Build a mechanism from a config dict with key `name`."""
    name = spec["name"].lower()
    if name in ("nonprivate", "non_private", "fedavg"):
        return NonPrivateMechanism(clip_norm=spec.get("clip_norm"))
    if name in ("fixed_noise", "fixed_dp", "dp_fedavg"):
        cfg = DPConfig(
            clip_norm=spec["clip_norm"],
            noise_multiplier=spec["noise_multiplier"],
            expected_participants=spec["expected_participants"],
        )
        return FixedNoiseDPMechanism(cfg, rng_seed=rng_seed)
    if name in ("adaptive_clip", "adaptive"):
        cfg = AdaptiveClipConfig(
            init_clip_norm=spec["init_clip_norm"],
            target_quantile=spec.get("target_quantile", 0.5),
            lr_clip=spec.get("lr_clip", 0.2),
            noise_multiplier=spec["noise_multiplier"],
            expected_participants=spec["expected_participants"],
            quantile_noise_frac=spec.get("quantile_noise_frac", 0.05),
        )
        return AdaptiveClipDPMechanism(cfg, rng_seed=rng_seed)
    if name in ("scale_aware", "scaleaware"):
        cfg = ScaleAwareConfig(
            clip_norm=spec["clip_norm"],
            reference_sigma=spec["reference_sigma"],
            reference_dim=spec["reference_dim"],
            effective_dim=spec["effective_dim"],
            expected_participants=spec["expected_participants"],
        )
        return ScaleAwareDPMechanism(cfg, rng_seed=rng_seed)
    raise ValueError(f"Unknown mechanism: {name}")
