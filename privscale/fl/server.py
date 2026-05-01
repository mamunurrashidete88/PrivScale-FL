"""FedAvg server loop with pluggable privacy mechanism.

The server:
  1. Samples a subset of clients per round (Poisson or uniform-without-replacement).
  2. For each sampled client: loads current global weights, runs local training,
     obtains the delta, and hands it to the privacy mechanism for clipping+noise.
  3. Aggregates the (privatized) deltas and applies the update to the global
     model. Participation weighting is uniform over sampled clients (standard
     DP-FedAvg).

The mechanism is a callable with signature
    mechanism(deltas: list[Tensor], round_idx: int) -> Tensor
that returns the aggregated, privatized update.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .flatten import flat_params, set_flat_params, trainable_size
from .client import LocalTrainConfig, LocalTrainerFn


AggregationFn = Callable[[list[torch.Tensor], int], torch.Tensor]
ClientLoaderFn = Callable[[int], DataLoader]  # client_id -> DataLoader
EvalFn = Callable[[nn.Module], dict]  # model -> {metric: value}


@dataclass
class ServerConfig:
    num_clients: int
    sample_fraction: float = 0.1
    rounds: int = 500
    sampling: str = "uniform"          # "uniform" or "poisson"
    eval_every: int = 10
    checkpoint_every: int = 0          # 0 disables
    save_update_trace: bool = False    # keep per-round aggregated delta for analysis
    log_per_client_stats: bool = False


class FederatedServer:
    def __init__(
        self,
        global_model: nn.Module,
        client_loader_fn: ClientLoaderFn,
        local_trainer_fn: LocalTrainerFn,
        local_cfg: LocalTrainConfig,
        aggregation_fn: AggregationFn,
        server_cfg: ServerConfig,
        eval_fn: EvalFn | None = None,
        device: torch.device | None = None,
        server_lr: float = 1.0,
        rng_seed: int = 0,
    ):
        self.model = global_model
        self.client_loader_fn = client_loader_fn
        self.local_trainer_fn = local_trainer_fn
        self.local_cfg = local_cfg
        self.aggregation_fn = aggregation_fn
        self.cfg = server_cfg
        self.eval_fn = eval_fn
        self.device = device or next(global_model.parameters()).device
        self.server_lr = server_lr
        self.rng = np.random.default_rng(rng_seed)
        self.d_flat = trainable_size(global_model)

    # ---- sampling ----
    def _sample_clients(self) -> list[int]:
        K = self.cfg.num_clients
        q = self.cfg.sample_fraction
        if self.cfg.sampling == "poisson":
            mask = self.rng.random(K) < q
            ids = np.where(mask)[0].tolist()
            if not ids:
                ids = [int(self.rng.integers(0, K))]
            return ids
        # uniform without replacement
        m = max(1, int(round(q * K)))
        return sorted(self.rng.choice(K, size=m, replace=False).tolist())

    # ---- one round ----
    def _run_round(self, round_idx: int) -> dict:
        t0 = time.time()
        sampled = self._sample_clients()
        global_flat = flat_params(self.model)

        deltas: list[torch.Tensor] = []
        client_stats: list[dict] = []

        for cid in sampled:
            loader = self.client_loader_fn(cid)
            if len(loader) == 0:
                continue
            # Load fresh global weights
            set_flat_params(self.model, global_flat)
            delta, stats = self.local_trainer_fn(self.model, loader, self.local_cfg, self.device)
            stats["client_id"] = cid
            client_stats.append(stats)
            deltas.append(delta)

        if not deltas:
            return {"round": round_idx, "sampled": 0, "elapsed": time.time() - t0}

        # Privacy mechanism: per-client clip, aggregate, add noise
        agg_update = self.aggregation_fn(deltas, round_idx)

        # Apply server update
        new_flat = global_flat + self.server_lr * agg_update
        set_flat_params(self.model, new_flat)

        record = {
            "round": round_idx,
            "sampled": len(deltas),
            "elapsed": time.time() - t0,
            "delta_norm_mean": float(torch.stack([d.norm() for d in deltas]).mean().item()),
            "agg_norm": float(agg_update.norm().item()),
        }
        if self.cfg.log_per_client_stats:
            record["client_stats"] = client_stats
        if self.cfg.save_update_trace:
            record["agg_delta"] = agg_update.numpy()
        return record

    # ---- main loop ----
    def run(self, log_callback: Callable[[dict], None] | None = None) -> None:
        for r in range(1, self.cfg.rounds + 1):
            rec = self._run_round(r)
            if log_callback is not None:
                log_callback(rec)
            if self.eval_fn is not None and (r % self.cfg.eval_every == 0 or r == self.cfg.rounds):
                metrics = self.eval_fn(self.model)
                eval_rec = {"round": r, "eval": True, **metrics}
                if log_callback is not None:
                    log_callback(eval_rec)
