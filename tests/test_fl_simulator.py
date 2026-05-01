"""End-to-end test: tiny FL run on synthetic data converges.

Uses a 2-class linearly separable dataset partitioned across 5 clients, a
1-layer linear classifier, and FedAvg. After a few rounds the model should
achieve high accuracy. This catches integration regressions in the server,
client trainer, and flatten/unflatten code.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from privscale.fl import (
    FederatedServer, ServerConfig, LocalTrainConfig,
    train_local_vision, eval_classification, trainable_size,
)
from privscale.privacy import NonPrivateMechanism


def _make_synthetic(n_per_client: int = 200, n_clients: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Two Gaussians in 8-D, well-separated
    mu = rng.normal(0, 1, size=8)
    loaders = []
    test_x, test_y = [], []
    for k in range(n_clients):
        x_pos = rng.normal(loc=+mu, scale=0.5, size=(n_per_client // 2, 8))
        x_neg = rng.normal(loc=-mu, scale=0.5, size=(n_per_client // 2, 8))
        x = np.concatenate([x_pos, x_neg], axis=0)
        y = np.concatenate([np.ones(n_per_client // 2), np.zeros(n_per_client // 2)]).astype(np.int64)
        ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y))
        loaders.append(DataLoader(ds, batch_size=32, shuffle=True))
    # Held-out test set with the same structure
    n_test = 400
    x_pos = rng.normal(loc=+mu, scale=0.5, size=(n_test // 2, 8))
    x_neg = rng.normal(loc=-mu, scale=0.5, size=(n_test // 2, 8))
    x = np.concatenate([x_pos, x_neg], axis=0)
    y = np.concatenate([np.ones(n_test // 2), np.zeros(n_test // 2)]).astype(np.int64)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y)),
        batch_size=64, shuffle=False,
    )
    return loaders, test_loader


def test_fedavg_converges_on_synthetic():
    torch.manual_seed(0)
    loaders, test_loader = _make_synthetic()
    model = nn.Linear(8, 2)

    server = FederatedServer(
        global_model=model,
        client_loader_fn=lambda cid: loaders[cid],
        local_trainer_fn=train_local_vision,
        local_cfg=LocalTrainConfig(epochs=1, lr=0.05, momentum=0.9, max_steps=10),
        aggregation_fn=NonPrivateMechanism(clip_norm=None),
        server_cfg=ServerConfig(
            num_clients=len(loaders), sample_fraction=1.0, rounds=20,
            sampling="uniform", eval_every=10,
        ),
        eval_fn=lambda m: eval_classification(m, test_loader, torch.device("cpu")),
        device=torch.device("cpu"),
        rng_seed=0,
    )

    accs = []
    server.run(log_callback=lambda rec: accs.append(rec.get("test_acc")) if rec.get("eval") else None)
    final_acc = [a for a in accs if a is not None][-1]
    assert final_acc > 0.85, final_acc
    assert trainable_size(model) == 8 * 2 + 2
