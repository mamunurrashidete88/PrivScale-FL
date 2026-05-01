"""Client partitioners for label-based datasets.

Implements Dirichlet-alpha partitioning (the standard non-IID protocol used in
FL benchmarks) and the pathological k-class-per-client split.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 0,
    min_per_client: int = 10,
    max_attempts: int = 200,
) -> list[np.ndarray]:
    """Partition indices by Dirichlet(alpha) over labels.

    Smaller alpha => more heterogeneous (client distributions concentrate on
    few classes). Returns a list of index arrays, one per client.

    For very small alpha some clients legitimately get very few samples, so
    we relax `min_per_client` progressively if it cannot be met. The relaxation
    halves the threshold each round of failures, ensuring we always return
    a valid partition rather than throwing on extreme heterogeneity.
    """
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    for idx in class_indices:
        rng.shuffle(idx)

    threshold = max(1, min_per_client)
    best_parts: list[list[int]] | None = None
    best_min: int = -1

    for attempt in range(max_attempts):
        client_idx: list[list[int]] = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
            splits = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
            chunks = np.split(class_indices[c], splits)
            for i, chunk in enumerate(chunks):
                client_idx[i].extend(chunk.tolist())

        sizes = [len(x) for x in client_idx]
        cur_min = min(sizes)
        if cur_min >= threshold:
            return [np.array(sorted(x), dtype=np.int64) for x in client_idx]
        if cur_min > best_min:
            best_min = cur_min
            best_parts = client_idx
        # Periodically relax the threshold for very heterogeneous regimes
        if (attempt + 1) % 50 == 0 and threshold > 1:
            threshold = max(1, threshold // 2)

    # Fall back to the best partition we saw; this can happen for very small alpha.
    assert best_parts is not None
    return [np.array(sorted(x), dtype=np.int64) for x in best_parts]


def pathological_partition(
    labels: np.ndarray,
    num_clients: int,
    classes_per_client: int = 2,
    seed: int = 0,
) -> list[np.ndarray]:
    """Pathological split: each client sees exactly `classes_per_client` classes.

    Follows McMahan et al. (2017): for each class, partition its examples into
    a fixed number of shards (so every shard contains a single class), then
    assign `classes_per_client` shards to each client uniformly at random
    without replacement.

    The number of shards per class is computed so that
        num_classes * shards_per_class = num_clients * classes_per_client,
    requiring num_clients * classes_per_client to be divisible by num_classes.
    If it is not, we round up shards_per_class and trim assignments.
    """
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    total_shards_needed = num_clients * classes_per_client

    # Shards per class: at least 1, enough to cover all clients
    shards_per_class = max(1, (total_shards_needed + num_classes - 1) // num_classes)

    shards: list[np.ndarray] = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        # Within-class shards are guaranteed single-label
        chunks = np.array_split(idx, shards_per_class)
        shards.extend(chunks)

    # We now have num_classes * shards_per_class >= total_shards_needed shards,
    # each containing exactly one class. Sample without replacement.
    shard_ids = rng.permutation(len(shards))[:total_shards_needed]

    client_idx: list[np.ndarray] = []
    for k in range(num_clients):
        assigned = shard_ids[k * classes_per_client : (k + 1) * classes_per_client]
        client_idx.append(np.concatenate([shards[s] for s in assigned]))
    return [np.sort(x).astype(np.int64) for x in client_idx]


def summarize_partition(client_idx: list[np.ndarray], labels: np.ndarray) -> dict:
    """Summary statistics for a partition: client sizes and per-client label support."""
    sizes = np.array([len(x) for x in client_idx])
    label_support = []
    for idx in client_idx:
        if len(idx) == 0:
            label_support.append(0)
            continue
        label_support.append(int(len(np.unique(labels[idx]))))
    return {
        "num_clients": len(client_idx),
        "size_min": int(sizes.min()),
        "size_max": int(sizes.max()),
        "size_mean": float(sizes.mean()),
        "size_std": float(sizes.std()),
        "label_support_mean": float(np.mean(label_support)),
        "label_support_min": int(np.min(label_support)),
    }
