"""Tests for client partitioners."""
from __future__ import annotations

import numpy as np

from privscale.data.partitioners import (
    dirichlet_partition, pathological_partition, summarize_partition,
)


def test_dirichlet_partition_covers_all():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, size=10000)
    parts = dirichlet_partition(labels, num_clients=20, alpha=0.5, seed=0)
    flat = np.concatenate(parts)
    assert sorted(flat.tolist()) == list(range(len(labels)))
    info = summarize_partition(parts, labels)
    assert info["num_clients"] == 20
    assert info["size_min"] >= 10


def test_dirichlet_smaller_alpha_more_skew():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, size=20000)
    parts_small = dirichlet_partition(labels, 50, alpha=0.05, seed=1)
    parts_large = dirichlet_partition(labels, 50, alpha=5.0, seed=1)
    sup_small = summarize_partition(parts_small, labels)["label_support_mean"]
    sup_large = summarize_partition(parts_large, labels)["label_support_mean"]
    # smaller alpha => fewer distinct labels per client on average
    assert sup_small < sup_large


def test_pathological_partition_classes_per_client():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, size=20000)
    parts = pathological_partition(labels, num_clients=50, classes_per_client=2, seed=0)
    for idx in parts:
        n_labels = len(np.unique(labels[idx]))
        assert n_labels <= 2
