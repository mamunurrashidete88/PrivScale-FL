"""CIFAR-10/100 loaders and client partition construction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .partitioners import dirichlet_partition, pathological_partition, summarize_partition


CIFAR_MEAN = {
    "cifar10": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.5071, 0.4865, 0.4409),
}
CIFAR_STD = {
    "cifar10": (0.2470, 0.2435, 0.2616),
    "cifar100": (0.2673, 0.2564, 0.2762),
}


@dataclass
class FederatedVisionSplit:
    train_set: Dataset
    test_set: Dataset
    client_indices: list[np.ndarray]
    num_classes: int
    partition_info: dict


def _build_transforms(dataset_name: str, augment: bool):
    mean = CIFAR_MEAN[dataset_name]
    std = CIFAR_STD[dataset_name]
    normalize = transforms.Normalize(mean, std)
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = transforms.Compose([transforms.ToTensor(), normalize])
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])
    return train_tf, test_tf


def load_cifar_federated(
    dataset_name: str,
    data_root: str,
    num_clients: int,
    partition: str = "dirichlet",
    alpha: float = 0.1,
    classes_per_client: int = 2,
    seed: int = 0,
    augment: bool = True,
) -> FederatedVisionSplit:
    """Load CIFAR-10/100 and partition the training set across clients."""
    assert dataset_name in ("cifar10", "cifar100")
    train_tf, test_tf = _build_transforms(dataset_name, augment=augment)
    ds_cls = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100

    data_root = str(Path(data_root).expanduser())
    train_set = ds_cls(root=data_root, train=True, download=True, transform=train_tf)
    test_set = ds_cls(root=data_root, train=False, download=True, transform=test_tf)

    labels = np.array(train_set.targets)

    if partition == "dirichlet":
        client_idx = dirichlet_partition(labels, num_clients, alpha=alpha, seed=seed)
    elif partition == "pathological":
        client_idx = pathological_partition(
            labels, num_clients, classes_per_client=classes_per_client, seed=seed
        )
    elif partition == "iid":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(labels))
        client_idx = [np.sort(c).astype(np.int64) for c in np.array_split(perm, num_clients)]
    else:
        raise ValueError(f"Unknown partition strategy: {partition}")

    info = summarize_partition(client_idx, labels)
    num_classes = 10 if dataset_name == "cifar10" else 100
    return FederatedVisionSplit(
        train_set=train_set,
        test_set=test_set,
        client_indices=client_idx,
        num_classes=num_classes,
        partition_info=info,
    )


def make_client_loader(
    split: FederatedVisionSplit,
    client_id: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    subset = Subset(split.train_set, split.client_indices[client_id].tolist())
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def make_test_loader(
    split: FederatedVisionSplit, batch_size: int = 256, num_workers: int = 0
) -> DataLoader:
    return DataLoader(
        split.test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
