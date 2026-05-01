"""FEMNIST loader (LEAF natural user split).

Expects preprocessed LEAF FEMNIST data in `data_root` with the standard JSON
layout:

    data_root/
        train/  (JSON files: {"users": [...], "user_data": {uid: {"x":..., "y":...}}})
        test/

See https://github.com/TalwalkarLab/leaf for preprocessing instructions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class FemnistSplit:
    train_per_user: dict[str, tuple[np.ndarray, np.ndarray]]
    test_per_user: dict[str, tuple[np.ndarray, np.ndarray]]
    users: list[str]
    num_classes: int = 62


class _FemnistTensorDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        # LEAF stores flattened 28x28 images in [0, 1]
        self.x = torch.from_numpy(x).float().view(-1, 1, 28, 28)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _read_leaf_dir(dir_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    per_user: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for f in sorted(dir_path.glob("*.json")):
        with open(f) as fh:
            blob = json.load(fh)
        for uid in blob["users"]:
            ud = blob["user_data"][uid]
            x = np.asarray(ud["x"], dtype=np.float32)
            y = np.asarray(ud["y"], dtype=np.int64)
            if uid in per_user:
                x0, y0 = per_user[uid]
                x = np.concatenate([x0, x], axis=0)
                y = np.concatenate([y0, y], axis=0)
            per_user[uid] = (x, y)
    return per_user


def load_femnist(data_root: str, max_users: int | None = None, seed: int = 0) -> FemnistSplit:
    root = Path(data_root).expanduser()
    train = _read_leaf_dir(root / "train")
    test = _read_leaf_dir(root / "test")
    users = sorted(set(train.keys()) & set(test.keys()))
    if max_users is not None and len(users) > max_users:
        rng = np.random.default_rng(seed)
        users = list(rng.choice(users, size=max_users, replace=False))
        users.sort()
    train = {u: train[u] for u in users}
    test = {u: test[u] for u in users}
    return FemnistSplit(train_per_user=train, test_per_user=test, users=users)


def make_femnist_client_loader(split: FemnistSplit, user: str, batch_size: int) -> DataLoader:
    x, y = split.train_per_user[user]
    ds = _FemnistTensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


def make_femnist_test_loader(split: FemnistSplit, batch_size: int = 256) -> DataLoader:
    xs = np.concatenate([x for x, _ in split.test_per_user.values()], axis=0)
    ys = np.concatenate([y for _, y in split.test_per_user.values()], axis=0)
    ds = _FemnistTensorDataset(xs, ys)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
