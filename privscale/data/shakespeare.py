"""Shakespeare federated dataset (LEAF split, next-character prediction).

LEAF processes the complete works of Shakespeare into per-speaker sequences.
Each "user" is a speaking role. We expose a character-level next-token
prediction dataset compatible with the FL simulator, and optionally a
word-level dataset for GPT-2 fine-tuning.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# LEAF Shakespeare character vocabulary
ALL_CHARS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)
CHAR_TO_IDX = {c: i for i, c in enumerate(ALL_CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(ALL_CHARS)}
VOCAB_SIZE = len(ALL_CHARS)


def text_to_ids(text: str) -> np.ndarray:
    return np.array([CHAR_TO_IDX.get(c, 0) for c in text], dtype=np.int64)


@dataclass
class ShakespeareSplit:
    train_per_user: dict[str, np.ndarray]
    test_per_user: dict[str, np.ndarray]
    users: list[str]
    vocab_size: int = VOCAB_SIZE
    seq_len: int = 80


class _CharLMSlidingDataset(Dataset):
    """Sliding-window next-character prediction over a single user's text."""

    def __init__(self, ids: np.ndarray, seq_len: int = 80):
        self.seq_len = seq_len
        # Clip to allow (seq_len + 1)-length windows
        self.ids = ids
        self.n_samples = max(0, len(ids) - seq_len - 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        window = self.ids[i : i + self.seq_len + 1]
        x = torch.from_numpy(window[:-1]).long()
        y = torch.from_numpy(window[1:]).long()
        return x, y


def _read_leaf_shakespeare(dir_path: Path) -> dict[str, np.ndarray]:
    per_user: dict[str, list[str]] = {}
    for f in sorted(dir_path.glob("*.json")):
        with open(f) as fh:
            blob = json.load(fh)
        for uid in blob["users"]:
            ud = blob["user_data"][uid]
            # LEAF stores lists of text snippets under "x" and "y"; we
            # concatenate into one long string per user.
            text = "".join(ud.get("raw", ud.get("x", [])))
            per_user.setdefault(uid, []).append(text)
    return {uid: text_to_ids("".join(chunks)) for uid, chunks in per_user.items()}


def load_shakespeare(
    data_root: str,
    seq_len: int = 80,
    max_users: int | None = None,
    seed: int = 0,
) -> ShakespeareSplit:
    root = Path(data_root).expanduser()
    train = _read_leaf_shakespeare(root / "train")
    test = _read_leaf_shakespeare(root / "test")
    users = sorted(set(train) & set(test))
    # Filter out users whose train text is too short for even one window
    users = [u for u in users if len(train[u]) > seq_len + 1]
    if max_users is not None and len(users) > max_users:
        rng = np.random.default_rng(seed)
        users = sorted(rng.choice(users, size=max_users, replace=False))
    train = {u: train[u] for u in users}
    test = {u: test[u] for u in users}
    return ShakespeareSplit(
        train_per_user=train, test_per_user=test, users=users, seq_len=seq_len
    )


def make_shakespeare_client_loader(
    split: ShakespeareSplit, user: str, batch_size: int
) -> DataLoader:
    ds = _CharLMSlidingDataset(split.train_per_user[user], seq_len=split.seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


def make_shakespeare_test_loader(
    split: ShakespeareSplit, batch_size: int = 256, max_tokens: int = 200_000
) -> DataLoader:
    # Pool all test sequences into a single evaluation stream (capped for speed)
    chunks = []
    total = 0
    for u in split.users:
        ids = split.test_per_user[u]
        if total + len(ids) > max_tokens:
            ids = ids[: max_tokens - total]
        chunks.append(ids)
        total += len(ids)
        if total >= max_tokens:
            break
    ids = np.concatenate(chunks) if chunks else np.zeros(split.seq_len + 2, dtype=np.int64)
    ds = _CharLMSlidingDataset(ids, seq_len=split.seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
