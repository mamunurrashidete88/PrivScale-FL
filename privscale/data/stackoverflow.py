"""Stack Overflow federated text subset.

We use a per-user shard layout. Expected preprocessing produces:

    data_root/
        users.txt                # one user id per line (the subset)
        train/{user_id}.txt      # plain text, one post per line
        test/{user_id}.txt

The preprocessing script (scripts/prepare_stackoverflow.py, not included here)
pulls from the TFF Stack Overflow federated split or a HuggingFace mirror,
filters to the top-N users by example count, and writes the layout above.

This keeps the loader simple and avoids a hard dependency on TFF at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class StackOverflowSplit:
    root: Path
    users: list[str]
    tokenizer: object  # transformers.PreTrainedTokenizerBase
    seq_len: int = 128


class _LMTokenizedTextDataset(Dataset):
    """Tokenize a list of strings on init, then expose fixed-length LM windows."""

    def __init__(self, texts: list[str], tokenizer, seq_len: int = 128):
        joined = tokenizer.eos_token.join(t.strip() for t in texts if t.strip())
        enc = tokenizer(joined, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"][0]
        self.seq_len = seq_len
        # Drop the tail that doesn't fill a full window
        n_windows = ids.numel() // seq_len
        self.ids = ids[: n_windows * seq_len].view(n_windows, seq_len)

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, i):
        x = self.ids[i]
        # Causal LM: labels = inputs (shift handled inside the model wrapper)
        return x, x.clone()


def _read_user_file(path: Path, limit: int | None = None) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if limit is not None:
        lines = lines[:limit]
    return lines


def load_stackoverflow(
    data_root: str,
    tokenizer,
    seq_len: int = 128,
    max_users: int | None = None,
) -> StackOverflowSplit:
    root = Path(data_root).expanduser()
    users_file = root / "users.txt"
    if not users_file.exists():
        raise FileNotFoundError(
            f"Expected users.txt under {root}. Run scripts/prepare_stackoverflow.py first."
        )
    users = [u.strip() for u in users_file.read_text().splitlines() if u.strip()]
    if max_users is not None:
        users = users[:max_users]
    return StackOverflowSplit(root=root, users=users, tokenizer=tokenizer, seq_len=seq_len)


def make_stackoverflow_client_loader(
    split: StackOverflowSplit, user: str, batch_size: int, max_posts: int = 500
) -> DataLoader:
    posts = _read_user_file(split.root / "train" / f"{user}.txt", limit=max_posts)
    if not posts:
        # Empty client loader: return an empty iterator but keep shape
        empty = _LMTokenizedTextDataset([split.tokenizer.eos_token], split.tokenizer, split.seq_len)
        return DataLoader(empty, batch_size=batch_size, shuffle=False)
    ds = _LMTokenizedTextDataset(posts, split.tokenizer, split.seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


def make_stackoverflow_test_loader(
    split: StackOverflowSplit, batch_size: int = 8, max_posts_per_user: int = 20
) -> DataLoader:
    all_posts = []
    for u in split.users:
        all_posts.extend(_read_user_file(split.root / "test" / f"{u}.txt", limit=max_posts_per_user))
    if not all_posts:
        all_posts = [split.tokenizer.eos_token]
    ds = _LMTokenizedTextDataset(all_posts, split.tokenizer, split.seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
