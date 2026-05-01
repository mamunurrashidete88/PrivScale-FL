"""Character-level LSTM used for Shakespeare (LEAF benchmark baseline)."""
from __future__ import annotations

import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 8, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] int64
        h = self.embed(x)
        h, _ = self.lstm(h)
        return self.fc(h)  # [B, T, V]
