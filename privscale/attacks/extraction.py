"""Canary-based extraction for language models.

Following Carlini et al. (2019) "The Secret Sharer": we insert unique,
low-frequency canary sequences into a small subset of clients' training data,
train as usual, and then measure whether the trained model can complete the
canary given a short prefix more often than baseline. We also report the
exposure metric:
    exposure(c) = log2(|R|) - log2(rank(c))
where |R| is the size of the random candidate set and rank(c) is the rank of
the canary by model likelihood.

Because canaries must be inserted at training time, this module provides:
  - `make_canaries`: deterministic canary generator (prefix + random suffix
    sampled from a known pattern);
  - `extraction_rate`: given trained model + canaries + prefix length,
    measure greedy-decode recovery rate.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Canary:
    prefix: str
    suffix: str
    full: str


def make_canaries(n: int, prefix_template: str = "The secret key is",
                  suffix_len: int = 6, seed: int = 0) -> list[Canary]:
    """Create `n` canaries with a fixed prefix and random uppercase-letter suffix."""
    rng = random.Random(seed)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out = []
    for _ in range(n):
        suffix = " ".join(rng.choice(alphabet) for _ in range(suffix_len))
        full = f"{prefix_template} {suffix}"
        out.append(Canary(prefix=prefix_template, suffix=suffix, full=full))
    return out


@torch.no_grad()
def greedy_complete(model, tokenizer, prefix: str, max_new_tokens: int,
                    device: torch.device) -> str:
    ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    out = model.model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def extraction_rate(
    model,
    tokenizer,
    canaries: list[Canary],
    device: torch.device,
    max_new_tokens: int = 24,
) -> dict:
    """Fraction of canaries whose suffix is reproduced verbatim by greedy decoding."""
    recovered = 0
    for c in canaries:
        completion = greedy_complete(
            model, tokenizer, c.prefix, max_new_tokens=max_new_tokens, device=device
        )
        if c.suffix.strip() in completion:
            recovered += 1
    rate = recovered / max(1, len(canaries))
    return {"extraction_rate": rate, "n_recovered": recovered, "n_canaries": len(canaries)}


@torch.no_grad()
def canary_exposure(
    model,
    tokenizer,
    canary: Canary,
    candidate_set_size: int,
    device: torch.device,
    seed: int = 0,
) -> float:
    """Exposure metric (Carlini et al. 2019).

    We sample `candidate_set_size - 1` random suffixes from the same alphabet,
    score all of them under the model, and compute exposure from the rank of
    the true canary.
    """
    rng = random.Random(seed)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    suffix_len = len(canary.suffix.split())

    candidates = [canary.suffix]
    for _ in range(candidate_set_size - 1):
        candidates.append(" ".join(rng.choice(alphabet) for _ in range(suffix_len)))

    losses = []
    for s in candidates:
        text = f"{canary.prefix} {s}"
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        out = model(input_ids=ids, labels=ids)
        losses.append(out.loss.item())

    losses = np.array(losses)
    order = np.argsort(losses)  # ascending loss
    rank = int(np.where(order == 0)[0][0]) + 1  # 1-indexed
    return math.log2(candidate_set_size) - math.log2(rank)
