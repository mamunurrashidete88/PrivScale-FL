"""Canary insertion for language client datasets.

Inserts each canary's full text as additional training "posts" into a small
randomly-chosen subset of clients, repeating each canary `repeats` times within
the chosen client's data. The canaries themselves are returned so the
extraction evaluator can score them after training.

Design notes:
  - We keep canary insertion separate from the main loader so the dataset on
    disk is not modified.
  - Each canary is assigned to one specific client (sampled uniformly); the
    repeats are inside that client's local stream. This matches the standard
    secret-sharer setup (rare strings appearing a small known number of times).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from privscale.attacks import Canary, make_canaries


@dataclass
class CanaryPlan:
    canaries: list[Canary]
    client_assignment: dict[int, list[int]]  # client_id -> list of canary indices
    repeats: int


def plan_canaries(
    num_canaries: int,
    num_clients: int,
    repeats: int = 3,
    seed: int = 0,
    suffix_len: int = 6,
) -> CanaryPlan:
    canaries = make_canaries(num_canaries, suffix_len=suffix_len, seed=seed)
    rng = random.Random(seed + 1)
    assignment: dict[int, list[int]] = {}
    for i in range(num_canaries):
        cid = rng.randrange(num_clients)
        assignment.setdefault(cid, []).append(i)
    return CanaryPlan(canaries=canaries, client_assignment=assignment, repeats=repeats)


def inject_into_text_list(posts: list[str], canary_indices: list[int],
                          plan: CanaryPlan) -> list[str]:
    """Append `repeats` copies of each assigned canary to a client's posts."""
    out = list(posts)
    for ci in canary_indices:
        text = plan.canaries[ci].full
        for _ in range(plan.repeats):
            out.append(text)
    return out
