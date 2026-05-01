"""GPT-2 pipeline construction for run_experiment.py.

Kept in a separate module to keep the main runner small and to make the
language-side wiring easy to test independently.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from privscale.attacks import CanaryPlan, plan_canaries, inject_into_text_list
from privscale.data.stackoverflow import (
    _LMTokenizedTextDataset, _read_user_file, load_stackoverflow,
    make_stackoverflow_test_loader,
)
from privscale.fl import eval_gpt2, get_trainer
from privscale.models import build_gpt2


def build_gpt2_pipeline(cfg: dict, device: torch.device, log) -> tuple:
    ds_cfg = cfg["dataset"]
    mcfg = cfg["model"]
    model = build_gpt2(
        variant=mcfg["name"],
        use_lora=mcfg.get("use_lora", False),
        lora_r=mcfg.get("lora_r", 8),
        lora_alpha=mcfg.get("lora_alpha", 16),
        lora_dropout=mcfg.get("lora_dropout", 0.05),
    ).to(device)
    log.info("Built %s (use_lora=%s)", mcfg["name"], mcfg.get("use_lora", False))

    split = load_stackoverflow(
        data_root=ds_cfg["data_root"],
        tokenizer=model.tokenizer,
        seq_len=ds_cfg.get("seq_len", 128),
        max_users=ds_cfg.get("max_users"),
    )
    log.info("Loaded Stack Overflow with %d users", len(split.users))

    # Canary plan is built once per run (deterministic given seed). The plan
    # is returned so the runner can score extraction at the end.
    canary_cfg = cfg.get("attacks", {}).get("canary", {})
    plan: CanaryPlan | None = None
    if canary_cfg.get("enabled"):
        plan = plan_canaries(
            num_canaries=canary_cfg.get("num_canaries", 100),
            num_clients=len(split.users),
            repeats=canary_cfg.get("repeats", 4),
            seed=cfg["seed"],
            suffix_len=canary_cfg.get("suffix_len", 6),
        )
        log.info("Canary plan: %d canaries assigned to %d clients (repeats=%d)",
                 len(plan.canaries), len(plan.client_assignment), plan.repeats)

    # Per-client loader factory: read posts from disk, inject canaries assigned
    # to this client (if any), tokenize, build a windowed LM dataset.
    user_ids = split.users
    bs = cfg["local"]["batch_size"]
    max_posts = ds_cfg.get("max_posts_per_client", 500)

    def client_loader_fn(cid: int) -> DataLoader:
        user = user_ids[cid]
        posts = _read_user_file(split.root / "train" / f"{user}.txt", limit=max_posts)
        if plan is not None and cid in plan.client_assignment:
            posts = inject_into_text_list(posts, plan.client_assignment[cid], plan)
        if not posts:
            empty = _LMTokenizedTextDataset(
                [model.tokenizer.eos_token], model.tokenizer, split.seq_len
            )
            return DataLoader(empty, batch_size=bs, shuffle=False)
        ds = _LMTokenizedTextDataset(posts, model.tokenizer, split.seq_len)
        return DataLoader(ds, batch_size=bs, shuffle=True)

    test_loader = make_stackoverflow_test_loader(split, batch_size=8)
    eval_fn = lambda m: eval_gpt2(m, test_loader, device)
    trainer = get_trainer("gpt2")

    return split, model, client_loader_fn, eval_fn, trainer, test_loader, plan
