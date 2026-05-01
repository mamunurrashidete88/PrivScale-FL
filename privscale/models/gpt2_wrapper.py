"""GPT-2 wrapper for federated causal LM with optional LoRA adaptation.

We expose only the trainable parameter set to the FL aggregator: for full
fine-tuning this is the whole model; for LoRA it is the adapter weights plus
any unfrozen heads. This keeps the privatization and aggregation logic
architecture-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


GPT2_VARIANTS = {
    "gpt2-small": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
}


@dataclass
class GPT2Config:
    variant: str = "gpt2-small"
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = ("c_attn",)


class GPT2LM(nn.Module):
    """Wraps GPT2LMHeadModel and exposes a uniform forward/loss API."""

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        hf_name = GPT2_VARIANTS[cfg.variant]
        self.cfg = cfg
        self.model = GPT2LMHeadModel.from_pretrained(hf_name)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(hf_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if cfg.use_lora:
            self._apply_lora()

    def _apply_lora(self):
        from peft import LoraConfig, get_peft_model, TaskType

        lconf = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=list(self.cfg.lora_targets),
            bias="none",
        )
        self.model = get_peft_model(self.model, lconf)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        return self.model(input_ids=input_ids, labels=labels)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def trainable_named_parameters(self):
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]


def build_gpt2(variant: str, use_lora: bool = False, **kwargs) -> GPT2LM:
    cfg = GPT2Config(variant=variant, use_lora=use_lora, **kwargs)
    return GPT2LM(cfg)
