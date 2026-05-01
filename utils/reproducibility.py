"""
utils/reproducibility.py
=========================
Seed management and environment reproducibility utilities.
"""
from __future__ import annotations

import os
import platform
import random
from typing import Any

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN is too slow for WRN/GPT-2; we accept minor
        # nondeterminism and rely on seed-averaged reporting.
        torch.backends.cudnn.benchmark = True


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_environment() -> dict[str, Any]:
    """Print and return system/package version info for reproducibility logging."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
    }
    for k, v in info.items():
        print(f"  {k:16s}: {v}")
    return info
