"""
evaluation/metrics.py
=====================
Metric definitions and tracking for the PrivScale-FL evaluation suite.

Tracked metrics (Section 4 of the paper):
  (i)   Test accuracy / perplexity — utility under each mechanism
  (ii)  MIA AUC                    — threshold-independent membership inference
  (iii) MIA advantage              — max |TPR − FPR| over thresholds
  (iv)  Extraction rate            — canary recovery rate (language only)
  (v)   Effective dimension d      — numerical rank of update covariance
  (vi)  Privacy debt               — ε_eff(d) / ε_target ratio
  (vii) Spectral diagnostics       — r̂(d), λ̄(d), tr(Â(d))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PrivacyMetrics:
    """Container for privacy evaluation results at a single scale point."""
    model_name: str = ""
    mechanism: str = ""
    effective_dim: int = 0
    trainable_params: int = 0

    # Utility
    test_acc: float | None = None
    test_ppl: float | None = None
    test_loss: float = 0.0

    # MIA
    mia_auc: float = 0.5
    mia_advantage: float = 0.0
    mia_n_members: int = 0
    mia_n_nonmembers: int = 0

    # Extraction (language only)
    extraction_rate: float = 0.0
    n_canaries: int = 0
    exposure_mean: float = 0.0

    # Spectral
    spectral_trace: float = 0.0
    participation_ratio: float = 0.0
    mean_informative_eigenvalue: float = 0.0

    # Privacy debt
    noise_multiplier: float = 0.0
    reference_sigma: float = 0.0
    reference_dim: int = 0

    def privacy_debt_ratio(self) -> float:
        """Compute ε_eff(d) / ε_target from Theorem 2."""
        d = max(2, self.effective_dim)
        d0 = max(2, self.reference_dim)
        if d0 == 0 or d == 0:
            return 1.0
        return math.sqrt((d * math.log(d0)) / (d0 * math.log(d)))

    def to_dict(self) -> dict[str, Any]:
        out = {
            "model": self.model_name,
            "mechanism": self.mechanism,
            "effective_dim": self.effective_dim,
            "trainable_params": self.trainable_params,
            "test_acc": self.test_acc,
            "test_ppl": self.test_ppl,
            "test_loss": self.test_loss,
            "mia_auc": self.mia_auc,
            "mia_advantage": self.mia_advantage,
            "extraction_rate": self.extraction_rate,
            "spectral_trace": self.spectral_trace,
            "participation_ratio": self.participation_ratio,
            "privacy_debt_ratio": self.privacy_debt_ratio(),
        }
        return {k: v for k, v in out.items() if v is not None}


@dataclass
class LadderResults:
    """Collects PrivacyMetrics across a model-scale ladder for one mechanism."""
    mechanism: str = ""
    points: list[PrivacyMetrics] = field(default_factory=list)

    def dims(self) -> np.ndarray:
        return np.array([p.effective_dim for p in self.points], dtype=float)

    def mia_aucs(self) -> np.ndarray:
        return np.array([p.mia_auc for p in self.points], dtype=float)

    def debt_ratios(self) -> np.ndarray:
        return np.array([p.privacy_debt_ratio() for p in self.points], dtype=float)

    def cross_scale_drift(self) -> float:
        """max MIA AUC − min MIA AUC across the ladder."""
        aucs = self.mia_aucs()
        return float(aucs.max() - aucs.min()) if len(aucs) >= 2 else 0.0
