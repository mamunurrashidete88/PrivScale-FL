"""
evaluation/evaluator.py
========================
Evaluation orchestrator that loads run outputs and computes aggregate metrics.

Takes a `runs/` directory tree and produces PrivacyMetrics for each run,
LadderResults for each mechanism, and summary statistics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import LadderResults, PrivacyMetrics


def load_run_metrics(run_dir: Path) -> PrivacyMetrics | None:
    """Load a single run's outputs into a PrivacyMetrics object."""
    cfg_path = run_dir / "config.json"
    sum_path = run_dir / "summary.json"
    if not cfg_path.exists() or not sum_path.exists():
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    with open(sum_path) as f:
        summary = json.load(f)

    pm = PrivacyMetrics(
        model_name=cfg.get("model", {}).get("name", ""),
        mechanism=cfg.get("mechanism", {}).get("name", ""),
        trainable_params=summary.get("trainable_parameters", 0),
        noise_multiplier=cfg.get("mechanism", {}).get("noise_multiplier", 0),
        reference_sigma=cfg.get("mechanism", {}).get("reference_sigma", 0),
        reference_dim=cfg.get("mechanism", {}).get("reference_dim", 0),
    )

    # Utility
    ev = summary.get("final_eval", {})
    pm.test_acc = ev.get("test_acc")
    pm.test_ppl = ev.get("test_ppl")
    pm.test_loss = ev.get("test_loss", 0.0)

    # MIA
    mia = summary.get("mia", {})
    pm.mia_auc = mia.get("auc", 0.5)
    pm.mia_advantage = mia.get("advantage", 0.0)
    pm.mia_n_members = mia.get("n_members", 0)
    pm.mia_n_nonmembers = mia.get("n_nonmembers", 0)

    # Extraction
    ext = summary.get("extraction", {})
    pm.extraction_rate = ext.get("extraction_rate", 0.0)
    pm.n_canaries = ext.get("n_canaries", 0)
    pm.exposure_mean = ext.get("exposure_mean", 0.0)

    # Spectral
    spec_path = run_dir / "spectral.json"
    if spec_path.exists():
        with open(spec_path) as f:
            spec = json.load(f)
        pm.effective_dim = spec.get("d_tau", 0)
        pm.spectral_trace = spec.get("trace", 0.0)
        pm.participation_ratio = spec.get("participation_ratio", 0.0)
        eigs = spec.get("top_eigs", [])
        if eigs and pm.participation_ratio > 0:
            pm.mean_informative_eigenvalue = pm.spectral_trace / max(1, pm.participation_ratio)

    return pm


def load_ladder(runs_dir: Path, mechanism: str | None = None) -> list[PrivacyMetrics]:
    """Load all runs under a directory, optionally filtering by mechanism."""
    results = []
    for cfg_path in sorted(runs_dir.rglob("config.json")):
        pm = load_run_metrics(cfg_path.parent)
        if pm is None:
            continue
        if mechanism is not None and pm.mechanism != mechanism:
            continue
        results.append(pm)
    return results


def build_ladders(runs_dir: Path) -> dict[str, LadderResults]:
    """Group all runs by mechanism into LadderResults."""
    all_runs = load_ladder(Path(runs_dir))
    ladders: dict[str, LadderResults] = {}
    for pm in all_runs:
        key = pm.mechanism
        if key not in ladders:
            ladders[key] = LadderResults(mechanism=key)
        ladders[key].points.append(pm)
    # Sort each ladder by effective dimension
    for lr in ladders.values():
        lr.points.sort(key=lambda p: p.effective_dim)
    return ladders


def print_summary(ladders: dict[str, LadderResults]) -> None:
    """Print a readable summary of all ladders."""
    for mech, lr in sorted(ladders.items()):
        print(f"\n{'='*60}")
        print(f"Mechanism: {mech}  ({len(lr.points)} scale points)")
        print(f"{'='*60}")
        print(f"  {'Model':<20s} {'d':>6s} {'MIA AUC':>9s} {'Debt':>8s} {'Util':>8s}")
        print(f"  {'-'*55}")
        for p in lr.points:
            util = f"{p.test_acc:.1f}" if p.test_acc is not None else (
                   f"{p.test_ppl:.1f}" if p.test_ppl is not None else "—")
            print(f"  {p.model_name:<20s} {p.effective_dim:6d} {p.mia_auc:9.4f} "
                  f"{p.privacy_debt_ratio():8.3f} {util:>8s}")
        print(f"  Cross-scale drift: {lr.cross_scale_drift():.4f}")
