"""
utils/visualization.py
=======================
Plotting utilities for generating all paper figures.

Figures generated:
  Fig 3: Spectral validation (r̂(d), λ̄(d), tr(Â(d)))
  Fig 4: Privacy risk scaling under fixed noise (MIA AUC, extraction, proxy)
  Fig 5: Scaling law fit comparison + privacy debt curves
  Fig 6: Mechanism comparison (MIA AUC, utility, privacy-utility trade-off)

Usage:
    python utils/visualization.py --results-dir results/ --output-dir figures/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# Paper-quality plot settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MECHANISM_LABELS = {
    "nonprivate": "Non-private",
    "fixed_noise": "Fixed-noise DP",
    "fixed_dp": "Fixed-noise DP",
    "adaptive_clip": "Adaptive-clip DP",
    "adaptive": "Adaptive-clip DP",
    "scale_aware": "Scale-aware DP",
    "scaleaware": "Scale-aware DP",
}
MECHANISM_COLORS = {
    "nonprivate": "#e74c3c",
    "fixed_noise": "#3498db",
    "fixed_dp": "#3498db",
    "adaptive_clip": "#f39c12",
    "adaptive": "#f39c12",
    "scale_aware": "#2ecc71",
    "scaleaware": "#2ecc71",
}
MECHANISM_MARKERS = {
    "nonprivate": "X",
    "fixed_noise": "o",
    "fixed_dp": "o",
    "adaptive_clip": "s",
    "adaptive": "s",
    "scale_aware": "D",
    "scaleaware": "D",
}


def _label(mech: str) -> str:
    return MECHANISM_LABELS.get(mech, mech)


def _color(mech: str) -> str:
    return MECHANISM_COLORS.get(mech, "#95a5a6")


def _marker(mech: str) -> str:
    return MECHANISM_MARKERS.get(mech, "^")


def plot_mia_vs_dim(df: pd.DataFrame, out_dir: Path, suffix: str = "") -> None:
    """Fig 4a: MIA AUC vs effective dimension, per mechanism."""
    if "mia_auc" not in df.columns or df["mia_auc"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for mech, sub in df.groupby("mechanism"):
        sub = sub.dropna(subset=["mia_auc"])
        if len(sub) < 2:
            continue
        grp = sub.groupby("dim")["mia_auc"].agg(["mean", "std"]).reset_index()
        ax.errorbar(grp["dim"], grp["mean"], yerr=grp["std"].fillna(0),
                    marker=_marker(mech), color=_color(mech),
                    capsize=3, label=_label(mech))
    ax.set_xlabel("Effective dimension $d$")
    ax.set_ylabel("MIA AUC")
    ax.legend()
    ax.set_xscale("log")
    fig.savefig(out_dir / f"mia_vs_dim{suffix}.png")
    fig.savefig(out_dir / f"mia_vs_dim{suffix}.pdf")
    plt.close(fig)


def plot_utility_vs_dim(df: pd.DataFrame, out_dir: Path, suffix: str = "") -> None:
    """Fig 6c: Utility vs effective dimension, per mechanism."""
    util_col = next((c for c in ("final_test_acc", "final_test_ppl", "final_test_loss")
                     if c in df.columns and df[c].notna().any()), None)
    if util_col is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for mech, sub in df.groupby("mechanism"):
        sub = sub.dropna(subset=[util_col])
        if len(sub) < 2:
            continue
        grp = sub.groupby("dim")[util_col].agg(["mean", "std"]).reset_index()
        ax.errorbar(grp["dim"], grp["mean"], yerr=grp["std"].fillna(0),
                    marker=_marker(mech), color=_color(mech),
                    capsize=3, label=_label(mech))
    ax.set_xlabel("Effective dimension $d$")
    ax.set_ylabel(util_col.replace("final_", "").replace("_", " ").title())
    ax.legend()
    ax.set_xscale("log")
    fig.savefig(out_dir / f"utility_vs_dim{suffix}.png")
    fig.savefig(out_dir / f"utility_vs_dim{suffix}.pdf")
    plt.close(fig)


def plot_scaling_law_fits(df: pd.DataFrame, out_dir: Path, suffix: str = "") -> None:
    """Fig 5: MIA AUC vs d with constant / linear / d/log d overlays."""
    from privscale.analysis.fitting import fit_candidates, select_best

    if "mia_auc" not in df.columns:
        return
    # Use fixed-noise mechanism for the fit comparison
    sub = df[df["mechanism"].isin(["fixed_noise", "fixed_dp"])].dropna(subset=["mia_auc"])
    if len(sub) < 3:
        return
    grp = sub.groupby("dim")["mia_auc"].mean().reset_index()
    d = grp["dim"].values
    y = grp["mia_auc"].values
    fits = fit_candidates(d, y)
    best = select_best(fits)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(d, y, color="#3498db", s=60, zorder=5, label="Observed (fixed-noise DP)")
    d_fine = np.linspace(d.min(), d.max(), 200)
    for name, fr in fits.items():
        style = "-" if name == best.name else "--"
        alpha = 1.0 if name == best.name else 0.4
        log_d = np.log(np.maximum(d_fine, 2.0))
        if name == "constant":
            pred = np.full_like(d_fine, fr.coefs[0])
        elif name == "linear":
            pred = fr.coefs[0] + fr.coefs[1] * d_fine
        else:
            pred = fr.coefs[0] + fr.coefs[1] * d_fine / log_d
        label = f"{name} ($R^2$={fr.r2:.3f})"
        ax.plot(d_fine, pred, style, alpha=alpha, label=label)
    ax.set_xlabel("Effective dimension $d$")
    ax.set_ylabel("MIA AUC")
    ax.legend(fontsize=8)
    ax.set_xscale("log")
    fig.savefig(out_dir / f"scaling_law_fits{suffix}.png")
    fig.savefig(out_dir / f"scaling_law_fits{suffix}.pdf")
    plt.close(fig)

    # Save fit summary as JSON
    fit_summary = {name: {"r2": f.r2, "aic": f.aic, "coefs": f.coefs.tolist()}
                   for name, f in fits.items()}
    fit_summary["selected"] = best.name
    with open(out_dir / f"fits{suffix}.json", "w") as fh:
        json.dump(fit_summary, fh, indent=2)


def plot_mechanism_comparison(df: pd.DataFrame, out_dir: Path, suffix: str = "") -> None:
    """Fig 6: Side-by-side bar chart of MIA AUC across mechanisms and scales."""
    if "mia_auc" not in df.columns:
        return
    sub = df.dropna(subset=["mia_auc"])
    if len(sub) < 2:
        return
    models = sorted(sub["model"].dropna().unique())
    mechs = sorted(sub["mechanism"].dropna().unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(models))
    width = 0.8 / max(1, len(mechs))
    for i, mech in enumerate(mechs):
        means, stds = [], []
        for m in models:
            vals = sub[(sub["model"] == m) & (sub["mechanism"] == mech)]["mia_auc"]
            means.append(vals.mean() if len(vals) else 0)
            stds.append(vals.std() if len(vals) > 1 else 0)
        ax.bar(x + i * width, means, width, yerr=stds, label=_label(mech),
               color=_color(mech), capsize=2)
    ax.set_xticks(x + width * (len(mechs) - 1) / 2)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("MIA AUC")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / f"mechanism_comparison{suffix}.png")
    fig.savefig(out_dir / f"mechanism_comparison{suffix}.pdf")
    plt.close(fig)


def plot_cross_scale_drift(df: pd.DataFrame, out_dir: Path, suffix: str = "") -> None:
    """Fig 6d: Cross-scale privacy drift (max MIA AUC − min MIA AUC per mechanism)."""
    if "mia_auc" not in df.columns:
        return
    sub = df.dropna(subset=["mia_auc"])
    mechs = sorted(sub["mechanism"].dropna().unique())
    drifts, labels = [], []
    for mech in mechs:
        ms = sub[sub["mechanism"] == mech]
        grp = ms.groupby("dim")["mia_auc"].mean()
        if len(grp) >= 2:
            drifts.append(grp.max() - grp.min())
            labels.append(_label(mech))
    if not drifts:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = [_color(m) for m in mechs if _label(m) in labels]
    ax.bar(labels, drifts, color=colors)
    ax.set_ylabel("Cross-scale MIA AUC drift")
    ax.set_xticklabels(labels, rotation=15, ha="right")
    fig.savefig(out_dir / f"cross_scale_drift{suffix}.png")
    fig.savefig(out_dir / f"cross_scale_drift{suffix}.pdf")
    plt.close(fig)


def generate_all(results_dir: str, output_dir: str) -> None:
    """Generate all paper figures from aggregated results."""
    res = Path(results_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = res / "results.csv"
    if not csv_path.exists():
        print(f"No results.csv found in {res}. Run aggregate_results.py first.")
        return

    df = pd.read_csv(csv_path)
    if "d_tau" in df.columns:
        df["dim"] = df["d_tau"].fillna(df.get("trainable_parameters"))
    else:
        df["dim"] = df.get("trainable_parameters")
    df = df.dropna(subset=["dim"])

    # Split by modality if present
    modalities = df["modality"].dropna().unique() if "modality" in df.columns else ["all"]
    for mod in modalities:
        sub = df[df["modality"] == mod] if mod != "all" else df
        sfx = f"_{mod}" if mod != "all" else ""
        plot_mia_vs_dim(sub, out, suffix=sfx)
        plot_utility_vs_dim(sub, out, suffix=sfx)
        plot_scaling_law_fits(sub, out, suffix=sfx)
        plot_mechanism_comparison(sub, out, suffix=sfx)
        plot_cross_scale_drift(sub, out, suffix=sfx)

    print(f"Figures written to {out}/")


def main():
    p = argparse.ArgumentParser(description="Generate paper figures.")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    generate_all(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
