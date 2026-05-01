"""Generate plots from aggregated results.

Reads `results.csv` and produces:
  - mia_vs_dim.png      : MIA AUC vs effective dimension, per mechanism
  - acc_vs_dim.png      : utility vs effective dimension, per mechanism
  - debt_vs_dim.png     : empirical effective-epsilon proxy vs effective dim
                          (computed only if 'd_tau' and 'mia_auc' are present)
  - fits.json           : fit summaries from privscale.analysis.fitting
                          for MIA AUC vs d, evaluated per mechanism

This script does not assume any particular trend is correct; it fits all three
candidate families on equal footing and reports them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privscale.analysis.fitting import fit_candidates, select_best


def _scatter_with_fit(ax, x, y, label):
    ax.scatter(x, y, label=label, s=40, alpha=0.7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Directory containing results.csv")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    res_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(res_dir / "results.csv")
    if "d_tau" not in df.columns:
        df["d_tau"] = df.get("trainable_parameters")
    df["dim"] = df["d_tau"].fillna(df["trainable_parameters"])
    df = df.dropna(subset=["dim"])

    # MIA AUC vs dim, by mechanism
    if "mia_auc" in df.columns and df["mia_auc"].notna().any():
        fig, ax = plt.subplots(figsize=(6, 4.5))
        all_fits = {}
        for mech, sub in df.groupby("mechanism"):
            sub = sub.dropna(subset=["mia_auc"])
            if len(sub) < 2:
                continue
            grp = sub.groupby("dim")["mia_auc"].agg(["mean", "std", "count"]).reset_index()
            ax.errorbar(grp["dim"], grp["mean"],
                        yerr=grp["std"].fillna(0.0),
                        marker="o", capsize=3, label=str(mech))
            if len(grp) >= 3:
                fits = fit_candidates(grp["dim"].values, grp["mean"].values)
                best = select_best(fits)
                all_fits[mech] = {
                    name: {"r2": f.r2, "aic": f.aic, "coefs": f.coefs.tolist()}
                    for name, f in fits.items()
                } | {"selected": best.name}
        ax.set_xlabel("Effective dimension d")
        ax.set_ylabel("MIA AUC")
        ax.set_title("MIA AUC vs effective dimension")
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(out_dir / "mia_vs_dim.png", dpi=150)
        plt.close(fig)

        with open(out_dir / "fits.json", "w") as f:
            json.dump(all_fits, f, indent=2)

    # Utility vs dim
    util_col = None
    for cand in ("final_test_acc", "final_test_ppl", "final_test_loss"):
        if cand in df.columns and df[cand].notna().any():
            util_col = cand
            break
    if util_col is not None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for mech, sub in df.groupby("mechanism"):
            sub = sub.dropna(subset=[util_col])
            if len(sub) < 2:
                continue
            grp = sub.groupby("dim")[util_col].agg(["mean", "std"]).reset_index()
            ax.errorbar(grp["dim"], grp["mean"],
                        yerr=grp["std"].fillna(0.0),
                        marker="s", capsize=3, label=str(mech))
        ax.set_xlabel("Effective dimension d")
        ax.set_ylabel(util_col)
        ax.set_title(f"{util_col} vs effective dimension")
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(out_dir / "utility_vs_dim.png", dpi=150)
        plt.close(fig)

    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
