"""Aggregate per-run results into a single CSV.

Walks `runs_dir` recursively, finds `summary.json` + `config.json` + optional
`spectral.json`, and produces a tidy DataFrame with one row per run. The CSV
is written to `out/results.csv` and the pivoted-by-mechanism CSV to
`out/by_mechanism.csv`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_run(run_dir: Path) -> dict | None:
    cfg_path = run_dir / "config.json"
    sum_path = run_dir / "summary.json"
    if not cfg_path.exists() or not sum_path.exists():
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    with open(sum_path) as f:
        summary = json.load(f)
    record = {
        "run_dir": str(run_dir),
        "run_name": cfg.get("run_name", run_dir.name),
        "seed": cfg.get("seed"),
        "modality": cfg.get("modality"),
        "dataset": cfg.get("dataset", {}).get("name"),
        "alpha": cfg.get("dataset", {}).get("alpha"),
        "partition": cfg.get("dataset", {}).get("partition"),
        "num_clients": cfg.get("dataset", {}).get("num_clients"),
        "model": cfg.get("model", {}).get("name"),
        "mechanism": cfg.get("mechanism", {}).get("name"),
        "noise_multiplier": cfg.get("mechanism", {}).get("noise_multiplier"),
        "clip_norm": cfg.get("mechanism", {}).get("clip_norm")
            or cfg.get("mechanism", {}).get("init_clip_norm"),
        "rounds": cfg.get("server", {}).get("rounds"),
        "trainable_parameters": summary.get("trainable_parameters"),
        "final_test_acc": summary.get("final_eval", {}).get("test_acc"),
        "final_test_loss": summary.get("final_eval", {}).get("test_loss"),
        "final_test_ppl": summary.get("final_eval", {}).get("test_ppl"),
        "mia_auc": summary.get("mia", {}).get("auc"),
        "mia_advantage": summary.get("mia", {}).get("advantage"),
        "extraction_rate": summary.get("extraction", {}).get("extraction_rate"),
    }
    spec_path = run_dir / "spectral.json"
    if spec_path.exists():
        with open(spec_path) as f:
            spec = json.load(f)
        record["d_tau"] = spec.get("d_tau")
        record["spectral_trace"] = spec.get("trace")
        record["participation_ratio"] = spec.get("participation_ratio")
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for cfg_path in runs_dir.rglob("config.json"):
        rec = load_run(cfg_path.parent)
        if rec is not None:
            records.append(rec)

    if not records:
        print(f"No completed runs found under {runs_dir}")
        return

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["dataset", "model", "mechanism", "seed"])
    df.to_csv(out_dir / "results.csv", index=False)
    print(f"Wrote {len(df)} run records to {out_dir / 'results.csv'}")

    # Aggregate over seeds
    group_cols = ["dataset", "alpha", "partition", "model", "mechanism"]
    metric_cols = [c for c in (
        "final_test_acc", "final_test_loss", "final_test_ppl",
        "mia_auc", "mia_advantage", "extraction_rate", "d_tau",
        "spectral_trace", "trainable_parameters",
    ) if c in df.columns]
    agg = (df.groupby(group_cols, dropna=False)[metric_cols]
             .agg(["mean", "std", "count"])
             .reset_index())
    agg.columns = [
        ("__".join(str(c) for c in col if c) if isinstance(col, tuple) else str(col)).rstrip("_")
        for col in agg.columns.values
    ]
    agg.to_csv(out_dir / "by_mechanism.csv", index=False)
    print(f"Wrote per-(dataset, model, mechanism) aggregates to {out_dir / 'by_mechanism.csv'}")


if __name__ == "__main__":
    main()
