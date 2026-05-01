"""Main experiment runner.

Usage:
    python scripts/run_experiment.py --config configs/vision/smoke_cifar10.yaml
    python scripts/run_experiment.py --config configs/vision/cifar100_resnet20_fixed.yaml \
        --seed 2 --output-dir runs/cifar100/resnet20/fixed_dp/seed2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

# Allow running without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privscale.utils import (
    JsonlLogger, ensure_dir, get_device, load_config, set_seed, setup_stdout_logger,
)
from privscale.data.cifar import load_cifar_federated, make_client_loader, make_test_loader
from privscale.data.femnist import load_femnist, make_femnist_client_loader, make_femnist_test_loader
from privscale.data.shakespeare import (
    load_shakespeare, make_shakespeare_client_loader, make_shakespeare_test_loader,
    VOCAB_SIZE as SHAKESPEARE_VOCAB,
)
from privscale.models import build_model
from privscale.fl import (
    FederatedServer, ServerConfig, LocalTrainConfig, get_trainer,
    eval_classification, eval_charlm, eval_gpt2, trainable_size, flat_params,
)
from privscale.privacy import build_mechanism
from privscale.analysis import UpdateTraceCollector, spectral_summary
from privscale.attacks import mia_vision, mia_gpt2, extraction_rate, canary_exposure


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run a single FL experiment.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None, help="Override config seed.")
    p.add_argument("--output-dir", default=None, help="Override config output_dir.")
    p.add_argument("--override-rounds", type=int, default=None,
                   help="Override server.rounds (useful for quick sanity checks).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline builders  (return: split, model, cl_fn, ev_fn, trainer, test_ld)
# ---------------------------------------------------------------------------

def build_vision_pipeline(cfg, device, log):
    ds = cfg["dataset"]
    split = load_cifar_federated(
        dataset_name=ds["name"], data_root=ds["data_root"],
        num_clients=ds["num_clients"], partition=ds.get("partition", "dirichlet"),
        alpha=ds.get("alpha", 0.1), classes_per_client=ds.get("classes_per_client", 2),
        seed=cfg["seed"], augment=ds.get("augment", True),
    )
    log.info("Loaded %s: %d clients, partition=%s, mean_size=%.0f",
             ds["name"], ds["num_clients"], ds.get("partition"),
             split.partition_info["size_mean"])
    model = build_model(cfg["model"]["name"], num_classes=split.num_classes,
                        dropout=cfg["model"].get("dropout", 0.3)).to(device)
    test_ld = make_test_loader(split, batch_size=256)
    cl_fn = lambda cid: make_client_loader(split, cid, batch_size=cfg["local"]["batch_size"])
    ev_fn = lambda m: eval_classification(m, test_ld, device)
    return split, model, cl_fn, ev_fn, get_trainer("vision"), test_ld


def build_femnist_pipeline(cfg, device, log):
    ds = cfg["dataset"]
    split = load_femnist(data_root=ds["data_root"], max_users=ds.get("max_users"),
                         seed=cfg["seed"])
    log.info("Loaded FEMNIST: %d users", len(split.users))
    model = build_model(cfg["model"]["name"], num_classes=split.num_classes).to(device)
    test_ld = make_femnist_test_loader(split, batch_size=256)
    users = split.users
    cl_fn = lambda cid: make_femnist_client_loader(
        split, users[cid], batch_size=cfg["local"]["batch_size"])
    ev_fn = lambda m: eval_classification(m, test_ld, device)
    return split, model, cl_fn, ev_fn, get_trainer("vision"), test_ld


def build_charlm_pipeline(cfg, device, log):
    ds = cfg["dataset"]
    split = load_shakespeare(data_root=ds["data_root"], seq_len=ds.get("seq_len", 80),
                             max_users=ds.get("max_users"), seed=cfg["seed"])
    log.info("Loaded Shakespeare: %d users", len(split.users))
    mc = cfg["model"]
    model = build_model(mc["name"], vocab_size=SHAKESPEARE_VOCAB,
                        embed_dim=mc.get("embed_dim", 8), hidden_dim=mc.get("hidden_dim", 256),
                        num_layers=mc.get("num_layers", 2)).to(device)
    test_ld = make_shakespeare_test_loader(split, batch_size=64)
    users = split.users
    cl_fn = lambda cid: make_shakespeare_client_loader(
        split, users[cid], batch_size=cfg["local"]["batch_size"])
    ev_fn = lambda m: eval_charlm(m, test_ld, device)
    return split, model, cl_fn, ev_fn, get_trainer("charlm"), test_ld


def build_pipeline(cfg, device, log):
    """Returns (split, model, cl_fn, ev_fn, trainer, test_ld, canary_plan)."""
    m = cfg["modality"]
    if m == "vision":
        return (*build_vision_pipeline(cfg, device, log), None)
    if m == "femnist":
        return (*build_femnist_pipeline(cfg, device, log), None)
    if m == "charlm":
        return (*build_charlm_pipeline(cfg, device, log), None)
    if m == "gpt2":
        from privscale.data.gpt2_pipeline import build_gpt2_pipeline
        return build_gpt2_pipeline(cfg, device, log)
    raise ValueError(f"Unknown modality: {m}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_num_clients(cfg, split):
    if "num_clients" in cfg["dataset"]:
        return cfg["dataset"]["num_clients"]
    if hasattr(split, "users"):
        return len(split.users)
    if hasattr(split, "client_indices"):
        return len(split.client_indices)
    raise ValueError("Cannot determine number of clients.")


def build_mia_loaders_vision(split, n_members, n_nonmembers, seed=0, batch_size=256):
    """Member/non-member loaders for MIA from a vision or femnist split."""
    rng = np.random.default_rng(seed + 17)
    all_idx = np.concatenate(split.client_indices)
    used = set(int(i) for i in all_idx.tolist())
    n_total = len(split.train_set)
    unused = np.array([i for i in range(n_total) if i not in used], dtype=np.int64)
    if len(unused) < n_nonmembers:
        rng.shuffle(all_idx)
        unused = all_idx[:n_nonmembers]
        all_idx = all_idx[n_nonmembers:]
    m_idx = rng.choice(all_idx, size=min(n_members, len(all_idx)), replace=False)
    nm_idx = rng.choice(unused, size=min(n_nonmembers, len(unused)), replace=False)
    m_ld = DataLoader(Subset(split.train_set, m_idx.tolist()),
                      batch_size=batch_size, shuffle=False)
    nm_ld = DataLoader(Subset(split.train_set, nm_idx.tolist()),
                       batch_size=batch_size, shuffle=False)
    return m_ld, nm_ld


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.override_rounds is not None:
        cfg["server"]["rounds"] = args.override_rounds

    out_dir = ensure_dir(cfg["output_dir"])
    set_seed(cfg["seed"])
    device = get_device(args.device)
    log = setup_stdout_logger("privscale.run")
    log.info("Run %s  seed=%d  device=%s  output=%s",
             cfg.get("run_name", ""), cfg["seed"], device, out_dir)

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ---- Build pipeline ----
    split, model, cl_fn, ev_fn, trainer, test_ld, canary_plan = build_pipeline(
        cfg, device, log)
    p_dim = trainable_size(model)
    log.info("Trainable parameters: %d", p_dim)

    # ---- Build mechanism ----
    K = _get_num_clients(cfg, split)
    q = cfg["server"]["sample_fraction"]
    mech_spec = dict(cfg["mechanism"])
    mech_spec.setdefault("expected_participants", q * K)
    if mech_spec["name"] == "scale_aware":
        if mech_spec.get("effective_dim", 0) <= 1:
            mech_spec["effective_dim"] = p_dim
        if mech_spec.get("reference_dim", 0) <= 1:
            mech_spec["reference_dim"] = p_dim
    mechanism = build_mechanism(mech_spec, rng_seed=cfg["seed"])
    log.info("Mechanism: %s  (expected_participants=%.1f)",
             mech_spec["name"], mech_spec.get("expected_participants", 0))

    # ---- Server ----
    scfg = ServerConfig(
        num_clients=K,
        sample_fraction=q,
        rounds=cfg["server"]["rounds"],
        sampling=cfg["server"].get("sampling", "uniform"),
        eval_every=cfg["server"]["eval_every"],
        save_update_trace=cfg["server"].get("save_update_trace", False),
    )
    lcfg = LocalTrainConfig(
        epochs=cfg["local"]["epochs"],
        lr=cfg["local"]["lr"],
        momentum=cfg["local"].get("momentum", 0.9),
        weight_decay=cfg["local"].get("weight_decay", 0.0),
        max_steps=cfg["local"].get("max_steps"),
    )
    server = FederatedServer(
        global_model=model, client_loader_fn=cl_fn, local_trainer_fn=trainer,
        local_cfg=lcfg, aggregation_fn=mechanism, server_cfg=scfg,
        eval_fn=ev_fn, device=device, rng_seed=cfg["seed"],
    )

    # ---- Trace collector ----
    trace = None
    if scfg.save_update_trace:
        ac = cfg.get("analysis", {})
        trace = UpdateTraceCollector(
            param_dim=p_dim,
            proj_dim=ac.get("trace_proj_dim", 4096),
            max_samples=ac.get("trace_max_samples", 256),
            seed=cfg["seed"],
            save_dir=out_dir,
        )

    # ---- Logger callback ----
    jsonl = JsonlLogger(out_dir, "metrics.jsonl")

    def cb(rec: dict) -> None:
        if "agg_delta" in rec and trace is not None:
            arr = rec.pop("agg_delta")
            trace.add(torch.from_numpy(arr), round_idx=rec["round"], client_id=-1)
        jsonl.log(rec)
        if rec.get("eval"):
            pretty = {k: round(v, 4) for k, v in rec.items()
                      if k not in ("round", "eval", "_t") and isinstance(v, (int, float))}
            log.info("Round %d eval: %s", rec["round"], pretty)

    # ---- Run FL ----
    server.run(log_callback=cb)
    jsonl.close()

    # ---- Spectral analysis ----
    if trace is not None and len(trace.samples) >= 4:
        try:
            summ = spectral_summary(
                [torch.from_numpy(s) for s in trace.samples], k=64)
            log.info("Spectral: d_tau=%d  trace=%.4g  pr=%.2f",
                     summ.d_tau, summ.trace, summ.participation_ratio)
            spectral_record = {
                "d_tau": summ.d_tau,
                "tau": summ.tau,
                "trace": summ.trace,
                "trace_sq": summ.trace_sq,
                "participation_ratio": summ.participation_ratio,
                "top_eigs": summ.eigenvalues[:50].tolist(),
            }
        except Exception as exc:
            log.warning("Spectral summary failed: %s", exc)
            spectral_record = {"error": str(exc)}
        with open(out_dir / "spectral.json", "w") as f:
            json.dump(spectral_record, f, indent=2)
        trace.save("trace")

    # ---- Attacks ----
    summary: dict = {"trainable_parameters": p_dim}
    attacks_cfg = cfg.get("attacks", {})
    mia_cfg = attacks_cfg.get("mia", {})

    # MIA for vision / femnist
    if mia_cfg.get("enabled") and cfg["modality"] in ("vision", "femnist"):
        m_ld, nm_ld = build_mia_loaders_vision(
            split,
            n_members=mia_cfg.get("n_members", 2048),
            n_nonmembers=mia_cfg.get("n_nonmembers", 2048),
            seed=cfg["seed"],
        )
        result = mia_vision(model, m_ld, nm_ld, device)
        log.info("MIA: AUC=%.4f  advantage=%.4f  (m=%d, nm=%d)",
                 result.auc, result.advantage, result.n_members, result.n_nonmembers)
        summary["mia"] = {
            "auc": result.auc, "advantage": result.advantage,
            "n_members": result.n_members, "n_nonmembers": result.n_nonmembers,
        }

    # MIA for GPT-2: pool train-user posts as members, test-user posts as non-members
    if mia_cfg.get("enabled") and cfg["modality"] == "gpt2":
        from privscale.data.stackoverflow import _LMTokenizedTextDataset, _read_user_file
        users = split.users
        n_mia = max(1, min(len(users) // 5, mia_cfg.get("n_users", 200)))
        m_ds_list, nm_ds_list = [], []
        for i in range(n_mia):
            posts_m = _read_user_file(split.root / "train" / f"{users[i]}.txt", limit=30)
            posts_nm = _read_user_file(split.root / "test" / f"{users[-(i+1)]}.txt", limit=30)
            if posts_m:
                m_ds_list.append(_LMTokenizedTextDataset(posts_m, model.tokenizer, split.seq_len))
            if posts_nm:
                nm_ds_list.append(_LMTokenizedTextDataset(posts_nm, model.tokenizer, split.seq_len))
        if m_ds_list and nm_ds_list:
            m_ld = DataLoader(ConcatDataset(m_ds_list), batch_size=4, shuffle=False)
            nm_ld = DataLoader(ConcatDataset(nm_ds_list), batch_size=4, shuffle=False)
            result = mia_gpt2(model, m_ld, nm_ld, device)
            log.info("MIA (GPT-2): AUC=%.4f  advantage=%.4f",
                     result.auc, result.advantage)
            summary["mia"] = {
                "auc": result.auc, "advantage": result.advantage,
                "n_members": result.n_members, "n_nonmembers": result.n_nonmembers,
            }

    # Canary extraction
    canary_cfg = attacks_cfg.get("canary", {})
    if canary_plan is not None and canary_cfg.get("enabled") and cfg["modality"] == "gpt2":
        ext = extraction_rate(
            model, model.tokenizer, canary_plan.canaries, device,
            max_new_tokens=canary_cfg.get("max_new_tokens", 24),
        )
        log.info("Extraction: rate=%.4f (%d/%d)",
                 ext["extraction_rate"], ext["n_recovered"], ext["n_canaries"])
        n_exposure = canary_cfg.get("exposure_subset", 0)
        if n_exposure > 0:
            cs = canary_cfg.get("candidate_set_size", 1024)
            exposures = [
                canary_exposure(model, model.tokenizer, c, candidate_set_size=cs,
                                device=device, seed=cfg["seed"])
                for c in canary_plan.canaries[:n_exposure]
            ]
            ext["exposure_mean"] = float(np.mean(exposures))
            ext["exposure_max"] = float(np.max(exposures))
        summary["extraction"] = ext

    # ---- Final eval and save ----
    final = ev_fn(model)
    summary["final_eval"] = final
    log.info("Final eval: %s", {k: round(v, 4) for k, v in final.items()})

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
