# Paper title: Privacy Risk Scales with Effective Dimension in Federated Learning


## Overview

Privacy mechanisms in federated learning are typically calibrated independently
of model scale, implicitly assuming that privacy risk remains stable as models
grow. This assumption is incorrect. We show that effective model dimension,
rather than raw parameter count, governs how much client-specific information
survives in privatized update trajectories — and that this quantity grows with
model scale, creating a privacy debt that fixed-noise mechanisms cannot repay.

PrivScale-FL provides:

- Federated leakage — a signal-level mutual-information quantity measuring
  how much client-specific information survives in privatized local update
  trajectories under an additive Gaussian channel model
- Effective model dimension — a threshold-based numerical rank of the
  aggregated local gradient covariance, identified as the operative scaling
  variable for client-level privacy risk
- Conditional leakage scaling law — federated leakage grows as
  Θ(d / log d) with effective dimension d under a spectral model of
  informative client-specific directions (Theorem 1)
- Privacy debt — fixed-noise DP-FL mechanisms become progressively
  miscalibrated as models scale; the effective privacy parameter diverges
  as d → ∞ (Theorem 2)
- Scale-aware calibration — a noise schedule σ(d) = Θ(√(d / log d))
  that preserves a target leakage regime under model expansion (Theorem 3)



## Requirements

---
Python ≥ 3.10, PyTorch ≥ 2.2. CUDA is optional but
recommended for full-scale runs. GPT-2 experiments additionally require
`transformers ≥ 4.40` and `peft ≥ 0.10`.

---

## Dataset Setup

### CIFAR-10 / CIFAR-100 (Dirichlet partitioning)

Downloaded automatically by `torchvision` on first run. Client partitions
(Dirichlet-α or pathological split) are constructed at runtime.

### FEMNIST (LEAF natural user split)

Requires the LEAF preprocessing pipeline:

```bash
git clone https://github.com/TalwalkarLab/leaf.git
cd leaf/data/femnist
./preprocess.sh -s niid --sf 0.1 -k 0 -t sample
```

Point `data_root` in the config to the resulting `data/` directory.

### Shakespeare (LEAF natural user split)

```bash
cd leaf/data/shakespeare
./preprocess.sh -s niid --sf 0.1 -k 0 -t sample
```

### Stack Overflow (subset, per-user shards)

```bash
python scripts/prepare_stackoverflow.py \
    --output data/stackoverflow_subset \
    --max-users 2000 --min-posts 50
```

Requires ~50 GB disk and 30–60 minutes. Uses the HuggingFace
`pacovaldez/stackoverflow-questions` mirror.

---

## Artifact Structure

```
PrivScale/
├── privscale/                        Core library
│   ├── data/
│   │   ├── partitioners.py           Dirichlet-α and pathological client partitioners
│   │   ├── cifar.py                  CIFAR-10/100 federated loader + partition
│   │   ├── femnist.py                FEMNIST loader (LEAF natural user split)
│   │   ├── shakespeare.py            Shakespeare char-level loader (LEAF split)
│   │   ├── stackoverflow.py          Stack Overflow per-user shard loader
│   │   └── gpt2_pipeline.py          GPT-2 pipeline builder with canary injection
│   ├── models/
│   │   ├── resnet_cifar.py           ResNet-20/32/44/56/110 for 32×32 inputs
│   │   ├── wrn.py                    WideResNet-28-10 (Zagoruyko & Komodakis 2016)
│   │   ├── femnist_cnn.py            LEAF benchmark CNN (62-way)
│   │   ├── char_lstm.py              Stacked character LSTM for Shakespeare
│   │   └── gpt2_wrapper.py           GPT-2 Small/Medium/Large with optional LoRA
│   ├── fl/
│   │   ├── flatten.py                Flat-parameter view for architecture-agnostic FL
│   │   ├── client.py                 Per-client local training (vision / charLM / GPT-2)
│   │   ├── evaluate.py               Test-set evaluation (accuracy / perplexity)
│   │   └── server.py                 FedAvg server with pluggable privacy mechanism
│   ├── privacy/
│   │   ├── clipping.py               Per-client L2 clipping
│   │   ├── mechanisms.py             Four aggregation mechanisms (see table above)
│   │   └── rdp_accountant.py         Rényi DP accountant (Mironov 2017; MTZ 2019)
│   ├── analysis/
│   │   ├── spectral.py               Effective dimension via randomized SVD + diagnostics
│   │   ├── trace_collector.py        JL-projected update reservoir for spectral analysis
│   │   └── fitting.py                Candidate trend families (constant / linear / d/log d)
│   ├── attacks/
│   │   ├── mia.py                    Shadow-model membership inference (AUC + advantage)
│   │   ├── extraction.py             Canary extraction rate + exposure metric
│   │   └── canary_injection.py       Canary insertion into language client data
│   └── utils/
│       ├── common.py                 Seeding, device selection, config loading
│       └── logging.py                Structured JSONL logger
├── configs/
│   ├── vision/                       24 CIFAR-100 configs (6 models × 4 mechanisms)
│   │   ├── cifar100_{resnet20,resnet32,resnet44,resnet56,resnet110,wrn_28_10}_{nonprivate,fixed,adaptive,scale_aware}.yaml
│   │   └── smoke_cifar10.yaml        Quick smoke test config
│   └── language/                     12 Stack Overflow configs (3 models × 4 mechanisms)
│       ├── so_gpt2_{small,medium_lora,large_lora}_{nonprivate,fixed,adaptive,scale_aware}.yaml
│       └── smoke_shakespeare.yaml    Quick smoke test config
├── experiments/                      High-level experiment orchestration
│   ├── run_full_comparison.py        Full mechanism comparison driver
│   ├── run_risk_scaling.py           Privacy-risk scaling experiments
│   ├── run_scaling_law.py            Scaling-law fitting experiments
│   ├── run_spectral_validation.py    Spectral diagnostic experiments
│   ├── run_calibration.py            Scale-aware calibration experiments
│   └── run_ablation.py              Ablation study driver
├── evaluation/                       Evaluation utilities
│   ├── evaluator.py                  Unified evaluator for privacy and utility metrics
│   └── metrics.py                    Metric computation helpers
├── scripts/
│   ├── run_experiment.py             Main entry point: config → data → model → FL → attacks → outputs
│   ├── calibrate_noise.py            Bisection search for σ at target (ε, δ) via RDP
│   ├── prepare_stackoverflow.py      Stack Overflow per-user shard preprocessor
│   ├── aggregate_results.py          Cross-run aggregation into tidy CSV
│   ├── generate_plots.py             Plots with equal-footing trend fitting
│   ├── run_vision_ladder.sh          Full 6-point vision ladder driver (6 models × 4 mechs × seeds)
│   ├── run_language_ladder.sh        Full language ladder driver (3 models × 4 mechs × seeds)
│   ├── run_all_experiments.sh        Master script for full experiment suite
│   ├── run_quick_test.sh             Quick integration test
│   ├── run_ablation.sh               Ablation study driver script
│   ├── run_calibration.sh            Calibration experiment driver script
│   ├── run_risk_scaling.sh           Risk-scaling experiment driver script
│   ├── run_scaling_law.sh            Scaling-law experiment driver script
│   └── run_spectral_validation.sh    Spectral validation driver script
├── tests/
│   ├── conftest.py                   Shared test fixtures
│   ├── test_rdp_accountant.py        RDP accountant invariant tests
│   ├── test_clipping.py              L2 clipping correctness
│   ├── test_partitioners.py          Dirichlet and pathological partition tests
│   ├── test_fitting.py               Trend-fitting selection tests
│   └── test_fl_simulator.py          End-to-end FedAvg convergence on synthetic data
├── utils/                            Top-level shared utilities
│   ├── reproducibility.py            Seed management and reproducibility helpers
│   └── visualization.py              Plotting and visualization utilities
├── figures/                          Generated plots (populated by generate_plots.py)
├── LICENSE                           Project license
├── Makefile                          Build and experiment targets
├── pyproject.toml                    Package metadata
├── setup.py                          Package installer
├── pytest.ini                        Test configuration
├── requirements.txt                  Python dependencies
└── README.md                         This file
```



---


## Quick Start


```bash
python scripts/run_experiment.py --config configs/vision/smoke_cifar10.yaml
```


### Calibrate noise at the reference scale

```bash
python scripts/calibrate_noise.py --eps 3.0 --delta 1e-5 --q 0.1 --rounds 500
```



### Run the full vision 

```bash
bash scripts/run_vision_ladder.sh
```


```bash
bash scripts/run_language_ladder.sh
```


## Compute Requirements

| Configuration | Scope | Approx. GPU-hours | Hardware |
|---|---|---|---|
| Smoke test | 1 model, 20 rounds | < 0.1 | Any GPU or CPU |
| Vision ladder | 6 models × 4 mechs × 5/3 seeds | 440–560 A100-hrs | 1× A100 80 GB |
| Language ladder | 3 models × 4 mechs × 3 seeds | 400–600 A100-hrs | 1× A100 80 GB |
| Full (both ladders) | 9 models × 4 mechs × mixed seeds | 840–1160 A100-hrs | 1–2× A100 80 GB |

GPT-2 Large with LoRA requires ≥ 40 GB GPU memory.
WRN-28-10 federated runs require ≥ 16 GB GPU memory.

---

## Run Outputs

Each run produces a self-contained directory under `runs/`:

| File | Content |
|---|---|
| `config.json` | Resolved experiment configuration |
| `metrics.jsonl` | Per-round training metrics (loss, accuracy, timing) |
| `spectral.json` | Effective dimension, spectral trace, participation ratio, top eigenvalues |
| `summary.json` | Final evaluation metrics, MIA results, extraction results |
| `trace.npz` | JL-projected client update reservoir for post-hoc spectral analysis |

Aggregation (`scripts/aggregate_results.py`) produces:

| File | Content |
|---|---|
| `results/results.csv` | One row per run with all metrics |
| `results/by_mechanism.csv` | Per-(dataset, model, mechanism) mean ± std across seeds |



## Configuration

All experiments are driven by YAML config files. Key fields:

```yaml
modality: vision | femnist | charlm | gpt2
dataset:
  name: cifar100
  partition: dirichlet        # dirichlet | pathological | iid
  alpha: 0.1                  # Dirichlet concentration (lower = more heterogeneous)
  num_clients: 100
model:
  name: resnet20              # resnet20 | resnet32 | resnet44 | resnet56 | resnet110 | wrn_28_10 | gpt2-small | ...
  use_lora: false             # LoRA adaptation for GPT-2 Medium / Large
server:
  rounds: 500
  sample_fraction: 0.1        # per-round client sampling rate q
  sampling: poisson           # poisson | uniform
mechanism:
  name: scale_aware           # nonprivate | fixed_noise | adaptive_clip | scale_aware
  clip_norm: 1.0
  reference_sigma: 3.94       # σ₀ from calibrate_noise.py
  reference_dim: 217          # d_τ measured at the smallest model (ResNet-20)
  effective_dim: 1162         # d_τ measured at this model's scale
attacks:
  mia:
    enabled: true
    n_members: 2048
    n_nonmembers: 2048
  canary:                     #
    enabled: true
    num_canaries: 100
    repeats: 4
```




## Open Science

Source code and experiment logs are available at:

> https://anonymous.4open.science/r/PrivScale-FL/

---

## License & Citation

Provided for academic peer review.
