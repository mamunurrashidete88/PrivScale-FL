# =============================================================================
# PrivScale-FL Makefile
# Usage: make <target>   (make help for full listing)
# =============================================================================

PYTHON     := python3
PYTEST     := python3 -m pytest
SEEDS      := 0 1 2
ROUNDS_V   := 500
ROUNDS_L   := 200
EPS        := 3.0
DELTA      := 1e-5

.PHONY: help install data-all data-stackoverflow \
        test test-fast \
        quick-test calibrate \
        spectral risk-scaling scaling-law calibration-compare ablation \
        all-experiments vision-ladder language-ladder \
        figures aggregate \
        clean-results clean-data clean-all

# -----------------------------------------------------------------------------
help:                        ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-24s\033[0m %s\n",$$1,$$2}'

# -----------------------------------------------------------------------------
install:                     ## Install all Python dependencies
	pip install -r requirements.txt
	pip install -e .

# -----------------------------------------------------------------------------
data-all: data-stackoverflow ## Prepare all datasets (CIFAR auto-downloads)

data-stackoverflow:          ## Preprocess Stack Overflow per-user shards
	$(PYTHON) scripts/prepare_stackoverflow.py \
	  --output data/stackoverflow_subset --max-users 2000 --min-posts 50

# -----------------------------------------------------------------------------
test:                        ## Run full test suite
	$(PYTEST) tests/ -v

test-fast:                   ## Run fast unit tests only (no integration)
	$(PYTEST) tests/ -v -m "not slow and not integration"

# -----------------------------------------------------------------------------
quick-test:                  ## Sanity check: CIFAR-10, ResNet-20, 20 rounds
	$(PYTHON) scripts/run_experiment.py \
	  --config configs/vision/smoke_cifar10.yaml

calibrate:                   ## Find σ₀ for (ε,δ)=(3.0,10⁻⁵) at q=0.1, T=500
	$(PYTHON) scripts/calibrate_noise.py \
	  --eps $(EPS) --delta $(DELTA) --q 0.1 --rounds $(ROUNDS_V)

# -----------------------------------------------------------------------------
# Individual sub-experiment targets (Section 4 of the paper)

spectral:                    ## Fig 3 — Spectral validation of Assumption 3
	$(PYTHON) experiments/run_spectral_validation.py \
	  --datasets cifar100 stackoverflow --seed 0

risk-scaling:                ## Fig 4 — Privacy risk scaling (RQ1)
	$(PYTHON) experiments/run_risk_scaling.py \
	  --datasets cifar100 stackoverflow --seed 0

scaling-law:                 ## Fig 5 — Scaling law fit + privacy debt (RQ2)
	$(PYTHON) experiments/run_scaling_law.py \
	  --datasets cifar100 stackoverflow --seed 0

calibration-compare:         ## Fig 6 + Table 2 — Mechanism comparison (RQ3)
	$(PYTHON) experiments/run_calibration.py \
	  --datasets cifar100 stackoverflow --seed 0

ablation:                    ## Appendix — Batch size, participation, FT vs LoRA
	$(PYTHON) experiments/run_ablation.py \
	  --datasets cifar100 stackoverflow --seed 0

# -----------------------------------------------------------------------------
vision-ladder:               ## Full 6-point vision ladder (6 models × 4 mechs × 5/3 seeds)
	bash scripts/run_vision_ladder.sh

language-ladder:             ## Full language ladder (3 models × 4 mechs × 3 seeds)
	bash scripts/run_language_ladder.sh

all-experiments: vision-ladder language-ladder aggregate figures  ## Run everything

# -----------------------------------------------------------------------------
aggregate:                   ## Aggregate per-run results into CSV
	$(PYTHON) scripts/aggregate_results.py --runs-dir runs/ --out results/

figures:                     ## Generate all paper figures from results/
	$(PYTHON) utils/visualization.py --results-dir results/ --output-dir figures/

# -----------------------------------------------------------------------------
clean-results:               ## Remove all run outputs and results
	rm -rf runs/ results/ figures/*.png figures/*.pdf
	find . -name "*.jsonl" -delete 2>/dev/null || true

clean-data:                  ## Remove preprocessed data
	rm -rf data/stackoverflow_subset
	find data/ -name "*.pt" -delete 2>/dev/null || true

clean-all: clean-results     ## Remove results, data caches, and Python cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
