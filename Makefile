SHELL := /bin/bash
.DEFAULT_GOAL := help

# ── Env vars (read from .env when present) ─────────────────────────────────────
USER_NAME ?= $(shell grep -s '^USER_NAME=' .env | cut -d= -f2-)
ifeq ($(strip $(USER_NAME)),)
	USER_NAME := $(shell whoami)
endif

MLFLOW_PORT     ?= $(shell grep -s '^MLFLOW_PORT='     .env | cut -d= -f2-)
BASE_DOCKERFILE ?= $(shell grep -s '^BASE_DOCKERFILE=' .env | cut -d= -f2-)
GPU_DEVICE      ?= $(shell grep -s '^GPU_DEVICE='      .env | cut -d= -f2-)
DATASETS_DIR    ?= $(shell grep -s '^DATASETS_DIR='    .env | cut -d= -f2-)

ifeq ($(strip $(MLFLOW_PORT)),)
	MLFLOW_PORT := 5000
endif
ifeq ($(strip $(BASE_DOCKERFILE)),)
	BASE_DOCKERFILE := docker/Dockerfile
endif
ifeq ($(strip $(GPU_DEVICE)),)
	GPU_DEVICE := all
endif
ifeq ($(strip $(GPU_DEVICE)),all)
	GPU_REQUEST := all
else
	GPU_REQUEST := device=$(GPU_DEVICE)
endif

OS := $(shell uname -s)
GPU_FLAGS ?= $(if $(filter Linux,$(OS)),--gpus $(GPU_REQUEST),)

# Optional override for the dataset mount; defaults to the in-repo data/ tree.
ifeq ($(strip $(DATASETS_DIR)),)
	DATASETS_VOLUME_MOUNTS :=
else
	DATASETS_VOLUME_MOUNTS := -v $(DATASETS_DIR):/datasets:ro
endif

IMAGE              := audio-mnist-$(USER_NAME)
CONFIG             ?= configs/experiments/exp_cnn_classifier_v1.yaml
LOCAL_TRACKING_URI := http://localhost:$(MLFLOW_PORT)

# Local Python runner — defaults to the pixi env. Override (e.g. PY=.venv/bin/python)
# to bypass pixi and use a pre-existing virtualenv instead.
PIXI_ENV ?= default
PY       ?= pixi run -e $(PIXI_ENV) python

LIMIT    ?=
MANIFEST ?=
WORKERS  ?=

DOCKER_RUN := docker run --rm \
	$(GPU_FLAGS) \
	-v $(PWD)/configs:/app/configs \
	-v $(PWD)/data:/app/data \
	$(DATASETS_VOLUME_MOUNTS) \
	--env-file .env \
	$(IMAGE)

# ============================================================
# Help
# ============================================================

.PHONY: help
help:
	@echo "Targets:"
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables: CONFIG, LIMIT, MANIFEST, WORKERS, MLFLOW_PORT, GPU_DEVICE"
	@echo "Examples:"
	@echo "  make prepare-local CONFIG=configs/experiments/exp_cnn_classifier_v1.yaml LIMIT=20"
	@echo "  make train         CONFIG=configs/experiments/exp_cnn_classifier_v1.yaml"

# ============================================================
# Setup
# ============================================================

.PHONY: setup-local
setup-local:  ## Install the local pixi environment (CPU/MPS on Mac, CUDA on Linux with PIXI_ENV=gpu)
	@command -v pixi >/dev/null 2>&1 || { \
		echo "pixi not found. Install it from https://pixi.sh (e.g. 'curl -fsSL https://pixi.sh/install.sh | bash')"; \
		exit 1; \
	}
	pixi install -e $(PIXI_ENV)
	@echo "Pixi env '$(PIXI_ENV)' ready. Test with: make gpu-check-local"

.PHONY: shell-local
shell-local:  ## Enter the pixi shell for the selected environment
	pixi shell -e $(PIXI_ENV)

.PHONY: build
build:  ## Build the Docker image (server)
	docker build -f $(BASE_DOCKERFILE) -t $(IMAGE) .

.PHONY: mlflow-up
mlflow-up:  ## Start the MLflow tracking server (docker compose)
	docker compose up -d mlflow
	@echo "MLflow UI → http://localhost:$(MLFLOW_PORT)"

.PHONY: mlflow-down
mlflow-down:  ## Stop the MLflow tracking server
	docker compose down

.PHONY: gpu-check
gpu-check:  ## Show torch device availability inside the image
	$(DOCKER_RUN) python -c "import torch; \
print('torch          =', torch.__version__); \
print('cuda_available =', torch.cuda.is_available()); \
print('cuda_devices   =', torch.cuda.device_count())"

.PHONY: gpu-check-local
gpu-check-local:  ## Show torch device availability on the host (cuda/mps/cpu)
	@$(PY) -c "import torch; \
print(f'torch              = {torch.__version__}'); \
print(f'cuda_available     = {torch.cuda.is_available()}'); \
print(f'mps_available      = {torch.backends.mps.is_available()}'); \
print(f'mps_built          = {torch.backends.mps.is_built()}')"

# ============================================================
# Pipeline — Docker (Linux server)
# ============================================================

.PHONY: prepare
prepare:  ## Stage 1: scan + speaker-disjoint split  (docker)
	$(DOCKER_RUN) python -m pipelines.prepare_data \
		--config $(CONFIG) \
		$(if $(LIMIT),--limit $(LIMIT))

.PHONY: features
features:  ## Stage 2: extract spectrograms  (docker)
	$(DOCKER_RUN) python -m pipelines.extract_features \
		--config $(CONFIG) \
		$(if $(MANIFEST),--manifest $(MANIFEST)) \
		$(if $(WORKERS),--workers $(WORKERS))

.PHONY: train
train:  ## Stage 3: train  (docker, uses GPU on Linux)
	$(DOCKER_RUN) python -m pipelines.train --config $(CONFIG)

.PHONY: evaluate
evaluate:  ## Stage 4: evaluate  (docker)
	$(DOCKER_RUN) python -m pipelines.evaluate --config $(CONFIG)

.PHONY: train-eval
train-eval: train evaluate  ## Stages 3+4

.PHONY: run-all
run-all: prepare features train evaluate  ## Full pipeline (docker)

# ============================================================
# Pipeline — Local (Mac, MPS)
# ============================================================

.PHONY: prepare-local
prepare-local:  ## Stage 1 on host
	MLFLOW_TRACKING_URI=$(LOCAL_TRACKING_URI) $(PY) -m pipelines.prepare_data \
		--config $(CONFIG) \
		$(if $(LIMIT),--limit $(LIMIT))

.PHONY: features-local
features-local:  ## Stage 2 on host
	MLFLOW_TRACKING_URI=$(LOCAL_TRACKING_URI) $(PY) -m pipelines.extract_features \
		--config $(CONFIG) \
		$(if $(MANIFEST),--manifest $(MANIFEST)) \
		$(if $(WORKERS),--workers $(WORKERS))

.PHONY: train-local
train-local:  ## Stage 3 on host (uses MPS if available)
	MLFLOW_TRACKING_URI=$(LOCAL_TRACKING_URI) $(PY) -m pipelines.train --config $(CONFIG)

.PHONY: evaluate-local
evaluate-local:  ## Stage 4 on host
	MLFLOW_TRACKING_URI=$(LOCAL_TRACKING_URI) $(PY) -m pipelines.evaluate --config $(CONFIG)

.PHONY: run-all-local
run-all-local: prepare-local features-local train-local evaluate-local  ## Full pipeline on host
