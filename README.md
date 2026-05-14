# audio-mnist

MLOps pipeline for 10-class spoken-digit classification on **AudioMNIST**
(30,000 WAVs, 60 speakers). Spectrogram CNN in PyTorch, MLflow for tracking,
docker for the Linux server, pixi for the local Mac (MPS) dev env.

Pipeline stages (each is a CLI under `pipelines/`):

1. **prepare**  — scan dataset → speaker-disjoint split → `data/manifests/*.json`
2. **features** — log-mel spectrograms → `data/features/.../*.npy` + feature manifest
3. **train**    — train CNN → `data/artifacts/{name}/{best_model.pt, model_summary.txt, config_snapshot.yaml, training_history.json, mlflow_run.json}`
4. **evaluate** — test-set inference → `data/artifacts/{name}/eval/` (metrics CSV/JSON + plots), appended to the training MLflow run

## Layout

```
configs/   experiment, dataset, features, model YAMLs
pipelines/ stage entry points (prepare_data, extract_features, train, evaluate)
src/       config, data, features, models, training, evaluation, mlflow_logger
docker/    Dockerfile (CPU) + Dockerfile.gpu (CUDA 12)
data/      audio-mnist/ (input) + manifests/, features/, artifacts/ (generated)
```

## Setup

```bash
# Mac dev (CPU/MPS via pixi)
make setup-local

# Linux server (CUDA via pixi)
make setup-local PIXI_ENV=gpu

# Or docker (Linux server)
make build
```

Start the MLflow tracking server (used by both flows):

```bash
make mlflow-up    # http://localhost:${MLFLOW_PORT:-8010}
make mlflow-down
```

## Run

`CONFIG` defaults to `configs/experiments/exp_cnn_classifier_v1.yaml`.

### Local (Mac, pixi)

```bash
make prepare-local                                 # stage 1
make features-local WORKERS=4                      # stage 2
make train-local                                   # stage 3
make evaluate-local                                # stage 4
make run-all-local                                 # 1–4 sequentially
```

Smoke test with a small subset (N files per speaker × digit):

```bash
make run-all-local LIMIT=10
```

### Docker (Linux server, `--gpus` is auto-added on Linux)

```bash
make prepare
make features WORKERS=8
make train
make evaluate
make run-all
```

### Variables

| Var         | Default                                              | Notes                                          |
|-------------|------------------------------------------------------|------------------------------------------------|
| `CONFIG`    | `configs/experiments/exp_cnn_classifier_v1.yaml`     | Experiment YAML                                |
| `LIMIT`     | (none)                                               | Max files per (speaker, digit) — smoke tests   |
| `MANIFEST`  | (auto-resolved)                                      | Override Stage 1 manifest for `features`       |
| `WORKERS`   | `1`                                                  | Parallel workers for feature extraction        |
| `PIXI_ENV`  | `default`                                            | Use `gpu` on the Linux server                  |
| `GPU_DEVICE`| `all`                                                | `--gpus device=X` instead of `--gpus all`      |

## Direct CLI

```bash
pixi run python -m pipelines.prepare_data     --config $CONFIG [--limit N]
pixi run python -m pipelines.extract_features --config $CONFIG [--workers N]
pixi run python -m pipelines.train            --config $CONFIG
pixi run python -m pipelines.evaluate         --config $CONFIG
```

## Outputs per run

```
data/manifests/{exp_name}_split.json
data/features/{features_name}/AudioMNIST/{train,val,test}/*.npy
data/features/{features_name}/AudioMNIST/feature_manifest.json
data/artifacts/{exp_name}/
  best_model.pt
  last_model.pt
  model_summary.txt
  config_snapshot.yaml
  training_history.json
  mlflow_run.json
  eval/
    eval_metrics.csv
    eval_report.json
    confusion_matrix.png
    per_class_accuracy.png
    per_speaker_accuracy.png
```
