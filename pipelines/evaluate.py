# pipelines/evaluate.py
"""
Stage 4: Evaluation

Load the best checkpoint, run inference on the test split, compute
multiclass metrics, save plots and a JSON report, and resume the training
MLflow run to append eval/* artifacts and metrics.

Usage:
    python -m pipelines.evaluate --config configs/experiments/exp_cnn_classifier_v1.yaml
    python -m pipelines.evaluate --config configs/experiments/exp_cnn_classifier_v1.yaml \
        --model-path data/artifacts/cnn_classifier_v1/best_model.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import torch

from src.config import load_experiment
from src.evaluation.evaluator import run_evaluation
from src.evaluation.metrics import compute_metrics, generate_plots
from src.mlflow_logger import MLflowLogger, resolve_mlflow_experiment_name
from src.models.cnn import build_model
from src.training.data_loader import build_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4: Evaluate trained model on test set")
    parser.add_argument("--config", type=str, required=True, help="Experiment config YAML")
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Trained model (.pt). If omitted, auto-resolved from data/artifacts/{name}/best_model.pt",
    )
    parser.add_argument(
        "--feature-manifest", type=str, default=None,
        help=("Stage 2 feature manifest. If omitted, auto-resolved from "
              "data/features/{features.name}/{dataset.name}/feature_manifest.json"),
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for evaluation artifacts (default: data/artifacts/{name}/eval)",
    )
    return parser.parse_args(argv)


def _resolve_model_path(experiment_name: str) -> Path:
    path = Path("data/artifacts") / experiment_name / "best_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run Stage 3 (train) first.")
    return path


def _resolve_feature_manifest(features_name: str, dataset_name: str) -> Path:
    path = Path("data/features") / features_name / dataset_name / "feature_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature manifest not found: {path}. Run Stage 2 (extract_features) first."
        )
    return path


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _save_metrics_csv(metrics: dict[str, float], output_dir: Path) -> Path:
    path = output_dir / "eval_metrics.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for name, value in metrics.items():
            writer.writerow([name, f"{value:.6f}"])
    logger.info("Saved metrics CSV: %s", path)
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_experiment(args.config)

    logger.info("=" * 60)
    logger.info("STAGE 4: EVALUATION")
    logger.info("=" * 60)
    logger.info("Experiment: %s", cfg.name)

    model_path = Path(args.model_path) if args.model_path else _resolve_model_path(cfg.name)
    logger.info("Model: %s", model_path)

    manifest_path = (
        Path(args.feature_manifest)
        if args.feature_manifest
        else _resolve_feature_manifest(cfg.features.name, cfg.dataset.name)
    )
    logger.info("Feature manifest: %s", manifest_path)

    artifact_dir = Path("data/artifacts") / cfg.name
    output_dir = Path(args.output_dir) if args.output_dir else (artifact_dir / "eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    device = _select_device()
    logger.info("Device: %s", device)

    # Data
    _train_loader, _val_loader, test_loader, test_entries = build_dataloaders(
        feature_manifest_path=manifest_path,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    logger.info("Test files: %d", len(test_entries))

    # Model
    model = build_model(cfg.model, cfg.input_shape).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state"])

    # Inference
    result = run_evaluation(model, test_loader, test_entries, device)

    # Metrics + plots
    metrics = compute_metrics(result.y_true, result.y_pred, cfg.evaluation.metrics)
    plot_paths = generate_plots(
        y_true=result.y_true,
        y_pred=result.y_pred,
        class_names=cfg.dataset.class_names,
        speaker_ids=result.speaker_ids,
        plot_names=cfg.evaluation.plots,
        output_dir=output_dir,
        run_name=cfg.name,
    )

    # Reports
    report = {
        "experiment": cfg.name,
        "model_path": str(model_path),
        "test_files": len(test_entries),
        "num_classes": cfg.dataset.num_classes,
        "metrics": metrics,
        "file_scores": result.file_scores(cfg.dataset.class_names),
    }
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved report: %s", report_path)

    metrics_csv = _save_metrics_csv(metrics, output_dir)

    # MLflow — resume the training run if we can, append eval/ artifacts.
    mlflow_experiment = resolve_mlflow_experiment_name(cfg.mlflow_experiment_name)
    mlf = MLflowLogger(experiment_name=mlflow_experiment)
    run_meta = MLflowLogger.load_run_meta(artifact_dir)
    if run_meta:
        mlf.resume_run(run_meta["run_id"])
    else:
        logger.warning("No training run found — creating a standalone eval run")
        mlf.start_run(run_name=f"{cfg.name}-eval")

    mlf.log_metrics({f"eval_{k}": v for k, v in metrics.items()})
    for p in plot_paths:
        mlf.log_artifact(p, artifact_path="eval")
    mlf.log_artifact(metrics_csv, artifact_path="eval")
    mlf.log_artifact(report_path, artifact_path="eval")
    mlf.end_run()

    logger.info("-" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("-" * 60)
    for name, value in metrics.items():
        logger.info("  %-25s %.4f", name, value)
    logger.info("-" * 60)
    logger.info("Plots : %s", [str(p) for p in plot_paths])
    logger.info("Report: %s", report_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
