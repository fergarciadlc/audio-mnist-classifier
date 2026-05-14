# pipelines/train.py
"""
Stage 3: Training

Builds the model from config, trains on Stage 2 features, saves the best
checkpoint, model summary, training history, and config snapshot, and logs
to MLflow when MLFLOW_TRACKING_URI is set.

Usage:
    python -m pipelines.train --config configs/experiments/exp_cnn_classifier_v1.yaml
    python -m pipelines.train --config configs/experiments/exp_cnn_classifier_v1.yaml \
        --feature-manifest data/features/mel_16k_64/AudioMNIST/feature_manifest.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import load_experiment
from src.training.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: Train model on extracted features")
    parser.add_argument("--config", type=str, required=True, help="Experiment config YAML")
    parser.add_argument(
        "--feature-manifest", type=str, default=None,
        help=("Stage 2 feature manifest. If omitted, auto-resolved from "
              "data/features/{features.name}/{dataset.name}/feature_manifest.json"),
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/artifacts",
        help="Root dir for training artifacts (default: data/artifacts)",
    )
    return parser.parse_args(argv)


def _resolve_feature_manifest(features_name: str, dataset_name: str) -> Path:
    path = Path("data/features") / features_name / dataset_name / "feature_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature manifest not found: {path}. Run Stage 2 (extract_features) first."
        )
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_experiment(args.config)

    logger.info("=" * 60)
    logger.info("STAGE 3: TRAINING")
    logger.info("=" * 60)
    logger.info("Experiment: %s", cfg.name)
    logger.info("Model     : %s (%s)", cfg.model.name, cfg.model.type)
    logger.info(
        "Epochs: %d, Batch: %d, LR: %s",
        cfg.training.epochs, cfg.training.batch_size, cfg.training.learning_rate,
    )

    feature_manifest_path = (
        Path(args.feature_manifest)
        if args.feature_manifest
        else _resolve_feature_manifest(cfg.features.name, cfg.dataset.name)
    )
    logger.info("Feature manifest: %s", feature_manifest_path)

    result = train(
        cfg=cfg,
        feature_manifest_path=feature_manifest_path,
        output_dir=Path(args.output_dir),
    )

    logger.info("-" * 60)
    logger.info("Training complete!")
    logger.info("  Epochs run        : %d", result["epochs_run"])
    logger.info("  Best epoch        : %d", result["best_epoch"])
    logger.info("  Best val metric   : %.4f", result["best_val_metric"])
    logger.info("  Final train loss  : %.4f", result["final_train_loss"])
    logger.info("  Final val loss    : %.4f", result["final_val_loss"])
    logger.info("  Final val accuracy: %.4f", result["final_val_accuracy"])
    logger.info("  Checkpoint        : %s", result["checkpoint_path"])
    logger.info("Done.")


if __name__ == "__main__":
    main()
