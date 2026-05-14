# pipelines/prepare_data.py
"""
Stage 1: Data Preparation

Scans the AudioMNIST directory tree, assigns labels (digit) and speaker IDs,
generates a speaker-disjoint train/val/test split, and writes a manifest JSON
to `data/manifests/{experiment_name}[_limitN]_split.json`.

Usage:
    python -m pipelines.prepare_data --config configs/experiments/exp_cnn_classifier_v1.yaml
    python -m pipelines.prepare_data --config configs/experiments/exp_cnn_classifier_v1.yaml --limit 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import load_experiment
from src.data.manifest import write_manifest
from src.data.scanner import scan_audiomnist
from src.data.splitter import speaker_disjoint_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: Scan AudioMNIST + speaker-disjoint split"
    )
    parser.add_argument("--config", type=str, required=True, help="Experiment config YAML")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max files per speaker (overrides dataset.limit). Useful for smoke tests.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/manifests",
        help="Directory for the manifest JSON (default: data/manifests)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_experiment(args.config)
    ds = cfg.dataset

    if ds.split.strategy != "speaker_disjoint":
        raise ValueError(
            f"This pipeline only supports split.strategy='speaker_disjoint' "
            f"(got '{ds.split.strategy}')"
        )

    limit = args.limit if args.limit is not None else ds.limit

    logger.info("=" * 60)
    logger.info("STAGE 1: DATA PREPARATION")
    logger.info("=" * 60)
    logger.info("Dataset : %s  (type: %s)", ds.name, ds.type)
    logger.info("Root    : %s", ds.path)
    logger.info("Limit   : %s files / speaker", limit or "all")

    files = scan_audiomnist(root=ds.path, format=ds.format, limit=limit)
    split = speaker_disjoint_split(files=files, ratios=ds.split.ratios, seed=ds.split.seed)

    suffix = f"_limit{limit}" if limit else ""
    manifest_name = f"{cfg.name}{suffix}_split.json"
    output_path = Path(args.output_dir) / manifest_name

    write_manifest(
        split=split,
        output_path=output_path,
        dataset_name=ds.name,
        metadata={
            "experiment": cfg.name,
            "dataset_type": ds.type,
            "limit": limit,
            "split_seed": ds.split.seed,
            "split_strategy": ds.split.strategy,
            "split_ratios": ds.split.ratios,
        },
    )

    logger.info("-" * 60)
    for split_name, stats in split.summary.items():
        logger.info(
            "  %-6s: %4d files (%d speakers)",
            split_name, stats["total"], stats["speakers"],
        )
    logger.info("-" * 60)
    logger.info("Manifest: %s", output_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
