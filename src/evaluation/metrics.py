# src/evaluation/metrics.py
"""
Multiclass metric computation and plot generation for AudioMNIST evaluation.

Metrics:
    accuracy, macro_f1, weighted_f1, macro_precision, macro_recall

Plots:
    confusion_matrix, per_class_accuracy, per_speaker_accuracy
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


_METRICS = {
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
}

_PLOTS = {
    "confusion_matrix",
    "per_class_accuracy",
    "per_speaker_accuracy",
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: list[str],
) -> dict[str, float]:
    """Compute requested multiclass metrics."""
    out: dict[str, float] = {}
    for name in metric_names:
        if name not in _METRICS:
            logger.warning("Unknown metric '%s', skipping", name)
            continue
        try:
            out[name] = float(_compute_one(name, y_true, y_pred))
        except Exception:
            logger.warning("Failed to compute metric '%s'", name, exc_info=True)
            out[name] = float("nan")
    return out


def _compute_one(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if name == "accuracy":
        return accuracy_score(y_true, y_pred)
    if name == "macro_f1":
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    if name == "weighted_f1":
        return f1_score(y_true, y_pred, average="weighted", zero_division=0)
    if name == "macro_precision":
        return precision_score(y_true, y_pred, average="macro", zero_division=0)
    if name == "macro_recall":
        return recall_score(y_true, y_pred, average="macro", zero_division=0)
    raise ValueError(f"Unknown metric: {name}")


def generate_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    speaker_ids: list[str] | None,
    plot_names: list[str],
    output_dir: Path,
    run_name: str | None = None,
) -> list[Path]:
    """Generate evaluation plots and save as PNG files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for name in plot_names:
        if name not in _PLOTS:
            logger.warning("Unknown plot '%s', skipping", name)
            continue
        try:
            path = _plot_one(name, y_true, y_pred, class_names, speaker_ids, output_dir, run_name)
            if path:
                saved.append(path)
        except Exception:
            logger.warning("Failed to generate plot '%s'", name, exc_info=True)
    return saved


def _plot_one(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    speaker_ids: list[str] | None,
    output_dir: Path,
    run_name: str | None,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(8, 6))

    if name == "confusion_matrix":
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title("Confusion Matrix")

    elif name == "per_class_accuracy":
        per_class = []
        for cls in range(len(class_names)):
            mask = y_true == cls
            if mask.any():
                per_class.append(float((y_pred[mask] == cls).mean()))
            else:
                per_class.append(0.0)
        x = np.arange(len(class_names))
        bars = ax.bar(x, per_class, color="steelblue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Digit")
        ax.set_ylabel("Accuracy")
        ax.set_title("Per-class accuracy")
        for bar, val in zip(bars, per_class):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", fontsize=8)

    elif name == "per_speaker_accuracy":
        if not speaker_ids:
            logger.warning("per_speaker_accuracy requires speaker_ids — skipping")
            plt.close(fig)
            return None
        speakers = sorted(set(speaker_ids))
        per_sp = []
        for sp in speakers:
            mask = np.array([s == sp for s in speaker_ids])
            if mask.any():
                per_sp.append(float((y_pred[mask] == y_true[mask]).mean()))
            else:
                per_sp.append(0.0)
        x = np.arange(len(speakers))
        ax.bar(x, per_sp, color="steelblue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(speakers, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Speaker (test split)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Per-speaker accuracy")
        ax.axhline(np.mean(per_sp), color="crimson", linestyle="--",
                   alpha=0.6, label=f"mean={np.mean(per_sp):.3f}")
        ax.legend()

    if run_name:
        fig.suptitle(run_name, fontsize=10, y=0.995)
    fig.tight_layout()
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", path)
    return path
