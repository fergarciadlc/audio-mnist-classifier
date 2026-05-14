# src/evaluation/evaluator.py
"""
Run inference on the test split and collect predictions for metrics.

The model is fed one batch at a time; each test-set entry's predicted class
and per-class softmax probability are recorded so downstream code can compute
metrics and per-file diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    y_true: np.ndarray         # (N,) int — ground-truth digit class
    y_pred: np.ndarray         # (N,) int — argmax class
    y_prob: np.ndarray         # (N, num_classes) float — softmax probabilities
    speaker_ids: list[str]     # length N
    feature_paths: list[str]   # length N

    def file_scores(self, class_names: list[str]) -> list[dict]:
        """Per-test-file record suitable for JSON/CSV export."""
        out: list[dict] = []
        for i in range(len(self.y_true)):
            true_cls = int(self.y_true[i])
            pred_cls = int(self.y_pred[i])
            out.append({
                "feature_path": self.feature_paths[i],
                "speaker_id": self.speaker_ids[i],
                "true_label": true_cls,
                "true_label_name": class_names[true_cls],
                "predicted_label": pred_cls,
                "predicted_label_name": class_names[pred_cls],
                "predicted_prob": float(self.y_prob[i, pred_cls]),
            })
        return out


@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    test_entries: list[dict],
    device: torch.device,
) -> EvalResult:
    """Run inference over the test loader, return aligned predictions.

    Important: the test_loader must NOT shuffle, so each batch corresponds
    in order to `test_entries`.
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    n_total = len(test_entries)
    seen = 0
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).cpu().numpy()
        all_logits.append(logits)
        all_targets.append(y.numpy())

        seen += x.size(0)
        if seen % (50 * test_loader.batch_size) == 0 or seen >= n_total:
            logger.info("  Evaluated %d/%d files", min(seen, n_total), n_total)

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_targets, axis=0).astype(np.int64)

    # Softmax
    logits_max = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - logits_max)
    y_prob = e / e.sum(axis=1, keepdims=True)
    y_pred = y_prob.argmax(axis=1).astype(np.int64)

    if len(test_entries) != len(y_true):
        raise RuntimeError(
            f"Inference produced {len(y_true)} predictions but test_entries has "
            f"{len(test_entries)} — did the test loader shuffle?"
        )

    speaker_ids = [e.get("speaker_id", "") for e in test_entries]
    feature_paths = [e["feature_path"] for e in test_entries]

    return EvalResult(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        speaker_ids=speaker_ids,
        feature_paths=feature_paths,
    )
