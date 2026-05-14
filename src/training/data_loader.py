# src/training/data_loader.py
"""
PyTorch Dataset + DataLoader builders for AudioMNIST spectrograms.

Reads a Stage 2 feature manifest (one .npy per audio file, fixed shape),
returns (features, label) tensors where:

    features: (1, n_frames, n_freq_bins) float32  — single channel
    label   : int64                                — digit class 0..9
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def load_feature_manifest(path: str | Path) -> tuple[dict, list[dict]]:
    """Load a Stage 2 feature manifest. Returns (metadata, entries)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature manifest not found: {path}")
    with open(path) as f:
        manifest = json.load(f)
    return manifest, manifest["entries"]


def _filter_entries(entries: list[dict], split: str) -> list[dict]:
    filtered = [e for e in entries if e["split"] == split]
    if not filtered:
        raise ValueError(f"No entries found for split '{split}'")
    return filtered


class AudioMnistDataset(Dataset):
    """Loads .npy spectrograms produced by Stage 2.

    Each .npy is shape (n_frames, n_freq_bins) and is returned as a
    (1, n_frames, n_freq_bins) float32 tensor with int64 label.
    """

    def __init__(self, entries: list[dict]) -> None:
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.entries[idx]
        spec = np.load(entry["feature_path"]).astype(np.float32)
        x = torch.from_numpy(spec).unsqueeze(0)
        y = torch.tensor(int(entry["label"]), dtype=torch.long)
        return x, y


def build_dataloaders(
    feature_manifest_path: str | Path,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, list[dict]]:
    """Build train/val/test DataLoaders from a feature manifest.

    Returns:
        (train_loader, val_loader, test_loader, test_entries) — test_entries
        is kept so the evaluator can map predictions back to per-file metadata
        (path, speaker_id, label_name).
    """
    _, entries = load_feature_manifest(feature_manifest_path)

    train_entries = _filter_entries(entries, "train")
    val_entries = _filter_entries(entries, "val")
    test_entries = _filter_entries(entries, "test")

    logger.info(
        "Datasets: train=%d, val=%d, test=%d",
        len(train_entries), len(val_entries), len(test_entries),
    )

    train_loader = DataLoader(
        AudioMnistDataset(train_entries),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        AudioMnistDataset(val_entries),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        AudioMnistDataset(test_entries),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, test_loader, test_entries
