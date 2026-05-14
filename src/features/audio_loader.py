# src/features/audio_loader.py
"""
Load and preprocess audio files using librosa.

Handles: mono conversion, resampling, and fixed-length pad/truncate.
High-frequency cut is applied post-STFT (frequency-bin slicing in spectrogram.py).
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from src.config import AudioConfig

logger = logging.getLogger(__name__)


def load_audio(
    path: str | Path,
    audio_cfg: AudioConfig,
) -> tuple[np.ndarray, float]:
    """Load an audio file, enforce mono+SR, and pad/truncate to a fixed length.

    Args:
        path: Path to the audio file.
        audio_cfg: Audio preprocessing config (sample_rate, mono, fixed_duration_sec).

    Returns:
        Tuple of (waveform as float32 1-D array, original duration in seconds).

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If librosa fails to decode the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        y, sr = librosa.load(
            path,
            sr=audio_cfg.sample_rate,
            mono=audio_cfg.mono,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio {path}: {exc}") from exc

    y = y.astype(np.float32)
    original_duration_sec = len(y) / sr

    if audio_cfg.fixed_duration_sec is not None:
        target_samples = int(round(audio_cfg.fixed_duration_sec * sr))
        if len(y) < target_samples:
            pad = target_samples - len(y)
            # Center-pad so the utterance sits roughly in the middle.
            left = pad // 2
            right = pad - left
            y = np.pad(y, (left, right), mode="constant")
        elif len(y) > target_samples:
            # Center-crop.
            start = (len(y) - target_samples) // 2
            y = y[start : start + target_samples]

    return y, original_duration_sec
