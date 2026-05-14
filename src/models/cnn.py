# src/models/cnn.py
"""
Spectrogram CNN for AudioMNIST digit classification.

Input shape:  (batch, 1, n_frames, n_freq)  — single-channel time x frequency.
Output:       (batch, num_classes) logits — caller pairs with CrossEntropyLoss.
"""

from __future__ import annotations

import torch.nn as nn

from src.config import ModelConfig


class AudioDigitCNN(nn.Module):
    """Stacked conv blocks + adaptive pool + MLP head."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        filters = cfg.backbone["filters"]
        kernel = int(cfg.backbone.get("kernels", 3))
        dropout = float(cfg.backbone.get("dropout", 0.3))
        use_bn = bool(cfg.backbone.get("batch_norm", True))
        head_units = int(cfg.head["units"])
        head_dropout = float(cfg.head.get("dropout", dropout))
        num_classes = int(cfg.output.num_classes)

        def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.extend([
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        in_channels = [1, *filters[:-1]]
        self.features = nn.Sequential(
            *[conv_block(ic, oc) for ic, oc in zip(in_channels, filters)]
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(head_dropout),
            nn.Linear(filters[-1], head_units),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_units, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(cfg: ModelConfig, input_shape: tuple[int, int, int]) -> nn.Module:
    """Factory for the configured CNN.

    `input_shape` is the feature map shape (n_frames, n_freq_bins, channels)
    as reported by ExperimentConfig.input_shape — informational because the
    network uses AdaptiveAvgPool2d and is otherwise shape-agnostic.
    """
    return AudioDigitCNN(cfg)
