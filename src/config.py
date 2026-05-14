# src/config.py
"""
Config loader for the AudioMNIST pipeline.

Loads an experiment YAML, resolves referenced sub-configs (dataset, features,
model), and returns a structured ExperimentConfig that the rest of the
pipeline consumes.

Usage:
    from src.config import load_experiment

    cfg = load_experiment("configs/experiments/exp_cnn_classifier_v1.yaml")
    print(cfg.features.stft.n_fft)
    print(cfg.model.backbone["filters"])
    print(cfg.training.batch_size)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(data)}")
    return data


def _find_project_root(start: Path) -> Path:
    """Walk up from start until we find a directory containing 'configs/'."""
    current = start.resolve()
    for parent in [current, *list(current.parents)]:
        if (parent / "configs").is_dir():
            return parent
    raise FileNotFoundError(
        f"Could not find project root (no 'configs/' dir) starting from {start}"
    )


def _resolve_path(ref_path: str, project_root: Path) -> Path:
    path = Path(ref_path)
    resolved = path if path.is_absolute() else (project_root / path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Referenced config not found: {resolved}")
    return resolved


# ── Dataset ────────────────────────────────────────────────────────────────────


@dataclass
class SplitConfig:
    strategy: str
    ratios: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    seed: int = 42


@dataclass
class DatasetConfig:
    name: str
    type: str                   # "classification"
    path: str
    format: str                 # "wav"
    num_classes: int
    class_names: list[str]
    split: SplitConfig
    description: str = ""
    limit: int | None = None


def _parse_dataset(d: dict) -> DatasetConfig:
    body = d["dataset"]
    split = SplitConfig(
        strategy=body["split"]["strategy"],
        ratios=body["split"].get("ratios", [0.8, 0.1, 0.1]),
        seed=body["split"].get("seed", 42),
    )
    return DatasetConfig(
        name=body["name"],
        type=body["type"],
        path=body["path"],
        format=body["format"],
        num_classes=int(body["num_classes"]),
        class_names=list(body["class_names"]),
        split=split,
        description=body.get("description", ""),
        limit=body.get("limit"),
    )


# ── Features ───────────────────────────────────────────────────────────────────


@dataclass
class AudioConfig:
    sample_rate: int
    mono: bool = True
    hf_cut_hz: int | None = None
    fixed_duration_sec: float | None = None


@dataclass
class StftConfig:
    n_fft: int
    win_length: int
    hop_length: int


@dataclass
class MelConfig:
    n_mels: int
    fmin: int
    fmax: int
    power: float = 2.0


@dataclass
class NormConfig:
    strategy: str        # "fixed" | "per_sample"
    mean: float | None = None
    std: float | None = None


@dataclass
class FeaturesConfig:
    name: str
    type: str            # "mel_spectrogram_db" | "stft_magnitude_db"
    audio: AudioConfig
    stft: StftConfig
    normalization: NormConfig
    mel: MelConfig | None = None
    storage: str = "full"
    output_format: str = "npy"


def _parse_features(d: dict) -> FeaturesConfig:
    body = d["features"]
    audio = AudioConfig(
        sample_rate=int(body["audio"]["sample_rate"]),
        mono=body["audio"].get("mono", True),
        hf_cut_hz=body["audio"].get("hf_cut_hz"),
        fixed_duration_sec=body["audio"].get("fixed_duration_sec"),
    )
    stft = StftConfig(
        n_fft=int(body["stft"]["n_fft"]),
        win_length=int(body["stft"]["win_length"]),
        hop_length=int(body["stft"]["hop_length"]),
    )
    norm = NormConfig(
        strategy=body["normalization"]["strategy"],
        mean=body["normalization"].get("mean"),
        std=body["normalization"].get("std"),
    )
    mel = None
    if body.get("mel"):
        mel = MelConfig(
            n_mels=int(body["mel"]["n_mels"]),
            fmin=int(body["mel"]["fmin"]),
            fmax=int(body["mel"]["fmax"]),
            power=float(body["mel"].get("power", 2.0)),
        )
    return FeaturesConfig(
        name=body["name"],
        type=body["type"],
        audio=audio,
        stft=stft,
        normalization=norm,
        mel=mel,
        storage=body.get("storage", "full"),
        output_format=body.get("output_format", "npy"),
    )


# ── Model ──────────────────────────────────────────────────────────────────────


@dataclass
class ModelOutputConfig:
    num_classes: int
    loss: str = "cross_entropy"


@dataclass
class ModelConfig:
    name: str
    type: str
    description: str
    backbone: dict
    head: dict
    output: ModelOutputConfig
    tags: dict[str, Any] = field(default_factory=dict)


def _parse_model(d: dict) -> ModelConfig:
    body = d["model"]
    out = ModelOutputConfig(
        num_classes=int(body["output"]["num_classes"]),
        loss=body["output"].get("loss", "cross_entropy"),
    )
    return ModelConfig(
        name=body["name"],
        type=body["type"],
        description=body.get("description", ""),
        backbone=body["backbone"],
        head=body["head"],
        output=out,
        tags=body.get("tags") or {},
    )


# ── Training / Evaluation ──────────────────────────────────────────────────────


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str = "adam"
    seed: int = 42
    num_workers: int = 4
    callbacks: dict = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    metrics: list[str] = field(default_factory=list)
    plots: list[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    name: str
    description: str
    mlflow_experiment_name: str
    tags: dict[str, str]
    dataset: DatasetConfig
    features: FeaturesConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Compute (n_frames, n_freq_bins, channels) for the saved features.

        For fixed-duration mel/stft, n_frames is determined by the duration
        and hop_length. n_freq_bins is n_mels (mel) or hf_cut_bin
        (stft, when hf_cut_hz is set) or n_fft//2 + 1.
        """
        feat = self.features
        sr = feat.audio.sample_rate
        hop = feat.stft.hop_length

        if feat.audio.fixed_duration_sec is None:
            raise ValueError(
                "input_shape requires features.audio.fixed_duration_sec "
                "(variable-length features are not supported by this pipeline)"
            )
        n_samples = int(round(feat.audio.fixed_duration_sec * sr))
        # librosa.stft uses center=True; n_frames = 1 + n_samples // hop_length
        n_frames = 1 + n_samples // hop

        if feat.type == "mel_spectrogram_db":
            if feat.mel is None:
                raise ValueError("Mel features require a `mel` section in features config")
            n_freq = feat.mel.n_mels
        elif feat.type == "stft_magnitude_db":
            if feat.audio.hf_cut_hz is not None:
                freq_per_bin = sr / feat.stft.n_fft
                n_freq = int(feat.audio.hf_cut_hz / freq_per_bin) + 1
            else:
                n_freq = feat.stft.n_fft // 2 + 1
        else:
            raise ValueError(f"Unknown feature type: {feat.type}")

        return (n_frames, n_freq, 1)

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten nested config into dot-keyed dict for MLflow params."""
        def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    _flatten(f"{prefix}.{k}" if prefix else k, v, out)
            elif isinstance(value, list):
                out[prefix] = ",".join(str(v) for v in value)
            else:
                out[prefix] = value

        flat: dict[str, Any] = {}
        full = {
            "name": self.name,
            "dataset": asdict(self.dataset),
            "features": asdict(self.features),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
        }
        _flatten("", full, flat)
        return flat


def load_experiment(config_path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML and all referenced sub-configs."""
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    project_root = _find_project_root(config_path.parent)
    raw = _load_yaml(config_path)

    # Snapshot files (written by training) inline dataset/features/model
    # instead of referencing paths. Detect and dispatch.
    if isinstance(raw.get("dataset"), dict) and "name" in raw["dataset"]:
        return _build_experiment_from_inline(raw)

    exp = raw["experiment"]
    dataset_cfg = _parse_dataset(_load_yaml(_resolve_path(raw["dataset"], project_root)))
    features_cfg = _parse_features(_load_yaml(_resolve_path(raw["features"], project_root)))
    model_cfg = _parse_model(_load_yaml(_resolve_path(raw["model"], project_root)))

    training = TrainingConfig(
        epochs=int(raw["training"]["epochs"]),
        batch_size=int(raw["training"]["batch_size"]),
        learning_rate=float(raw["training"]["learning_rate"]),
        optimizer=raw["training"].get("optimizer", "adam"),
        seed=int(raw["training"].get("seed", 42)),
        num_workers=int(raw["training"].get("num_workers", 4)),
        callbacks=raw["training"].get("callbacks", {}) or {},
    )
    eval_cfg = EvaluationConfig(
        metrics=list(raw.get("evaluation", {}).get("metrics", [])),
        plots=list(raw.get("evaluation", {}).get("plots", [])),
    )

    if model_cfg.output.num_classes != dataset_cfg.num_classes:
        raise ValueError(
            f"model.output.num_classes ({model_cfg.output.num_classes}) "
            f"!= dataset.num_classes ({dataset_cfg.num_classes})"
        )

    return ExperimentConfig(
        name=exp["name"],
        description=exp.get("description", ""),
        mlflow_experiment_name=exp.get("mlflow_experiment", ""),
        tags=exp.get("tags") or {},
        dataset=dataset_cfg,
        features=features_cfg,
        model=model_cfg,
        training=training,
        evaluation=eval_cfg,
    )


def _build_experiment_from_inline(raw: dict) -> ExperimentConfig:
    """Build an ExperimentConfig from a training-time snapshot (flat YAML).

    Snapshots inline `dataset`/`features`/`model` blocks (see trainer's
    `_save_config_snapshot`), so there are no referenced paths to resolve.
    """
    dataset_cfg = _parse_dataset({"dataset": raw["dataset"]})
    features_cfg = _parse_features({"features": raw["features"]})
    model_cfg = _parse_model({"model": raw["model"]})

    training = TrainingConfig(
        epochs=int(raw["training"]["epochs"]),
        batch_size=int(raw["training"]["batch_size"]),
        learning_rate=float(raw["training"]["learning_rate"]),
        optimizer=raw["training"].get("optimizer", "adam"),
        seed=int(raw["training"].get("seed", 42)),
        num_workers=int(raw["training"].get("num_workers", 4)),
        callbacks=raw["training"].get("callbacks", {}) or {},
    )
    eval_cfg = EvaluationConfig(
        metrics=list(raw.get("evaluation", {}).get("metrics", [])),
        plots=list(raw.get("evaluation", {}).get("plots", [])),
    )

    return ExperimentConfig(
        name=raw["name"],
        description=raw.get("description", ""),
        mlflow_experiment_name=raw.get("mlflow_experiment", ""),
        tags=raw.get("tags") or {},
        dataset=dataset_cfg,
        features=features_cfg,
        model=model_cfg,
        training=training,
        evaluation=eval_cfg,
    )
