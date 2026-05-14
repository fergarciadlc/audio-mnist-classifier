# pipelines/infer.py
"""
Stage 5: Inference

Run a trained model on a single audio file or a directory of audio files,
print top-k predictions, and optionally write a CSV.

The experiment config is always resolved from `config_snapshot.yaml`
sitting next to the checkpoint (written by training).

Usage:
    python -m pipelines.infer \
        --checkpoint data/artifacts/cnn_classifier_v1/best_model.pt \
        --input path/to/audio.wav
    python -m pipelines.infer \
        --checkpoint data/artifacts/cnn_classifier_v1/best_model.pt \
        --input path/to/folder --top-k 3 --output preds.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import load_experiment
from src.features.audio_loader import load_audio
from src.features.spectrogram import compute_spectrogram
from src.models.cnn import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5: Inference on audio file(s)")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Trained model checkpoint (.pt). The config is auto-resolved "
             "from config_snapshot.yaml next to it.",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Audio file OR directory of audio files",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k predictions to display")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional CSV path for per-file predictions",
    )
    return parser.parse_args(argv)


def _resolve_snapshot(ckpt_path: Path) -> Path:
    snapshot = ckpt_path.parent / "config_snapshot.yaml"
    if not snapshot.is_file():
        raise FileNotFoundError(
            f"Expected config_snapshot.yaml next to checkpoint: {snapshot}"
        )
    return snapshot


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _gather_audio_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in AUDIO_EXTS:
            raise ValueError(f"Unsupported audio extension: {input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            f for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS
        )
        if not files:
            raise ValueError(f"No audio files in {input_path} (extensions: {sorted(AUDIO_EXTS)})")
        return files
    raise FileNotFoundError(input_path)


def _file_to_tensor(path: Path, cfg) -> torch.Tensor:
    """Load a single audio file and return a (1, n_frames, n_freq) float32 tensor."""
    y, _ = load_audio(path, cfg.features.audio)
    spec = compute_spectrogram(y, cfg.features)
    return torch.from_numpy(spec).unsqueeze(0)


def _write_csv(
    out_path: Path,
    files: list[Path],
    top_idx: list[list[int]],
    top_vals: list[list[float]],
    class_names: list[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    k = len(top_idx[0]) if top_idx else 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["file", "prediction", "confidence"]
        for i in range(1, k):
            header += [f"top{i + 1}_label", f"top{i + 1}_prob"]
        writer.writerow(header)
        for fp, idx, val in zip(files, top_idx, top_vals):
            row = [str(fp), class_names[idx[0]], f"{val[0]:.6f}"]
            for i in range(1, k):
                row += [class_names[idx[i]], f"{val[i]:.6f}"]
            writer.writerow(row)
    logger.info("Wrote predictions CSV: %s", out_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    snapshot_path = _resolve_snapshot(ckpt_path)
    cfg = load_experiment(snapshot_path)

    logger.info("=" * 60)
    logger.info("STAGE 5: INFERENCE")
    logger.info("=" * 60)
    logger.info("Experiment: %s", cfg.name)
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Config:     %s", snapshot_path)

    files = _gather_audio_files(Path(args.input))
    logger.info("Files: %d", len(files))

    device = _select_device()
    logger.info("Device: %s", device)

    # Model
    model = build_model(cfg.model, cfg.input_shape).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    # Feature extraction (sequential — librosa isn't safely parallel here)
    tensors: list[torch.Tensor] = []
    for fp in tqdm(files, desc="extracting", leave=False):
        try:
            tensors.append(_file_to_tensor(fp, cfg))
        except Exception as exc:
            logger.warning("Skipping %s: %s", fp, exc)
            tensors.append(None)  # placeholder, filtered next

    valid_pairs = [(fp, t) for fp, t in zip(files, tensors) if t is not None]
    if not valid_pairs:
        raise RuntimeError("No audio files could be loaded.")
    files = [fp for fp, _ in valid_pairs]
    x_all = torch.stack([t for _, t in valid_pairs])  # (N, 1, n_frames, n_freq)

    # Batched inference
    k = max(1, min(args.top_k, cfg.dataset.num_classes))
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(x_all), args.batch_size), desc="predicting", leave=False):
            batch = x_all[i : i + args.batch_size].to(device)
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()
            all_probs.append(probs)
    probs_arr = np.concatenate(all_probs, axis=0)

    # Top-k per file
    top_idx_arr = np.argsort(-probs_arr, axis=1)[:, :k]
    top_vals_arr = np.take_along_axis(probs_arr, top_idx_arr, axis=1)
    top_idx = top_idx_arr.tolist()
    top_vals = top_vals_arr.tolist()

    # Print
    class_names = cfg.dataset.class_names
    name_w = min(60, max(len(fp.name) for fp in files))
    logger.info("-" * 60)
    logger.info("PREDICTIONS")
    logger.info("-" * 60)
    for fp, idx, val in zip(files, top_idx, top_vals):
        topk_str = ", ".join(f"{class_names[c]}:{p:.3f}" for c, p in zip(idx, val))
        print(f"  {fp.name.ljust(name_w)}  pred={class_names[idx[0]]}  top-{k}=[{topk_str}]")

    if args.output:
        _write_csv(Path(args.output), files, top_idx, top_vals, class_names)

    logger.info("Done.")


if __name__ == "__main__":
    main()
