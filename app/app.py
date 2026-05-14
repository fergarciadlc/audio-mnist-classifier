# app/app.py
"""Gradio demo for the AudioMNIST digit classifier.

Reuses the project's preprocessing (src.features.audio_loader,
src.features.spectrogram) so what the demo feeds the model is identical
to what training saw. The checkpoint and its sibling config_snapshot.yaml
fully determine the model + preprocessing — same contract as
pipelines/infer.py.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Make `src/...` importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import load_experiment  # noqa: E402
from src.features.audio_loader import load_audio  # noqa: E402
from src.features.spectrogram import compute_spectrogram  # noqa: E402
from src.models.cnn import build_model  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audio-mnist-demo")


DEFAULT_CKPT = REPO_ROOT / "data" / "artifacts" / "cnn_classifier_v1" / "best_model.pt"
CKPT_PATH = Path(os.environ.get("AUDIO_MNIST_CHECKPOINT", DEFAULT_CKPT)).resolve()
SNAPSHOT_PATH = CKPT_PATH.parent / "config_snapshot.yaml"

if not CKPT_PATH.is_file():
    raise FileNotFoundError(
        f"Checkpoint not found: {CKPT_PATH}\n"
        f"Set AUDIO_MNIST_CHECKPOINT to override."
    )
if not SNAPSHOT_PATH.is_file():
    raise FileNotFoundError(f"config_snapshot.yaml not found next to checkpoint: {SNAPSHOT_PATH}")


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


cfg = load_experiment(SNAPSHOT_PATH)
device = _select_device()

model = build_model(cfg.model, cfg.input_shape).to(device)
state = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state["model_state"])
model.eval()

CLASS_NAMES: list[str] = list(cfg.dataset.class_names)

logger.info("Checkpoint: %s", CKPT_PATH)
logger.info("Snapshot:   %s", SNAPSHOT_PATH)
logger.info("Device:     %s", device)
logger.info("Classes:    %s", CLASS_NAMES)


def predict(audio_path: str | None) -> dict[str, float]:
    """Run inference on a microphone or uploaded audio file.

    Gradio passes a filepath (temp WAV) when `type='filepath'`. We hand it
    straight to the same `load_audio` the training pipeline uses, so
    resampling, mono conversion, and center-pad/crop to 1.0 s all match.
    """
    if not audio_path:
        return {}

    y, _ = load_audio(audio_path, cfg.features.audio)
    spec = compute_spectrogram(y, cfg.features)

    x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T, F)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


with gr.Blocks(title="AudioMNIST — spoken digit classifier") as demo:
    gr.Markdown(
        "# AudioMNIST — spoken digit classifier\n"
        "Tap the mic, say a digit between **0 and 9**, then stop. "
        "Or upload a short audio file. Top-3 predictions appear on the right."
    )
    with gr.Row():
        audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Audio (≤ 1 s recommended)",
        )
        label = gr.Label(num_top_classes=3, label="Predictions")
    audio.change(predict, inputs=audio, outputs=label)
    audio.stop_recording(predict, inputs=audio, outputs=label)


if __name__ == "__main__":
    demo.launch()
