# app/app.py
"""Gradio demo for the AudioMNIST digit classifier.

Two modes (Tabs):

  1. Push-to-talk — record/upload a single clip, get top-3 predictions.
     Uses the project's `load_audio` so preprocessing exactly matches
     training (16 kHz mono, center-pad/crop to 1.0 s, log-mel, normalize).

  2. Always listening — streaming mic + simple energy-based VAD with
     adaptive noise floor. Auto-detects voice onset/offset, captures the
     utterance (with pre-roll), then runs the same preprocessing on the
     in-memory PCM array and fires a prediction.

The checkpoint and its sibling config_snapshot.yaml fully determine the
model + preprocessing — same contract as pipelines/infer.py.
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
import librosa  # noqa: E402
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


# ── Resolve checkpoint + config ────────────────────────────────────────────────

DEFAULT_CKPT = REPO_ROOT / "data" / "artifacts" / "cnn_classifier_v1" / "best_model.pt"
CKPT_PATH = Path(os.environ.get("AUDIO_MNIST_CHECKPOINT", DEFAULT_CKPT)).resolve()
SNAPSHOT_PATH = CKPT_PATH.parent / "config_snapshot.yaml"

if not CKPT_PATH.is_file():
    raise FileNotFoundError(
        f"Checkpoint not found: {CKPT_PATH}\nSet AUDIO_MNIST_CHECKPOINT to override."
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
state_dict = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state_dict["model_state"])
model.eval()

CLASS_NAMES: list[str] = list(cfg.dataset.class_names)
TARGET_SR: int = int(cfg.features.audio.sample_rate)
FIXED_DUR_SEC: float | None = cfg.features.audio.fixed_duration_sec

logger.info("Checkpoint: %s", CKPT_PATH)
logger.info("Snapshot:   %s", SNAPSHOT_PATH)
logger.info("Device:     %s", device)
logger.info("Classes:    %s", CLASS_NAMES)
logger.info("Target SR:  %d Hz | fixed duration: %s s", TARGET_SR, FIXED_DUR_SEC)


# ── Shared inference helpers ───────────────────────────────────────────────────


def _spec_to_probs(spec: np.ndarray) -> dict[str, float]:
    x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T, F)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


def _center_to_fixed_length(pcm: np.ndarray, sr: int) -> np.ndarray:
    """Center-pad/crop to FIXED_DUR_SEC. Mirrors src.features.audio_loader exactly."""
    if FIXED_DUR_SEC is None:
        return pcm
    target = int(round(FIXED_DUR_SEC * sr))
    if len(pcm) < target:
        pad = target - len(pcm)
        left = pad // 2
        right = pad - left
        return np.pad(pcm, (left, right), mode="constant")
    if len(pcm) > target:
        start = (len(pcm) - target) // 2
        return pcm[start : start + target]
    return pcm


def predict_filepath(audio_path: str | None) -> dict[str, float]:
    """Phase 1: predict from a filepath given by Gradio (mic or upload)."""
    if not audio_path:
        return {}
    y, _ = load_audio(audio_path, cfg.features.audio)
    spec = compute_spectrogram(y, cfg.features)
    return _spec_to_probs(spec)


def predict_pcm(pcm: np.ndarray, native_sr: int) -> dict[str, float]:
    """Phase 2: predict from an in-memory PCM array (any sample rate)."""
    if native_sr != TARGET_SR:
        pcm = librosa.resample(pcm.astype(np.float32), orig_sr=native_sr, target_sr=TARGET_SR)
    pcm = _center_to_fixed_length(pcm, TARGET_SR)
    spec = compute_spectrogram(pcm, cfg.features)
    return _spec_to_probs(spec)


# ── Phase 2: streaming + energy VAD ────────────────────────────────────────────

FRAME_MS = 30                # VAD frame size
TAIL_MS = 200                # pre-roll captured before voice onset
CALIB_FRAMES = 30            # ~900 ms of initial calibration
ONSET_THRESH_MULT = 4.0      # energy > noise_floor * this → frame is voiced
OFFSET_THRESH_MULT = 2.5     # hysteresis: lower threshold once in voice
ONSET_FRAMES = 3             # ~90 ms voiced → confirm onset
OFFSET_FRAMES = 10           # ~300 ms unvoiced → confirm offset
COOLDOWN_FRAMES = 17         # ~500 ms suppression after a prediction
HISTORY_KEEP = 5
NOISE_EMA_ALPHA = 0.03       # slow update of noise floor during silence


def _initial_stream_state() -> dict:
    return {
        "pending": np.zeros(0, dtype=np.float32),  # samples not yet framed
        "tail": np.zeros(0, dtype=np.float32),     # pre-roll buffer (silence only)
        "voice_pcm": np.zeros(0, dtype=np.float32),
        "vad_state": "calibrating",
        "noise_floor": 1e-4,
        "calib_frames_seen": 0,
        "consec_voiced": 0,
        "consec_unvoiced": 0,
        "cooldown_frames": 0,
        "native_sr": 0,
        "last_prediction": {},
        "history": [],                              # list[(label, prob)]
        "status": "Calibrating noise floor — stay quiet for ~1 second…",
    }


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def _format_history(history: list[tuple[str, float]]) -> str:
    if not history:
        return "<i style='color:#888'>No predictions yet — speak a digit 0–9.</i>"
    items = list(reversed(history[-HISTORY_KEEP:]))
    rows = "".join(
        "<div style='display:flex;gap:1.5em;padding:6px 0;border-bottom:1px solid #eee;'>"
        f"<span style='font-size:1.4em;font-weight:600;width:1.5em;'>{label}</span>"
        f"<span style='color:#666;align-self:center;'>{prob:.0%}</span>"
        "</div>"
        for label, prob in items
    )
    return f"<div>{rows}</div>"


def _process_frame(frame: np.ndarray, state: dict) -> None:
    """Single-frame VAD state machine."""
    energy = _rms(frame)
    vad = state["vad_state"]

    # Pre-roll buffer only fills during silence/calibration.
    if vad in ("silence", "calibrating"):
        tail_max = int(TAIL_MS / 1000 * state["native_sr"])
        state["tail"] = np.concatenate([state["tail"], frame])[-tail_max:]

    if vad == "calibrating":
        # Track max during calibration so we don't underestimate ambient noise.
        state["noise_floor"] = max(state["noise_floor"], energy)
        state["calib_frames_seen"] += 1
        if state["calib_frames_seen"] >= CALIB_FRAMES:
            state["vad_state"] = "silence"
            state["status"] = "Listening — say a digit"
        return

    if vad == "silence":
        is_voiced = energy > state["noise_floor"] * ONSET_THRESH_MULT
        if is_voiced:
            state["consec_voiced"] += 1
            if state["consec_voiced"] >= ONSET_FRAMES:
                # ONSET — seed voice region with pre-roll (already includes
                # the frames that triggered detection, since tail was being
                # filled during silence).
                state["voice_pcm"] = state["tail"].copy()
                state["vad_state"] = "voice"
                state["consec_voiced"] = 0
                state["consec_unvoiced"] = 0
                state["status"] = "Voice detected…"
        else:
            state["consec_voiced"] = 0
            # Slow EMA update of noise floor when the room is genuinely quiet.
            state["noise_floor"] = (
                (1 - NOISE_EMA_ALPHA) * state["noise_floor"] + NOISE_EMA_ALPHA * energy
            )
        return

    if vad == "voice":
        state["voice_pcm"] = np.concatenate([state["voice_pcm"], frame])
        is_unvoiced = energy < state["noise_floor"] * OFFSET_THRESH_MULT
        if is_unvoiced:
            state["consec_unvoiced"] += 1
            if state["consec_unvoiced"] >= OFFSET_FRAMES:
                clip = state["voice_pcm"]
                # Reset capture state before the (slow-ish) inference call.
                state["voice_pcm"] = np.zeros(0, dtype=np.float32)
                state["tail"] = np.zeros(0, dtype=np.float32)
                state["vad_state"] = "cooldown"
                state["cooldown_frames"] = COOLDOWN_FRAMES
                state["consec_unvoiced"] = 0
                state["status"] = "Predicting…"
                try:
                    pred = predict_pcm(clip, state["native_sr"])
                    state["last_prediction"] = pred
                    top_label = max(pred, key=pred.get)
                    state["history"].append((top_label, pred[top_label]))
                except Exception:
                    logger.exception("Inference failed on captured utterance")
        else:
            state["consec_unvoiced"] = 0
        return

    if vad == "cooldown":
        state["cooldown_frames"] -= 1
        if state["cooldown_frames"] <= 0:
            state["vad_state"] = "silence"
            state["status"] = "Listening — say a digit"


def stream_step(audio_chunk, state):
    """Called by Gradio on every streamed audio chunk."""
    if state is None:
        state = _initial_stream_state()

    if audio_chunk is None:
        return (
            state["last_prediction"],
            state["status"],
            _format_history(state["history"]),
            state,
        )

    sr, x = audio_chunk
    if x is None or len(x) == 0:
        return (
            state["last_prediction"],
            state["status"],
            _format_history(state["history"]),
            state,
        )

    # Normalize to mono float32 in [-1, 1].
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = x.astype(np.float32)

    state["native_sr"] = sr

    # Drain pending+new samples into fixed-size frames.
    pending = np.concatenate([state["pending"], x])
    frame_len = int(FRAME_MS / 1000 * sr)
    if frame_len > 0:
        n_frames = len(pending) // frame_len
        for i in range(n_frames):
            _process_frame(pending[i * frame_len : (i + 1) * frame_len], state)
        state["pending"] = pending[n_frames * frame_len :]
    else:
        state["pending"] = pending

    return (
        state["last_prediction"],
        state["status"],
        _format_history(state["history"]),
        state,
    )


# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="AudioMNIST — spoken digit classifier") as demo:
    gr.Markdown("# AudioMNIST — spoken digit classifier")

    with gr.Tab("Push to talk"):
        gr.Markdown(
            "Tap the mic, say a digit **0–9**, then stop. "
            "Or upload an audio file. Top-3 predictions appear on the right."
        )
        with gr.Row():
            ptt_audio = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Audio (≤ 1 s recommended)",
            )
            ptt_label = gr.Label(num_top_classes=3, label="Predictions")
        ptt_audio.change(predict_filepath, inputs=ptt_audio, outputs=ptt_label)
        ptt_audio.stop_recording(predict_filepath, inputs=ptt_audio, outputs=ptt_label)

    with gr.Tab("Always listening"):
        gr.Markdown(
            "Click the mic to start streaming. Stay quiet briefly while we "
            "calibrate the noise floor (~1 s), then speak digits — each "
            "utterance is auto-detected, captured with a short pre-roll, and "
            "classified."
        )
        stream_state = gr.State(None)
        with gr.Row():
            with gr.Column():
                live_audio = gr.Audio(
                    sources="microphone",
                    streaming=True,
                    type="numpy",
                    label="Live mic",
                )
                status_box = gr.Textbox(
                    label="Status",
                    value="Click the mic above to start.",
                    interactive=False,
                )
            with gr.Column():
                live_label = gr.Label(num_top_classes=3, label="Current prediction")
                history_html = gr.HTML(
                    value="<i style='color:#888'>No predictions yet.</i>",
                    label="Recent",
                )

        live_audio.stream(
            stream_step,
            inputs=[live_audio, stream_state],
            outputs=[live_label, status_box, history_html, stream_state],
        )


if __name__ == "__main__":
    demo.launch()
