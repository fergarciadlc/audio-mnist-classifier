# src/training/trainer.py
"""
PyTorch training loop for AudioMNIST.

Builds the model from config, trains with CrossEntropy, validates each epoch,
saves best + last checkpoints, and logs per-epoch metrics to MLflow when the
tracking URI is set.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.optim import Optimizer

from src.config import ExperimentConfig
from src.mlflow_logger import MLflowLogger, resolve_mlflow_experiment_name
from src.models.cnn import build_model
from src.training.data_loader import build_dataloaders

logger = logging.getLogger(__name__)


# ── Device / optimizer helpers ─────────────────────────────────────────────────


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_optimizer(name: str, params, learning_rate: float) -> Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=learning_rate)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate)
    if name == "sgd":
        return torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=learning_rate)
    raise ValueError(f"Unknown optimizer: {name}")


# ── Train / validate loops ─────────────────────────────────────────────────────


def _train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_n = y.size(0)
        total_loss += loss.item() * batch_n
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += batch_n

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        batch_n = y.size(0)
        total_loss += loss.item() * batch_n
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += batch_n
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


# ── Artifact helpers ───────────────────────────────────────────────────────────


def _save_training_history(history: dict[str, list[float]], output_dir: Path) -> Path:
    path = output_dir / "training_history.json"
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved training history: %s", path)
    return path


def _save_model_summary(model: nn.Module, input_shape: tuple[int, int, int], output_dir: Path) -> Path:
    """Write a text summary of the model: structure + parameter counts.

    If torchinfo is available we use it for a rich Keras-style summary;
    otherwise we fall back to the plain repr + a manual parameter count.
    """
    lines: list[str] = []
    try:
        from torchinfo import summary

        n_frames, n_freq, _ = input_shape
        info = summary(
            model,
            input_size=(1, 1, n_frames, n_freq),
            col_names=("input_size", "output_size", "num_params"),
            verbose=0,
        )
        lines.append(str(info))
    except Exception:
        buf = io.StringIO()
        print(model, file=buf)
        lines.append(buf.getvalue())
        n_params = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lines.append(f"\nTotal params:     {n_params:,}")
        lines.append(f"Trainable params: {n_train:,}")

    path = output_dir / "model_summary.txt"
    path.write_text("\n".join(lines))
    logger.info("Saved model summary: %s", path)
    return path


def _save_config_snapshot(cfg: ExperimentConfig, output_dir: Path) -> Path:
    """Freeze the full experiment config for reproducibility."""
    path = output_dir / "config_snapshot.yaml"
    snapshot = {
        "name": cfg.name,
        "description": cfg.description,
        "mlflow_experiment": cfg.mlflow_experiment_name,
        "tags": dict(cfg.tags),
        "dataset": asdict(cfg.dataset),
        "features": asdict(cfg.features),
        "model": asdict(cfg.model),
        "training": asdict(cfg.training),
        "evaluation": asdict(cfg.evaluation),
    }
    with open(path, "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved config snapshot: %s", path)
    return path


# ── Public entry point ─────────────────────────────────────────────────────────


def train(
    cfg: ExperimentConfig,
    feature_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the full training pipeline.

    Saves under `output_dir / cfg.name`:
        best_model.pt, last_model.pt, model_summary.txt,
        config_snapshot.yaml, training_history.json, mlflow_run.json
    """
    torch.manual_seed(cfg.training.seed)

    device = _select_device()
    logger.info("Device: %s", device)

    input_shape = cfg.input_shape
    logger.info("Input shape (n_frames, n_freq, ch): %s", input_shape)

    # Data
    train_loader, val_loader, _test_loader, _ = build_dataloaders(
        feature_manifest_path=feature_manifest_path,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = build_model(cfg.model, input_shape).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(cfg.training.optimizer, model.parameters(), cfg.training.learning_rate)

    # LR scheduler from cfg.training.callbacks.reduce_lr (optional)
    cb_cfg = cfg.training.callbacks or {}
    scheduler = None
    if "reduce_lr" in cb_cfg:
        rl = cb_cfg["reduce_lr"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=rl.get("mode", "min"),
            factor=float(rl.get("factor", 0.5)),
            patience=int(rl.get("patience", 3)),
        )

    # Early stopping config (default: monitor val_accuracy max)
    es = cb_cfg.get("early_stopping") or {}
    es_monitor = es.get("monitor", "val_accuracy")
    es_mode = es.get("mode", "max")
    es_patience = int(es.get("patience", 10))

    # Output dir
    checkpoint_dir = output_dir / cfg.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pt"
    last_path = checkpoint_dir / "last_model.pt"

    # MLflow
    mlflow_experiment = resolve_mlflow_experiment_name(cfg.mlflow_experiment_name)
    logger.info("MLflow experiment: %s", mlflow_experiment)
    mlf = MLflowLogger(experiment_name=mlflow_experiment)
    mlf.start_run(run_name=cfg.name)
    mlf.log_params(cfg.to_flat_dict())
    if cfg.tags:
        mlf.log_tags(cfg.tags)

    # History
    history: dict[str, list[float]] = {
        "train_loss": [], "train_accuracy": [],
        "val_loss":   [], "val_accuracy":   [],
        "lr":         [],
    }

    best_metric = -float("inf") if es_mode == "max" else float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = _validate(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["lr"].append(lr_now)
        epochs_run = epoch

        logger.info(
            "Epoch %3d/%d  train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f | lr=%.2e",
            epoch, cfg.training.epochs,
            train_loss, train_acc, val_loss, val_acc, lr_now,
        )

        mlf.log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "lr": lr_now,
        }, step=epoch)

        # Scheduler step
        if scheduler is not None:
            sched_monitor = (cb_cfg.get("reduce_lr") or {}).get("monitor", "val_loss")
            sched_value = val_acc if sched_monitor == "val_accuracy" else val_loss
            scheduler.step(sched_value)

        # Checkpointing
        current = val_acc if es_monitor == "val_accuracy" else val_loss
        is_better = (current > best_metric) if es_mode == "max" else (current < best_metric)
        if is_better:
            best_metric = current
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "metric": best_metric,
                    "monitor": es_monitor,
                },
                best_path,
            )
            logger.info("  ✓ new best %s=%.4f — saved %s", es_monitor, best_metric, best_path.name)
        else:
            epochs_without_improvement += 1

        torch.save({"model_state": model.state_dict(), "epoch": epoch}, last_path)

        if epochs_without_improvement >= es_patience:
            logger.info(
                "Early stopping: no improvement in %s for %d epochs (best at epoch %d)",
                es_monitor, es_patience, best_epoch,
            )
            break

    # Persist artifacts
    history_path = _save_training_history(history, checkpoint_dir)
    summary_path = _save_model_summary(model, input_shape, checkpoint_dir)
    config_path = _save_config_snapshot(cfg, checkpoint_dir)

    # Re-load best weights so the in-memory model logged to MLflow is the best.
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state"])

    mlf.log_artifact(best_path)
    mlf.log_artifact(summary_path)
    mlf.log_artifact(config_path)
    mlf.log_artifact(history_path)

    if mlf.active:
        try:
            # input_example shape matches Dataset output (single sample, no batch dim)
            example_x, _ = next(iter(val_loader))
            example_np = example_x[:1].cpu().numpy()
            mlf.log_pytorch_model(
                model.cpu(),
                artifact_path="model",
                registered_model_name=cfg.model.name,
                description=cfg.model.description,
                tags={str(k): str(v) for k, v in cfg.model.tags.items()} if cfg.model.tags else None,
                input_example=example_np,
            )
            model.to(device)
        except Exception:
            logger.warning("Failed to log model to MLflow", exc_info=True)

    mlf.save_run_meta(checkpoint_dir)
    mlf.end_run()

    final_train_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else float("nan")
    final_val_acc = history["val_accuracy"][-1] if history["val_accuracy"] else float("nan")

    return {
        "checkpoint_path": str(best_path),
        "last_checkpoint_path": str(last_path),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_val_accuracy": final_val_acc,
    }
