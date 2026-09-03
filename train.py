"""
train.py — Training entry point for the Sainsbury's WiFi CSI Sensing project.

Usage
-----
    python train.py                              # use default config.yaml
    python train.py --config config.yaml \\
                    --dataset uthar \\
                    --model resnet \\
                    --epochs 100 \\
                    --device cuda

The script expects two factory functions to be importable from the src package:
  - src.datasets.build_dataset(cfg) → (train_ds, val_ds, test_ds, class_names)
  - src.models.build_model(cfg)     → torch.nn.Module
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Project utilities — available once src/utils/ is in place
from src.utils import compute_metrics
from src.utils.metrics import EarlyStopping
from src.utils.visualise import plot_training_curves

# Dataset and model factories (implemented in src/datasets/ and src/models/)
from src.datasets import build_dataset  # type: ignore[import]
from src.models import build_model      # type: ignore[import]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CSI-based activity classifier."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override data.dataset from config (e.g. synthetic, uthar, entrance).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model.architecture from config (e.g. cnn, resnet, lstm, transformer).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs from config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to train on: cpu, cuda, mps. Auto-detected if omitted.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a YAML config file and return as a dict."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Run from the project root or pass --config <path>."
        )
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Merge CLI overrides into the loaded config dict (in-place)."""
    if args.dataset is not None:
        cfg["data"]["dataset"] = args.dataset
    if args.model is not None:
        cfg["model"]["architecture"] = args.model
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    return cfg


def resolve_device(requested: str | None) -> torch.device:
    """Pick the best available device if none is specified."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (slight speed cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Training / validation epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch.

    Returns
    -------
    (avg_loss, accuracy) as floats.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # Gradient clipping helps stability with LSTM / Transformer
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list, list]:
    """Run one evaluation epoch (no gradient computation).

    Returns
    -------
    (avg_loss, accuracy, all_preds, all_labels)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy, all_preds, all_labels


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    num_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    sched_name = cfg["training"].get("scheduler", "cosine").lower()
    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
    elif sched_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["training"].get("step_size", 10),
            gamma=cfg["training"].get("gamma", 0.5),
        )
    elif sched_name == "none":
        return None
    else:
        print(f"[train] Unknown scheduler '{sched_name}', using none.")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Config ---
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    dataset_name: str = cfg["data"]["dataset"]
    arch_name: str = cfg["model"]["architecture"]

    # Sync num_classes between data and model sections
    ds_key = dataset_name if dataset_name in cfg.get("classes", {}) else "synthetic"
    class_names: list[str] = cfg.get("classes", {}).get(ds_key, [])
    if class_names:
        cfg["model"]["num_classes"] = len(class_names)

    # --- Reproducibility ---
    seed: int = cfg["training"].get("seed", 42)
    set_seed(seed)

    # --- Device ---
    device = resolve_device(args.device)
    print(f"[train] Device: {device}")

    # --- Directories ---
    log_dir = Path(cfg["training"].get("log_dir", "./logs"))
    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "./checkpoints"))
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Datasets & DataLoaders ---
    print(f"[train] Building dataset: {dataset_name}")
    train_ds, val_ds, test_ds, class_names = build_dataset(cfg)

    num_workers = cfg["data"].get("num_workers", 4)
    batch_size = cfg["data"].get("batch_size", 64)

    # Pin memory only makes sense for CUDA
    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    print(
        f"[train] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} samples"
    )

    # --- Model ---
    print(f"[train] Building model: {arch_name}")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Trainable parameters: {n_params:,}")

    # --- Loss, optimiser, scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"].get("learning_rate", 1e-3),
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    num_epochs: int = cfg["training"].get("epochs", 50)
    scheduler = build_scheduler(optimizer, cfg, num_epochs)

    # --- Early stopping ---
    patience = cfg["training"].get("early_stopping_patience", 10)
    best_ckpt_path = ckpt_dir / f"best_{arch_name}_{dataset_name}.pt"
    stopper = EarlyStopping(patience=patience, mode="min", verbose=True)

    # --- Training loop ---
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    print(f"\n[train] Starting training for {num_epochs} epochs …\n")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, _, _ = evaluate_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"loss={t_loss:.4f} acc={t_acc:.4f} | "
            f"val_loss={v_loss:.4f} val_acc={v_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Check early stopping; save if best
        should_stop = stopper(v_loss)
        if stopper.is_best:
            stopper.save_checkpoint(model, str(best_ckpt_path))
        if should_stop:
            print(f"[train] Early stopping at epoch {epoch}.")
            break

    total_time = time.time() - start_time
    print(f"\n[train] Training complete in {total_time / 60:.1f} min.")

    # --- Plot training curves ---
    curves_path = log_dir / f"training_curves_{arch_name}_{dataset_name}.png"
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs, save_path=str(curves_path)
    )

    # --- Final test evaluation ---
    print("\n[train] Loading best checkpoint for test evaluation …")
    if best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    _, test_acc, test_preds, test_labels = evaluate_epoch(
        model, test_loader, criterion, device
    )
    metrics = compute_metrics(test_labels, test_preds, class_names)

    print("\n" + "=" * 60)
    print(f"TEST ACCURACY: {metrics['accuracy']:.4f}")
    print("=" * 60)
    print(metrics["report"])

    # --- Save results JSON ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "dataset": dataset_name,
        "model": arch_name,
        "epochs_trained": len(train_losses),
        "best_val_loss": float(min(val_losses)),
        "best_val_acc": float(max(val_accs)),
        "test_accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class": metrics["per_class"],
        "config": cfg,
    }
    results_path = log_dir / f"results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[train] Results saved → {results_path}")


if __name__ == "__main__":
    main()
