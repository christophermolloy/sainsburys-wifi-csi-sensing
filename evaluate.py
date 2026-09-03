"""
evaluate.py — Evaluation entry point for the Sainsbury's WiFi CSI Sensing project.

Usage
-----
    python evaluate.py \\
        --checkpoint checkpoints/best_cnn_synthetic.pt \\
        --config config.yaml \\
        --dataset uthar \\
        --device cpu \\
        --output-dir logs/eval_20260903/

The script:
  1. Loads the saved model checkpoint.
  2. Runs inference on the test split.
  3. Prints a full sklearn classification report.
  4. Saves a confusion matrix plot.
  5. Shows per-class example CSI heatmaps with their predictions.
  6. Reports throughput (samples / second).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.utils import compute_metrics, plot_confusion_matrix
from src.utils.metrics import ConfusionMatrixTracker
from src.utils.visualise import plot_class_csi_comparison, plot_csi_heatmap
from src.datasets import build_dataset  # type: ignore[import]
from src.models import build_model      # type: ignore[import]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved CSI model checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the saved .pt model state-dict file.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override data.dataset from config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: cpu | cuda | mps. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots and reports. Defaults to logs/eval_<timestamp>/.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for inference (default: use config).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collect_class_examples(
    test_ds,
    class_names: list[str],
    n_per_class: int = 1,
) -> dict[str, np.ndarray]:
    """Return one representative CSI sample per class (averaged over n_per_class).

    Parameters
    ----------
    test_ds : torch Dataset
        Each item is (tensor, label_int).
    class_names : list of str
    n_per_class : int
        Number of samples to average per class for a cleaner representative view.

    Returns
    -------
    dict {class_name -> np.ndarray of shape (T, S)}
    """
    buckets: dict[int, list[np.ndarray]] = {i: [] for i in range(len(class_names))}

    for sample, label in test_ds:
        lbl = int(label)
        if len(buckets[lbl]) < n_per_class:
            arr = np.squeeze(np.asarray(sample, dtype=np.float32))
            if arr.ndim == 2:
                buckets[lbl].append(arr)
            # If still has a channel dim (C, T, S), take first channel
            elif arr.ndim == 3:
                buckets[lbl].append(arr[0])

        # Stop early if all classes are full
        if all(len(v) >= n_per_class for v in buckets.values()):
            break

    samples_by_class: dict[str, np.ndarray] = {}
    for i, name in enumerate(class_names):
        if buckets[i]:
            samples_by_class[name] = np.mean(buckets[i], axis=0)
    return samples_by_class


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], float]:
    """Run inference over the full loader.

    Returns
    -------
    (all_preds, all_labels, throughput_samples_per_sec)
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_samples = 0

    start = time.perf_counter()
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        total_samples += inputs.size(0)
    elapsed = time.perf_counter() - start

    throughput = total_samples / elapsed if elapsed > 0 else float("inf")
    return all_preds, all_labels, throughput


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Config ---
    cfg = load_config(args.config)
    if args.dataset is not None:
        cfg["data"]["dataset"] = args.dataset
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size

    dataset_name: str = cfg["data"]["dataset"]
    arch_name: str = cfg["model"]["architecture"]

    # Resolve class names
    ds_key = dataset_name if dataset_name in cfg.get("classes", {}) else "synthetic"
    class_names: list[str] = cfg.get("classes", {}).get(ds_key, [])
    if class_names:
        cfg["model"]["num_classes"] = len(class_names)

    # --- Device ---
    device = resolve_device(args.device)
    print(f"[evaluate] Device: {device}")

    # --- Output directory ---
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(cfg["training"].get("log_dir", "./logs")) / f"eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[evaluate] Outputs → {out_dir}")

    # --- Dataset ---
    print(f"[evaluate] Loading dataset: {dataset_name}")
    _, _, test_ds, class_names = build_dataset(cfg)
    print(f"[evaluate] Test samples: {len(test_ds)}")

    batch_size = cfg["data"].get("batch_size", 64)
    num_workers = cfg["data"].get("num_workers", 4)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --- Model ---
    print(f"[evaluate] Building model: {arch_name}")
    model = build_model(cfg).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[evaluate] Checkpoint loaded from {ckpt_path}")

    # --- Inference ---
    print("[evaluate] Running inference …")
    all_preds, all_labels, throughput = run_inference(model, test_loader, device)
    print(f"[evaluate] Throughput: {throughput:.0f} samples/sec")

    # --- Metrics ---
    metrics = compute_metrics(all_labels, all_preds, class_names)
    print("\n" + "=" * 60)
    print(f"TEST ACCURACY : {metrics['accuracy']:.4f}")
    print(f"MACRO F1      : {metrics['macro_f1']:.4f}")
    print(f"WEIGHTED F1   : {metrics['weighted_f1']:.4f}")
    print("=" * 60)
    print(metrics["report"])

    # Save text report
    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Checkpoint : {ckpt_path}\n")
        f.write(f"Dataset    : {dataset_name}\n")
        f.write(f"Model      : {arch_name}\n")
        f.write(f"Device     : {device}\n")
        f.write(f"Throughput : {throughput:.0f} samples/sec\n\n")
        f.write(metrics["report"])
    print(f"[evaluate] Report saved → {report_path}")

    # --- Confusion matrix ---
    tracker = ConfusionMatrixTracker(num_classes=len(class_names))
    tracker.add_batch(all_preds, all_labels)
    cm = tracker.compute()

    cm_path = str(out_dir / "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, save_path=cm_path, title="Test Confusion Matrix")

    cm_norm_path = str(out_dir / "confusion_matrix_normalised.png")
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=cm_norm_path,
        title="Test Confusion Matrix (Normalised)",
        normalize=True,
    )

    # --- Per-class CSI example heatmaps ---
    print("[evaluate] Collecting per-class CSI examples …")
    samples_by_class = _collect_class_examples(test_ds, class_names, n_per_class=5)

    # Individual heatmaps
    for cls_name, data in samples_by_class.items():
        safe_name = cls_name.replace(" ", "_").replace("/", "-")
        hm_path = str(out_dir / f"csi_example_{safe_name}.png")
        plot_csi_heatmap(
            data,
            title=f"CSI Example — {cls_name}",
            save_path=hm_path,
        )

    # Comparison grid
    if len(samples_by_class) > 1:
        comparison_path = str(out_dir / "csi_class_comparison.png")
        plot_class_csi_comparison(
            samples_by_class,
            class_names,
            save_path=comparison_path,
        )

    # --- Summary ---
    print("\n[evaluate] Summary")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Macro F1  : {metrics['macro_f1']:.4f}")
    print(f"  Throughput: {throughput:.0f} samples/sec")
    print(f"  Outputs   : {out_dir}/")


if __name__ == "__main__":
    main()
