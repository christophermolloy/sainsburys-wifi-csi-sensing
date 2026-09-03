"""
visualise.py — Visualisation utilities for CSI sensing experiments.

Sainsbury's WiFi CSI Sensing project.
All functions follow a consistent save_path convention:
  - save_path=None  → plt.show() is called
  - save_path given → figure is saved to that path and plt.close() is called
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

_STYLE = "seaborn-v0_8-whitegrid"

# Colour palette used consistently across plots
_CLASS_PALETTE = [
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#F44336",  # red
    "#00BCD4",  # cyan
    "#8BC34A",  # light green
]


def _get_class_color(idx: int) -> str:
    return _CLASS_PALETTE[idx % len(_CLASS_PALETTE)]


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save figure to file or display it, then close."""
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[visualise] Saved → {save_path}")
    else:
        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# 1. plot_training_curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
) -> None:
    """Plot loss and accuracy curves over training epochs.

    The epoch with the best validation accuracy is highlighted with a vertical
    dashed line on both sub-plots.

    Parameters
    ----------
    train_losses, val_losses : list of float
        Per-epoch training and validation losses.
    train_accs, val_accs : list of float
        Per-epoch training and validation accuracies (0–1 range or 0–100, auto-detected).
    save_path : str or None
        If given, save the figure here; otherwise call plt.show().
    """
    with plt.style.context(_STYLE):
        epochs = range(1, len(train_losses) + 1)

        # Normalise accuracy to 0–100 for display
        scale = 100.0 if max(train_accs + val_accs) <= 1.0 else 1.0
        train_accs_disp = [a * scale for a in train_accs]
        val_accs_disp = [a * scale for a in val_accs]

        best_epoch = int(np.argmax(val_accs_disp)) + 1  # 1-indexed

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

        # --- Loss panel ---
        ax_loss.plot(epochs, train_losses, label="Train Loss", color="#2196F3", linewidth=2)
        ax_loss.plot(
            epochs, val_losses, label="Val Loss", color="#FF5722", linewidth=2, linestyle="--"
        )
        ax_loss.axvline(
            best_epoch, color="gray", linestyle=":", linewidth=1.5,
            label=f"Best epoch ({best_epoch})"
        )
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss")
        ax_loss.legend()
        ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # --- Accuracy panel ---
        ax_acc.plot(
            epochs, train_accs_disp, label="Train Acc", color="#4CAF50", linewidth=2
        )
        ax_acc.plot(
            epochs, val_accs_disp, label="Val Acc", color="#FF9800", linewidth=2, linestyle="--"
        )
        ax_acc.axvline(
            best_epoch, color="gray", linestyle=":", linewidth=1.5,
            label=f"Best epoch ({best_epoch})"
        )
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_title("Accuracy")
        ax_acc.legend()
        ax_acc.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax_acc.set_ylim(0, 105)

        # Annotate best val accuracy
        best_val_acc = val_accs_disp[best_epoch - 1]
        ax_acc.annotate(
            f"{best_val_acc:.1f}%",
            xy=(best_epoch, best_val_acc),
            xytext=(best_epoch + 0.5, best_val_acc - 5),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9,
        )

        plt.tight_layout()
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 2. plot_confusion_matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    """Plot a confusion matrix as an annotated seaborn heatmap.

    Each cell shows the raw count and the row-normalised percentage.

    Parameters
    ----------
    cm : np.ndarray, shape (n_classes, n_classes)
        Confusion matrix from sklearn or ConfusionMatrixTracker.
    class_names : sequence of str
        Labels for each class, in the same order as cm rows/columns.
    save_path : str or None
    title : str
    normalize : bool
        If True, normalise rows to sum to 1 before plotting.
    """
    cm_raw = cm.copy()
    with np.errstate(all="ignore"):
        cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    display_cm = cm_norm if normalize else cm_raw

    # Build annotation strings: count + %
    annot = np.empty_like(cm_raw, dtype=object)
    for i in range(cm_raw.shape[0]):
        for j in range(cm_raw.shape[1]):
            pct = cm_norm[i, j] * 100
            annot[i, j] = f"{cm_raw[i, j]}\n({pct:.1f}%)"

    n = len(class_names)
    fig_size = max(6, n * 1.2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        display_cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Proportion" if normalize else "Count"},
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 3. plot_csi_heatmap
# ---------------------------------------------------------------------------

def plot_csi_heatmap(
    csi_data: np.ndarray,
    title: str = "CSI Amplitude",
    save_path: Optional[str] = None,
) -> None:
    """Visualise a single CSI sample as a heatmap.

    Parameters
    ----------
    csi_data : np.ndarray, shape (time_steps, num_subcarriers)
        2-D CSI amplitude matrix. If shape is (1, T, S) or (T, S, 1) the
        channel dimension is squeezed automatically.
    title : str
    save_path : str or None
    """
    data = np.squeeze(csi_data)
    if data.ndim != 2:
        raise ValueError(
            f"Expected 2-D array (time_steps, subcarriers) after squeeze, "
            f"got shape {data.shape}"
        )

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        data,
        aspect="auto",
        origin="upper",
        cmap="viridis",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Amplitude", fontsize=10)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Subcarrier Index", fontsize=11)
    ax.set_ylabel("Time Step", fontsize=11)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 4. plot_class_csi_comparison
# ---------------------------------------------------------------------------

def plot_class_csi_comparison(
    samples_by_class: Dict[str, np.ndarray],
    class_names: Sequence[str],
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side CSI heatmaps, one representative sample per class.

    Parameters
    ----------
    samples_by_class : dict {class_name -> np.ndarray, shape (T, S)}
        One (or averaged) CSI sample per class. Keys should match class_names.
    class_names : sequence of str
    save_path : str or None
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4), sharey=True)

    if n_classes == 1:
        axes = [axes]

    # Find global colour scale
    all_vals = np.concatenate(
        [np.asarray(samples_by_class[c]).ravel() for c in class_names if c in samples_by_class]
    )
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    for ax, cls in zip(axes, class_names):
        data = np.squeeze(np.asarray(samples_by_class.get(cls, np.zeros((1, 1)))))
        if data.ndim != 2:
            data = data.reshape(1, -1)
        im = ax.imshow(
            data,
            aspect="auto",
            origin="upper",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(cls, fontsize=11, fontweight="bold")
        ax.set_xlabel("Subcarrier", fontsize=9)

    axes[0].set_ylabel("Time Step", fontsize=10)

    # Shared colourbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Amplitude")

    fig.suptitle("CSI Comparison by Class", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.90, 1])
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 5. plot_subcarrier_variance
# ---------------------------------------------------------------------------

def plot_subcarrier_variance(
    dataset: Dict[str, np.ndarray],
    class_names: Sequence[str],
    save_path: Optional[str] = None,
) -> None:
    """Per-subcarrier mean amplitude ± 1 std, one line per class.

    Useful for identifying which subcarriers carry the most class-discriminative
    information.

    Parameters
    ----------
    dataset : dict {class_name -> np.ndarray, shape (N, T, S) or (N, 1, T, S)}
        Collection of CSI samples grouped by class.
    class_names : sequence of str
    save_path : str or None
    """
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))

        for idx, cls in enumerate(class_names):
            if cls not in dataset:
                continue

            samples = np.asarray(dataset[cls])
            # Collapse channel and time dimensions → shape (N, S)
            while samples.ndim > 2:
                # Average over the second-to-last axis (time) first
                samples = samples.mean(axis=-2)

            mean_amp = samples.mean(axis=0)   # (S,)
            std_amp = samples.std(axis=0)     # (S,)
            subcarriers = np.arange(len(mean_amp))

            color = _get_class_color(idx)
            ax.plot(subcarriers, mean_amp, label=cls, color=color, linewidth=1.8)
            ax.fill_between(
                subcarriers,
                mean_amp - std_amp,
                mean_amp + std_amp,
                alpha=0.2,
                color=color,
            )

        ax.set_xlabel("Subcarrier Index", fontsize=11)
        ax.set_ylabel("Mean Amplitude ± 1 Std", fontsize=11)
        ax.set_title(
            "Per-Subcarrier Amplitude Distribution by Class",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        _save_or_show(fig, save_path)
