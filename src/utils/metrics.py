"""
metrics.py — CSI classification metrics utilities.

Sainsbury's WiFi CSI Sensing project.
Provides:
  - compute_metrics      : single-shot dict of classification metrics
  - ConfusionMatrixTracker : accumulates predictions across mini-batches
  - EarlyStopping        : patience-based early-stopping with checkpoint saving
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> Dict:
    """Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth class indices.
    y_pred : array-like of int
        Predicted class indices.
    class_names : sequence of str
        Human-readable label for each class index.

    Returns
    -------
    dict with keys:
        accuracy          (float)
        per_class         (dict[str -> {precision, recall, f1, support}])
        macro_precision   (float)
        macro_recall      (float)
        macro_f1          (float)
        weighted_precision(float)
        weighted_recall   (float)
        weighted_f1       (float)
        report            (str)  — full sklearn text report
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = list(range(len(class_names)))

    accuracy = float(accuracy_score(y_true, y_pred))

    per_class: Dict[str, Dict] = {}
    precision_per = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    recall_per = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    f1_per = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    support_per = np.bincount(y_true, minlength=len(class_names))

    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision_per[i]),
            "recall": float(recall_per[i]),
            "f1": float(f1_per[i]),
            "support": int(support_per[i]),
        }

    macro_precision = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    macro_recall = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    weighted_precision = float(
        precision_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    weighted_recall = float(
        recall_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    weighted_f1 = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=labels,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "report": report,
    }


# ---------------------------------------------------------------------------
# ConfusionMatrixTracker
# ---------------------------------------------------------------------------

class ConfusionMatrixTracker:
    """Accumulate predictions across mini-batches, compute confusion matrix on demand.

    Usage
    -----
    tracker = ConfusionMatrixTracker(num_classes=4)
    for preds, labels in dataloader:
        tracker.add_batch(preds.numpy(), labels.numpy())
    cm = tracker.compute()   # shape (num_classes, num_classes)
    tracker.reset()
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self._all_preds: List[np.ndarray] = []
        self._all_labels: List[np.ndarray] = []

    def add_batch(
        self,
        preds: Sequence[int],
        labels: Sequence[int],
    ) -> None:
        """Append a batch of predictions and ground-truth labels.

        Parameters
        ----------
        preds : array-like of int, shape (N,)
            Predicted class indices for the batch.
        labels : array-like of int, shape (N,)
            Ground-truth class indices for the batch.
        """
        self._all_preds.append(np.asarray(preds, dtype=np.int64).ravel())
        self._all_labels.append(np.asarray(labels, dtype=np.int64).ravel())

    def compute(self) -> np.ndarray:
        """Return the accumulated confusion matrix.

        Returns
        -------
        np.ndarray, shape (num_classes, num_classes)
            cm[i, j] = number of samples with true label i predicted as j.
        """
        if not self._all_preds:
            return np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        y_pred = np.concatenate(self._all_preds)
        y_true = np.concatenate(self._all_labels)
        return confusion_matrix(
            y_true, y_pred, labels=list(range(self.num_classes))
        )

    def reset(self) -> None:
        """Clear all accumulated predictions."""
        self._all_preds = []
        self._all_labels = []

    @property
    def total_samples(self) -> int:
        """Number of samples accumulated so far."""
        return sum(len(p) for p in self._all_preds)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Patience-based early stopping with best-model checkpointing.

    Tracks validation loss. When the loss fails to improve by more than
    ``min_delta`` for ``patience`` consecutive epochs, signals that training
    should stop.

    Usage
    -----
    stopper = EarlyStopping(patience=10, min_delta=1e-4)
    for epoch in range(max_epochs):
        val_loss = validate(model)
        if stopper(val_loss):
            print("Early stopping triggered.")
            break
        if stopper.is_best:
            stopper.save_checkpoint(model, "checkpoints/best.pt")
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        patience : int
            Number of epochs with no improvement before stopping.
        min_delta : float
            Minimum change in monitored value to qualify as improvement.
        mode : {'min', 'max'}
            Whether lower ('min') or higher ('max') values are better.
        verbose : bool
            Print messages when a new best is found or counter increases.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.is_best: bool = False
        self._stop: bool = False

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def __call__(self, val_metric: float) -> bool:
        """Check whether training should stop.

        Parameters
        ----------
        val_metric : float
            Current epoch validation metric (loss or accuracy depending on mode).

        Returns
        -------
        bool
            ``True`` if training should be stopped, ``False`` otherwise.
        """
        self.is_best = False

        if self._is_improvement(val_metric):
            self.best_score = val_metric
            self.counter = 0
            self.is_best = True
            if self.verbose:
                print(
                    f"[EarlyStopping] New best score: {val_metric:.6f}"
                )
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs "
                    f"(best={self.best_score:.6f}, current={val_metric:.6f})"
                )
            if self.counter >= self.patience:
                self._stop = True
                if self.verbose:
                    print(
                        f"[EarlyStopping] Patience exhausted after {self.patience} epochs. "
                        "Stopping training."
                    )

        return self._stop

    def save_checkpoint(self, model, path: str) -> None:
        """Save model weights to ``path`` when a new best score is achieved.

        Parameters
        ----------
        model : torch.nn.Module
            The model whose state_dict will be saved.
        path : str
            Destination file path (e.g. ``checkpoints/best_model.pt``).
        """
        import torch  # deferred import — avoids hard dep at module load time

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(model.state_dict(), path)
        if self.verbose:
            print(f"[EarlyStopping] Checkpoint saved → {path}")

    def reset(self) -> None:
        """Reset all internal state (useful when resuming or reusing the object)."""
        self.best_score = None
        self.counter = 0
        self.is_best = False
        self._stop = False
