"""
datasets/uthar.py
-----------------
PyTorch Dataset for the UT-HAR WiFi CSI human activity recognition dataset.

The SenseFi-distributed version stores data as NumPy arrays with a misleading
.csv extension. Each file is a proper .npy binary:

    UT_HAR/data/X_train.csv  →  shape (3977, 250, 90), float64
    UT_HAR/data/X_val.csv    →  shape ( 496, 250, 90), float64
    UT_HAR/data/X_test.csv   →  shape ( 500, 250, 90), float64
    UT_HAR/label/y_train.csv →  shape (3977,), int labels 0-6
    UT_HAR/label/y_val.csv   →  shape ( 496,), int labels 0-6
    UT_HAR/label/y_test.csv  →  shape ( 500,), int labels 0-6

Classes (label index → activity):
    0  lie_down   |  1  fall     |  2  walk
    3  pickup     |  4  run      |  5  sit_down  |  6  stand_up

Download via:
    python data/download.py
which fetches the SenseFi-processed zip from Google Drive and extracts here.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class UTHARDataset(Dataset):
    """UT-HAR WiFi CSI activity recognition dataset.

    Parameters
    ----------
    config : dict
        Project config dict. Uses ``config["data"]["data_dir"]`` as the root.
    split : str
        One of ``"train"``, ``"val"``, or ``"test"``.
    transform : callable, optional
        Optional transform applied to the CSI tensor after loading.
    """

    CLASSES = [
        "lie_down",
        "fall",
        "walk",
        "pickup",
        "run",
        "sit_down",
        "stand_up",
    ]
    NUM_CLASSES = 7

    def __init__(
        self,
        config: dict,
        split: str = "train",
        transform=None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

        data_dir = Path(config["data"]["data_dir"]) / "uthar" / "UT_HAR"
        x_path = data_dir / "data" / f"X_{split}.csv"
        y_path = data_dir / "label" / f"y_{split}.csv"

        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"UT-HAR data not found at {data_dir}.\n"
                "Run:  python data/download.py\n"
                "to fetch the pre-processed dataset from SenseFi's Google Drive."
            )

        # Files use .csv extension but are numpy binary format
        X_raw = np.load(str(x_path)).astype(np.float32)  # (N, 250, 90)
        y_raw = np.load(str(y_path)).astype(np.int64)     # (N,)

        # Validate shapes
        assert X_raw.ndim == 3 and X_raw.shape[1:] == (250, 90), (
            f"Unexpected X shape: {X_raw.shape}. Expected (N, 250, 90)."
        )

        # Per-subcarrier z-score normalisation across the time axis
        mean = X_raw.mean(axis=1, keepdims=True)   # (N, 1, 90)
        std  = X_raw.std(axis=1, keepdims=True) + 1e-8
        X_norm = (X_raw - mean) / std

        # Add channel dim: (N, 1, 250, 90) — expected by CNN models
        self.X = torch.from_numpy(X_norm[:, np.newaxis, :, :])  # (N, 1, 250, 90)
        self.y = torch.from_numpy(y_raw)
        self.transform = transform
        self.split = split

        log.info(
            "[UT-HAR] %s split: %d samples | classes: %s",
            split,
            len(self),
            self.CLASSES,
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]  # (1, 250, 90)
        y = self.y[idx]  # scalar
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    # ------------------------------------------------------------------
    @classmethod
    def download(cls, data_dir: str = "./data/raw") -> None:
        """Download the SenseFi-processed UT-HAR zip from Google Drive."""
        import zipfile
        try:
            import gdown
        except ImportError:
            raise ImportError("Install gdown first:  pip install gdown")

        out = Path(data_dir) / "uthar"
        out.mkdir(parents=True, exist_ok=True)
        zip_path = out / "UT_HAR.zip"

        print("[download] Fetching UT_HAR.zip from SenseFi Google Drive …")
        gdown.download(
            id="1fEiI3nAoOsddR5qcJQXqz4ocM3aMAcwz",
            output=str(zip_path),
            quiet=False,
        )
        print("[download] Extracting …")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out)
        zip_path.unlink()
        print(f"[download] Done. Data at {out}/UT_HAR/")
