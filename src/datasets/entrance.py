"""
EntranceCSIDataset — Real collected entrance CSI data loader.

This class is the production template users fill in once they have collected
data with ESP32 or Raspberry Pi hardware.

---------------------------------------------------------------------------
Expected directory structure
---------------------------------------------------------------------------

    data/raw/entrance/
    ├── empty/
    │   ├── session_2024_01_15_10_00.csv
    │   ├── session_2024_01_15_10_05.csv
    │   └── ...
    ├── person_only/
    │   └── ...
    ├── person_basket/
    │   └── ...
    └── person_trolley/
        └── ...

---------------------------------------------------------------------------
CSV format (one file per recording session)
---------------------------------------------------------------------------

Each row is one CSI packet (one snapshot of all subcarriers at one instant):

    timestamp, subcarrier_0_amp, subcarrier_1_amp, ..., subcarrier_N_amp

Example (30 subcarriers, 1 antenna pair → 30 amplitude columns):

    1705312800.123,  0.412, 0.389, 0.401, ..., 0.376
    1705312800.223,  0.415, 0.391, 0.398, ..., 0.379
    ...

The timestamp column is dropped during loading.  If your CSIKit / csiread
output includes other metadata columns (e.g. RSSI, noise floor), add them
to the METADATA_COLUMNS list below.

---------------------------------------------------------------------------
Collecting data with ESP32 + csiread
---------------------------------------------------------------------------

1. Flash the ESP32 with the CSI-extraction firmware:
       https://github.com/StevenMHernandez/ESP32-CSI-Tool

2. Stream CSI packets over serial:
       python -m csiread.tool --port /dev/ttyUSB0 --output session.csv

   Or use the provided capture helper:
       python scripts/capture_csi.py --class person_only --duration 60 \\
           --output data/raw/entrance/person_only/session_001.csv

3. Data quality guidelines:
   - Minimum 60 seconds per session (≥ 6000 packets at 100 Hz)
   - Collect at least 5 sessions per class
   - Ensure consistent hardware placement across sessions

---------------------------------------------------------------------------
Collecting data with Raspberry Pi + nexmon_csi
---------------------------------------------------------------------------

1. Install nexmon_csi on a Raspberry Pi 4:
       https://github.com/seemoo-lab/nexmon_csi

2. Capture with mcp:
       sudo mcp -i wlan0 -c 6 -C 1 -N 1 -m FF:FF:FF:FF:FF:FF -f capture.pcap

3. Convert with CSIKit:
       python -c "from CSIKit.reader import get_reader; \\
           r = get_reader('capture.pcap'); \\
           r.read_file('capture.pcap', scaled=True); \\
           r.export_csv('session.csv')"

4. Place the resulting CSV in the appropriate class directory.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Class names in alphabetical order — determines integer label assignment
CLASSES: List[str] = ["empty", "person_basket", "person_only", "person_trolley"]
NUM_CLASSES: int   = len(CLASSES)

# Columns that are NOT subcarrier amplitudes — will be dropped on load.
# Extend this list if your hardware adds extra metadata columns.
METADATA_COLUMNS: List[str] = ["timestamp", "rssi", "noise_floor", "channel"]

_DEFAULT_WINDOW_SIZE = 250   # time steps per sample (matches UT-HAR)
_DEFAULT_STRIDE      = 125   # 50% overlap by default


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class EntranceCSIDataset(Dataset):
    """
    PyTorch Dataset for real CSI data collected at a Sainsbury's entrance.

    Parameters
    ----------
    config : dict
        Expected keys under config["data"]:
          data_dir       (str)   root data directory (will look under data_dir/raw/entrance)
          window_size    (int)   samples per window (default 250)
          stride         (int)   window stride in samples (default 125, i.e. 50% overlap)
          val_fraction   (float) fraction of windows for validation (default 0.15)
          test_fraction  (float) fraction of windows for test (default 0.15)
          seed           (int)   random seed for split (default 42)
          fs             (float) sampling frequency in Hz (default 100.0)
          lowcut         (float) bandpass low cutoff Hz (default 0.1)
          highcut        (float) bandpass high cutoff Hz (default 10.0)
    split : str
        "train", "val", or "test".
    """

    CLASSES    : List[str] = CLASSES
    NUM_CLASSES: int       = NUM_CLASSES

    def __init__(self, config: dict, split: str = "train") -> None:
        super().__init__()
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.split = split
        data_cfg   = config.get("data", {})

        data_root      = Path(data_cfg.get("data_dir", "data")) / "raw" / "entrance"
        self.window_sz = int(data_cfg.get("window_size", _DEFAULT_WINDOW_SIZE))
        self.stride    = int(data_cfg.get("stride",      _DEFAULT_STRIDE))
        val_frac       = float(data_cfg.get("val_fraction",  0.15))
        test_frac      = float(data_cfg.get("test_fraction", 0.15))
        seed           = int(data_cfg.get("seed", 42))
        self.fs        = float(data_cfg.get("fs",      100.0))
        self.lowcut    = float(data_cfg.get("lowcut",    0.1))
        self.highcut   = float(data_cfg.get("highcut",  10.0))

        self.validate_data_dir(data_root)

        X_all, y_all = self._load_all(data_root)

        if len(X_all) == 0:
            raise RuntimeError(
                f"No CSI windows loaded from {data_root}. "
                "Check that CSV files exist in the expected class subdirectories."
            )

        # Stratified split
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(y_all))
        idx_trainval, idx_test = train_test_split(
            idx, test_size=test_frac, random_state=seed, stratify=y_all
        )
        adjusted_val = val_frac / (1.0 - test_frac)
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=adjusted_val,
            random_state=seed,
            stratify=y_all[idx_trainval],
        )

        split_idx = {"train": idx_train, "val": idx_val, "test": idx_test}[split]
        X = X_all[split_idx]
        y = y_all[split_idx]

        # Per-subcarrier z-score normalisation
        X = self._normalize(X)

        self.X = torch.from_numpy(X).float()   # (N, 1, window_sz, n_sc)
        self.y = torch.from_numpy(y).long()    # (N,)

        logger.info(
            "EntranceCSIDataset | split=%s | samples=%d | tensor shape=%s",
            split, len(self.y), tuple(self.X.shape),
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all(
        self, data_root: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scan class directories, load CSVs, window them, and return arrays.

        Returns
        -------
        X : (N, 1, window_size, n_subcarriers)  float32
        y : (N,)                                 int64
        """
        import pandas as pd

        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for class_idx, class_name in enumerate(CLASSES):
            class_dir = data_root / class_name
            if not class_dir.is_dir():
                warnings.warn(
                    f"Class directory not found, skipping: {class_dir}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            csv_files = sorted(class_dir.glob("*.csv"))
            if not csv_files:
                warnings.warn(
                    f"No CSV files in {class_dir}, skipping class '{class_name}'.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            logger.info(
                "  Loading class '%s' (%d files) …", class_name, len(csv_files)
            )

            for csv_path in csv_files:
                try:
                    windows = self._load_csv(csv_path)     # (n_windows, window_sz, n_sc)
                except Exception as exc:
                    warnings.warn(
                        f"Failed to load {csv_path}: {exc}. Skipping.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

                if windows is None or len(windows) == 0:
                    continue

                n_windows = windows.shape[0]
                X_list.append(windows[:, None, :, :])         # add channel dim
                y_list.append(np.full(n_windows, class_idx, dtype=np.int64))
                logger.debug(
                    "    %s → %d windows", csv_path.name, n_windows
                )

        if not X_list:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

        X = np.concatenate(X_list, axis=0).astype(np.float32)
        y = np.concatenate(y_list, axis=0)
        return X, y

    def _load_csv(self, csv_path: Path) -> Optional[np.ndarray]:
        """
        Load one CSV recording session and return windowed segments.

        Steps:
            1. Read CSV with pandas.
            2. Drop metadata columns (timestamp, etc.).
            3. Convert to float32.
            4. Apply bandpass filter along time axis.
            5. Slide window to produce fixed-length segments.

        Returns
        -------
        windows : (n_windows, window_size, n_subcarriers) float32, or None
        """
        import pandas as pd
        from scipy.signal import butter, sosfilt

        df = pd.read_csv(str(csv_path))

        # Drop metadata columns (case-insensitive)
        drop_cols = [
            c for c in df.columns
            if c.lower() in [m.lower() for m in METADATA_COLUMNS]
        ]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        if df.empty or len(df.columns) == 0:
            logger.warning("  %s: no amplitude columns after dropping metadata.", csv_path.name)
            return None

        data = df.values.astype(np.float32)  # (n_packets, n_subcarriers)

        if len(data) < self.window_sz:
            logger.warning(
                "  %s: only %d rows — shorter than window_size=%d, skipping.",
                csv_path.name, len(data), self.window_sz,
            )
            return None

        # Bandpass filter: remove DC and high-frequency noise
        data = self._bandpass(data)

        # Sliding window
        windows = self._window(data)   # (n_windows, window_size, n_sc)
        return windows

    def _bandpass(self, data: np.ndarray) -> np.ndarray:
        """
        Apply a 4th-order Butterworth bandpass filter along the time axis.

        data : (T, C)  →  filtered (T, C)
        """
        try:
            from scipy.signal import butter, sosfilt
            nyq = self.fs / 2.0
            low  = max(self.lowcut  / nyq, 1e-4)
            high = min(self.highcut / nyq, 1.0 - 1e-4)
            sos  = butter(4, [low, high], btype="bandpass", output="sos")
            # Apply filter to each subcarrier column independently
            filtered = sosfilt(sos, data, axis=0).astype(np.float32)
        except Exception as exc:
            logger.warning("Bandpass filter failed (%s); returning unfiltered data.", exc)
            filtered = data
        return filtered

    def _window(self, data: np.ndarray) -> np.ndarray:
        """
        Sliding window over the time axis.

        data    : (T, C)
        returns : (n_windows, window_size, C)
        """
        T, C     = data.shape
        windows  = []
        start    = 0
        while start + self.window_sz <= T:
            windows.append(data[start : start + self.window_sz])
            start += self.stride
        if not windows:
            return np.empty((0, self.window_sz, C), dtype=np.float32)
        return np.stack(windows, axis=0)   # (n_windows, window_size, C)

    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        """
        Per-subcarrier z-score normalisation.

        X : (N, 1, T, C)
        """
        N, _, T, C = X.shape
        flat  = X.reshape(-1, C)
        mean  = flat.mean(axis=0, keepdims=True)
        std   = flat.std(axis=0, keepdims=True)
        std   = np.where(std < 1e-8, 1.0, std)
        return ((flat - mean) / std).reshape(N, 1, T, C)

    # ------------------------------------------------------------------
    # Validation / reporting
    # ------------------------------------------------------------------

    @staticmethod
    def validate_data_dir(data_root: Path) -> Dict[str, int]:
        """
        Check the data directory structure and report what it finds.

        Parameters
        ----------
        data_root : Path
            Path to the entrance data root (expected to contain class subdirs).

        Returns
        -------
        report : dict  mapping class_name → number of CSV files found.

        Also prints a human-readable summary to stdout.
        """
        data_root = Path(data_root)
        print(f"\nValidating entrance data directory: {data_root}")
        print("-" * 60)

        if not data_root.exists():
            print(
                f"  [MISSING] Directory does not exist: {data_root}\n"
                "  Create it and populate with class subdirectories.\n"
                "  Expected structure:\n"
                "    data/raw/entrance/\n"
                "    ├── empty/           *.csv\n"
                "    ├── person_only/     *.csv\n"
                "    ├── person_basket/   *.csv\n"
                "    └── person_trolley/  *.csv\n"
            )
            return {}

        report: Dict[str, int] = {}
        total_files = 0

        for class_name in CLASSES:
            class_dir = data_root / class_name
            if not class_dir.is_dir():
                n_files = 0
                status  = "[MISSING]"
            else:
                csv_files = list(class_dir.glob("*.csv"))
                n_files   = len(csv_files)
                if n_files == 0:
                    status = "[EMPTY  ]"
                elif n_files < 3:
                    status = "[SPARSE ]"
                else:
                    status = "[OK     ]"

            report[class_name] = n_files
            total_files       += n_files
            print(f"  {status}  {class_name:<20s}  {n_files} CSV file(s)")

        print("-" * 60)
        print(f"  Total CSV files found: {total_files}")

        if total_files == 0:
            print(
                "\n  No data found. Populate the directories with CSV files.\n"
                "  See the module docstring for hardware collection instructions."
            )
        elif total_files < 12:
            print(
                "\n  WARNING: Very few files detected. Recommend at least 5 recording\n"
                "  sessions per class (≥ 60 seconds each) for reliable training."
            )
        else:
            print("\n  Data directory looks good.")

        print()
        return report
