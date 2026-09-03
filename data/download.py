"""
download.py — Download and preprocess the UT-HAR CSI dataset.

Sainsbury's WiFi CSI Sensing project.

UT-HAR reference
----------------
  Repository : https://github.com/ermongroup/Wifi_Activity_Recognition
  Paper      : "A Survey on Behaviour Recognition Using WiFi Channel State Information"
  Classes    : lie_down, fall, pick_up, run, sit_down, stand_up, walk
  Shape      : Each sample is a 250×90 CSI amplitude matrix (250 time frames,
               30 subcarriers × 3 antenna pairs = 90 features).

SenseFi benchmark (alternative)
--------------------------------
  Paper      : "SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi
               Human Sensing" (2022)
  Google Drive data:
    https://drive.google.com/drive/folders/1EUEE4NnJLi1I30ojxmGBHHFVQ2uXNDCL

Usage
-----
    python data/download.py                         # download UT-HAR
    python data/download.py --output-dir data/raw/uthar
    python data/download.py --skip-download         # convert existing CSVs only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import requests
    from tqdm import tqdm

    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = (
    "https://raw.githubusercontent.com/ermongroup/"
    "Wifi_Activity_Recognition/master/data/"
)
FILES = ["X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv"]

# UT-HAR label mapping (0-indexed, matches the repository)
UTHAR_CLASSES = [
    "lie_down",   # 0
    "fall",       # 1
    "pick_up",    # 2
    "run",        # 3
    "sit_down",   # 4
    "stand_up",   # 5
    "walk",       # 6
]

# CSI array dimensions for UT-HAR
TIME_STEPS = 250
NUM_SUBCARRIERS = 90  # 30 subcarriers × 3 antenna pairs


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and preprocess the UT-HAR WiFi CSI dataset."
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/uthar",
        help="Directory to save downloaded and converted files (default: data/raw/uthar).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading; convert already-present CSV files only.",
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Download CSVs but do not convert to .npy.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _check_dependencies() -> None:
    if not _REQUESTS_OK:
        print("[download] ERROR: 'requests' and/or 'tqdm' not installed.")
        print("           Run: pip install requests tqdm")
        sys.exit(1)


def _download_file(url: str, dest: Path) -> bool:
    """Download a single file with a tqdm progress bar.

    Returns True on success, False on failure.
    """
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"[download] FAILED to fetch {url}\n           {exc}")
        return False

    total = int(response.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest.name,
        ncols=80,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return True


def download_uthar(output_dir: Path) -> bool:
    """Download all four UT-HAR CSV files.

    Returns True if all downloads succeeded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for fname in FILES:
        dest = output_dir / fname
        if dest.exists():
            print(f"[download] {fname} already exists, skipping.")
            continue
        url = BASE_URL + fname
        print(f"[download] Downloading {url}")
        ok = _download_file(url, dest)
        if not ok:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _load_csv_with_progress(csv_path: Path, dtype: type) -> np.ndarray:
    """Load a CSV file into a numpy array, reporting progress."""
    print(f"[convert] Reading {csv_path.name} …", end=" ", flush=True)
    arr = np.loadtxt(str(csv_path), delimiter=",", dtype=dtype)
    print(f"done. Shape: {arr.shape}")
    return arr


def convert_to_npy(output_dir: Path) -> dict[str, np.ndarray]:
    """Convert the four CSV files to .npy files.

    X shape after conversion: (N, 1, TIME_STEPS, NUM_SUBCARRIERS)
    y shape after conversion: (N,)

    Returns a dict with keys 'X_train', 'y_train', 'X_test', 'y_test'.
    """
    arrays: dict[str, np.ndarray] = {}

    for split in ("train", "test"):
        x_csv = output_dir / f"X_{split}.csv"
        y_csv = output_dir / f"y_{split}.csv"

        for p in [x_csv, y_csv]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Required file not found: {p}\n"
                    "Run without --skip-download to fetch the data first."
                )

        # X: each row is TIME_STEPS * NUM_SUBCARRIERS = 22,500 floats
        X_raw = _load_csv_with_progress(x_csv, np.float32)
        y_raw = _load_csv_with_progress(y_csv, np.int64)

        n_samples = X_raw.shape[0]
        expected_features = TIME_STEPS * NUM_SUBCARRIERS

        if X_raw.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features per row in {x_csv.name}, "
                f"got {X_raw.shape[1]}. "
                f"Check TIME_STEPS ({TIME_STEPS}) and NUM_SUBCARRIERS ({NUM_SUBCARRIERS})."
            )

        # Reshape: (N, T*S) → (N, 1, T, S) — channel-first for PyTorch CNNs
        X = X_raw.reshape(n_samples, 1, TIME_STEPS, NUM_SUBCARRIERS)

        # UT-HAR labels are 1-indexed in the repository; convert to 0-indexed
        y = y_raw.ravel()
        if y.min() == 1:
            y = y - 1

        arrays[f"X_{split}"] = X
        arrays[f"y_{split}"] = y

        # Save
        x_npy = output_dir / f"X_{split}.npy"
        y_npy = output_dir / f"y_{split}.npy"
        np.save(str(x_npy), X)
        np.save(str(y_npy), y)
        print(f"[convert] Saved {x_npy.name}  shape={X.shape}  dtype={X.dtype}")
        print(f"[convert] Saved {y_npy.name}  shape={y.shape}  dtype={y.dtype}")

    return arrays


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_statistics(arrays: dict[str, np.ndarray]) -> None:
    """Print a human-readable summary of the downloaded dataset."""
    print("\n" + "=" * 60)
    print("UT-HAR Dataset Statistics")
    print("=" * 60)

    for split in ("train", "test"):
        X = arrays.get(f"X_{split}")
        y = arrays.get(f"y_{split}")
        if X is None or y is None:
            continue

        print(f"\n{split.upper()} SET")
        print(f"  Samples   : {X.shape[0]:,}")
        print(f"  Shape     : {X.shape}  (N, C, T, S)")
        print(f"  Value min : {X.min():.4f}")
        print(f"  Value max : {X.max():.4f}")
        print(f"  Value mean: {X.mean():.4f}")
        print(f"  Value std : {X.std():.4f}")
        print(f"  Classes   :")

        unique, counts = np.unique(y, return_counts=True)
        for cls_idx, count in zip(unique, counts):
            cls_name = UTHAR_CLASSES[cls_idx] if cls_idx < len(UTHAR_CLASSES) else f"class_{cls_idx}"
            pct = 100.0 * count / len(y)
            print(f"    [{cls_idx}] {cls_name:<14} : {count:5d}  ({pct:.1f}%)")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Manual download instructions
# ---------------------------------------------------------------------------

def print_manual_instructions(output_dir: Path) -> None:
    print("\n" + "!" * 60)
    print("AUTOMATIC DOWNLOAD FAILED")
    print("!" * 60)
    print("\nManual download instructions:")
    print()
    print("  Option A — Git clone:")
    print("    git clone https://github.com/ermongroup/Wifi_Activity_Recognition")
    print(f"    cp -r Wifi_Activity_Recognition/data/ {output_dir}/")
    print()
    print("  Option B — SenseFi Benchmark (broader CSI dataset):")
    print("    Paper: https://arxiv.org/abs/2207.07859")
    print(
        "    Data : https://drive.google.com/drive/folders/"
        "1EUEE4NnJLi1I30ojxmGBHHFVQ2uXNDCL"
    )
    print()
    print("  After downloading, place the four CSV files here:")
    print(f"    {output_dir}/X_train.csv")
    print(f"    {output_dir}/y_train.csv")
    print(f"    {output_dir}/X_test.csv")
    print(f"    {output_dir}/y_test.csv")
    print()
    print("  Then re-run with --skip-download to convert CSVs to .npy:")
    print(f"    python data/download.py --skip-download --output-dir {output_dir}")
    print("!" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    print(f"[download] Output directory: {output_dir.resolve()}")

    # --- Download ---
    if not args.skip_download:
        _check_dependencies()
        print("[download] Downloading UT-HAR dataset …")
        ok = download_uthar(output_dir)
        if not ok:
            print_manual_instructions(output_dir)
            # Check if we can at least attempt conversion with existing files
            existing = [f for f in FILES if (output_dir / f).exists()]
            if len(existing) < 4:
                print(
                    "[download] Cannot proceed: not all CSV files present. "
                    "Follow the manual instructions above."
                )
                sys.exit(1)
            print(f"[download] Partial download: found {len(existing)}/4 files. Continuing …")
    else:
        print("[download] Skipping download (--skip-download set).")

    # --- Convert ---
    if not args.no_convert:
        print("[convert] Converting CSV → .npy …")
        try:
            arrays = convert_to_npy(output_dir)
        except FileNotFoundError as exc:
            print(f"[convert] ERROR: {exc}")
            print_manual_instructions(output_dir)
            sys.exit(1)
        except ValueError as exc:
            print(f"[convert] ERROR during reshape: {exc}")
            sys.exit(1)

        # --- Statistics ---
        print_statistics(arrays)
    else:
        print("[download] Skipping conversion (--no-convert set).")

    print("\n[download] Done.")
    print(f"  NPY files saved to: {output_dir.resolve()}")
    print(
        "\n  To use with the training pipeline, set config.yaml:\n"
        "    data:\n"
        "      dataset: uthar\n"
        f"      data_dir: {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
