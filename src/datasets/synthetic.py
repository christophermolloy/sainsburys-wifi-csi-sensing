"""
Synthetic CSI Dataset for Sainsbury's Entrance Monitoring.

Generates realistic-but-artificial CSI amplitude data for 4 classes that
mirror the physical entrance use case:

    0: empty          — no one in the area
    1: person_only    — single shopper, no items
    2: person_basket  — shopper carrying a hand basket
    3: person_trolley — shopper pushing a shopping trolley

Purpose
-------
This dataset is designed for **Step 0 pipeline validation**.  It lets you
confirm that the entire training pipeline (data loading, preprocessing,
model, loss, optimiser, eval loop) runs end-to-end before any real data is
collected.  The data is discriminative enough that a simple CNN should reach
>85% accuracy with default hyperparameters.

Usage
-----
    config = {
        "data": {
            "dataset": "synthetic",
            "n_samples": 2000,     # total (split equally across classes)
            "seed": 42,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
        }
    }
    ds_train = SyntheticCSIDataset(config, split="train")
    ds_val   = SyntheticCSIDataset(config, split="val")
    ds_test  = SyntheticCSIDataset(config, split="test")

Physical modelling notes
------------------------
CSI amplitude (|H_k|) for subcarrier k is modelled as:

    |H_k|(t) = path_loss_k + body_signal_k(t) + basket/trolley_k(t) + noise(t)

- path_loss_k  : static, per-subcarrier background level
- body_signal  : slow sinusoidal variation due to body movement / respiration
- basket/trolley: additional multipath components from metal objects
- noise        : AWGN + occasional phase-flip artefacts + slow drift

All values are kept in a physically plausible normalised amplitude range [0, 1]
before z-score normalisation is applied.
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

CLASSES: List[str] = ["empty", "person_only", "person_basket", "person_trolley"]
NUM_CLASSES: int   = len(CLASSES)

_N_TIMESTEPS   = 250   # time steps per sample
_N_SUBCARRIERS = 90    # 30 subcarriers × 3 antenna pairs


# ---------------------------------------------------------------------------
# Signal generation helpers
# ---------------------------------------------------------------------------

def _background_envelope(rng: np.random.Generator, n_sc: int = _N_SUBCARRIERS) -> np.ndarray:
    """
    Return a smooth per-subcarrier background amplitude envelope in [0.05, 0.35].

    This simulates the static multipath environment (walls, shelving, etc.)
    common to all classes.
    """
    # Use a sum of low-frequency cosines to get a smooth, realistic spectral
    # shape rather than a flat or purely random envelope.
    x    = np.linspace(0, 2 * np.pi, n_sc)
    env  = 0.20 + 0.08 * np.cos(x) + 0.04 * np.cos(2 * x + rng.uniform(0, np.pi))
    env += rng.uniform(-0.02, 0.02, size=n_sc)  # small per-subcarrier jitter
    return np.clip(env, 0.05, 0.35)


def _awgn(rng: np.random.Generator, shape: Tuple, sigma: float = 0.10) -> np.ndarray:
    """Additive White Gaussian Noise."""
    return rng.normal(0.0, sigma, size=shape).astype(np.float32)


def _slow_drift(
    rng: np.random.Generator,
    n_time: int = _N_TIMESTEPS,
    n_sc: int = _N_SUBCARRIERS,
    amplitude: float = 0.05,
) -> np.ndarray:
    """
    Very-low-frequency drift common in real CSI (temperature / hardware changes).
    Shape: (n_time, n_sc)
    """
    # One random walk per subcarrier, heavily smoothed
    raw  = rng.standard_normal((n_time, n_sc)).cumsum(axis=0)
    raw -= raw.mean(axis=0, keepdims=True)
    raw /= (np.abs(raw).max(axis=0, keepdims=True) + 1e-8)
    return (raw * amplitude).astype(np.float32)


def _phase_jumps(
    rng: np.random.Generator,
    n_time: int = _N_TIMESTEPS,
    n_sc: int = _N_SUBCARRIERS,
    jump_prob: float = 0.005,
    jump_amplitude: float = 0.15,
) -> np.ndarray:
    """
    Occasional phase-flip artefacts seen in real CSI hardware.
    Modelled as sparse amplitude spikes affecting random (time, subcarrier) pairs.
    """
    mask    = rng.random((n_time, n_sc)) < jump_prob
    jumps   = mask * rng.choice([-1, 1], size=(n_time, n_sc)) * jump_amplitude
    return jumps.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-class CSI generators   shape returned: (n_time, n_sc)
# ---------------------------------------------------------------------------

def _gen_empty(rng: np.random.Generator) -> np.ndarray:
    """
    Class 0 — empty scene.

    Characteristics:
    - Low, stable amplitude dominated by static reflections.
    - Very little temporal variation (σ_time ~ 0.03).
    - Smooth subcarrier envelope (multipath from walls only).
    """
    n_t, n_s  = _N_TIMESTEPS, _N_SUBCARRIERS
    background = _background_envelope(rng, n_s)   # (n_s,)

    # Replicate across time + tiny AWGN only
    data  = np.tile(background, (n_t, 1))          # (n_t, n_s)
    data += _awgn(rng, (n_t, n_s), sigma=0.02)    # very quiet
    data += _slow_drift(rng, n_t, n_s, amplitude=0.02)
    return data.astype(np.float32)


def _gen_person_only(rng: np.random.Generator) -> np.ndarray:
    """
    Class 1 — person only.

    Characteristics:
    - Moderate amplitude perturbation from body blockage / scattering.
    - 2-3 dominant subcarrier groups show slow, coherent sinusoidal variation
      (body movement, ~0.3–1.5 Hz).
    - Walking micro-Doppler: periodic amplitude modulation at ~1.0 Hz.
    """
    n_t, n_s = _N_TIMESTEPS, _N_SUBCARRIERS
    t        = np.arange(n_t, dtype=np.float32)
    background = _background_envelope(rng, n_s)

    data = np.tile(background, (n_t, 1))

    # Body movement: 2 dominant sinusoidal components, random frequency & phase
    n_components = rng.integers(2, 4)
    for _ in range(n_components):
        freq    = rng.uniform(0.3, 1.5) / n_t * 2 * np.pi   # cycles/sample
        phase   = rng.uniform(0, 2 * np.pi)
        amp     = rng.uniform(0.04, 0.10)
        # Affect a contiguous band of subcarriers (body reflection is wideband)
        center  = rng.integers(5, n_s - 5)
        width   = rng.integers(10, 25)
        sc_mask = np.zeros(n_s, dtype=np.float32)
        lo, hi  = max(0, center - width // 2), min(n_s, center + width // 2)
        sc_mask[lo:hi] = 1.0
        signal  = (amp * np.sin(freq * t + phase))[:, None] * sc_mask[None, :]
        data   += signal

    data += _awgn(rng, (n_t, n_s), sigma=0.08)
    data += _slow_drift(rng, n_t, n_s, amplitude=0.04)
    data += _phase_jumps(rng, n_t, n_s)
    return data.astype(np.float32)


def _gen_person_basket(rng: np.random.Generator) -> np.ndarray:
    """
    Class 2 — person carrying a hand basket.

    Characteristics:
    - Similar to person_only, but the metal basket creates additional
      specular reflections concentrated in the MID-frequency subcarrier range
      (indices 30-60, corresponding to 1.0–2.0 GHz mid-band).
    - Slightly higher amplitude perturbations overall (more scatterers).
    - Basket swing introduces an additional periodic component at ~0.8–1.2 Hz.
    """
    n_t, n_s = _N_TIMESTEPS, _N_SUBCARRIERS
    t         = np.arange(n_t, dtype=np.float32)
    background = _background_envelope(rng, n_s)

    data = np.tile(background, (n_t, 1))

    # Inherit body movement from person_only style
    n_body = rng.integers(2, 4)
    for _ in range(n_body):
        freq    = rng.uniform(0.3, 1.5) / n_t * 2 * np.pi
        phase   = rng.uniform(0, 2 * np.pi)
        amp     = rng.uniform(0.04, 0.10)
        center  = rng.integers(5, n_s - 5)
        width   = rng.integers(10, 25)
        sc_mask = np.zeros(n_s, dtype=np.float32)
        lo, hi  = max(0, center - width // 2), min(n_s, center + width // 2)
        sc_mask[lo:hi] = 1.0
        data   += (amp * np.sin(freq * t + phase))[:, None] * sc_mask[None, :]

    # Basket metal reflection — concentrated in mid-subcarrier band (30–60)
    basket_freq  = rng.uniform(0.8, 1.2) / n_t * 2 * np.pi
    basket_phase = rng.uniform(0, 2 * np.pi)
    basket_amp   = rng.uniform(0.08, 0.15)   # noticeably higher than body
    basket_mask  = np.zeros(n_s, dtype=np.float32)
    basket_mask[30:61] = np.hanning(31).astype(np.float32)  # smooth window
    data += (basket_amp * np.sin(basket_freq * t + basket_phase))[:, None] * basket_mask[None, :]

    # Higher background level in that band (metal increases path amplitude)
    metal_boost = np.zeros(n_s, dtype=np.float32)
    metal_boost[30:61] = 0.06
    data += metal_boost[None, :]

    data += _awgn(rng, (n_t, n_s), sigma=0.09)
    data += _slow_drift(rng, n_t, n_s, amplitude=0.04)
    data += _phase_jumps(rng, n_t, n_s)
    return data.astype(np.float32)


def _gen_person_trolley(rng: np.random.Generator) -> np.ndarray:
    """
    Class 3 — person pushing a shopping trolley.

    Characteristics:
    - High-amplitude multipath from the large metal frame.
    - Strong perturbations across ALL subcarriers (large reflector).
    - Clear periodic pattern from trolley wheel rotation (~2–4 Hz).
    - Distinctive spike pattern in HIGH-frequency subcarrier range (60–90),
      consistent with the trolley's extended metal structure.
    - Significantly higher overall amplitude than other classes.
    """
    n_t, n_s = _N_TIMESTEPS, _N_SUBCARRIERS
    t         = np.arange(n_t, dtype=np.float32)
    background = _background_envelope(rng, n_s)

    # Boosted background — trolley frame increases reflected energy everywhere
    data = np.tile(background * 1.4, (n_t, 1))

    # Body movement (person pushing)
    n_body = rng.integers(2, 4)
    for _ in range(n_body):
        freq    = rng.uniform(0.3, 1.0) / n_t * 2 * np.pi   # slightly slower (pushing)
        phase   = rng.uniform(0, 2 * np.pi)
        amp     = rng.uniform(0.06, 0.12)
        center  = rng.integers(5, n_s - 5)
        width   = rng.integers(15, 30)
        sc_mask = np.zeros(n_s, dtype=np.float32)
        lo, hi  = max(0, center - width // 2), min(n_s, center + width // 2)
        sc_mask[lo:hi] = 1.0
        data   += (amp * np.sin(freq * t + phase))[:, None] * sc_mask[None, :]

    # Trolley wheel rotation: strong periodic signal across ALL subcarriers
    wheel_freq  = rng.uniform(2.0, 4.0) / n_t * 2 * np.pi   # 2–4 Hz
    wheel_phase = rng.uniform(0, 2 * np.pi)
    wheel_amp   = rng.uniform(0.12, 0.20)
    data += (wheel_amp * np.abs(np.sin(wheel_freq * t + wheel_phase)))[:, None]

    # High-subcarrier spike pattern (60–90): trolley metal structure
    trolley_freq  = wheel_freq * 2                             # harmonic
    trolley_phase = rng.uniform(0, 2 * np.pi)
    trolley_amp   = rng.uniform(0.10, 0.18)
    trolley_mask  = np.zeros(n_s, dtype=np.float32)
    trolley_mask[60:] = np.linspace(0.5, 1.0, n_s - 60).astype(np.float32)
    data += (trolley_amp * np.abs(np.sin(trolley_freq * t + trolley_phase)))[:, None] * trolley_mask[None, :]

    # Occasional large multipath spikes (trolley re-orientation / turn)
    data += _phase_jumps(rng, n_t, n_s, jump_prob=0.012, jump_amplitude=0.20)
    data += _awgn(rng, (n_t, n_s), sigma=0.10)
    data += _slow_drift(rng, n_t, n_s, amplitude=0.06)
    return data.astype(np.float32)


_GENERATORS = {
    0: _gen_empty,
    1: _gen_person_only,
    2: _gen_person_basket,
    3: _gen_person_trolley,
}


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class SyntheticCSIDataset(Dataset):
    """
    Synthetic CSI amplitude dataset for Sainsbury's entrance monitoring.

    Parameters
    ----------
    config : dict
        Expected keys under config["data"]:
          n_samples      (int)   total samples, split equally per class (default 2000)
          seed           (int)   master random seed (default 42)
          val_fraction   (float) fraction for val split (default 0.15)
          test_fraction  (float) fraction for test split (default 0.15)
    split : str
        "train", "val", or "test".
    """

    CLASSES    : List[str] = CLASSES
    NUM_CLASSES : int      = NUM_CLASSES

    def __init__(self, config: dict, split: str = "train") -> None:
        super().__init__()
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        data_cfg      = config.get("data", {})
        n_samples     = int(data_cfg.get("n_samples", 2000))
        seed          = int(data_cfg.get("seed", 42))
        val_frac      = float(data_cfg.get("val_fraction", 0.15))
        test_frac     = float(data_cfg.get("test_fraction", 0.15))

        X_all, y_all  = self._generate_all(n_samples, seed)

        # Stratified split: test first, then val from the remainder
        from sklearn.model_selection import train_test_split
        idx           = np.arange(len(y_all))
        idx_trainval, idx_test = train_test_split(
            idx, test_size=test_frac, random_state=seed, stratify=y_all
        )
        adjusted_val  = val_frac / (1.0 - test_frac)
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=adjusted_val,
            random_state=seed,
            stratify=y_all[idx_trainval],
        )

        split_idx = {"train": idx_train, "val": idx_val, "test": idx_test}[split]
        X         = X_all[split_idx]
        y         = y_all[split_idx]

        # Per-subcarrier z-score normalisation
        X = self._normalize(X)

        self.X     = torch.from_numpy(X).float()   # (N, 1, 250, 90)
        self.y     = torch.from_numpy(y).long()    # (N,)

        logger.info(
            "SyntheticCSIDataset | split=%s | samples=%d",
            split, len(self.y),
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_all(n_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate all samples for all classes.

        Returns
        -------
        X : (n_samples, 1, 250, 90) float32
        y : (n_samples,) int64
        """
        rng             = np.random.default_rng(seed)
        per_class       = n_samples // NUM_CLASSES
        remainder       = n_samples - per_class * NUM_CLASSES

        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for class_idx, gen_fn in _GENERATORS.items():
            n = per_class + (1 if class_idx < remainder else 0)
            samples = np.stack(
                [gen_fn(rng) for _ in range(n)], axis=0
            )  # (n, T, C)
            samples = samples[:, None, :, :]  # (n, 1, T, C)
            X_list.append(samples)
            y_list.append(np.full(n, class_idx, dtype=np.int64))

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        # Shuffle while keeping X-y correspondence
        perm = rng.permutation(len(y))
        return X[perm].astype(np.float32), y[perm]

    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        """
        Per-subcarrier z-score normalisation.

        X : (N, 1, T, C)
        """
        N, _, T, C = X.shape
        flat        = X.reshape(-1, C)
        mean        = flat.mean(axis=0, keepdims=True)
        std         = flat.std(axis=0, keepdims=True)
        std         = np.where(std < 1e-8, 1.0, std)
        return ((flat - mean) / std).reshape(N, 1, T, C)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualise(self, save_path: Optional[str] = None) -> None:
        """
        Plot one sample per class as a CSI amplitude heatmap.

        Parameters
        ----------
        save_path : str, optional
            File path to save the figure (PNG/PDF).  If None, calls plt.show().
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            warnings.warn(
                "matplotlib is required for visualise(). "
                "Install with: pip install matplotlib"
            )
            return

        fig  = plt.figure(figsize=(14, 8))
        gs   = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

        for class_idx, class_name in enumerate(CLASSES):
            mask  = (self.y == class_idx).nonzero(as_tuple=True)[0]
            if len(mask) == 0:
                continue
            sample = self.X[mask[0]].squeeze(0).numpy()   # (T, C)

            ax = fig.add_subplot(gs[class_idx // 2, class_idx % 2])
            im = ax.imshow(
                sample.T,           # (C, T) — subcarriers on y-axis
                aspect="auto",
                origin="lower",
                cmap="viridis",
                interpolation="nearest",
            )
            ax.set_title(f"Class {class_idx}: {class_name}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Subcarrier index")
            plt.colorbar(im, ax=ax, label="Norm. amplitude")

        fig.suptitle(
            "Synthetic CSI Amplitude Heatmaps — One Sample per Class",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("Saved visualisation to %s", save_path)
        else:
            plt.show()

        plt.close(fig)
