"""
CSI Preprocessing Utilities.

All functions operate on NumPy arrays.  No PyTorch dependencies here — this
keeps the preprocessing code reusable in non-training contexts (e.g., real-
time inference on embedded hardware, exploratory analysis in notebooks).

Conventions
-----------
- "subcarrier axis" is always the LAST axis of input arrays.
- "time axis" is always axis=0 (or axis=-2 for batched arrays).
- Complex CSI has dtype complex64 or complex128.
- Amplitude / phase outputs are float32 unless the input is float64.

Shapes used in docstrings:
    (T, C)        — T time steps, C subcarriers (single sample, no batch)
    (N, T, C)     — N samples batch
    (N, 1, T, C)  — PyTorch-style with channel dimension
"""

import logging
import warnings
from typing import Optional, Tuple, Union

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Amplitude and phase extraction
# ---------------------------------------------------------------------------


def extract_amplitude(csi_complex: np.ndarray) -> np.ndarray:
    """
    Compute CSI amplitude (magnitude) from complex channel coefficients.

    Parameters
    ----------
    csi_complex : np.ndarray, dtype complex
        Complex CSI array.  Shape can be anything; amplitude is computed
        element-wise so the output shape matches the input.
        Typical shapes: (T, C), (N, T, C), (N, 1, T, C).

    Returns
    -------
    amplitude : np.ndarray, dtype float32
        |H| — element-wise absolute value.  Same shape as input.

    Notes
    -----
    The magnitude of a complex channel coefficient H = a + jb is:
        |H| = sqrt(a² + b²)
    This is related to received signal strength but retains spatial
    (subcarrier-level) structure that RSSI discards.
    """
    amplitude = np.abs(csi_complex)
    if amplitude.dtype != np.float32:
        amplitude = amplitude.astype(np.float32)
    return amplitude


def extract_phase(csi_complex: np.ndarray) -> np.ndarray:
    """
    Compute unwrapped phase from complex CSI channel coefficients.

    Raw phase from np.angle() is wrapped to [-π, π].  Phase unwrapping
    removes discontinuities by adding multiples of 2π where jumps exceed π,
    producing a continuous phase trajectory useful for velocity estimation.

    Parameters
    ----------
    csi_complex : np.ndarray, dtype complex
        Complex CSI array.  Shape: (..., T, C) or any shape where the
        time axis is the second-to-last axis.

    Returns
    -------
    phase_unwrapped : np.ndarray, dtype float32
        Unwrapped phase in radians.  Same shape as input.

    Notes
    -----
    Unwrapping is applied along the LAST axis (subcarrier axis) by default.
    For time-domain unwrapping, transpose the array before calling.
    """
    phase     = np.angle(csi_complex)                    # wrapped, float64
    unwrapped = np.unwrap(phase, axis=-1)                # unwrap over subcarriers
    return unwrapped.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Normalisation
# ---------------------------------------------------------------------------


def normalize_csi(
    csi: np.ndarray,
    method: str = "zscore",
    axis: int = 0,
) -> np.ndarray:
    """
    Normalise CSI amplitude data per subcarrier.

    Parameters
    ----------
    csi : np.ndarray, dtype float
        CSI amplitude array.  Shape: (T, C) — T time steps, C subcarriers.
        For batched input (N, T, C) or (N, 1, T, C) the caller should loop
        over samples or reshape first.
    method : str
        Normalisation strategy:
        - 'zscore'  : subtract mean, divide by std (per subcarrier)
        - 'minmax'  : scale to [0, 1] (per subcarrier)
        - 'robust'  : subtract median, divide by IQR (per subcarrier);
                      less sensitive to outliers / spike artefacts
    axis : int
        Axis along which statistics are computed.  Default 0 (time axis for
        shape (T, C)).

    Returns
    -------
    normalised : np.ndarray, same shape as input, dtype float32
    """
    csi = csi.astype(np.float32)

    if method == "zscore":
        mean = csi.mean(axis=axis, keepdims=True)
        std  = csi.std(axis=axis, keepdims=True)
        std  = np.where(std < 1e-8, 1.0, std)
        return (csi - mean) / std

    elif method == "minmax":
        mn = csi.min(axis=axis, keepdims=True)
        mx = csi.max(axis=axis, keepdims=True)
        rng = np.where((mx - mn) < 1e-8, 1.0, mx - mn)
        return (csi - mn) / rng

    elif method == "robust":
        median = np.median(csi, axis=axis, keepdims=True)
        q75    = np.percentile(csi, 75, axis=axis, keepdims=True)
        q25    = np.percentile(csi, 25, axis=axis, keepdims=True)
        iqr    = np.where((q75 - q25) < 1e-8, 1.0, q75 - q25)
        return (csi - median) / iqr

    else:
        raise ValueError(
            f"Unknown normalisation method '{method}'. "
            "Choose from: 'zscore', 'minmax', 'robust'."
        )


# ---------------------------------------------------------------------------
# 3. Hampel filter (spike / outlier removal)
# ---------------------------------------------------------------------------


def hampel_filter(
    series: np.ndarray,
    window_size: int = 5,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """
    Apply a Hampel identifier to remove spike outliers from a time series.

    The Hampel filter replaces outliers with the local median.  Each point
    is flagged as an outlier if it deviates from the local median by more
    than n_sigma times the local median absolute deviation (MAD).

    When applied to CSI data (shape T × C), this function operates
    **per subcarrier** (column-wise) so that spike removal decisions are
    independent per subcarrier.

    Parameters
    ----------
    series : np.ndarray
        Input data.  Shape: (T,) for a single series, or (T, C) to apply
        per-subcarrier.
    window_size : int
        Half-window size (total window = 2 * window_size + 1).  Default 5
        gives an 11-sample window at 100 Hz → 110 ms, suitable for
        removing hardware glitches without disturbing body movement.
    n_sigma : float
        Outlier threshold in units of estimated standard deviation.
        Default 3.0 (flag points > 3σ from median).

    Returns
    -------
    filtered : np.ndarray, same shape as input, dtype float32
        Copy of series with spike samples replaced by local medians.
    """
    one_d = series.ndim == 1
    if one_d:
        series = series[:, None]

    T, C = series.shape
    out  = series.astype(np.float32).copy()

    # MAD scale factor to make MAD equivalent to σ for Gaussian distributions
    k = 1.4826

    for c in range(C):
        col = out[:, c]
        for t in range(T):
            lo = max(0, t - window_size)
            hi = min(T, t + window_size + 1)
            nbr = col[lo:hi]
            med = np.median(nbr)
            mad = k * np.median(np.abs(nbr - med))
            if mad < 1e-10:
                # All-identical neighbourhood; only replace exact zeros if
                # context looks like a hardware dropout
                continue
            if np.abs(col[t] - med) > n_sigma * mad:
                col[t] = med
        out[:, c] = col

    if one_d:
        out = out[:, 0]
    return out


# ---------------------------------------------------------------------------
# 4. Butterworth filters
# ---------------------------------------------------------------------------


def butter_lowpass_filter(
    data: np.ndarray,
    cutoff: float = 10.0,
    fs: float = 100.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter along the time axis (axis=0).

    Removes high-frequency measurement noise while preserving body-movement
    signals (typically 0–10 Hz for human activity).

    Parameters
    ----------
    data : np.ndarray
        CSI amplitude data.  Shape: (T,) or (T, C).
        Filter is applied along axis=0 (time axis).
    cutoff : float
        Low-pass cutoff frequency in Hz.  Default 10.0 Hz.
    fs : float
        Sampling frequency in Hz.  Default 100.0 Hz.
    order : int
        Filter order.  Default 4 (good balance of roll-off and stability).

    Returns
    -------
    filtered : np.ndarray, same shape as input, dtype float32

    Notes
    -----
    Uses second-order sections (SOS) representation for numerical stability
    (avoids the instability of high-order transfer functions).
    """
    nyq    = fs / 2.0
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        warnings.warn(
            f"cutoff={cutoff} Hz >= Nyquist={nyq} Hz. "
            "Returning unfiltered data.",
            RuntimeWarning,
        )
        return data.astype(np.float32)

    sos      = signal.butter(order, normal_cutoff, btype="low", output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=0)
    return filtered.astype(np.float32)


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.1,
    highcut: float = 10.0,
    fs: float = 100.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter along the time axis (axis=0).

    Simultaneously removes DC drift (below lowcut) and high-frequency noise
    (above highcut).  Recommended for preprocessing raw entrance CSI before
    windowing.

    Parameters
    ----------
    data : np.ndarray
        CSI amplitude data.  Shape: (T,) or (T, C).
    lowcut : float
        High-pass cutoff in Hz (removes very slow drift).  Default 0.1 Hz.
    highcut : float
        Low-pass cutoff in Hz.  Default 10.0 Hz.
    fs : float
        Sampling frequency in Hz.  Default 100.0 Hz.
    order : int
        Filter order per band (total 2*order poles).  Default 4.

    Returns
    -------
    filtered : np.ndarray, same shape as input, dtype float32
    """
    nyq   = fs / 2.0
    low   = max(lowcut  / nyq, 1e-5)
    high  = min(highcut / nyq, 1.0 - 1e-5)

    if low >= high:
        raise ValueError(
            f"lowcut ({lowcut} Hz) must be less than highcut ({highcut} Hz)."
        )

    sos      = signal.butter(order, [low, high], btype="bandpass", output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=0)
    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Windowing
# ---------------------------------------------------------------------------


def window_csi(
    data: np.ndarray,
    window_size: int = 250,
    stride: int = 125,
) -> np.ndarray:
    """
    Slide a fixed-length window over the time axis to produce segments.

    Parameters
    ----------
    data : np.ndarray
        CSI amplitude array.  Shape: (T, C) — T time steps, C subcarriers.
    window_size : int
        Number of time steps per window.  Default 250 (matches UT-HAR and
        the model's expected input size).
    stride : int
        Number of time steps between consecutive window starts.  Default 125
        gives 50% overlap, which is a good trade-off between data augmentation
        and independence between windows.

    Returns
    -------
    windows : np.ndarray
        Shape: (n_windows, window_size, C), dtype float32.
        Returns empty array (0, window_size, C) if data is shorter than window.

    Examples
    --------
    >>> data = np.random.randn(1000, 90).astype(np.float32)
    >>> windows = window_csi(data, window_size=250, stride=125)
    >>> windows.shape
    (7, 250, 90)
    """
    if data.ndim == 1:
        data = data[:, None]

    T, C     = data.shape
    segments = []
    start    = 0

    while start + window_size <= T:
        segments.append(data[start : start + window_size])
        start += stride

    if not segments:
        return np.empty((0, window_size, C), dtype=np.float32)

    return np.stack(segments, axis=0).astype(np.float32)  # (n_windows, W, C)


# ---------------------------------------------------------------------------
# 6. Denoising convenience function
# ---------------------------------------------------------------------------


def denoise_csi(
    data: np.ndarray,
    fs: float = 100.0,
    hampel_window: int = 5,
    hampel_sigma: float = 3.0,
    lowpass_cutoff: float = 10.0,
    lowpass_order: int = 4,
) -> np.ndarray:
    """
    Standard two-stage CSI denoising pipeline.

    Stage 1: Hampel filter — removes hardware spike artefacts per subcarrier.
    Stage 2: Butterworth low-pass filter — removes high-frequency thermal noise.

    Parameters
    ----------
    data : np.ndarray
        CSI amplitude array.  Shape: (T,) or (T, C).
    fs : float
        Sampling frequency in Hz.  Default 100.0.
    hampel_window : int
        Half-window for Hampel filter.  Default 5 → 11-sample window.
    hampel_sigma : float
        Outlier threshold for Hampel filter in σ units.  Default 3.0.
    lowpass_cutoff : float
        Low-pass cutoff frequency in Hz.  Default 10.0.
    lowpass_order : int
        Low-pass Butterworth filter order.  Default 4.

    Returns
    -------
    denoised : np.ndarray, same shape as input, dtype float32
    """
    # Stage 1: spike removal
    denoised = hampel_filter(data, window_size=hampel_window, n_sigma=hampel_sigma)

    # Stage 2: smoothing
    denoised = butter_lowpass_filter(
        denoised, cutoff=lowpass_cutoff, fs=fs, order=lowpass_order
    )
    return denoised


# ---------------------------------------------------------------------------
# 7. Doppler frequency estimation via STFT
# ---------------------------------------------------------------------------


def compute_doppler(
    csi_amplitude: np.ndarray,
    fs: float = 100.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    subcarrier_axis: int = -1,
    aggregate: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Doppler frequency content via Short-Time Fourier Transform (STFT).

    The Doppler frequency shift in received WiFi signals is proportional to
    the velocity of moving objects in the environment.  By computing an STFT
    of the CSI amplitude time series, we obtain a time-frequency spectrogram
    that captures motion dynamics (walking ~0.5–2 Hz, running ~2–4 Hz,
    trolley wheels ~2–4 Hz).

    Parameters
    ----------
    csi_amplitude : np.ndarray
        CSI amplitude array.  Shape: (T, C) — time × subcarriers.
        The STFT is computed per subcarrier and then aggregated.
    fs : float
        Sampling frequency in Hz.  Default 100.0.
    nperseg : int, optional
        Length of each STFT segment in samples.
        Default: min(T // 4, 64), producing ~4 time bins.
    noverlap : int, optional
        Number of points to overlap between segments.
        Default: nperseg // 2 (50% overlap).
    nfft : int, optional
        FFT length (zero-padding if > nperseg).  Default: nperseg.
    subcarrier_axis : int
        Axis index for subcarriers.  Default -1.
    aggregate : str
        How to combine STFT magnitudes across subcarriers before returning:
        - 'mean'   : average over subcarriers (reduces to 2-D spectrogram)
        - 'max'    : max over subcarriers (highlights strongest Doppler components)
        - 'none'   : return all subcarriers (3-D array)

    Returns
    -------
    frequencies : np.ndarray, shape (n_freq_bins,)
        Array of Doppler frequency bins in Hz.
    times : np.ndarray, shape (n_time_bins,)
        Array of time-segment centres in seconds.
    spectrogram : np.ndarray
        STFT magnitude spectrogram.
        - If aggregate in ('mean', 'max'): shape (n_freq_bins, n_time_bins)
        - If aggregate == 'none': shape (C, n_freq_bins, n_time_bins)

    Examples
    --------
    >>> amp = np.random.randn(250, 90).astype(np.float32)
    >>> freqs, times, spec = compute_doppler(amp, fs=100.0)
    >>> freqs.shape, times.shape, spec.shape
    ((33,), (7,), (33, 7))
    """
    if csi_amplitude.ndim == 1:
        csi_amplitude = csi_amplitude[:, None]

    T, C = csi_amplitude.shape

    # Defaults for STFT parameters
    if nperseg is None:
        nperseg = min(max(T // 4, 8), 64)
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg

    # Compute STFT per subcarrier
    # scipy.signal.stft returns (f, t, Zxx) where Zxx is complex
    all_specs = []
    for c in range(C):
        f, t, Zxx = signal.stft(
            csi_amplitude[:, c],
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
        )
        mag = np.abs(Zxx)  # (n_freq_bins, n_time_bins)
        all_specs.append(mag)

    # all_specs: list of C arrays, each (n_freq_bins, n_time_bins)
    spec_array = np.stack(all_specs, axis=0)  # (C, n_freq_bins, n_time_bins)

    if aggregate == "mean":
        spectrogram = spec_array.mean(axis=0)
    elif aggregate == "max":
        spectrogram = spec_array.max(axis=0)
    elif aggregate == "none":
        spectrogram = spec_array
    else:
        raise ValueError(
            f"Unknown aggregate='{aggregate}'. Choose from: 'mean', 'max', 'none'."
        )

    return f.astype(np.float32), t.astype(np.float32), spectrogram.astype(np.float32)
