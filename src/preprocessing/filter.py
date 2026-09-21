"""
EMG Preprocessing Module.
Implements DC offset removal, Butterworth bandpass filtering, and train normalization.
"""

import numpy as np
import scipy.signal


def remove_dc(window: np.ndarray) -> np.ndarray:
    """Subtracts channel-wise mean from the window."""
    return window - np.mean(window, axis=0, keepdims=True)


def butterworth_bandpass(
    window: np.ndarray,
    lowcut: float = 20.0,
    highcut: float = 450.0,
    fs: float = 1000.0,
    order: int = 4
) -> np.ndarray:
    """Applies a zero-phase 4th-order Butterworth bandpass filter."""
    sos = scipy.signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return scipy.signal.sosfiltfilt(sos, window, axis=0)


def normalize(window: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Applies Z-score normalization using provided mean and std."""
    return (window - mean) / (std + eps)


def preprocess_window(
    window: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    lowcut: float = 20.0,
    highcut: float = 450.0,
    fs: float = 1000.0
) -> tuple[np.ndarray, float]:
    """
    Executes full preprocessing sequence:
    1. Compute raw RMS energy
    2. Remove DC offset
    3. 20-450 Hz bandpass filter
    4. Normalize using training stats
    """
    raw_rms = float(np.sqrt(np.mean(window ** 2)))
    dc_removed = remove_dc(window)
    filtered = butterworth_bandpass(dc_removed, lowcut=lowcut, highcut=highcut, fs=fs)
    normed = normalize(filtered, mean, std)
    return normed.astype(np.float32), raw_rms
