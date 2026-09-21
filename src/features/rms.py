"""
EMG Feature Extraction Module.
Computes time-domain energy and statistical features from sEMG windows.
"""

import numpy as np


def compute_rms(window: np.ndarray) -> float:
    """Computes the overall Root Mean Square (RMS) across all channels."""
    return float(np.sqrt(np.mean(window ** 2)))


def compute_channel_rms(window: np.ndarray) -> np.ndarray:
    """Computes RMS for each channel individually."""
    return np.sqrt(np.mean(window ** 2, axis=0))


def compute_mav(window: np.ndarray) -> np.ndarray:
    """Computes Mean Absolute Value (MAV) for each channel."""
    return np.mean(np.abs(window), axis=0)


def compute_waveform_length(window: np.ndarray) -> np.ndarray:
    """Computes Waveform Length (WL) for each channel."""
    return np.sum(np.abs(np.diff(window, axis=0)), axis=0)


def compute_zero_crossings(window: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Computes Zero Crossings (ZC) with noise deadband threshold."""
    diff = np.diff(np.sign(window), axis=0)
    amplitude_diff = np.abs(np.diff(window, axis=0))
    return np.sum((diff != 0) & (amplitude_diff > threshold), axis=0)
