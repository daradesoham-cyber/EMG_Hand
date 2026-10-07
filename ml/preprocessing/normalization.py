"""
Normalization Module for sEMG and Kinematic Joint Targets.

Guarantees:
- Normalization parameters (mean, std, quantile bounds) are fitted ONLY on training data (A1).
- Frozen parameters applied unchanged to validation (A2) and strictly held-out test data (A3).
- Zero data leakage between sessions.
"""

from typing import Tuple, Dict, Optional, List, Any
import numpy as np


class EMGNormalizer:
    """
    Channel-wise Z-score normalizer: z = (x - mean) / std.
    Strictly fitted on training data only.
    """

    def __init__(self, num_channels: int = 16):
        self.num_channels = int(num_channels)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, emg_data: np.ndarray) -> "EMGNormalizer":
        """Fit channel-wise mean and std on training data (N, C)."""
        if emg_data.ndim != 2 or emg_data.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected shape (N, {self.num_channels}), got {emg_data.shape}"
            )
        self.mean = np.mean(emg_data, axis=0, keepdims=True).astype(np.float32)
        self.std = np.std(emg_data, axis=0, keepdims=True).astype(np.float32)
        self.std = np.maximum(self.std, 1e-6)
        self.is_fitted = True
        return self

    def set_stats(self, mean: np.ndarray, std: np.ndarray) -> "EMGNormalizer":
        """Manually load frozen normalization statistics."""
        self.mean = np.array(mean, dtype=np.float32).reshape(1, self.num_channels)
        self.std = np.array(std, dtype=np.float32).reshape(1, self.num_channels)
        self.is_fitted = True
        return self

    def transform(self, emg_data: np.ndarray) -> np.ndarray:
        """Apply z-score normalization to array (N, C)."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform.")
        return ((emg_data - self.mean) / self.std).astype(np.float32)

    def get_stats(self) -> Dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Normalizer is not fitted.")
        return {"mean": self.mean.copy(), "std": self.std.copy()}


CHANNEL_COLS_MAP = {
    "index": [4, 5],
    "middle": [6, 7],
    "ring": [9, 10],
    "little": [12, 13],
    "pinky": [12, 13],
    "thumb": [1, 15],
    "thumb_flexion": [1, 15],
    "thumb_opposition": 0,
    "thumb_opp": 0,
}


def fit_target_quantiles(
    glove_data: np.ndarray,
    target_names: Optional[List[str]] = None,
    q_low: float = 1.0,
    q_high: float = 99.0,
) -> Dict[str, Tuple[float, float]]:
    """
    Fit robust percentile min/max extrema strictly on training acquisition (A1).
    """
    if target_names is None:
        target_names = ["index", "middle", "ring", "little", "thumb_flexion", "thumb_opposition"]

    stats = {}
    for name in target_names:
        cols = CHANNEL_COLS_MAP.get(name)
        if cols is None:
            raise ValueError(f"Unknown target name: {name}")
        if isinstance(cols, int):
            angle = glove_data[:, cols]
        else:
            angle = np.mean(glove_data[:, cols], axis=1)

        q_min = float(np.percentile(angle, q_low))
        q_max = float(np.percentile(angle, q_high))
        if q_max - q_min < 1.0:
            q_max = q_min + 1.0
        stats[name] = (q_min, q_max)

    return stats


def normalize_continuous_targets(
    glove_data: np.ndarray,
    norm_stats: Dict[str, Tuple[float, float]],
    target_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Normalize CyberGlove angles to [0, 1] range using frozen training bounds.
    """
    if target_names is None:
        target_names = ["index", "middle", "ring", "little", "thumb_flexion", "thumb_opposition"]

    N = len(glove_data)
    num_outputs = len(target_names)
    targets = np.zeros((N, num_outputs), dtype=np.float32)

    for i, name in enumerate(target_names):
        cols = CHANNEL_COLS_MAP[name]
        if isinstance(cols, int):
            raw_angle = glove_data[:, cols]
        else:
            raw_angle = np.mean(glove_data[:, cols], axis=1)

        q_min, q_max = norm_stats[name]
        norm = (raw_angle - q_min) / (q_max - q_min)
        targets[:, i] = np.clip(norm, 0.0, 1.0)

    return targets
