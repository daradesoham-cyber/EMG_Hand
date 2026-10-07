"""
NinaPro DB8 Dataset Loader and PyTorch Dataset Module.

Guarantees:
- Strict session-level separation (A1 = Train, A2 = Val, A3 = Strict Held-Out Test).
- Normalization (sEMG z-score, glove angle range) is fitted ONLY on Acquisition 1 (A1).
- No test-set data is accessed during fitting or tuning.
- Preserves all 18 raw CyberGlove channels in memory.
"""

import os
from typing import Tuple, Dict, List, Optional, Any
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

from ml.preprocessing import EMGPreprocessor
from ml.windowing import WindowConfig, extract_windows_numpy


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


def compute_target_normalization_stats(
    glove_data: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute joint angle min/max ranges strictly from training acquisition (A1).
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

        q_min = float(np.percentile(angle, 1.0))
        q_max = float(np.percentile(angle, 99.0))
        if q_max - q_min < 1.0:
            q_max = q_min + 1.0
        stats[name] = (q_min, q_max)

    return stats


def extract_continuous_targets(
    glove_data: np.ndarray,
    norm_stats: Dict[str, Tuple[float, float]],
    target_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Convert raw 18-channel CyberGlove degrees to normalized [0, 1] continuous targets.
    Supports arbitrary target configurations including full 6-actuator H1-H6 mapping.
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


class NinaProDB8SessionDataset:
    """Container for a single subject's multi-acquisition DB8 data."""

    def __init__(
        self,
        data_dir: str = "data/ninapro_db8",
        subject_id: int = 1,
        window_config: Optional[WindowConfig] = None,
        num_outputs: int = 5,
        target_names: Optional[List[str]] = None,
    ):
        self.data_dir = data_dir
        self.subject_id = subject_id
        self.window_config = window_config or WindowConfig()
        self.num_outputs = num_outputs
        if target_names is not None:
            self.target_names = target_names
        elif num_outputs == 6:
            self.target_names = ["index", "middle", "ring", "little", "thumb_flexion", "thumb_opposition"]
        else:
            self.target_names = ["thumb", "index", "middle", "ring", "pinky"]

        self.preprocessor = EMGPreprocessor(
            sampling_rate=self.window_config.fs,
            bandpass_low=20.0,
            bandpass_high=450.0,
            notch_freq=50.0,
            num_channels=16,
        )
        self.target_norm_stats: Optional[Dict[str, Tuple[float, float]]] = None

    def _load_mat(self, session_id: str) -> Dict[str, np.ndarray]:
        """Load .mat file for subject and acquisition."""
        fname = f"S{self.subject_id}_E1_{session_id}.mat"
        fpath = os.path.join(self.data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing NinaPro DB8 file: {fpath}")
        mat = sio.loadmat(fpath)
        return {
            "emg": mat["emg"].astype(np.float32),
            "glove": mat["glove"].astype(np.float32),
            "stimulus": mat["stimulus"].astype(np.int8),
            "repetition": mat["repetition"].astype(np.int8),
        }

    def prepare_data(
        self,
        downsample_stride: Optional[int] = None,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """
        Full causal data preparation pipeline with strict train/val/test isolation.

        A1 -> Fit preprocessor & target stats -> Filter & Window -> Train Set
        A2 -> Use frozen preprocessor & stats -> Filter & Window -> Val Set
        A3 -> Use frozen preprocessor & stats -> Filter & Window -> Test Set
        """
        # 1. Load raw data
        raw_a1 = self._load_mat("A1")
        raw_a2 = self._load_mat("A2")
        raw_a3 = self._load_mat("A3")

        # 2. Fit preprocessor and target normalizers ONLY on A1
        self.preprocessor.fit_normalization(raw_a1["emg"])
        self.target_norm_stats = compute_target_normalization_stats(
            raw_a1["glove"], self.target_names
        )

        # 3. Process each acquisition
        datasets = []
        for raw, name in [(raw_a1, "A1"), (raw_a2, "A2"), (raw_a3, "A3")]:
            emg_proc = self.preprocessor.process_offline(raw["emg"])
            targets = extract_continuous_targets(
                raw["glove"],
                self.target_norm_stats,
                self.target_names,
            )

            # Extract causal sliding windows
            X, Y = extract_windows_numpy(emg_proc, targets, self.window_config)

            # Optional downsampling of windows for quick benchmarking
            if downsample_stride and downsample_stride > 1:
                X = X[::downsample_stride]
                Y = Y[::downsample_stride]

            datasets.append((X, Y))

        return datasets[0], datasets[1], datasets[2]


class NumpyTensorDataset(Dataset):
    """Memory-efficient PyTorch Dataset wrapping numpy arrays."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]
