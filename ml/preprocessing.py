"""
sEMG Signal Preprocessing Pipeline for Prosthetic Hand Continuous Control.

Provides causal, reproducible signal conditioning:
- Baseline / DC removal
- Causal IIR Bandpass filtering (20 - 450 Hz)
- Causal IIR Notch filtering (50 Hz / 60 Hz)
- Train-set only fitted normalization (Z-score / MinMax)
- Strict causal execution: NO zero-phase (filtfilt) filtering in causal mode.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy import signal


class EMGPreprocessor:
    """
    Causal sEMG Signal Preprocessor.

    Guarantees temporal causality:
    - Only past and current samples influence current filtered values.
    - Uses causal IIR filters (scipy.signal.lfilter or direct form II).
    - Preserves filter initial conditions (zi) for streaming real-time inference.
    """

    def __init__(
        self,
        sampling_rate: float = 2000.0,
        bandpass_low: float = 20.0,
        bandpass_high: float = 450.0,
        bandpass_order: int = 4,
        notch_freq: float = 50.0,
        notch_q: float = 30.0,
        num_channels: int = 16,
    ):
        self.fs = float(sampling_rate)
        self.bp_low = float(bandpass_low)
        self.bp_high = float(bandpass_high)
        self.bp_order = int(bandpass_order)
        self.notch_freq = float(notch_freq)
        self.notch_q = float(notch_q)
        self.num_channels = int(num_channels)

        # Design causal bandpass filter
        nyq = 0.5 * self.fs
        low = self.bp_low / nyq
        high = self.bp_high / nyq
        # Ensure high cutoff < 1.0
        high = min(high, 0.99)
        self.bp_b, self.bp_a = signal.butter(
            self.bp_order, [low, high], btype="bandpass"
        )

        # Design causal notch filter
        self.notch_b, self.notch_a = signal.iirnotch(
            self.notch_freq / nyq, self.notch_q
        )

        # Normalization statistics (fitted on training set only)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fitted: bool = False

        # Streaming state buffers for real-time inference
        self.reset_streaming_state()

    def reset_streaming_state(self) -> None:
        """Reset internal filter delay registers for streaming."""
        self.bp_zi = np.zeros(
            (max(len(self.bp_a), len(self.bp_b)) - 1, self.num_channels),
            dtype=np.float32,
        )
        self.notch_zi = np.zeros(
            (max(len(self.notch_a), len(self.notch_b)) - 1, self.num_channels),
            dtype=np.float32,
        )

    def fit_normalization(self, emg_data: np.ndarray) -> "EMGPreprocessor":
        """
        Fit channel-wise mean and standard deviation on training data.

        Args:
            emg_data: (N, C) array of sEMG signals from training set.
        """
        if emg_data.ndim != 2 or emg_data.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected shape (N, {self.num_channels}), got {emg_data.shape}"
            )
        self.mean = np.mean(emg_data, axis=0, keepdims=True).astype(np.float32)
        self.std = np.std(emg_data, axis=0, keepdims=True).astype(np.float32)
        # Prevent division by zero
        self.std = np.maximum(self.std, 1e-6)
        self.is_fitted = True
        return self

    def set_normalization_stats(
        self, mean: np.ndarray, std: np.ndarray
    ) -> "EMGPreprocessor":
        """Set precomputed normalization statistics."""
        self.mean = np.array(mean, dtype=np.float32).reshape(1, self.num_channels)
        self.std = np.array(std, dtype=np.float32).reshape(1, self.num_channels)
        self.is_fitted = True
        return self

    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        """Export normalization statistics for deployment."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted.")
        return {"mean": self.mean.copy(), "std": self.std.copy()}

    def filter_batch_causal(
        self, emg_data: np.ndarray, apply_notch: bool = True
    ) -> np.ndarray:
        """
        Apply strictly causal forward-only IIR filtering to batch of data.

        Uses signal.lfilter (Direct Form II Transposed).
        DOES NOT USE filtfilt, ensuring zero temporal leakage from the future.
        """
        # 1. Bandpass filter (20 - 450 Hz)
        filtered = signal.lfilter(self.bp_b, self.bp_a, emg_data, axis=0)

        # 2. Notch filter (50 Hz power line)
        if apply_notch:
            filtered = signal.lfilter(
                self.notch_b, self.notch_a, filtered, axis=0
            )

        return filtered.astype(np.float32)

    def normalize(self, emg_data: np.ndarray) -> np.ndarray:
        """Apply fitted Z-score normalization."""
        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor must be fitted on training data before normalization."
            )
        return ((emg_data - self.mean) / self.std).astype(np.float32)

    def process_offline(
        self, emg_data: np.ndarray, apply_notch: bool = True
    ) -> np.ndarray:
        """Complete causal offline conditioning pipeline: Filter -> Normalize."""
        filtered = self.filter_batch_causal(emg_data, apply_notch=apply_notch)
        if self.is_fitted:
            return self.normalize(filtered)
        return filtered

    def process_streaming_sample(
        self, sample: np.ndarray, apply_notch: bool = True
    ) -> np.ndarray:
        """
        Causal filtering for a single multi-channel time step in streaming mode.

        Args:
            sample: (C,) or (1, C) array of 16-channel sEMG at time t.
        Returns:
            (C,) conditioned sample.
        """
        x = np.atleast_2d(sample).astype(np.float32)
        # Step 1: Bandpass
        out_bp, self.bp_zi = signal.lfilter(
            self.bp_b, self.bp_a, x, axis=0, zi=self.bp_zi
        )
        # Step 2: Notch
        if apply_notch:
            out_notch, self.notch_zi = signal.lfilter(
                self.notch_b, self.notch_a, out_bp, axis=0, zi=self.notch_zi
            )
            out = out_notch
        else:
            out = out_bp

        if self.is_fitted:
            out = (out - self.mean) / self.std

        return out[0].astype(np.float32)
