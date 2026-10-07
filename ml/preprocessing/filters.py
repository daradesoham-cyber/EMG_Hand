"""
Causal Filtering Module for sEMG Signals.

Implements:
- 4th-order causal Butterworth IIR bandpass (20 - 450 Hz)
- 2nd-order causal IIR notch filter at 50 Hz (Q = 30)
- Direct Form II Transposed filtering via scipy.signal.lfilter (strictly past-only)
- Streaming delay state registers (zi) for single-sample real-time execution.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy import signal


class CausalEMGFilter:
    """
    Causal IIR filter cascade for multi-channel sEMG acquisition.
    Enforces strictly causal execution: no zero-phase (filtfilt) backward pass.
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

        # Design causal Butterworth bandpass filter
        nyq = 0.5 * self.fs
        low = self.bp_low / nyq
        high = min(self.bp_high / nyq, 0.99)
        self.bp_b, self.bp_a = signal.butter(
            self.bp_order, [low, high], btype="bandpass"
        )

        # Design causal IIR notch filter
        self.notch_b, self.notch_a = signal.iirnotch(
            self.notch_freq / nyq, self.notch_q
        )

        # Streaming state delay registers
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

    def filter_batch(self, emg_data: np.ndarray, apply_notch: bool = True) -> np.ndarray:
        """
        Apply strictly causal forward-only IIR filtering to batch array (N, C).
        Uses scipy.signal.lfilter (Direct Form II Transposed).
        """
        filtered = signal.lfilter(self.bp_b, self.bp_a, emg_data, axis=0)
        if apply_notch:
            filtered = signal.lfilter(self.notch_b, self.notch_a, filtered, axis=0)
        return filtered.astype(np.float32)

    def filter_sample_streaming(
        self, sample: np.ndarray, apply_notch: bool = True
    ) -> np.ndarray:
        """
        Causal filtering for a single multi-channel time step in streaming mode.
        """
        x = np.atleast_2d(sample).astype(np.float32)
        out_bp, self.bp_zi = signal.lfilter(
            self.bp_b, self.bp_a, x, axis=0, zi=self.bp_zi
        )
        if apply_notch:
            out_notch, self.notch_zi = signal.lfilter(
                self.notch_b, self.notch_a, out_bp, axis=0, zi=self.notch_zi
            )
            out = out_notch
        else:
            out = out_bp
        return out[0].astype(np.float32)
