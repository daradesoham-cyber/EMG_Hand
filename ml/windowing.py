"""
Windowing and Feature Slicing Module for Causal sEMG-to-Kinematic Regression.

Verifies:
- 2 kHz sampling rate
- 128 ms window duration -> 256 samples
- 32 ms stride duration -> 64 samples
- Expected inference update rate: 31.25 Hz
- Causality: Window [t - W + 1 : t] predicts target state at time t.
"""

from typing import Tuple, Optional, Generator
import numpy as np


class WindowConfig:
    """Configurable window parameters with programmatic verification."""

    def __init__(
        self,
        sampling_rate: int = 2000,
        window_ms: float = 128.0,
        stride_ms: float = 32.0,
    ):
        self.fs = int(sampling_rate)
        self.window_ms = float(window_ms)
        self.stride_ms = float(stride_ms)

        # Programmatic sample calculation
        self.window_samples = int(round(self.window_ms * self.fs / 1000.0))
        self.stride_samples = int(round(self.stride_ms * self.fs / 1000.0))
        self.inference_freq_hz = float(self.fs) / float(self.stride_samples)

        # Verify target parameters
        self._verify_parameters()

    def _verify_parameters(self) -> None:
        """Verify standard 128 ms / 32 ms configuration."""
        if self.fs == 2000:
            assert self.window_samples == 256, (
                f"Expected 256 samples for 128 ms at 2 kHz, got {self.window_samples}"
            )
            assert self.stride_samples == 64, (
                f"Expected 64 samples for 32 ms at 2 kHz, got {self.stride_samples}"
            )
            assert abs(self.inference_freq_hz - 31.25) < 1e-4, (
                f"Expected 31.25 Hz inference frequency, got {self.inference_freq_hz}"
            )

    def __repr__(self) -> str:
        return (
            f"WindowConfig(fs={self.fs}Hz, window={self.window_ms}ms ({self.window_samples}s), "
            f"stride={self.stride_ms}ms ({self.stride_samples}s), rate={self.inference_freq_hz:.2f}Hz)"
        )


def extract_windows_numpy(
    emg: np.ndarray,
    targets: np.ndarray,
    config: WindowConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract causal sliding windows from continuous data.

    Args:
        emg: (N, C) preprocessed continuous sEMG signals.
        targets: (N, K) normalized continuous kinematic targets.
        config: WindowConfig instance.

    Returns:
        X: (num_windows, C, window_samples) array formatted for Conv1D (B, C, L).
        Y: (num_windows, K) array of target values at the window's final time step t.
    """
    N, C = emg.shape
    W = config.window_samples
    S = config.stride_samples

    if N < W:
        raise ValueError(f"Data length {N} is shorter than window length {W}")

    num_windows = (N - W) // S + 1

    # Preallocate output arrays for maximum speed
    K = targets.shape[1] if targets.ndim > 1 else 1
    X = np.empty((num_windows, C, W), dtype=np.float32)
    Y = np.empty((num_windows, K), dtype=np.float32)

    for i in range(num_windows):
        start_idx = i * S
        end_idx = start_idx + W
        # Window: [start_idx : end_idx] (W samples)
        # Transpose from (W, C) to (C, W) for 1D convolution
        X[i] = emg[start_idx:end_idx].T
        # Target: value at final causal sample of the window (end_idx - 1)
        Y[i] = targets[end_idx - 1]

    return X, Y


class StreamingWindowBuffer:
    """
    Real-time rolling ring buffer for streaming inference.

    Maintains a 256-sample window buffer and yields a feature window every 64 samples.
    """

    def __init__(self, config: WindowConfig, num_channels: int = 16):
        self.config = config
        self.num_channels = num_channels
        self.W = config.window_samples
        self.S = config.stride_samples

        self.buffer = np.zeros((self.num_channels, self.W), dtype=np.float32)
        self.samples_since_update = 0
        self.total_samples_received = 0

    def push_sample(self, sample: np.ndarray) -> Optional[np.ndarray]:
        """
        Push single multi-channel sample [C] into ring buffer.

        Returns:
            (C, W) window if stride reached and buffer is full, else None.
        """
        # Shift buffer left by 1 and insert new sample at the end
        self.buffer[:, :-1] = self.buffer[:, 1:]
        self.buffer[:, -1] = sample

        self.samples_since_update += 1
        self.total_samples_received += 1

        if (
            self.total_samples_received >= self.W
            and self.samples_since_update >= self.S
        ):
            self.samples_since_update = 0
            return self.buffer.copy()
        return None

    def push_chunk(self, chunk: np.ndarray) -> Generator[np.ndarray, None, None]:
        """
        Push chunk of samples [L, C] and yield ready windows.
        """
        for s in chunk:
            window = self.push_sample(s)
            if window is not None:
                yield window
