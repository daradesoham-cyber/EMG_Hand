"""
sEMG Preprocessing, Windowing, Normalization, and Causal Alignment Package.
"""

from ml.preprocessing.filters import CausalEMGFilter
from ml.preprocessing.normalization import (
    EMGNormalizer,
    fit_target_quantiles,
    normalize_continuous_targets,
    CHANNEL_COLS_MAP,
)
from ml.preprocessing.windowing import (
    WindowConfig,
    extract_windows_numpy,
    StreamingWindowBuffer,
)
from ml.preprocessing.alignment import (
    verify_temporal_alignment,
    compute_cross_correlation_lag,
)

# Backward-compatibility alias
class EMGPreprocessor:
    """Wrapper uniting CausalEMGFilter and EMGNormalizer for backward compatibility."""

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
        self.filter = CausalEMGFilter(
            sampling_rate=sampling_rate,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
            bandpass_order=bandpass_order,
            notch_freq=notch_freq,
            notch_q=notch_q,
            num_channels=num_channels,
        )
        self.normalizer = EMGNormalizer(num_channels=num_channels)
        self.num_channels = num_channels
        self.fs = sampling_rate

    @property
    def is_fitted(self) -> bool:
        return self.normalizer.is_fitted

    @property
    def mean(self):
        return self.normalizer.mean

    @property
    def std(self):
        return self.normalizer.std

    def reset_streaming_state(self):
        self.filter.reset_streaming_state()

    def fit_normalization(self, emg_data):
        self.normalizer.fit(emg_data)
        return self

    def set_normalization_stats(self, mean, std):
        self.normalizer.set_stats(mean, std)
        return self

    def get_normalization_stats(self):
        return self.normalizer.get_stats()

    def filter_batch_causal(self, emg_data, apply_notch: bool = True):
        return self.filter.filter_batch(emg_data, apply_notch=apply_notch)

    def normalize(self, emg_data):
        return self.normalizer.transform(emg_data)

    def process_offline(self, emg_data, apply_notch: bool = True):
        filtered = self.filter.filter_batch(emg_data, apply_notch=apply_notch)
        if self.normalizer.is_fitted:
            return self.normalizer.transform(filtered)
        return filtered

    def process_streaming_sample(self, sample, apply_notch: bool = True):
        filtered_sample = self.filter.filter_sample_streaming(sample, apply_notch=apply_notch)
        if self.normalizer.is_fitted:
            return ((filtered_sample - self.normalizer.mean[0]) / self.normalizer.std[0]).astype(float)
        return filtered_sample

__all__ = [
    "CausalEMGFilter",
    "EMGNormalizer",
    "fit_target_quantiles",
    "normalize_continuous_targets",
    "CHANNEL_COLS_MAP",
    "WindowConfig",
    "extract_windows_numpy",
    "StreamingWindowBuffer",
    "verify_temporal_alignment",
    "compute_cross_correlation_lag",
    "EMGPreprocessor",
]
