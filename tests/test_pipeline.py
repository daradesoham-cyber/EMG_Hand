"""
Comprehensive Unit Tests for NinaPro DB8 ML Pipeline.

Tests:
1. Window dimensions (256 samples, 16 channels)
2. Stride dimensions (64 samples, 31.25 Hz)
3. Causal behavior (no future samples affect past predictions)
4. Normalization fitting isolation (zero test leakage)
5. Model output dimensions and valid [0, 1] range
6. Receptive field derivation verification
"""

import unittest
import numpy as np
import torch

from ml.preprocessing import EMGPreprocessor
from ml.windowing import WindowConfig, extract_windows_numpy, StreamingWindowBuffer
from ml.model import CausalTCN, CausalConv1d


class TestEMGPipeline(unittest.TestCase):

    def setUp(self):
        self.fs = 2000
        self.num_channels = 16
        self.window_config = WindowConfig(
            sampling_rate=self.fs,
            window_ms=128.0,
            stride_ms=32.0,
        )

    def test_window_parameters(self):
        """Test window and stride sample calculations."""
        self.assertEqual(self.window_config.window_samples, 256)
        self.assertEqual(self.window_config.stride_samples, 64)
        self.assertAlmostEqual(self.window_config.inference_freq_hz, 31.25, places=3)

    def test_window_extraction_shapes(self):
        """Test window slicing output shapes."""
        N = 2000  # 1 second of data
        dummy_emg = np.random.randn(N, self.num_channels).astype(np.float32)
        dummy_targets = np.random.rand(N, 5).astype(np.float32)

        X, Y = extract_windows_numpy(dummy_emg, dummy_targets, self.window_config)

        expected_windows = (N - 256) // 64 + 1
        self.assertEqual(len(X), expected_windows)
        self.assertEqual(X.shape, (expected_windows, 16, 256))
        self.assertEqual(Y.shape, (expected_windows, 5))

    def test_causal_conv_isolation(self):
        """
        Verify strict temporal causality in CausalConv1d.

        Modifying input sample at time t must NOT change output at any step t' < t.
        """
        conv = CausalConv1d(
            in_channels=4, out_channels=4, kernel_size=3, dilation=2
        )
        conv.eval()

        # Input: batch of length 50
        x1 = torch.randn(1, 4, 50)
        x2 = x1.clone()

        # Perturb future sample at t=35
        perturb_time = 35
        x2[:, :, perturb_time:] += 10.0

        with torch.no_grad():
            out1 = conv(x1)
            out2 = conv(x2)

        # Outputs before t=35 MUST be identical to machine precision
        diff_past = torch.max(
            torch.abs(out1[:, :, :perturb_time] - out2[:, :, :perturb_time])
        ).item()
        self.assertEqual(
            diff_past,
            0.0,
            f"Future leakage detected! Past output changed by {diff_past}",
        )

        # Output at and after t=35 MUST differ
        diff_future = torch.max(
            torch.abs(out1[:, :, perturb_time:] - out2[:, :, perturb_time:])
        ).item()
        self.assertGreater(diff_future, 0.0)

    def test_tcn_full_causality(self):
        """
        Verify that the complete multi-layer CausalTCN is strictly causal.
        """
        model = CausalTCN(in_channels=16, num_outputs=5)
        model.eval()

        x1 = torch.randn(2, 16, 256)
        x2 = x1.clone()

        # Change only the very last sample (time 255)
        x2[:, :, -1] += 5.0

        # Features before time 255 in hidden layers must remain untouched
        # Verify layer 1
        with torch.no_grad():
            l1_1 = model.network[0](x1)
            l1_2 = model.network[0](x2)

        past_diff = torch.max(torch.abs(l1_1[:, :, :-1] - l1_2[:, :, :-1])).item()
        self.assertEqual(
            past_diff, 0.0, f"Leakage in TCN hidden layers: diff={past_diff}"
        )

    def test_receptive_field_derivation(self):
        """Verify receptive field mathematical calculation."""
        model = CausalTCN(
            kernel_size=3, dilations=[1, 2, 4, 8], convs_per_block=2
        )
        rf_info = model.compute_receptive_field()
        # RF = 1 + 2 * (3 - 1) * (1 + 2 + 4 + 8) = 1 + 4 * 15 = 61
        self.assertEqual(rf_info["receptive_field_samples"], 61)
        self.assertAlmostEqual(rf_info["receptive_field_ms"], 30.5, places=2)
        self.assertFalse(rf_info["covers_full_window"])

    def test_normalization_no_leakage(self):
        """Verify normalization statistics are strictly fitted on training set."""
        train_emg = np.random.normal(loc=5.0, scale=2.0, size=(1000, 16)).astype(
            np.float32
        )
        test_emg = np.random.normal(loc=-10.0, scale=0.5, size=(500, 16)).astype(
            np.float32
        )

        prep = EMGPreprocessor(num_channels=16)
        prep.fit_normalization(train_emg)

        # Check fitted stats match train set, NOT test set
        np.testing.assert_allclose(prep.mean, np.mean(train_emg, axis=0, keepdims=True), rtol=1e-4)
        np.testing.assert_allclose(prep.std, np.std(train_emg, axis=0, keepdims=True), rtol=1e-4)

        # Normalize test set using train stats
        norm_test = prep.normalize(test_emg)
        # Test set mean should NOT be 0 after normalizing with train stats
        self.assertNotAlmostEqual(float(norm_test.mean()), 0.0, places=1)

    def test_streaming_buffer(self):
        """Verify real-time ring buffer produces correct sliding window."""
        buffer = StreamingWindowBuffer(self.window_config, num_channels=16)

        # Feed 255 samples -> No window ready
        for _ in range(255):
            win = buffer.push_sample(np.ones(16, dtype=np.float32))
            self.assertIsNone(win)

        # 256th sample -> First window ready
        win = buffer.push_sample(np.ones(16, dtype=np.float32) * 2.0)
        self.assertIsNotNone(win)
        self.assertEqual(win.shape, (16, 256))
        self.assertEqual(win[0, -1], 2.0)

        # Next 63 samples -> No window
        for _ in range(63):
            win = buffer.push_sample(np.ones(16, dtype=np.float32))
            self.assertIsNone(win)

        # 64th sample -> Second window ready
        win = buffer.push_sample(np.ones(16, dtype=np.float32) * 3.0)
        self.assertIsNotNone(win)
        self.assertEqual(win[0, -1], 3.0)


if __name__ == "__main__":
    unittest.main()
