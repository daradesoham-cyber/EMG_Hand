"""
Pure NumPy V1 1D CNN Inference Engine.
Vectorized execution matching the trained Keras model without framework dependencies.
"""

import numpy as np


class V1CompactCNN:
    """
    Pure NumPy forward pass implementation of the V1 Compact 1D CNN.
    3 Conv1D layers with Batch Normalization and MaxPooling, followed by GlobalAvgPool and Dense.
    """

    CLASSES = ['PINCH', 'FIST', 'OPEN']

    def __init__(self, weights_path: str):
        data = np.load(weights_path)
        self.weights = {k: data[k] for k in data.files}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Input: (400, 4) or (batch, 400, 4)
        Output: (batch, 3) softmax probabilities
        """
        if x.ndim == 2:
            x = x[np.newaxis, ...]

        w = self.weights

        # Layer 1
        c1 = self._conv1d_same_relu(x, w['c1_w'], w['c1_b'])
        bn1 = self._batch_norm(c1, w['bn1_gamma'], w['bn1_beta'], w['bn1_mean'], w['bn1_var'])
        p1 = self._max_pool1d(bn1, stride=4)

        # Layer 2
        c2 = self._conv1d_same_relu(p1, w['c2_w'], w['c2_b'])
        bn2 = self._batch_norm(c2, w['bn2_gamma'], w['bn2_beta'], w['bn2_mean'], w['bn2_var'])
        p2 = self._max_pool1d(bn2, stride=4)

        # Layer 3
        c3 = self._conv1d_same_relu(p2, w['c3_w'], w['c3_b'])
        bn3 = self._batch_norm(c3, w['bn3_gamma'], w['bn3_beta'], w['bn3_mean'], w['bn3_var'])
        gap = np.mean(bn3, axis=1)

        # Dense layers
        d1 = np.maximum(gap @ w['d1_w'] + w['d1_b'], 0.0)
        logits = d1 @ w['d2_w'] + w['d2_b']
        return self._softmax(logits)

    def classify(
        self,
        normed_window: np.ndarray,
        raw_rms: float = None,
        rms_threshold: float = 0.01,
        confidence_threshold: float = 0.40
    ) -> tuple[str, dict[str, float], float, str]:
        """Classifies preprocessed window with REST safeguards."""
        if raw_rms is not None and raw_rms < rms_threshold:
            return 'REST', {'PINCH': 0.0, 'FIST': 0.0, 'OPEN': 0.0, 'REST': 1.0}, 1.0, f"Inactivity (RMS {raw_rms:.5f} < {rms_threshold})"

        probs = self.forward(normed_window)[0]
        probs_dict = {self.CLASSES[i]: float(probs[i]) for i in range(len(self.CLASSES))}
        best_idx = int(np.argmax(probs))
        pred = self.CLASSES[best_idx]
        conf = float(probs[best_idx])

        if conf < confidence_threshold:
            probs_dict['REST'] = 1.0 - conf
            return 'REST', probs_dict, conf, f"Low Confidence ({conf:.2f} < {confidence_threshold})"

        return pred, probs_dict, conf, "CNN Inference"

    @staticmethod
    def _conv1d_same_relu(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        batch, time_len, in_c = x.shape
        k, _, out_c = W.shape
        pad_total = k - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        x_padded = np.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant')
        windows = np.lib.stride_tricks.sliding_window_view(x_padded, window_shape=k, axis=1)
        windows = np.transpose(windows, (0, 1, 3, 2))
        windows_flat = windows.reshape(batch * time_len, k * in_c)
        W_flat = W.reshape(k * in_c, out_c)
        out = (windows_flat @ W_flat + b).reshape(batch, time_len, out_c)
        return np.maximum(out, 0.0)

    @staticmethod
    def _batch_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, mean: np.ndarray, var: np.ndarray, eps: float = 0.001) -> np.ndarray:
        inv_std = 1.0 / np.sqrt(var + eps)
        return (x - mean) * (inv_std * gamma) + beta

    @staticmethod
    def _max_pool1d(x: np.ndarray, stride: int = 4) -> np.ndarray:
        B, T, C = x.shape
        new_T = T // stride
        trimmed = x[:, : new_T * stride, :]
        reshaped = trimmed.reshape(B, new_T, stride, C)
        return np.max(reshaped, axis=2)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
