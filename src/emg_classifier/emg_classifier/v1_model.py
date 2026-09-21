#!/usr/bin/env python3
"""
V1 EMG CNN Model and Preprocessing Module for BiDexHand.

Model Specifications:
  - 3-class 1D CNN: PINCH (0), FIST (1), OPEN (2)
  - Selected dataset channels: [12, 11, 7, 2] (1-based -> indices 11, 10, 6, 1)
  - Target sampling rate: 1000 Hz
  - Window: 400 ms (400 samples)
  - DC removal -> 20-450 Hz bandpass -> train-only mean/std normalization -> CNN forward pass
  - Inactivity/low-confidence state maps to REST (REST is not a trained V1 class)
"""

import json
import os
import zipfile
import numpy as np
import scipy.signal


class V1EMGModel:
    """
    Handles V1 EMG preprocessing, model loading, and CNN inference.
    """

    CLASSES = ['PINCH', 'FIST', 'OPEN']
    SELECTED_CHANNELS_1BASED = [12, 11, 7, 2]
    SELECTED_CHANNELS_0BASED = [11, 10, 6, 1]
    TARGET_FS = 1000
    WINDOW_SAMPLES = 400

    def __init__(self, model_path: str, mean_path: str, std_path: str, logger=None):
        self.logger = logger
        self.model_path = os.path.expanduser(model_path)
        self.mean_path = os.path.expanduser(mean_path)
        self.std_path = os.path.expanduser(std_path)

        # Build Butterworth bandpass filter coefficients (20 - 450 Hz @ 1000 Hz)
        self.sos = scipy.signal.butter(
            4, [20.0, 450.0], btype='bandpass', fs=float(self.TARGET_FS), output='sos'
        )

        # Load normalization stats
        self.train_mean = self._load_array_or_json(self.mean_path, key='mean')
        self.train_std = self._load_array_or_json(self.std_path, key='std')

        # Load CNN weights
        self.weights = self._load_model_weights(self.model_path)
        self._log(f"V1 EMG Model loaded successfully from: {self.model_path}")
        self._log(f"Normalization stats: mean shape {self.train_mean.shape}, std shape {self.train_std.shape}")

    def _log(self, message: str):
        if self.logger:
            self.logger.info(message)
        else:
            print(f"[V1EMGModel] {message}", flush=True)

    def _load_array_or_json(self, path: str, key: str = None) -> np.ndarray:
        """Load float32 numpy array from .npy, .npz, or .json file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Normalization file not found: {path}")

        if path.endswith('.npy'):
            arr = np.load(path)
        elif path.endswith('.npz'):
            data = np.load(path)
            arr = data[key] if key in data else data[list(data.keys())[0]]
        elif path.endswith('.json'):
            with open(path, 'r') as f:
                data = json.load(f)
            if key and key in data:
                arr = np.array(data[key], dtype=np.float32)
            else:
                arr = np.array(data, dtype=np.float32)
        else:
            arr = np.load(path)

        return arr.astype(np.float32).reshape(-1)

    def _load_model_weights(self, model_path: str) -> dict:
        """
        Load model weights directly from .keras or .npz.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Check for cached .npz weights alongside the .keras file
        npz_candidate = model_path.replace('.keras', '_weights.npz')
        if os.path.exists(npz_candidate):
            self._log(f"Loading pre-extracted weights from: {npz_candidate}")
            npz = np.load(npz_candidate)
            return {k: npz[k] for k in npz.files}

        # If .npz directly passed
        if model_path.endswith('.npz'):
            npz = np.load(model_path)
            return {k: npz[k] for k in npz.files}

        # If .keras file passed, extract model.weights.h5 using h5py
        if model_path.endswith('.keras'):
            self._log(f"Extracting weights from Keras archive: {model_path}")
            try:
                import h5py
            except ImportError:
                raise ImportError(
                    "h5py is required to inspect .keras model files. "
                    "Ensure python3-h5py is installed or use an .npz weights export."
                )

            z = zipfile.ZipFile(model_path, 'r')
            h5_bytes = z.read('model.weights.h5')
            temp_h5 = f"/tmp/_tmp_v1_weights_{os.getpid()}.h5"
            with open(temp_h5, 'wb') as f:
                f.write(h5_bytes)

            try:
                f = h5py.File(temp_h5, 'r')
                weights = {
                    'c1_w': np.array(f['layers/conv1d/vars/0'], dtype=np.float32),
                    'c1_b': np.array(f['layers/conv1d/vars/1'], dtype=np.float32),
                    'bn1_gamma': np.array(f['layers/batch_normalization/vars/0'], dtype=np.float32),
                    'bn1_beta': np.array(f['layers/batch_normalization/vars/1'], dtype=np.float32),
                    'bn1_mean': np.array(f['layers/batch_normalization/vars/2'], dtype=np.float32),
                    'bn1_var': np.array(f['layers/batch_normalization/vars/3'], dtype=np.float32),
                    'c2_w': np.array(f['layers/conv1d_1/vars/0'], dtype=np.float32),
                    'c2_b': np.array(f['layers/conv1d_1/vars/1'], dtype=np.float32),
                    'bn2_gamma': np.array(f['layers/batch_normalization_1/vars/0'], dtype=np.float32),
                    'bn2_beta': np.array(f['layers/batch_normalization_1/vars/1'], dtype=np.float32),
                    'bn2_mean': np.array(f['layers/batch_normalization_1/vars/2'], dtype=np.float32),
                    'bn2_var': np.array(f['layers/batch_normalization_1/vars/3'], dtype=np.float32),
                    'c3_w': np.array(f['layers/conv1d_2/vars/0'], dtype=np.float32),
                    'c3_b': np.array(f['layers/conv1d_2/vars/1'], dtype=np.float32),
                    'bn3_gamma': np.array(f['layers/batch_normalization_2/vars/0'], dtype=np.float32),
                    'bn3_beta': np.array(f['layers/batch_normalization_2/vars/1'], dtype=np.float32),
                    'bn3_mean': np.array(f['layers/batch_normalization_2/vars/2'], dtype=np.float32),
                    'bn3_var': np.array(f['layers/batch_normalization_2/vars/3'], dtype=np.float32),
                    'd1_w': np.array(f['layers/dense/vars/0'], dtype=np.float32),
                    'd1_b': np.array(f['layers/dense/vars/1'], dtype=np.float32),
                    'd2_w': np.array(f['layers/dense_1/vars/0'], dtype=np.float32),
                    'd2_b': np.array(f['layers/dense_1/vars/1'], dtype=np.float32),
                }
                f.close()
                return weights
            finally:
                if os.path.exists(temp_h5):
                    os.remove(temp_h5)

        raise ValueError(f"Unsupported model format: {model_path}")

    def preprocess(self, raw_emg_data) -> tuple[np.ndarray, float]:
        """
        V1 Pipeline Preprocessing:
          1. Channel selection (if >4 channels, select [11, 10, 6, 1])
          2. Central 400 ms window extraction (if trial > 400 samples)
          3. DC removal
          4. 20-450 Hz bandpass filter
          5. Train-only normalization (train_mean, train_std)

        Returns:
          normalized_window: np.ndarray shape (400, 4)
          raw_rms: float (RMS energy for inactivity detection)
        """
        arr = np.asarray(raw_emg_data, dtype=np.float32)

        # Handle 1D flat buffer
        if arr.ndim == 1:
            if arr.size == self.WINDOW_SAMPLES * 4:
                arr = arr.reshape(self.WINDOW_SAMPLES, 4)
            elif arr.size % 4 == 0:
                arr = arr.reshape(-1, 4)
            elif arr.size % 12 == 0:
                arr = arr.reshape(-1, 12)
            else:
                raise ValueError(f"Flat raw EMG buffer size {arr.size} cannot be reshaped to 4 or 12 channels.")

        # Channel selection
        if arr.shape[1] >= 12:
            arr = arr[:, self.SELECTED_CHANNELS_0BASED]
        elif arr.shape[1] != 4:
            raise ValueError(f"Expected 4 or >=12 channels, but received shape {arr.shape}.")

        # Window extraction: central 400 ms
        n_samples = arr.shape[0]
        if n_samples > self.WINDOW_SAMPLES:
            start_idx = (n_samples - self.WINDOW_SAMPLES) // 2
            window = arr[start_idx : start_idx + self.WINDOW_SAMPLES, :]
        elif n_samples == self.WINDOW_SAMPLES:
            window = arr
        else:
            # Pad if shorter than 400 samples
            pad_needed = self.WINDOW_SAMPLES - n_samples
            window = np.pad(arr, ((0, pad_needed), (0, 0)), mode='edge')

        # Compute raw RMS energy across all 4 channels
        raw_rms = float(np.sqrt(np.mean(window ** 2)))

        # Step 1: DC removal
        dc_removed = window - np.mean(window, axis=0, keepdims=True)

        # Step 2: 20-450 Hz Butterworth bandpass filter
        filtered = scipy.signal.sosfiltfilt(self.sos, dc_removed, axis=0)

        # Step 3: Train-only normalization
        normed = (filtered - self.train_mean) / (self.train_std + 1e-8)

        return normed.astype(np.float32), raw_rms

    def forward(self, normed_window: np.ndarray) -> np.ndarray:
        """
        Pure NumPy forward pass matching the exact trained Keras CNN architecture.
        Input: (400, 4) or (batch, 400, 4)
        Output: (batch, 3) softmax probabilities
        """
        if normed_window.ndim == 2:
            x = normed_window[np.newaxis, ...]
        else:
            x = normed_window

        w = self.weights

        # Layer 1: Conv1D (15, 32, relu) -> BatchNorm -> MaxPool(4)
        c1 = self._conv1d_same_relu(x, w['c1_w'], w['c1_b'])
        bn1 = self._batch_norm(c1, w['bn1_gamma'], w['bn1_beta'], w['bn1_mean'], w['bn1_var'])
        p1 = self._max_pool1d(bn1, stride=4)

        # Layer 2: Conv1D (11, 64, relu) -> BatchNorm -> MaxPool(4)
        c2 = self._conv1d_same_relu(p1, w['c2_w'], w['c2_b'])
        bn2 = self._batch_norm(c2, w['bn2_gamma'], w['bn2_beta'], w['bn2_mean'], w['bn2_var'])
        p2 = self._max_pool1d(bn2, stride=4)

        # Layer 3: Conv1D (7, 128, relu) -> BatchNorm -> GlobalAveragePooling1D
        c3 = self._conv1d_same_relu(p2, w['c3_w'], w['c3_b'])
        bn3 = self._batch_norm(c3, w['bn3_gamma'], w['bn3_beta'], w['bn3_mean'], w['bn3_var'])
        gap = np.mean(bn3, axis=1)  # (batch, 128)

        # Dense 1: 64 relu
        d1 = np.maximum(gap @ w['d1_w'] + w['d1_b'], 0.0)

        # Dense 2: 3 softmax
        logits = d1 @ w['d2_w'] + w['d2_b']
        probs = self._softmax(logits)

        return probs

    def classify(
        self,
        normed_window: np.ndarray,
        raw_rms: float = None,
        rms_threshold: float = 0.01,
        confidence_threshold: float = 0.40,
    ) -> tuple[str, dict[str, float], float, str]:
        """
        Classifies preprocessed EMG window.

        REST handling:
          - V1 was NOT trained with REST as a gesture class.
          - If raw_rms < rms_threshold, classified as REST (inactivity state).
          - If confidence < confidence_threshold, falls back to REST (uncertainty state).

        Returns:
          (gesture_name, probabilities_dict, confidence, status_reason)
        """
        # Inactivity threshold check for REST
        if raw_rms is not None and raw_rms < rms_threshold:
            probs_dict = {'PINCH': 0.0, 'FIST': 0.0, 'OPEN': 0.0, 'REST': 1.0}
            return 'REST', probs_dict, 1.0, f"Inactivity (RMS {raw_rms:.5f} < {rms_threshold})"

        # CNN forward pass
        probs = self.forward(normed_window)[0]
        probs_dict = {
            'PINCH': float(probs[0]),
            'FIST': float(probs[1]),
            'OPEN': float(probs[2]),
        }

        best_idx = int(np.argmax(probs))
        predicted_gesture = self.CLASSES[best_idx]
        confidence = float(probs[best_idx])

        # Low-confidence threshold check
        if confidence < confidence_threshold:
            probs_dict['REST'] = 1.0 - confidence
            return 'REST', probs_dict, confidence, f"Low Confidence ({confidence:.2f} < {confidence_threshold})"

        return predicted_gesture, probs_dict, confidence, "CNN Inference"

    @staticmethod
    def _conv1d_same_relu(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Vectorized 1D Convolution with 'same' padding and ReLU activation."""
        batch, time_len, in_c = x.shape
        k, _, out_c = W.shape
        pad_total = k - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        x_padded = np.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant')
        windows = np.lib.stride_tricks.sliding_window_view(x_padded, window_shape=k, axis=1)
        windows = np.transpose(windows, (0, 1, 3, 2))  # (B, T, K, C)
        windows_flat = windows.reshape(batch * time_len, k * in_c)
        W_flat = W.reshape(k * in_c, out_c)
        out = (windows_flat @ W_flat + b).reshape(batch, time_len, out_c)
        return np.maximum(out, 0.0)

    @staticmethod
    def _batch_norm(
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
        eps: float = 0.001,
    ) -> np.ndarray:
        """Batch Normalization inference pass."""
        inv_std = 1.0 / np.sqrt(var + eps)
        return (x - mean) * (inv_std * gamma) + beta

    @staticmethod
    def _max_pool1d(x: np.ndarray, stride: int = 4) -> np.ndarray:
        """1D Max Pooling with pool_size=stride."""
        B, T, C = x.shape
        new_T = T // stride
        trimmed = x[:, : new_T * stride, :]
        reshaped = trimmed.reshape(B, new_T, stride, C)
        return np.max(reshaped, axis=2)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation along last axis."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
