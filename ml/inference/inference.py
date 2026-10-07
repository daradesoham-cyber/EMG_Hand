"""
Real-Time Streaming Inference Pipeline & MuJoCo HASEL Command Interface.

Pipeline:
16-channel sEMG (2 kHz)
    ↓
Causal IIR Filter (Bandpass 20-450 Hz + 50 Hz Notch)
    ↓
256-sample Ring Buffer (128 ms temporal history)
    ↓ (every 64 samples = 32 ms)
Causal TCN Model
    ↓
Continuous Output [thumb, index, middle, ring, pinky] in [0, 1]
    ↓
EMA Filter (alpha = 0.20)
    ↓
Six-Channel HASEL Actuator Interface [H1, H2, H3, H4, H5, H6] in [0, 1]
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch

from ml.preprocessing import EMGPreprocessor
from ml.windowing import WindowConfig, StreamingWindowBuffer
from ml.model import CausalTCN


class RealTimeEMGController:
    """
    Complete streaming controller for real-time prosthetic hand deployment.
    """

    def __init__(
        self,
        checkpoint_path: str = "results/best_model_6ch.pt",
        device_str: str = "cpu",
        ema_alpha: float = 0.20,
    ):
        self.device = torch.device(device_str)
        self.ema_alpha = float(ema_alpha)

        # 1. Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})

        self.target_names = ckpt.get("target_names") or cfg.get("features", {}).get(
            "target_names", ["index", "middle", "ring", "little", "thumb_flexion", "thumb_opposition"]
        )
        self.num_outputs = len(self.target_names)

        # 2. Reconstruct Model
        m_cfg = cfg.get("model", {})
        self.model = CausalTCN(
            in_channels=m_cfg.get("in_channels", 16),
            hidden_channels=m_cfg.get("hidden_channels", [32, 64, 64, 128, 128, 128]),
            kernel_size=m_cfg.get("kernel_size", 3),
            dilations=m_cfg.get("dilations", [1, 2, 4, 8, 16, 32]),
            num_outputs=self.num_outputs,
            convs_per_block=2,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # 3. Setup Preprocessor with trained normalization stats
        prep_stats = ckpt.get("preprocessor_stats", {})
        self.preprocessor = EMGPreprocessor(
            sampling_rate=2000.0,
            bandpass_low=20.0,
            bandpass_high=450.0,
            notch_freq=50.0,
            num_channels=16,
        )
        if "mean" in prep_stats and "std" in prep_stats:
            self.preprocessor.set_normalization_stats(
                prep_stats["mean"], prep_stats["std"]
            )

        # 4. Ring Buffer
        self.window_config = WindowConfig(sampling_rate=2000, window_ms=128.0, stride_ms=32.0)
        self.ring_buffer = StreamingWindowBuffer(self.window_config, num_channels=16)

        # 5. EMA filter state
        self.last_ema_output = np.zeros(self.num_outputs, dtype=np.float32)
        self.last_hasel_commands = np.zeros(6, dtype=np.float32)

    def process_sample(self, raw_emg_sample: np.ndarray) -> Optional[np.ndarray]:
        """
        Process single 16-channel sample arriving at 2 kHz.

        Returns:
            (6,) HASEL actuator command vector if stride interval (64 samples) is met,
            else None.
        """
        # Step 1: Causal IIR conditioning
        conditioned = self.preprocessor.process_streaming_sample(raw_emg_sample)

        # Step 2: Push into ring buffer
        window = self.ring_buffer.push_sample(conditioned)
        if window is None:
            return None

        # Step 3: Inference (every 32 ms)
        x_tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)  # (1, 16, 256)
        with torch.no_grad():
            pred = self.model(x_tensor).cpu().numpy()[0]  # (num_outputs,)

        # Step 4: EMA Post-Filter
        self.last_ema_output = (
            self.ema_alpha * pred + (1.0 - self.ema_alpha) * self.last_ema_output
        )

        # Step 5: Map to Six HASEL Channels
        self.last_hasel_commands = self.map_to_hasel_actuators(self.last_ema_output)
        return self.last_hasel_commands.copy()

    def map_to_hasel_actuators(self, finger_flexion: np.ndarray) -> np.ndarray:
        """
        Map continuous ML predictions to six HASEL actuator channels.

        Channels:
        H1: Index flexion
        H2: Middle flexion
        H3: Ring flexion
        H4: Little flexion
        H5: Thumb flexion
        H6: Thumb opposition / CMC circumduction
        """
        hasel = np.zeros(6, dtype=np.float32)

        if len(finger_flexion) == 6:
            # Standard 6-output model: [index, middle, ring, little, thumb_flexion, thumb_opposition]
            hasel[0] = finger_flexion[0]  # H1 Index
            hasel[1] = finger_flexion[1]  # H2 Middle
            hasel[2] = finger_flexion[2]  # H3 Ring
            hasel[3] = finger_flexion[3]  # H4 Little
            hasel[4] = finger_flexion[4]  # H5 Thumb Flexion
            hasel[5] = finger_flexion[5]  # H6 Thumb Opposition
        elif len(finger_flexion) == 5:
            # Legacy 5-output fallback: [thumb, index, middle, ring, pinky]
            th, idx, mid, rng, pnk = finger_flexion
            hasel[0] = idx  # H1 Index
            hasel[1] = mid  # H2 Middle
            hasel[2] = rng  # H3 Ring
            hasel[3] = pnk  # H4 Little
            hasel[4] = th   # H5 Thumb Flexion
            mean_fingers = (idx + mid + rng + pnk) / 4.0
            hasel[5] = np.clip(0.7 * th + 0.3 * mean_fingers, 0.0, 1.0)
        elif len(finger_flexion) > 6:
            hasel[:6] = finger_flexion[:6]

        return np.clip(hasel, 0.0, 1.0)
