"""
Causal Dilated Temporal Convolutional Network (TCN) for Continuous EMG Regression.

Architecture Specifications:
- Input: 16 sEMG channels (B, 16, L)
- 4 Causal Residual Dilated Conv1D Blocks
- Dilations: [1, 2, 4, 8]
- Kernel Size: 3
- Normalization & Regularization: BatchNorm1d, LeakyReLU, Dropout(0.2)
- Channel Progression: 16 -> 32 -> 64 -> 64 -> 128
- Regression Head: Linear layer -> 5 (or 6) continuous outputs in [0, 1]
- Strict Causality: Past-only padding guarantees no future sample access.
"""

from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    1D Convolution with strict causal left-padding.

    Pads (kernel_size - 1) * dilation zeros strictly on the past side.
    Output sequence length equals input sequence length.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad strictly on the left (time t - dt)
        x_padded = F.pad(x, (self.causal_padding, 0))
        return self.conv(x_padded)


class CausalResidualBlock(nn.Module):
    """
    Causal Residual Dilated Block with BatchNorm, LeakyReLU, and Dropout.

    Supports 1 or 2 convolutions per block (configurable).
    Default convs_per_block=2 follows standard Bai et al. 2018 TCN design.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
        convs_per_block: int = 2,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.convs_per_block = convs_per_block

        # Layer 1
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=negative_slope)
        self.drop1 = nn.Dropout(dropout)

        # Layer 2 (optional standard TCN structure)
        if convs_per_block == 2:
            self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.act2 = nn.LeakyReLU(negative_slope=negative_slope)
            self.drop2 = nn.Dropout(dropout)

        # Residual shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        if self.convs_per_block == 2:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.act2(out)
            out = self.drop2(out)

        return out + residual


class CausalTCN(nn.Module):
    """
    Causal Dilated Temporal Convolutional Network for continuous EMG regression.
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: List[int] = [32, 64, 64, 128],
        kernel_size: int = 3,
        dilations: List[int] = [1, 2, 4, 8],
        dropout: float = 0.2,
        num_outputs: int = 5,
        convs_per_block: int = 2,
        head_activation: str = "linear",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.num_outputs = num_outputs
        self.convs_per_block = convs_per_block
        self.head_activation = head_activation

        assert len(hidden_channels) == len(dilations), (
            f"Mismatch: {len(hidden_channels)} channels vs {len(dilations)} dilations"
        )

        blocks = []
        curr_in = in_channels
        for out_ch, d in zip(hidden_channels, dilations):
            blocks.append(
                CausalResidualBlock(
                    in_channels=curr_in,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                    convs_per_block=convs_per_block,
                )
            )
            curr_in = out_ch

        self.network = nn.Sequential(*blocks)
        self.regression_head = nn.Linear(curr_in, num_outputs)

        if head_activation == "sigmoid":
            self.output_act = nn.Sigmoid()
        else:
            self.output_act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, C, L) input batch.
        Returns:
            (B, num_outputs) continuous joint prediction at final causal sample.
        """
        features = self.network(x)  # (B, 128, L)
        # Extract features at the final causal time step t (last column)
        last_timestep = features[:, :, -1]  # (B, 128)
        pred = self.regression_head(last_timestep)  # (B, num_outputs)
        pred = self.output_act(pred)
        if self.head_activation == "linear":
            pred = torch.clamp(pred, 0.0, 1.0)
        return pred

    def compute_receptive_field(self) -> Dict[str, Any]:
        """
        Derive exact mathematical receptive field.

        Formula:
        RF = 1 + convs_per_block * (kernel_size - 1) * sum(dilations)
        """
        sum_dilations = sum(self.dilations)
        rf_samples = 1 + self.convs_per_block * (self.kernel_size - 1) * sum_dilations
        rf_ms_at_2khz = (rf_samples / 2000.0) * 1000.0

        return {
            "convs_per_block": self.convs_per_block,
            "kernel_size": self.kernel_size,
            "dilations": self.dilations,
            "sum_dilations": sum_dilations,
            "receptive_field_samples": rf_samples,
            "receptive_field_ms": rf_ms_at_2khz,
            "input_window_samples": 256,
            "input_window_ms": 128.0,
            "covers_full_window": rf_samples >= 256,
            "coverage_ratio": rf_samples / 256.0,
        }

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Baselines for comparison
class PersistenceBaseline:
    """
    Oracle Persistence / Temporal Smoothness Reference.

    Predicts the next state using the previous ground-truth kinematic state y[t-1].
    NOTE: This is an ORACLE reference serving as an empirical upper bound on kinematic
    continuity, NOT an sEMG-only causal predictor. It requires real-time glove feedback.
    """

    def __init__(self, num_outputs: int = 6):
        self.num_outputs = num_outputs

    def predict(self, current_target_at_t_minus_1: np.ndarray) -> np.ndarray:
        return current_target_at_t_minus_1.copy()


class LinearRegressionBaseline(nn.Module):
    """
    Linear regression baseline mapping causal sEMG statistics to targets.

    Extracts 4 standard time-domain features per channel:
    - Mean Absolute Value (MAV)
    - Root Mean Square (RMS)
    - Variance (VAR)
    - Waveform Length (WL)
    Total input dimension = 16 * 4 = 64 features.
    """

    def __init__(self, in_features: int = 16 * 4, num_outputs: int = 6):
        super().__init__()
        self.linear = nn.Linear(in_features, num_outputs)

    def forward(self, x_features: torch.Tensor) -> torch.Tensor:
        out = self.linear(x_features)
        return torch.clamp(out, 0.0, 1.0)

    @staticmethod
    def extract_features(x_raw: torch.Tensor) -> torch.Tensor:
        """
        Extract time-domain features from (B, C, L) sEMG window.
        Returns (B, C * 4) feature tensor.
        """
        mav = torch.mean(torch.abs(x_raw), dim=-1)
        rms = torch.sqrt(torch.mean(x_raw ** 2, dim=-1) + 1e-8)
        var = torch.var(x_raw, dim=-1)
        wl = torch.sum(torch.abs(torch.diff(x_raw, dim=-1)), dim=-1) / x_raw.size(-1)
        return torch.cat([mav, rms, var, wl], dim=-1)


class SimpleMLPBaseline(nn.Module):
    """Simple 2-layer MLP on flattened temporal sEMG windows."""

    def __init__(
        self, in_channels: int = 16, window_samples: int = 256, num_outputs: int = 6
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * window_samples, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return torch.clamp(out, 0.0, 1.0)


class CausalGRUBaseline(nn.Module):
    """
    Causal Gated Recurrent Unit (GRU) baseline for continuous sEMG regression.

    Processes the (B, 16, L) sequence strictly forward in time (B, L, 16)
    and predicts continuous joint states at the final causal time step.
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_outputs: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> transpose to (B, L, C)
        x_seq = x.transpose(1, 2)
        out, _ = self.gru(x_seq)
        last_step = out[:, -1, :]  # (B, hidden_dim)
        pred = self.fc(last_step)
        return torch.clamp(pred, 0.0, 1.0)
