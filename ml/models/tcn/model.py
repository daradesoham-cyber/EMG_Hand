"""
Causal Dilated Temporal Convolutional Network (TCN) for Continuous EMG Regression.

Architecture:
- Input: 16 sEMG channels (B, 16, L)
- 6 Causal Residual Dilated Conv1D Blocks (Model C)
- Dilations: [1, 2, 4, 8, 16, 32]
- Kernel Size: 3, Convs per Block: 2
- Hidden Channels: [32, 64, 64, 128, 128, 128]
- Receptive Field: 253 samples (126.5 ms at 2 kHz), covering 98.8% of the 256-sample window.
- Parameter Count: 332,966 trainable parameters.
- Normalization & Regularization: BatchNorm1d, LeakyReLU (0.01), Dropout (0.20), 1x1 Conv shortcut projections.
- Regression Head: Linear(128, 6) with clamp to [0.0, 1.0].
"""

from typing import List, Tuple, Optional, Dict, Any
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
        # Pad strictly on the left (past samples)
        x_padded = F.pad(x, (self.causal_padding, 0))
        return self.conv(x_padded)


class CausalResidualBlock(nn.Module):
    """
    Causal Residual Dilated Block with BatchNorm, LeakyReLU, and Dropout.
    Supports 1 or 2 convolutions per block (default 2 follows Bai et al. 2018).
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

        # Layer 2
        if convs_per_block == 2:
            self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.act2 = nn.LeakyReLU(negative_slope=negative_slope)
            self.drop2 = nn.Dropout(dropout)

        # Residual shortcut projection
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
    Frozen Causal Dilated Temporal Convolutional Network for continuous sEMG-to-actuator regression.
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: List[int] = [32, 64, 64, 128, 128, 128],
        kernel_size: int = 3,
        dilations: List[int] = [1, 2, 4, 8, 16, 32],
        dropout: float = 0.2,
        num_outputs: int = 6,
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
            x: (B, C, L) batch of sEMG windows.
        Returns:
            (B, num_outputs) continuous joint prediction at final causal sample.
        """
        features = self.network(x)  # (B, 128, L)
        last_timestep = features[:, :, -1]  # (B, 128)
        pred = self.regression_head(last_timestep)  # (B, num_outputs)
        pred = self.output_act(pred)
        if self.head_activation == "linear":
            pred = torch.clamp(pred, 0.0, 1.0)
        return pred

    def compute_receptive_field(self) -> Dict[str, Any]:
        """
        Derive exact mathematical receptive field.
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
            "coverage_ratio": float(rf_samples) / 256.0,
        }
