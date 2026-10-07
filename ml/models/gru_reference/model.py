"""
Causal GRU Reference / Benchmark Model.

Evaluated as an accuracy benchmark against the Causal TCN:
- 2-layer causal Gated Recurrent Unit (41,094 parameters).
- Processed sequence forward in time: (B, 16, 256) -> (B, 256, 16).
- Accuracy reference: Achieved higher predictive R² (0.6483 with EMA) on Session 3,
  but failed the 32 ms real-time deadline with measured CPU latency of 57.35 ms (p95: 91.04 ms).
- Retained strictly as an offline accuracy benchmark to justify the TCN selection.
"""

import torch
import torch.nn as nn


class CausalGRUBaseline(nn.Module):
    """
    Causal Gated Recurrent Unit (GRU) baseline for continuous sEMG regression.
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
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_outputs = num_outputs

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
