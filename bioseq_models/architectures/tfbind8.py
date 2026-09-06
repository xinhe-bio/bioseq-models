"""CNN used for the TFBind8 lower-tail surrogate ensemble."""

from __future__ import annotations

import torch
from torch import nn


class TFBindCNN(nn.Module):
    """Two-convolution CNN used for TFBind8 and TFBind10 weak models."""

    def __init__(
        self,
        input_channels: int = 4,
        output_dim: int = 1,
        channels: tuple[int, int] = (32, 64),
        kernel_size: int = 3,
        hidden_units: int = 128,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv1d(
                input_channels,
                channels[0],
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Conv1d(
                channels[0],
                channels[1],
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.regressor = nn.Sequential(
            nn.LazyLinear(hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim),
        )

    def forward(self, sequence_tensor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.encoder(sequence_tensor))
