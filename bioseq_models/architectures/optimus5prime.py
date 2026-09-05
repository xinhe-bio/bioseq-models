"""PyTorch implementation of the Optimus 5-Prime CNN."""

from __future__ import annotations

import torch
from torch import nn


class Optimus5Prime(nn.Module):
    """Sequence-only CNN from Sample et al. (2019)."""

    def __init__(self, input_channels: int = 4, output_dim: int = 1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 120, 8, padding="same"),
            nn.ReLU(),
            nn.Conv1d(120, 120, 8, padding="same"),
            nn.ReLU(),
            nn.Conv1d(120, 120, 8, padding="same"),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(120 * 50, 40),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(40, output_dim),
        )

    def forward(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(sequence_tensor))
