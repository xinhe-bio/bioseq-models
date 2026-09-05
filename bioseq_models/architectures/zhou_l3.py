"""Position-aware CNN for fixed-length L3 promoters."""

from __future__ import annotations

import torch
from torch import nn


class L3PositionAwareCNN(nn.Module):
    """CNN retaining the absolute locations of the -35 and -10 boxes.

    L3 has a fixed 29-nt construct and its assay deliberately varies the
    position-specific -35/-10 sequence pairs.  Unlike a motif detector with
    global max pooling, this CNN flattens the final feature map before its
    regression head so distinct promoter positions cannot be interchanged.
    """

    def __init__(self, output_dim: int = 1, length: int = 29) -> None:
        super().__init__()
        self.length = length
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Conv1d(96, 96, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * length, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

    def forward(self, sequence_tensor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(sequence_tensor))


class L3ConditionalCNN(nn.Module):
    """Position-aware CNN conditioned on the documented L3 sublibrary."""

    def __init__(
        self, num_sublibraries: int, output_dim: int = 1, length: int = 29
    ) -> None:
        super().__init__()
        if num_sublibraries < 2:
            raise ValueError("L3ConditionalCNN requires the s1--s4 sublibrary context")
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(96 * length, 256),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        # Separate small heads let the −10/−35 interaction function differ by
        # the explicitly documented source library without leaking test labels.
        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, output_dim))
                for _ in range(num_sublibraries)
            ]
        )

    def forward(
        self,
        sequence_tensor: torch.Tensor,
        cell_type_id: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encoder(sequence_tensor)
        predictions = torch.stack([head(features) for head in self.heads], dim=1)
        indices = (
            cell_type_id.long().view(-1, 1, 1).expand(-1, 1, predictions.shape[-1])
        )
        return predictions.gather(1, indices).squeeze(1)
