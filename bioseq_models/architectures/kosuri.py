"""Two-branch CNN for Kosuri promoter/RBS composability."""

from __future__ import annotations

import torch
from torch import nn


class KosuriTwoBranchCNN(nn.Module):
    """Encode promoter and RBS separately before joint two-task regression."""

    def __init__(self, output_dim: int = 2) -> None:
        super().__init__()

        def branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(4, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 96, kernel_size=5, padding=2),
                nn.BatchNorm1d(96),
                nn.GELU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Flatten(),
            )

        self.promoter_encoder = branch()
        self.rbs_encoder = branch()
        self.head = nn.Sequential(
            nn.Linear(96 * 2 + 2, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, output_dim),
        )

    def forward(
        self,
        promoter_tensor: torch.Tensor,
        rbs_tensor: torch.Tensor,
        component_lengths: torch.Tensor,
    ) -> torch.Tensor:
        promoter = self.promoter_encoder(promoter_tensor)
        rbs = self.rbs_encoder(rbs_tensor)
        return self.head(torch.cat([promoter, rbs, component_lengths], dim=1))
