"""Small sequence-only CNN for the Zhao 2022 Ptrc promoter library.

This is a BioSeqBench differentiable comparator, not an architecture claimed by
Zhao et al.  The original paper selected a gradient-boosted tree predictor.
"""

from __future__ import annotations

import torch
from torch import nn


class PtrcPromoterCNN(nn.Module):
    """Compact motif-CNN for 67--76 nt Ptrc-derived promoter sequences."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(sequence_tensor)).squeeze(1)
