"""PyTorch architecture for the 1,001-bp tissue-specific DeepSTARR models."""

from __future__ import annotations

import torch
from torch import nn


class DeepSTARRTissueNetwork(nn.Module):
    """Shared architecture used by DeepSTARR-Mouse and DeepSTARR-Embryo.

    The transpose before flattening preserves the feature order of the
    channels-last Keras models from which the checkpoints were converted.
    """

    sequence_length = 1001

    def __init__(self, *, output_kind: str = "activity") -> None:
        super().__init__()
        kind = str(output_kind).strip().lower()
        if kind not in {"accessibility", "activity"}:
            raise ValueError("output_kind must be 'accessibility' or 'activity'")
        self.output_kind = kind

        batch_norm = {"eps": 1e-3, "momentum": 0.01}
        channels = (4, 256, 120, 60, 60)
        kernels = (7, 3, 3, 3)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(in_channels, out_channels, kernel, padding="same")
            for in_channels, out_channels, kernel in zip(
                channels[:-1], channels[1:], kernels
            )
        )
        self.conv_batch_norms = nn.ModuleList(
            nn.BatchNorm1d(out_channels, **batch_norm) for out_channels in channels[1:]
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)

        self.dense_1 = nn.Linear(12 * 60, 64)
        self.dense_1_batch_norm = nn.BatchNorm1d(64, **batch_norm)
        self.dense_2 = nn.Linear(64, 256)
        self.dense_2_batch_norm = nn.BatchNorm1d(256, **batch_norm)
        self.dropout = nn.Dropout(0.4)
        self.output = nn.Linear(256, 1)

    def forward_features(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        hidden = sequence_tensor
        for convolution, batch_norm in zip(self.convolutions, self.conv_batch_norms):
            hidden = self.pool(torch.relu(batch_norm(convolution(hidden))))
        hidden = hidden.transpose(1, 2).contiguous().flatten(start_dim=1)
        hidden = self.dropout(torch.relu(self.dense_1_batch_norm(self.dense_1(hidden))))
        return self.dropout(torch.relu(self.dense_2_batch_norm(self.dense_2(hidden))))

    def forward(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        output = self.output(self.forward_features(sequence_tensor))
        return torch.sigmoid(output) if self.output_kind == "activity" else output


class DeepSTARRMouseNetwork(DeepSTARRTissueNetwork):
    """DeepSTARR-Mouse architecture from Chen et al. (2026)."""


class DeepSTARREmbryoNetwork(DeepSTARRTissueNetwork):
    """Drosophila embryo architecture from de Almeida et al. (2023)."""
