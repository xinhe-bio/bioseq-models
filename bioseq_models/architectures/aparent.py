"""APARENT alternative-polyadenylation network."""

from __future__ import annotations

import torch
from torch import nn


class APARENTNetwork(nn.Module):
    """Original APARENT MPRA architecture for 205-nt poly(A) windows."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(4, 96, kernel_size=8)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv_2 = nn.Conv1d(96, 128, kernel_size=6)
        self.dense_1 = nn.Linear(94 * 128 + 1, 512)
        self.dropout_1 = nn.Dropout(0.1)
        self.dense_2 = nn.Linear(512, 256)
        self.dropout_2 = nn.Dropout(0.1)
        # The source model concatenates a 13-value library indicator. The
        # sequence-only release uses an all-zero indicator at inference time.
        self.cleavage_head = nn.Linear(256 + 13, 206)
        self.isoform_head = nn.Linear(256 + 13, 1)

    def _shared_features(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.conv_1(sequence_tensor))
        hidden = self.pool(hidden)
        hidden = torch.relu(self.conv_2(hidden))
        # Keras Flatten consumed the channels-last [position, channel] tensor.
        hidden = hidden.transpose(1, 2).flatten(start_dim=1)
        distal_pas = torch.ones(
            (hidden.shape[0], 1), dtype=hidden.dtype, device=hidden.device
        )
        hidden = torch.relu(self.dense_1(torch.cat([hidden, distal_pas], dim=1)))
        hidden = self.dropout_1(hidden)
        hidden = torch.relu(self.dense_2(hidden))
        hidden = self.dropout_2(hidden)
        library = torch.zeros(
            (hidden.shape[0], 13), dtype=hidden.dtype, device=hidden.device
        )
        return torch.cat([hidden, library], dim=1)

    def forward(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        """Return 205 positional and one distal cleavage probability."""
        return torch.softmax(
            self.cleavage_head(self._shared_features(sequence_tensor)), dim=1
        )

    def isoform_probability(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        """Return the source model's non-normalized isoform probability head."""
        return torch.sigmoid(self.isoform_head(self._shared_features(sequence_tensor)))
