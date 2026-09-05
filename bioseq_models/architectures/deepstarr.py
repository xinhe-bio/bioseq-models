"""Faithful PyTorch implementation of the published DeepSTARR network."""

from __future__ import annotations

import torch
from torch import nn


class DeepSTARRPaper(nn.Module):
    """Basset-derived DeepSTARR from de Almeida et al. (2022).

    The author notebook uses the Keras order ``Conv/Dense -> BatchNorm ->
    ReLU -> (pool/dropout)``.  Keras BatchNormalization's momentum of 0.99
    means that 1% of each new batch updates the running statistics, which is
    ``momentum=0.01`` in PyTorch's convention.  These details are intentional
    here, rather than relying on PyTorch defaults.
    """

    def __init__(self, input_channels: int = 4, output_dim: int = 2) -> None:
        if output_dim != 2:
            raise ValueError(
                "DeepSTARRPaper has exactly two outputs: developmental and housekeeping activity"
            )
        super().__init__()
        batch_norm = {"eps": 1e-3, "momentum": 0.01}
        channels = [input_channels, 256, 60, 60, 120]
        kernels = [7, 3, 5, 3]
        blocks: list[nn.Module] = []
        for in_channels, out_channels, kernel in zip(
            channels[:-1], channels[1:], kernels
        ):
            blocks.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel, padding="same"),
                    nn.BatchNorm1d(out_channels, **batch_norm),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                ]
            )
        # 249 -> 124 -> 62 -> 31 -> 15 with four pool-2 layers.
        blocks.append(nn.Flatten())
        self.encoder = nn.Sequential(*blocks)
        self.dense_1 = nn.Linear(120 * 15, 256)
        self.dense_1_bn = nn.BatchNorm1d(256, **batch_norm)
        self.dense_1_activation = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.4)
        self.dense_2 = nn.Linear(256, 256)
        self.dense_2_bn = nn.BatchNorm1d(256, **batch_norm)
        self.dense_2_activation = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.4)
        # The source model has two named, independent Dense(1) heads rather
        # than one shared Dense(2) layer.
        self.developmental_head = nn.Linear(256, 1)
        self.housekeeping_head = nn.Linear(256, 1)
        self.apply(self._keras_glorot_initialize)

    @staticmethod
    def _keras_glorot_initialize(module: nn.Module) -> None:
        """Match Keras Conv1D/Dense defaults (Glorot-uniform, zero bias)."""
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, sequence_tensor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(sequence_tensor)
        hidden = self.dropout_1(
            self.dense_1_activation(self.dense_1_bn(self.dense_1(hidden)))
        )
        hidden = self.dropout_2(
            self.dense_2_activation(self.dense_2_bn(self.dense_2(hidden)))
        )
        return torch.cat(
            [self.developmental_head(hidden), self.housekeeping_head(hidden)], dim=1
        )
