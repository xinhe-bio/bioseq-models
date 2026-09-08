"""MPRALegNet architecture for Agarwal lentiMPRA constructs."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class _SELayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.fc(inputs.mean(dim=2)).unsqueeze(-1)


class _EffBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, resize_factor: int) -> None:
        super().__init__()
        inner = channels * resize_factor
        self.block = nn.Sequential(
            nn.Conv1d(channels, inner, kernel_size=1, padding="same", bias=False),
            nn.BatchNorm1d(inner),
            nn.SiLU(),
            nn.Conv1d(
                inner,
                inner,
                kernel_size=kernel_size,
                groups=inner,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(inner),
            nn.SiLU(),
            _SELayer(inner, reduction=resize_factor),
            nn.Conv1d(inner, channels, kernel_size=1, padding="same", bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class _LegNetBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        resize_factor: int,
        pool_size: int,
    ) -> None:
        super().__init__()
        self.efficient = _EffBlock(input_channels, kernel_size, resize_factor)
        self.local = nn.Sequential(
            nn.Conv1d(
                2 * input_channels,
                output_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(output_channels),
            nn.SiLU(),
        )
        self.pool = nn.MaxPool1d(pool_size) if pool_size != 1 else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([self.efficient(inputs), inputs], dim=1)
        return self.pool(self.local(combined))


class MPRALegNet(nn.Module):
    """Author architecture trained here without the optional direction channel."""

    def __init__(
        self,
        input_channels: int = 4,
        stem_channels: int = 64,
        stem_kernel_size: int = 11,
        efficient_kernel_size: int = 9,
        block_sizes: tuple[int, ...] = (80, 96, 112, 128),
        pool_sizes: tuple[int, ...] = (2, 2, 2, 2),
        resize_factor: int = 4,
    ) -> None:
        super().__init__()
        if len(block_sizes) != len(pool_sizes):
            raise ValueError("block_sizes and pool_sizes must have equal length")
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                stem_channels,
                kernel_size=stem_kernel_size,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(stem_channels),
            nn.SiLU(),
        )
        current = stem_channels
        blocks = []
        for channels, pool_size in zip(block_sizes, pool_sizes):
            blocks.append(
                _LegNetBlock(
                    current,
                    channels,
                    efficient_kernel_size,
                    resize_factor,
                    pool_size,
                )
            )
            current = channels
        self.blocks = nn.Sequential(*blocks)
        self.mapper = nn.Sequential(
            nn.BatchNorm1d(current),
            nn.Conv1d(current, 2 * current, kernel_size=1),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * current, 2 * current),
            nn.BatchNorm1d(2 * current),
            nn.SiLU(),
            nn.Linear(2 * current, 1),
        )

    def forward(self, sequence_tensor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        hidden = self.mapper(self.blocks(self.stem(sequence_tensor)))
        return self.head(functional.adaptive_avg_pool1d(hidden, 1).squeeze(-1))
