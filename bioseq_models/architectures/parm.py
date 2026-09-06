"""Exact PyTorch port of the published PARM CNN."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from einops.layers.torch import Rearrange
from torch import nn


class PARMGELU(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(1.702 * inputs) * inputs


class _Residual(nn.Module):
    def __init__(self, function: nn.Module) -> None:
        super().__init__()
        self.function = function

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.function(inputs) + inputs


def _conv_block(
    input_channels: int,
    output_channels: int | None = None,
    kernel_size: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.BatchNorm1d(input_channels),
        PARMGELU(),
        nn.Conv1d(
            input_channels,
            output_channels or input_channels,
            kernel_size,
            padding=kernel_size // 2,
        ),
    )


class PARMAttentionPool(nn.Module):
    def __init__(self, channels: int, pool_size: int = 2) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.pool = Rearrange("b d (n p) -> b d n p", p=pool_size)
        self.attention_logits = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, length = inputs.shape
        remainder = length % self.pool_size
        if remainder:
            padding = self.pool_size - remainder
            inputs = functional.pad(inputs, (0, padding), value=0.0)
            mask = torch.zeros(
                (batch_size, 1, length), dtype=torch.bool, device=inputs.device
            )
            mask = functional.pad(mask, (0, padding), value=True)
        pooled = self.pool(inputs)
        logits = self.attention_logits(pooled)
        if remainder:
            logits = logits.masked_fill(self.pool(mask), -torch.finfo(logits.dtype).max)
        return (pooled * logits.softmax(dim=-1)).sum(dim=-1)


class PARMExactCNN(nn.Module):
    """Publication architecture: 125 channels and five residual blocks."""

    def __init__(self, filters: int = 125, blocks: int = 5) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, filters, 7, padding="same"),
            _Residual(_conv_block(filters)),
            PARMAttentionPool(filters, pool_size=2),
        )
        tower: list[nn.Module] = []
        previous_filters = filters
        for block in range(blocks):
            output_filters = int(filters * 0.2) if block > 4 else filters
            tower.append(
                nn.Sequential(
                    _conv_block(previous_filters, output_filters, kernel_size=7),
                    _Residual(
                        _conv_block(output_filters, output_filters, kernel_size=1)
                    ),
                    PARMAttentionPool(output_filters, pool_size=2),
                )
            )
            previous_filters = output_filters
        self.conv_tower = nn.Sequential(*tower)
        self.output = nn.Linear(previous_filters, 1)
        self.relu = nn.ReLU()

    def forward(self, sequence_tensor: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        encoded = self.conv_tower(self.stem(sequence_tensor))
        encoded = torch.max(encoded, dim=-1).values
        return self.relu(self.output(encoded))
