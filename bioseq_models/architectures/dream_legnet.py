"""Autosome.org's original DREAM 2022 LegNet architecture in PyTorch."""

from __future__ import annotations

import math
from itertools import pairwise

import torch
import torch.nn.functional as F
from tltorch import TRL
from torch import nn


class _Bilinear(nn.Module):
    """Low-rank tensor-regression interaction layer used in the original SE block."""

    def __init__(self, size: int, output_size: int, rank: float = 0.5) -> None:
        super().__init__()
        self.trl = TRL((size, size), output_size, bias=True, rank=rank)
        self.trl.weight = self.trl.weight.normal_(std=0.00075)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.unsqueeze(-1)
        return self.trl(inputs @ inputs.transpose(-1, -2))


class _ConcatenateInteraction(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.interaction = _Bilinear(size, size // 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([inputs, self.interaction(inputs)], dim=-1)


class _ComplexSE(nn.Module):
    """The legacy LegNet squeeze-excitation block, including bilinear features."""

    def __init__(
        self, input_channels: int, output_channels: int, reduction: int = 4
    ) -> None:
        super().__init__()
        hidden = input_channels // reduction
        self.fc = nn.Sequential(
            nn.Linear(output_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            _ConcatenateInteraction(hidden),
            nn.SiLU(),
            nn.Linear(hidden + hidden // 2, output_channels),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate = self.fc(inputs.mean(dim=2)).unsqueeze(-1)
        return inputs * gate


class _EfficientResidualBlock(nn.Module):
    """EfficientNetV2-like grouped-convolution block from autosome-ru/LegNet."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        resize_factor: int,
        se_reduction: int,
    ) -> None:
        super().__init__()
        inner = output_channels * resize_factor
        self.transform = nn.Sequential(
            nn.Conv1d(input_channels, inner, kernel_size=1, padding="same", bias=False),
            nn.BatchNorm1d(inner),
            nn.SiLU(),
            nn.Conv1d(
                inner,
                inner,
                kernel_size=kernel_size,
                padding="same",
                groups=inner // 2,
                bias=False,
            ),
            nn.BatchNorm1d(inner),
            nn.SiLU(),
            _ComplexSE(input_channels, inner, reduction=se_reduction),
            nn.Conv1d(inner, input_channels, kernel_size=1, padding="same", bias=False),
            nn.BatchNorm1d(input_channels),
            nn.SiLU(),
        )
        self.resize = nn.Sequential(
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.resize(torch.cat([inputs, self.transform(inputs)], dim=1))


class LegNetAutosome(nn.Module):
    """Published autosome.org DREAM 2022 model (LegNet's original configuration).

    Input has five channels: A/G/C/T one-hot in the source ordering plus a
    constant ``is_singleton`` channel. Outputs are log probabilities over the
    18 FACS bins and their expected bin value.
    """

    def __init__(
        self,
        block_sizes: tuple[int, ...] = (256, 128, 128, 64, 64, 64, 64),
        kernel_size: int = 7,
        resize_factor: int = 4,
        se_reduction: int = 4,
        final_channels: int = 18,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(
                5, block_sizes[0], kernel_size=kernel_size, padding="same", bias=False
            ),
            nn.BatchNorm1d(block_sizes[0]),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[
                _EfficientResidualBlock(
                    previous, current, kernel_size, resize_factor, se_reduction
                )
                for previous, current in pairwise(block_sizes)
            ]
        )
        self.mapper = nn.Sequential(
            nn.Conv1d(block_sizes[-1], final_channels, kernel_size=1, padding="same"),
            nn.SiLU(),
        )
        self.register_buffer("bins", torch.arange(final_channels, dtype=torch.float32))

    def forward(
        self, sequence_tensor: torch.Tensor, **_: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.blocks(self.stem(sequence_tensor))
        logits = F.adaptive_avg_pool1d(self.mapper(hidden), 1).squeeze(-1)
        log_probabilities = F.log_softmax(logits, dim=1)
        score = (log_probabilities.exp() * self.bins).sum(dim=1)
        return log_probabilities, score


def initialise_autosome_weights(module: nn.Module) -> None:
    """Match the initialiser in the original AutosomE training script."""
    if isinstance(module, nn.Conv1d):
        module.weight.data.normal_(
            0, math.sqrt(2 / (module.kernel_size[0] * module.out_channels))
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(0, 0.001)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
