"""Standardized lightweight surrogate CNN for DNA/RNA tasks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as functional


class MaskedGroupNorm1d(nn.GroupNorm):
    """Group normalization that excludes padded positions."""

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is None or bool(mask.all()):
            return super().forward(inputs)
        if mask.shape != (inputs.shape[0], inputs.shape[2]):
            raise ValueError("mask must have shape [batch, length]")
        batch, channels, length = inputs.shape
        grouped = inputs.reshape(
            batch, self.num_groups, channels // self.num_groups, length
        )
        valid = mask[:, None, None, :].to(dtype=inputs.dtype, device=inputs.device)
        denominator = (valid.sum(dim=(2, 3)) * (channels // self.num_groups)).clamp_min(
            1.0
        )
        mean = (grouped * valid).sum(dim=(2, 3)) / denominator
        centered = grouped - mean[:, :, None, None]
        variance = (centered.square() * valid).sum(dim=(2, 3)) / denominator
        normalized = centered * torch.rsqrt(variance[:, :, None, None] + self.eps)
        normalized = normalized.reshape(batch, channels, length)
        if self.affine:
            normalized = (
                normalized * self.weight[None, :, None] + self.bias[None, :, None]
            )
        return normalized * mask[:, None, :].to(normalized.dtype)


class ResidualDilatedBlock(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = MaskedGroupNorm1d(8, channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = MaskedGroupNorm1d(8, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        normalization_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        branch = self.conv1(inputs)
        branch = self.dropout(functional.gelu(self.norm1(branch, normalization_mask)))
        branch = self.conv2(branch)
        branch = self.dropout(functional.gelu(self.norm2(branch, normalization_mask)))
        output = functional.gelu(inputs + branch)
        return output * mask[:, None, :].to(output.dtype)


class WeakSeqCNN(nn.Module):
    """194,882-parameter WeakSeqCNN-v1 regressor."""

    def __init__(
        self,
        alphabet_size: int = 4,
        channels: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 5,
        dilations: Sequence[int] = (1, 2, 4, 8),
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if alphabet_size != 4:
            raise ValueError("WeakSeqCNN-v1 requires four A/C/G/T channels")
        if channels != 64 or num_blocks != 4 or kernel_size != 5:
            raise ValueError(
                "WeakSeqCNN-v1 fixes channels=64, num_blocks=4, and kernel_size=5"
            )
        if tuple(dilations) != (1, 2, 4, 8) or hidden_dim != 128 or dropout != 0.1:
            raise ValueError("WeakSeqCNN-v1 fixes dilations, hidden_dim, and dropout")
        self.input_projection = nn.Conv1d(alphabet_size, channels, kernel_size=1)
        self.position_projection = nn.Conv1d(channels + 2, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            ResidualDilatedBlock(channels, kernel_size, dilation, dropout)
            for dilation in dilations
        )
        self.attention_score = nn.Linear(channels, 1)
        self.head = nn.Sequential(
            nn.Linear(channels * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _relative_positions(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if bool(mask.all()):
            forward = torch.linspace(
                0, 1, mask.shape[1], dtype=dtype, device=mask.device
            )
            forward = forward[None, :].expand(mask.shape[0], -1)
            return torch.stack((forward, 1.0 - forward), dim=1)
        valid = mask.to(dtype)
        ordinal = (valid.cumsum(dim=1) - 1).clamp_min(0)
        denominator = (valid.sum(dim=1, keepdim=True) - 1).clamp_min(1)
        forward = (ordinal / denominator) * valid
        reverse = (1.0 - forward) * valid
        return torch.stack((forward, reverse), dim=1)

    def forward(
        self,
        x: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        sequence_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x is None:
            x = sequence_tensor
        if x is None or x.ndim != 3:
            raise ValueError("x must have shape [batch, length, 4]")
        if x.shape[-1] == 4:
            sequence = x
        elif x.shape[1] == 4:
            sequence = x.transpose(1, 2)
        else:
            raise ValueError("x must contain four nucleotide channels")
        if not sequence.is_floating_point():
            raise TypeError("WeakSeqCNN input must be floating point")
        batch, length, _ = sequence.shape
        normalization_mask: torch.Tensor | None
        if mask is None:
            mask = torch.ones((batch, length), dtype=torch.bool, device=sequence.device)
            normalization_mask = None
        else:
            mask = mask.to(device=sequence.device, dtype=torch.bool)
            if mask.shape != (batch, length):
                raise ValueError("mask must have shape [batch, length]")
            if not bool(mask.any(dim=1).all()):
                raise ValueError("every sequence must contain an unmasked nucleotide")
            normalization_mask = mask
        hidden = self.input_projection(sequence.transpose(1, 2))
        positions = self._relative_positions(mask, hidden.dtype)
        hidden = self.position_projection(torch.cat((hidden, positions), dim=1))
        hidden = hidden * mask[:, None, :].to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, mask, normalization_mask)
        mask_float = mask[:, None, :].to(hidden.dtype)
        mean_pool = (hidden * mask_float).sum(dim=-1) / mask_float.sum(
            dim=-1
        ).clamp_min(1)
        max_pool = (
            hidden.masked_fill(~mask[:, None, :], torch.finfo(hidden.dtype).min)
            .max(dim=-1)
            .values
        )
        transposed = hidden.transpose(1, 2)
        attention_logits = self.attention_score(transposed).squeeze(-1)
        attention_logits = attention_logits.masked_fill(
            ~mask, torch.finfo(hidden.dtype).min
        )
        attention = torch.softmax(attention_logits, dim=1)
        attention_pool = (attention[:, :, None] * transposed).sum(dim=1)
        return self.head(
            torch.cat((mean_pool, max_pool, attention_pool), dim=-1)
        ).squeeze(-1)
