"""Shared tensor utilities for checkpoint-backed nucleotide models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


def encode_acgt_sequences(
    sequences: Sequence[str] | np.ndarray,
    *,
    exact_length: int | None = None,
    padded_length: int | None = None,
) -> torch.Tensor:
    """One-hot encode A/C/G/T strings; N padding becomes an all-zero vector."""
    values = np.asarray(sequences, dtype=str).reshape(-1)
    lengths = np.char.str_len(values)
    if exact_length is not None and np.any(lengths != exact_length):
        row = int(np.flatnonzero(lengths != exact_length)[0])
        raise ValueError(
            f"model requires {exact_length}-nt sequences; row {row} has "
            f"length {int(lengths[row])}"
        )
    output_length = padded_length or exact_length
    if output_length is None:
        output_length = int(lengths.max(initial=0))
    if np.any(lengths > output_length):
        row = int(np.flatnonzero(lengths > output_length)[0])
        raise ValueError(
            f"row {row} has length {int(lengths[row])}, exceeding {output_length}"
        )
    encoded = np.zeros((len(values), 4, output_length), dtype=np.float32)
    lookup = {base: channel for channel, base in enumerate("ACGT")}
    for row, sequence in enumerate(values):
        for position, base in enumerate(sequence):
            if base == "N":
                continue
            if base not in lookup:
                raise ValueError(
                    f"invalid nucleotide at row {row}, position {position}: {base!r}"
                )
            encoded[row, lookup[base], position] = 1.0
    return torch.from_numpy(encoded)


def encode_acgt_last_channel(
    sequences: Sequence[str] | np.ndarray,
    *,
    exact_length: int | None = None,
    padded_length: int | None = None,
) -> torch.Tensor:
    """One-hot encode to canonical ``[batch, length, A/C/G/T]`` layout."""
    return encode_acgt_sequences(
        sequences,
        exact_length=exact_length,
        padded_length=padded_length,
    ).transpose(1, 2)


def prepare_acgt_probabilities(
    probabilities,
    *,
    exact_length: int,
    channel_order: str = "ACGT",
):
    """Convert canonical ``[batch, length, A/C/G/T]`` soft input for a CNN."""
    import torch

    if not isinstance(probabilities, torch.Tensor):
        raise TypeError("probabilities must be a PyTorch tensor")
    if probabilities.ndim != 3 or probabilities.shape[2] != 4:
        raise ValueError(
            "probabilities must have shape [batch, length, 4] in A/C/G/T order"
        )
    if probabilities.shape[1] != exact_length:
        raise ValueError(
            f"model requires {exact_length}-nt soft inputs; received {probabilities.shape[1]}"
        )
    if len(channel_order) != 4 or set(channel_order) != set("ACGT"):
        raise ValueError("channel_order must be a permutation of ACGT")
    indices = ["ACGT".index(base) for base in channel_order]
    return probabilities[:, :, indices].transpose(1, 2)


class RestoredTargetScale(nn.Module):
    """Restore standardized model predictions to their measured target scale."""

    def __init__(self, model: nn.Module, mean, scale) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("target_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer(
            "target_scale", torch.as_tensor(scale, dtype=torch.float32)
        )

    def forward(self, sequence_tensor):
        return self.model(sequence_tensor) * self.target_scale + self.target_mean


def task_matrix(values: torch.Tensor) -> torch.Tensor:
    """Normalize a model output to ``[batch, tasks]``."""
    return values[:, None] if values.ndim == 1 else values


class ScaledEnsemble(nn.Module):
    """Average independently scaled ensemble members."""

    def __init__(self, models, means, scales) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer("means", torch.as_tensor(means, dtype=torch.float32))
        self.register_buffer("scales", torch.as_tensor(scales, dtype=torch.float32))

    def forward(self, inputs):
        predictions = []
        for index, model in enumerate(self.models):
            value = task_matrix(model(inputs))
            predictions.append(value * self.scales[index] + self.means[index])
        stacked = torch.stack(predictions)
        return stacked.mean(dim=0)
