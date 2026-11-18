"""Saluki inference architecture corresponding to the released Keras checkpoints.

Architecture derived from Calico's basenji/rnann.py (Apache-2.0).
Copyright 2021 Calico LLC. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import torch
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, filters: int, kernel_size: int, epsilon: float):
        super().__init__()
        self.norm = nn.LayerNorm(filters, eps=epsilon)
        self.conv = nn.Conv1d(filters, filters, kernel_size)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.pool(self.conv(torch.relu(x)))


class SalukiNetwork(nn.Module):
    """Six-channel RNA CNN/GRU, with a right-to-left recurrent aggregation."""

    def __init__(
        self,
        filters=64,
        kernel_size=5,
        num_layers=6,
        ln_epsilon=0.007,
        bn_epsilon=0.001,
        seq_length=12288,
    ):
        super().__init__()
        self.seq_length = int(seq_length)
        self.initial = nn.Conv1d(6, filters, kernel_size, bias=False)
        self.blocks = nn.ModuleList(
            _ConvBlock(filters, kernel_size, ln_epsilon) for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(filters, eps=ln_epsilon)
        self.gru = nn.GRU(filters, filters, batch_first=True)
        self.bn1 = nn.BatchNorm1d(filters, eps=bn_epsilon)
        self.dense = nn.Linear(filters, filters)
        self.bn2 = nn.BatchNorm1d(filters, eps=bn_epsilon)
        self.output = nn.Linear(filters, 1)

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1:] != (6, self.seq_length):
            raise ValueError(f"Saluki expects [batch, 6, {self.seq_length}] inputs")
        x = self.initial(inputs)
        for block in self.blocks:
            x = block(x)
        x = torch.relu(self.final_norm(x.transpose(1, 2))).flip(1)
        _, hidden = self.gru(x)
        x = torch.relu(self.bn1(hidden[-1]))
        x = self.dense(x)
        return self.output(torch.relu(self.bn2(x)))


class SalukiEnsemble(nn.Module):
    """Arithmetic mean of explicitly selected official fold/cross models."""

    def __init__(self, models):
        super().__init__()
        if not models:
            raise ValueError("Provide at least one Saluki model")
        self.models = nn.ModuleList(models)

    def forward(self, inputs):
        total = self.models[0](inputs)
        for model in self.models[1:]:
            total = total + model(inputs)
        return total / len(self.models)
