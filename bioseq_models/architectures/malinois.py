"""Malinois BassetBranched architecture.

Adapted from the MIT-licensed BODA implementation by Sagar Gosai and
Rodrigo Castro. See ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class Conv1dNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=True)
        self.bn_layer = nn.BatchNorm1d(out_channels)

    def forward(self, inputs):
        return self.bn_layer(self.conv(inputs))


class LinearNorm(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.bn_layer = nn.BatchNorm1d(out_features)

    def forward(self, inputs):
        return self.bn_layer(self.linear(inputs))


class GroupedLinear(nn.Module):
    def __init__(self, in_group_size: int, out_group_size: int, groups: int):
        super().__init__()
        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.groups = groups
        self.weight = nn.Parameter(torch.zeros(groups, in_group_size, out_group_size))
        self.bias = nn.Parameter(torch.zeros(groups, 1, out_group_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(3))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        nn.init.uniform_(self.bias, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))

    def forward(self, inputs):
        grouped = (
            inputs.permute(1, 0)
            .reshape(self.groups, self.in_group_size, -1)
            .permute(0, 2, 1)
        )
        grouped = torch.bmm(grouped, self.weight) + self.bias
        return (
            grouped.permute(0, 2, 1)
            .reshape(self.out_group_size * self.groups, -1)
            .permute(1, 0)
        )


class BranchedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_group_size: int,
        out_group_size: int,
        n_branches: int,
        n_layers: int,
        activation: str,
        dropout_p: float,
    ):
        super().__init__()
        self.n_branches = n_branches
        self.n_layers = n_layers
        self.nonlin = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_p)
        current_size = in_features
        for index in range(n_layers):
            output_size = out_group_size if index + 1 == n_layers else hidden_group_size
            setattr(
                self,
                f"branched_layer_{index + 1}",
                GroupedLinear(current_size, output_size, n_branches),
            )
            current_size = hidden_group_size

    def forward(self, inputs):
        output = inputs.repeat(1, self.n_branches)
        for index in range(self.n_layers - 1):
            output = getattr(self, f"branched_layer_{index + 1}")(output)
            output = self.dropout(self.nonlin(output))
        return getattr(self, f"branched_layer_{self.n_layers}")(output)


class MalinoisNetwork(nn.Module):
    """Official three-output Malinois network for 600-nt MPRA constructs."""

    def __init__(self):
        super().__init__()
        self.pad1 = nn.ConstantPad1d((9, 9), 0.0)
        self.conv1 = Conv1dNorm(4, 300, 19)
        self.pad2 = nn.ConstantPad1d((5, 5), 0.0)
        self.conv2 = Conv1dNorm(300, 200, 11)
        self.pad3 = nn.ConstantPad1d((3, 3), 0.0)
        self.conv3 = Conv1dNorm(200, 200, 7)
        self.pad4 = nn.ConstantPad1d((1, 1), 0.0)
        self.maxpool_3 = nn.MaxPool1d(3)
        self.maxpool_4 = nn.MaxPool1d(4)
        self.linear1 = LinearNorm(200 * 13, 1000)
        self.branched = BranchedLinear(
            1000,
            140,
            140,
            n_branches=3,
            n_layers=3,
            activation="ReLU",
            dropout_p=0.5757068086404574,
        )
        self.output = GroupedLinear(140, 1, 3)
        self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout(0.11625456877954289)

    def forward(self, inputs):
        output = self.nonlin(self.conv1(self.pad1(inputs)))
        output = self.maxpool_3(output)
        output = self.nonlin(self.conv2(self.pad2(output)))
        output = self.maxpool_4(output)
        output = self.nonlin(self.conv3(self.pad3(output)))
        output = self.maxpool_4(self.pad4(output))
        output = torch.flatten(output, start_dim=1)
        output = self.dropout(self.nonlin(self.linear1(output)))
        output = self.branched(output)
        return self.output(output)
