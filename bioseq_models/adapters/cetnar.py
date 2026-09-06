"""Optimization-ready Cetnar sequence CNN-BiLSTM adapter."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from sequence_optimization import DifferentiableTorchSequenceScorer

from ..architectures import CetnarStabilityCNNBiLSTM
from .common import CHECKPOINT_DIR, encode_acgt_last_channel


DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "cetnar_stability_cnn.pt"


class _DecayAndStability(nn.Module):
    def __init__(self, model: nn.Module, mean: float, scale: float) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("target_mean", torch.tensor(float(mean)))
        self.register_buffer("target_scale", torch.tensor(float(scale)))

    def forward(self, inputs):
        mask = inputs.abs().sum(dim=-1) > 0
        standardized = self.model(inputs.transpose(1, 2), mask)
        decay = standardized * self.target_scale + self.target_mean
        return torch.cat([decay, -decay], dim=1)


class CetnarStability(DifferentiableTorchSequenceScorer):
    """Return measured decay rate and its higher-is-better negative."""

    def __init__(self, model: nn.Module, **kwargs) -> None:
        kwargs.setdefault("task_names", ("decay_rate_k_fit", "stability"))
        kwargs.setdefault("pad_to_length", 80)
        super().__init__(
            model,
            lambda values: encode_acgt_last_channel(values, exact_length=80),
            **kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        **kwargs,
    ) -> "CetnarStability":
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Cetnar checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        network = CetnarStabilityCNNBiLSTM()
        network.load_state_dict(state["model_state"])
        model = _DecayAndStability(
            network,
            state["label_mean"],
            state["label_scale"],
        )
        return cls(model, **kwargs)
