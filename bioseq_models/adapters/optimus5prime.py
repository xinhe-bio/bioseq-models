"""Optimus 5-Prime checkpoint adapter."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ..architectures import Optimus5Prime as Optimus5PrimeNetwork
from ..inference import TorchSequenceModel
from .common import CHECKPOINT_DIR, RestoredTargetScale, encode_acgt_sequences


DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "optimus5prime.pt"


class Optimus5Prime(TorchSequenceModel):
    """Predict MRL from 50-position inputs, including optional N padding."""

    def __init__(self, model: nn.Module | None = None, **kwargs):
        kwargs.setdefault("task_names", ("mrl",))
        super().__init__(
            model or Optimus5PrimeNetwork(),
            lambda values: encode_acgt_sequences(values, exact_length=50),
            **kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        **kwargs,
    ) -> "Optimus5Prime":
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Optimus checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        model: nn.Module = Optimus5PrimeNetwork()
        model.load_state_dict(state["model_state"])
        mean, scale = state.get("target_mean"), state.get("target_std")
        if mean is not None and scale is not None:
            model = RestoredTargetScale(model, mean, scale)
        return cls(model=model, **kwargs)
