"""Optimization-ready Zhou L3 checkpoint adapter."""

from __future__ import annotations

from pathlib import Path

import torch

from sequence_optimization import DifferentiableTorchSequenceScorer

from ..architectures import L3PositionAwareCNN
from .common import CHECKPOINT_DIR, RestoredTargetScale, encode_acgt_sequences


DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "zhou_l3.pt"


class ZhouL3(DifferentiableTorchSequenceScorer):
    def __init__(self, model=None, **kwargs):
        kwargs.setdefault("task_names", ("promoter_activity",))
        super().__init__(
            model or L3PositionAwareCNN(length=29),
            lambda values: encode_acgt_sequences(values, exact_length=29),
            soft_input_transform=lambda values: values.transpose(1, 2),
            **kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        **kwargs,
    ) -> "ZhouL3":
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Zhou L3 checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = L3PositionAwareCNN(output_dim=1, length=29)
        model.load_state_dict(state["model_state"])
        mean, scale = state.get("target_mean"), state.get("target_std")
        if mean is not None and scale is not None:
            model = RestoredTargetScale(model, mean, scale)
        return cls(model=model, **kwargs)
