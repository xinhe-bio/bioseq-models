"""Optimization-ready DeepSTARR checkpoint adapter."""

from __future__ import annotations

from pathlib import Path

import torch

from sequence_optimization import DifferentiableTorchSequenceScorer

from ..architectures import DeepSTARRPaper
from .common import CHECKPOINT_DIR, encode_acgt_sequences


DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "deepstarr.pt"


class DeepSTARR(DifferentiableTorchSequenceScorer):
    def __init__(self, model: DeepSTARRPaper | None = None, **kwargs):
        kwargs.setdefault("task_names", ("developmental", "housekeeping"))
        super().__init__(
            model or DeepSTARRPaper(),
            lambda values: encode_acgt_sequences(values, exact_length=249),
            soft_input_transform=lambda values: values.transpose(1, 2),
            **kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        **kwargs,
    ) -> "DeepSTARR":
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"DeepSTARR checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = DeepSTARRPaper()
        model.load_state_dict(state["model_state"])
        return cls(model=model, **kwargs)
