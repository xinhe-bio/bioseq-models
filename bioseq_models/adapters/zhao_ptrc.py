"""Optimization-ready Zhao Ptrc checkpoint adapter."""

from __future__ import annotations

from pathlib import Path

import torch

from sequence_optimization import DifferentiableTorchSequenceScorer

from ..architectures import PtrcPromoterCNN
from .common import CHECKPOINT_DIR, channels_first_with_padding, encode_acgt_sequences


DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "zhao_ptrc.pt"


class ZhaoPtrc(DifferentiableTorchSequenceScorer):
    """Predict log10 fluorescence/OD600 for Ptrc-derived promoters."""

    def __init__(self, model: PtrcPromoterCNN | None = None, **kwargs):
        kwargs.setdefault("task_names", ("log10_fluorescence_per_od600",))
        super().__init__(
            model or PtrcPromoterCNN(),
            lambda values: encode_acgt_sequences(values, padded_length=76),
            soft_input_transform=lambda values: channels_first_with_padding(values, 76),
            **kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        **kwargs,
    ) -> "ZhaoPtrc":
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Zhao Ptrc checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = PtrcPromoterCNN()
        model.load_state_dict(state["model_state"])
        return cls(model=model, **kwargs)
