"""DeepSTARR checkpoint adapter."""

from __future__ import annotations

from pathlib import Path

import torch

from ..architectures import DeepSTARRPaper
from ..inference import TorchSequenceModel
from ..resources import weight_path
from .common import encode_acgt_sequences, prepare_acgt_probabilities


class DeepSTARR(TorchSequenceModel):
    def __init__(self, model: DeepSTARRPaper | None = None, **kwargs):
        kwargs.setdefault("task_names", ("developmental", "housekeeping"))
        super().__init__(
            model or DeepSTARRPaper(),
            lambda values: encode_acgt_sequences(values, exact_length=249),
            soft_encoder=lambda values: prepare_acgt_probabilities(
                values, exact_length=249
            ),
            **kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path | None = None,
        weights_dir: str | Path | None = None,
        **kwargs,
    ) -> DeepSTARR:
        path = Path(checkpoint) if checkpoint else weight_path(
            "checkpoints/deepstarr.pt", weights_dir
        )
        if not path.is_file():
            raise FileNotFoundError(f"DeepSTARR checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = DeepSTARRPaper()
        model.load_state_dict(state["model_state"])
        return cls(model=model, **kwargs)
