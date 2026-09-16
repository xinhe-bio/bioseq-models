"""APARENT cleavage-site probability checkpoint adapter."""

from __future__ import annotations

from pathlib import Path

import torch

from ..architectures import APARENTNetwork
from ..inference import TorchSequenceModel
from ..resources import weight_path
from .common import encode_acgt_sequences

TASK_NAMES = tuple(
    [f"cleavage_probability_{position}" for position in range(205)]
    + ["distal_cleavage_probability"]
)


class APARENT(TorchSequenceModel):
    """Predict a cleavage distribution from a 205-nt poly(A)-site window."""

    def __init__(self, model: APARENTNetwork | None = None, **kwargs) -> None:
        kwargs.setdefault("task_names", TASK_NAMES)
        super().__init__(
            model or APARENTNetwork(),
            lambda values: encode_acgt_sequences(values, exact_length=205),
            **kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path | None = None,
        weights_dir: str | Path | None = None,
        **kwargs,
    ) -> APARENT:
        path = (
            Path(checkpoint)
            if checkpoint
            else weight_path("checkpoints/aparent.pt", weights_dir)
        )
        if not path.is_file():
            raise FileNotFoundError(f"APARENT checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("architecture") != "aparent_site_probabilities":
            raise ValueError(
                f"checkpoint architecture is {state.get('architecture')!r}; "
                "expected 'aparent_site_probabilities'"
            )
        model = APARENTNetwork()
        model.load_state_dict(state["model_state"], strict=True)
        return cls(model=model, **kwargs)
