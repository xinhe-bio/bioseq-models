"""TFBind8 weak-ensemble adapter."""

from __future__ import annotations

from pathlib import Path

import torch

from ..architectures import TFBindCNN
from ..inference import TorchSequenceModel
from .common import CHECKPOINT_DIR, ScaledEnsemble, encode_acgt_sequences


DEFAULT_CHECKPOINTS = tuple(
    CHECKPOINT_DIR / f"tfbind8_weak_{member:02d}.pt" for member in range(10)
)


class TFBind8Weak(TorchSequenceModel):
    """Ten-member CNN trained on the lower 95% of the TFBind8 landscape."""

    def __init__(self, model, **kwargs) -> None:
        kwargs.setdefault("task_names", ("binding_score",))
        kwargs.setdefault("pad_to_length", 8)
        super().__init__(
            model,
            lambda values: encode_acgt_sequences(values, exact_length=8),
            **kwargs,
        )
        self.train_fraction = 0.95

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint=None,
        **kwargs,
    ) -> "TFBind8Weak":
        paths = DEFAULT_CHECKPOINTS if checkpoint is None else (Path(checkpoint),)
        models, means, scales = [], [], []
        for path in paths:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state["dataset"] != "tfbind8" or int(state["sequence_length"]) != 8:
                raise ValueError(f"not a TFBind8 checkpoint: {path}")
            model = TFBindCNN()
            model(torch.zeros(1, 4, 8))
            model.load_state_dict(state["model_state"])
            models.append(model)
            means.append(float(state["train_label_mean"]))
            scales.append(float(state["train_label_std"]))
        return cls(ScaledEnsemble(models, means, scales), **kwargs)
