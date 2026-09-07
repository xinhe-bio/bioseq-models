"""Five-fold PARM K562 ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..architectures import PARMExactCNN
from ..inference import TorchSequenceModel
from .common import CHECKPOINT_DIR, ScaledEnsemble, encode_acgt_sequences


MODEL_LENGTH = 600
ADAPTER_5 = "CAGTGAT"
ADAPTER_3 = "CACGACG"
MAX_BIOLOGICAL_LENGTH = MODEL_LENGTH - len(ADAPTER_5) - len(ADAPTER_3)
DEFAULT_CHECKPOINTS = tuple(
    CHECKPOINT_DIR / f"parm_k562_fold{fold}.pt" for fold in range(5)
)


def _validate_side(side: str) -> str:
    side = str(side).lower()
    if side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")
    return side


def _prepare_strings(values, side: str):
    prepared = []
    for sequence in np.asarray(values, dtype=str).reshape(-1):
        if len(sequence) > MAX_BIOLOGICAL_LENGTH:
            raise ValueError(
                f"PARM biological sequence exceeds {MAX_BIOLOGICAL_LENGTH} nt"
            )
        construct = ADAPTER_5 + sequence + ADAPTER_3
        if side == "left":
            construct = construct.rjust(MODEL_LENGTH, "N")
        else:
            construct = construct.ljust(MODEL_LENGTH, "N")
        prepared.append(construct)
    return np.asarray(prepared, dtype=str)


class PARMK562(TorchSequenceModel):
    """Published PARM architecture averaged across five K562 folds."""

    def __init__(self, model, *, padding_side: str = "right", **kwargs) -> None:
        side = _validate_side(padding_side)
        if kwargs.pop("pad_to_length", None) is not None:
            raise ValueError(
                "PARMK562 adds adapters before its internal 600-nt padding"
            )
        kwargs.pop("padding_side", None)
        kwargs.setdefault("task_names", ("log2rpm_k562",))
        super().__init__(
            model,
            lambda values: encode_acgt_sequences(
                _prepare_strings(values, side), exact_length=MODEL_LENGTH
            ),
            **kwargs,
        )
        self.padding_side = side
        self.model_input_length = MODEL_LENGTH

    @classmethod
    def from_checkpoint(cls, checkpoint=None, **kwargs) -> "PARMK562":
        paths = DEFAULT_CHECKPOINTS if checkpoint is None else (Path(checkpoint),)
        models = []
        for path in paths:
            state = torch.load(path, map_location="cpu", weights_only=True)
            model = PARMExactCNN()
            model.load_state_dict(state["model_state"])
            models.append(model)
        return cls(
            ScaledEnsemble(models, [0.0] * len(models), [1.0] * len(models)), **kwargs
        )
