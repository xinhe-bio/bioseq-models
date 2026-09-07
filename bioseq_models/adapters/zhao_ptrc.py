"""Zhao Ptrc checkpoint adapters."""

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np

from ..architectures import PtrcPromoterCNN
from ..inference import CallableSequenceModel, TorchSequenceModel
from .common import CHECKPOINT_DIR, encode_acgt_sequences


DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "zhao_ptrc.pt"
DEFAULT_XGBOOST_CHECKPOINT = CHECKPOINT_DIR / "zhao_ptrc_xgboost.json"


def _xgboost_features(values, length: int = 76) -> np.ndarray:
    values = np.asarray(values, dtype=str).reshape(-1)
    encoded = np.zeros((len(values), 4, length), dtype=np.float32)
    lookup = {base: channel for channel, base in enumerate("ACGT")}
    for row, sequence in enumerate(values):
        if len(sequence) > length:
            raise ValueError(f"Ptrc sequence exceeds {length} nucleotides")
        for position, base in enumerate(sequence):
            if base == "N":
                continue
            if base not in lookup:
                raise ValueError(f"invalid nucleotide {base!r}")
            encoded[row, lookup[base], position] = 1.0
    return encoded.reshape(len(values), -1)


class ZhaoPtrc(TorchSequenceModel):
    """Predict log10 fluorescence/OD600 for Ptrc-derived promoters."""

    def __init__(self, model: PtrcPromoterCNN | None = None, **kwargs):
        kwargs.setdefault("task_names", ("log10_fluorescence_per_od600",))
        kwargs.setdefault("pad_to_length", 76)
        super().__init__(
            model or PtrcPromoterCNN(),
            lambda values: encode_acgt_sequences(values, padded_length=76),
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


class ZhaoPtrcXGBoost(CallableSequenceModel):
    """Functional reproduction of the paper's selected tree-model family."""

    def __init__(self, model, *, batch_size=None, **kwargs) -> None:
        kwargs.setdefault("task_names", ("log10_fluorescence_per_od600",))
        kwargs.setdefault("pad_to_length", 76)
        # ``load_model`` accepts one common set of runtime options.  XGBoost
        # evaluates the full matrix directly, so this option is intentionally
        # accepted but does not affect prediction.
        del batch_size
        super().__init__(
            lambda values: model.inplace_predict(_xgboost_features(values)),
            **kwargs,
        )
        self.model = model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_XGBOOST_CHECKPOINT,
        *,
        device=None,
        **kwargs,
    ) -> "ZhaoPtrcXGBoost":
        from xgboost import Booster

        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Zhao XGBoost checkpoint not found: {path}")
        model = Booster(params={"nthread": 1})
        model.load_model(path)
        del device
        return cls(model, **kwargs)
