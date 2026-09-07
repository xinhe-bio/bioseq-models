"""Checkpoint ensembles for the standardized WeakSeqCNN-v1 architecture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..architectures import WeakSeqCNN
from ..inference import TorchSequenceModel
from .common import CHECKPOINT_DIR, encode_acgt_last_channel, task_matrix


UTR5_CHECKPOINTS = tuple(
    CHECKPOINT_DIR / f"utr5_weak_{member:02d}.pt" for member in range(5)
)
DREAM_CHECKPOINTS = tuple(
    CHECKPOINT_DIR / f"dream_weak_{member:02d}.pt" for member in range(5)
)


class _WeakEnsemble(nn.Module):
    def __init__(self, models, means, scales) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer("means", torch.as_tensor(means, dtype=torch.float32))
        self.register_buffer("scales", torch.as_tensor(scales, dtype=torch.float32))

    def forward(self, inputs):
        mask = inputs.abs().sum(dim=-1) > 0
        predictions = []
        for index, model in enumerate(self.models):
            standardized = task_matrix(model(inputs, mask=mask))
            predictions.append(standardized * self.scales[index] + self.means[index])
        stacked = torch.stack(predictions)
        return stacked.mean(dim=0)


def _load_weak_ensemble(paths, expected_dataset: str):
    models, means, scales, lengths, fractions = [], [], [], set(), set()
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["model"] != "weakseqcnn_v1" or state["dataset"] != expected_dataset:
            raise ValueError(f"unexpected WeakSeqCNN checkpoint: {path}")
        model = WeakSeqCNN(**state["model_config"])
        model.load_state_dict(state["model_state"])
        models.append(model)
        means.append(float(state["train_label_mean"]))
        scales.append(float(state["train_label_std"]))
        lengths.add(int(state["sequence_length"]))
        fractions.add(float(state["train_fraction"]))
    if len(lengths) != 1 or len(fractions) != 1:
        raise ValueError("WeakSeqCNN ensemble metadata does not agree")
    return _WeakEnsemble(models, means, scales), lengths.pop(), fractions.pop()


class UTR5Weak(TorchSequenceModel):
    """Five-member 5′UTR WeakSeqCNN trained on the lower 95% pool."""

    def __init__(self, model, **kwargs) -> None:
        kwargs.setdefault("task_names", ("mrl",))
        kwargs.setdefault("pad_to_length", 50)
        super().__init__(
            model,
            lambda values: encode_acgt_last_channel(values, exact_length=50),
            **kwargs,
        )
        self.train_fraction = 0.95

    @classmethod
    def from_checkpoint(cls, checkpoint=None, **kwargs) -> "UTR5Weak":
        paths = UTR5_CHECKPOINTS if checkpoint is None else (Path(checkpoint),)
        model, length, fraction = _load_weak_ensemble(paths, "utr5_mpra")
        if length != 50 or fraction != 0.95:
            raise ValueError("expected lower-95% 50-nt UTR5 checkpoints")
        return cls(model, **kwargs)


class DreamWeak(TorchSequenceModel):
    """Five-member variable-length DREAM WeakSeqCNN lower-95% ensemble."""

    maximum_length = 142

    def __init__(self, model, **kwargs) -> None:
        kwargs.setdefault("task_names", ("promoter_activity",))

        def encode(values):
            lengths = np.char.str_len(np.asarray(values, dtype=str))
            if np.any(lengths > self.maximum_length):
                raise ValueError("DREAM WeakSeqCNN accepts at most 142 nucleotides")
            return encode_acgt_last_channel(values)

        super().__init__(
            model,
            encode,
            **kwargs,
        )
        self.train_fraction = 0.95

    @classmethod
    def from_checkpoint(cls, checkpoint=None, **kwargs) -> "DreamWeak":
        paths = DREAM_CHECKPOINTS if checkpoint is None else (Path(checkpoint),)
        model, length, fraction = _load_weak_ensemble(paths, "yeast_gpra_dream2022")
        if length != cls.maximum_length or fraction != 0.95:
            raise ValueError("expected lower-95% DREAM checkpoints")
        return cls(model, **kwargs)
