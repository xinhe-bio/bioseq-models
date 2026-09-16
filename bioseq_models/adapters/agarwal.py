"""Agarwal lentiMPRA MPRALegNet cross-validation ensembles."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ..architectures import MPRALegNet
from ..inference import TorchSequenceModel
from ..resources import weight_path
from .common import encode_acgt_sequences, prepare_acgt_probabilities, task_matrix


def _checkpoint_paths(cell_type: str, weights_dir=None) -> tuple[Path, ...]:
    return tuple(
        weight_path(f"checkpoints/agarwal_{cell_type}_fold{fold}.pt", weights_dir)
        for fold in range(1, 11)
    )


class _MPRATTAEnsemble(nn.Module):
    def __init__(self, models) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, inputs):
        reverse = inputs[:, [3, 2, 1, 0], :].flip(dims=[2])
        predictions = []
        for model in self.models:
            if torch.is_grad_enabled() and inputs.requires_grad:
                from torch.utils.checkpoint import checkpoint

                forward_value = task_matrix(
                    checkpoint(model, inputs, use_reentrant=False)
                )
                reverse_value = task_matrix(
                    checkpoint(model, reverse, use_reentrant=False)
                )
            else:
                forward_value = task_matrix(model(inputs))
                reverse_value = task_matrix(model(reverse))
            predictions.append((forward_value + reverse_value) / 2)
        stacked = torch.stack(predictions)
        return stacked.mean(dim=0)


class AgarwalMPRALegNet(TorchSequenceModel):
    """Base adapter; use a cell-specific subclass from the registry."""

    cell_type = ""

    def __init__(self, model, **kwargs) -> None:
        kwargs.setdefault("task_names", ("log2_rna_dna",))
        super().__init__(
            model,
            lambda values: encode_acgt_sequences(values, exact_length=230)[
                :, [0, 2, 1, 3], :
            ],
            soft_encoder=lambda values: prepare_acgt_probabilities(
                values,
                exact_length=230,
                channel_order="AGCT",
            ),
            **kwargs,
        )
        self.cell_type = type(self).cell_type
        self.cv_folds = 10

    @classmethod
    def from_checkpoint(cls, checkpoint=None, weights_dir=None, **kwargs):
        paths = (
            _checkpoint_paths(cls.cell_type, weights_dir)
            if checkpoint is None
            else (Path(checkpoint),)
        )
        models = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Agarwal checkpoint not found: {path}")
            state = torch.load(path, map_location="cpu", weights_only=True)
            model = MPRALegNet(input_channels=4)
            model.load_state_dict(state["model_state"])
            models.append(model)
        return cls(_MPRATTAEnsemble(models), **kwargs)


class AgarwalHepG2(AgarwalMPRALegNet):
    cell_type = "hepg2"


class AgarwalK562(AgarwalMPRALegNet):
    cell_type = "k562"


class AgarwalWTC11(AgarwalMPRALegNet):
    cell_type = "wtc11"
