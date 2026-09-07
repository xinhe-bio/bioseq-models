"""DREAM 2022 LegNet checkpoint adapter."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ..architectures import LegNetAutosome
from ..inference import TorchSequenceModel
from .common import CHECKPOINT_DIR, encode_acgt_sequences


DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "dream_legnet.pt"
LEFT_ADAPTER_LENGTH = 17
FULL_SEQUENCE_LENGTH = 110
VARIABLE_REGION_LENGTH = FULL_SEQUENCE_LENGTH - LEFT_ADAPTER_LENGTH
MODEL_INPUT_LENGTH = 150
DREAM_CONTEXT_150 = (
    "AGTCCACTATTAAAGAACGTGGACTCCAACGTCAAAGGGCGAAAAACCGTCTATCAGGGCGATGGCCCA"
    "CTACGTGAACCATCACCCTAATCAAGTGCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCA"
    "TTTTTTTCACATC"
)
CONTEXT_PREFIX = DREAM_CONTEXT_150[-(MODEL_INPUT_LENGTH - VARIABLE_REGION_LENGTH) :]


def _variable_regions(sequences, variable_region_only: bool):
    expected_length = (
        VARIABLE_REGION_LENGTH if variable_region_only else FULL_SEQUENCE_LENGTH
    )
    values = []
    for row, sequence in enumerate(sequences):
        if len(sequence) != expected_length:
            mode = "variable region" if variable_region_only else "full construct"
            raise ValueError(
                f"DREAM {mode} requires {expected_length}-nt sequences; "
                f"row {row} has length {len(sequence)}"
            )
        values.append(
            sequence if variable_region_only else sequence[LEFT_ADAPTER_LENGTH:]
        )
    return values


def _encode_dream_sequences(sequences, *, variable_region_only: bool = False):
    processed = [
        CONTEXT_PREFIX + sequence
        for sequence in _variable_regions(sequences, variable_region_only)
    ]
    acgt = encode_acgt_sequences(processed, exact_length=MODEL_INPUT_LENGTH)
    # The published LegNet uses A/G/C/T rather than A/C/G/T channel order.
    agct = acgt[:, [0, 2, 1, 3], :]
    singleton = torch.ones((len(processed), 1, MODEL_INPUT_LENGTH), dtype=agct.dtype)
    return torch.cat([agct, singleton], dim=1)


class _ReverseComplementAverage(nn.Module):
    def __init__(self, model: LegNetAutosome):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        _, forward_score = self.model(sequence_tensor=inputs)
        reverse_dna = inputs[:, :4, :][:, [3, 2, 1, 0], :].flip(dims=[2])
        reverse_inputs = torch.cat([reverse_dna, inputs[:, 4:5, :]], dim=1)
        _, reverse_score = self.model(sequence_tensor=reverse_inputs)
        return (forward_score + reverse_score) / 2


class DreamLegNet(TorchSequenceModel):
    def __init__(
        self,
        model: LegNetAutosome | None = None,
        *,
        variable_region_only: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("task_names", ("promoter_activity",))
        super().__init__(
            _ReverseComplementAverage(model or LegNetAutosome()),
            lambda sequences: _encode_dream_sequences(
                sequences, variable_region_only=variable_region_only
            ),
            **kwargs,
        )
        self.variable_region_only = bool(variable_region_only)
        self.sequence_length = (
            VARIABLE_REGION_LENGTH
            if self.variable_region_only
            else FULL_SEQUENCE_LENGTH
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        **kwargs,
    ) -> "DreamLegNet":
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"DREAM LegNet checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = LegNetAutosome()
        model.load_state_dict(state["model_state"])
        return cls(model=model, **kwargs)
