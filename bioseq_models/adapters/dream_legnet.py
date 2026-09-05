"""Optimization-ready DREAM 2022 LegNet checkpoint adapter."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from sequence_optimization import TorchSequenceScorer

from ..architectures import LegNetAutosome
from .common import CHECKPOINT_DIR, encode_acgt_sequences


DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "dream_legnet.pt"
LEFT_ADAPTER_LENGTH = 17
DREAM_CONTEXT_150 = (
    "AGTCCACTATTAAAGAACGTGGACTCCAACGTCAAAGGGCGAAAAACCGTCTATCAGGGCGATGGCCCA"
    "CTACGTGAACCATCACCCTAATCAAGTGCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCA"
    "TTTTTTTCACATC"
)


def _encode_dream_sequences(sequences):
    processed = [
        (DREAM_CONTEXT_150 + sequence[LEFT_ADAPTER_LENGTH:])[-150:]
        for sequence in sequences
    ]
    acgt = encode_acgt_sequences(processed, exact_length=150)
    # The published LegNet uses A/G/C/T rather than A/C/G/T channel order.
    agct = acgt[:, [0, 2, 1, 3], :]
    singleton = torch.ones((len(processed), 1, 150), dtype=agct.dtype)
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


class DreamLegNet(TorchSequenceScorer):
    def __init__(self, model: LegNetAutosome | None = None, **kwargs):
        kwargs.setdefault("task_names", ("promoter_activity",))
        super().__init__(
            _ReverseComplementAverage(model or LegNetAutosome()),
            _encode_dream_sequences,
            **kwargs,
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
