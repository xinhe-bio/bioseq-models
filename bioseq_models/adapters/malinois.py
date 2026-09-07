"""Malinois checkpoint adapter."""

from __future__ import annotations

from pathlib import Path

import torch

from ..architectures import MalinoisNetwork
from ..inference import TorchSequenceModel
from ..resources import weight_path
from .common import encode_acgt_sequences

MPRA_UPSTREAM = "ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG"
MPRA_DOWNSTREAM = "CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT"
LEFT_CONTEXT = MPRA_UPSTREAM[-200:]
RIGHT_CONTEXT = MPRA_DOWNSTREAM[:200]


def _encode_with_context(sequences):
    constructs = [LEFT_CONTEXT + sequence + RIGHT_CONTEXT for sequence in sequences]
    return encode_acgt_sequences(constructs, exact_length=600)


class Malinois(TorchSequenceModel):
    """Predict K562, HepG2, and SK-N-SH enhancer activity from 200-mers."""

    def __init__(self, model: MalinoisNetwork | None = None, **kwargs):
        kwargs.setdefault("task_names", ("K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC"))
        super().__init__(model or MalinoisNetwork(), _encode_with_context, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path | None = None,
        weights_dir: str | Path | None = None,
        **kwargs,
    ) -> Malinois:
        path = (
            Path(checkpoint)
            if checkpoint
            else weight_path("checkpoints/malinois.pt", weights_dir)
        )
        if not path.is_file():
            raise FileNotFoundError(f"Malinois checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = MalinoisNetwork()
        model.load_state_dict(state["model_state"])
        return cls(model=model, **kwargs)
