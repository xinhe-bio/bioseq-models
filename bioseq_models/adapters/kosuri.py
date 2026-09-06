"""Optimization-ready Kosuri promoter/RBS composability adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from sequence_optimization import DifferentiableTorchSequenceScorer

from ..architectures import KosuriTwoBranchCNN
from .common import CHECKPOINT_DIR, encode_acgt_last_channel


DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "kosuri_composability.pt"
MAX_PROMOTER_LENGTH = 75
MAX_RBS_LENGTH = 34
COMBINED_LENGTH = MAX_PROMOTER_LENGTH + MAX_RBS_LENGTH


class _CombinedKosuriModel(nn.Module):
    def __init__(self, model: nn.Module, mean, scale) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("target_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer(
            "target_scale", torch.as_tensor(scale, dtype=torch.float32)
        )

    def forward(self, combined):
        promoter = combined[:, :MAX_PROMOTER_LENGTH]
        rbs = combined[:, MAX_PROMOTER_LENGTH:]
        component_lengths = torch.stack(
            [
                (promoter.abs().sum(dim=-1) > 0).sum(dim=1) / MAX_PROMOTER_LENGTH,
                (rbs.abs().sum(dim=-1) > 0).sum(dim=1) / MAX_RBS_LENGTH,
            ],
            dim=1,
        ).to(combined.dtype)
        standardized = self.model(
            promoter.transpose(1, 2),
            rbs.transpose(1, 2),
            component_lengths,
        )
        return standardized * self.target_scale + self.target_mean


class KosuriComposability(DifferentiableTorchSequenceScorer):
    """Use a 75-nt promoter slot followed by a 34-nt RBS slot."""

    def __init__(self, model: nn.Module, **kwargs) -> None:
        kwargs.setdefault(
            "task_names",
            ("log10_protein_expression", "log10_rna_expression"),
        )
        super().__init__(
            model,
            lambda values: encode_acgt_last_channel(
                values, exact_length=COMBINED_LENGTH
            ),
            **kwargs,
        )

    def score_pairs(self, promoters, rbs_sequences):
        """Score variable-length promoter/RBS pairs using canonical slots."""
        promoters = np.asarray(promoters, dtype=str).reshape(-1)
        rbs_sequences = np.asarray(rbs_sequences, dtype=str).reshape(-1)
        if len(promoters) != len(rbs_sequences):
            raise ValueError("promoters and rbs_sequences must have equal lengths")
        combined = []
        for promoter, rbs in zip(promoters, rbs_sequences):
            promoter = promoter.upper().replace("U", "T")
            rbs = rbs.upper().replace("U", "T")
            if len(promoter) > MAX_PROMOTER_LENGTH or len(rbs) > MAX_RBS_LENGTH:
                raise ValueError("a Kosuri component exceeds its canonical slot")
            combined.append(
                promoter.ljust(MAX_PROMOTER_LENGTH, "N")
                + rbs.ljust(MAX_RBS_LENGTH, "N")
            )
        return self.score(combined)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        **kwargs,
    ) -> "KosuriComposability":
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Kosuri checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (
            int(state["max_promoter_length"]) != MAX_PROMOTER_LENGTH
            or int(state["max_rbs_length"]) != MAX_RBS_LENGTH
        ):
            raise ValueError("unexpected Kosuri checkpoint component lengths")
        network = KosuriTwoBranchCNN()
        network.load_state_dict(state["model_state"])
        model = _CombinedKosuriModel(
            network,
            state["target_mean"],
            state["target_scale"],
        )
        return cls(model, **kwargs)
