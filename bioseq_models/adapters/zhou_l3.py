"""Zhou L3 checkpoint adapter."""

from __future__ import annotations

from pathlib import Path

import torch

from ..architectures import L3ConditionalCNN, L3PositionAwareCNN
from ..inference import TorchSequenceModel
from .common import CHECKPOINT_DIR, RestoredTargetScale, encode_acgt_sequences


DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "zhou_l3.pt"
DEFAULT_CONDITIONAL_CHECKPOINT = CHECKPOINT_DIR / "zhou_l3_conditional.pt"


class _FixedSublibrary(torch.nn.Module):
    def __init__(self, model, sublibrary_index: int, mean, scale) -> None:
        super().__init__()
        self.model = model
        self.sublibrary_index = int(sublibrary_index)
        self.register_buffer("target_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer(
            "target_scale", torch.as_tensor(scale, dtype=torch.float32)
        )

    def forward(self, inputs):
        identifiers = torch.full(
            (len(inputs),),
            self.sublibrary_index,
            dtype=torch.long,
            device=inputs.device,
        )
        return self.model(inputs, identifiers) * self.target_scale + self.target_mean


class ZhouL3(TorchSequenceModel):
    def __init__(self, model=None, **kwargs):
        kwargs.setdefault("task_names", ("promoter_activity",))
        super().__init__(
            model or L3PositionAwareCNN(length=29),
            lambda values: encode_acgt_sequences(values, exact_length=29),
            **kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        **kwargs,
    ) -> "ZhouL3":
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Zhou L3 checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = L3PositionAwareCNN(output_dim=1, length=29)
        model.load_state_dict(state["model_state"])
        mean, scale = state.get("target_mean"), state.get("target_std")
        if mean is not None and scale is not None:
            model = RestoredTargetScale(model, mean, scale)
        return cls(model=model, **kwargs)


class ZhouL3Conditional(TorchSequenceModel):
    """Canonical L3 model with one fixed documented source sublibrary."""

    def __init__(self, model, *, sublibrary: str, **kwargs) -> None:
        kwargs.setdefault("task_names", ("promoter_activity",))
        super().__init__(
            model,
            lambda values: encode_acgt_sequences(values, exact_length=29),
            **kwargs,
        )
        self.sublibrary = sublibrary

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_CONDITIONAL_CHECKPOINT,
        *,
        sublibrary: str = "L3_s1",
        **kwargs,
    ) -> "ZhouL3Conditional":
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Zhou L3 conditional checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        vocabulary = state["cell_type_vocabulary"]
        if sublibrary not in vocabulary:
            raise ValueError(
                f"unknown L3 sublibrary {sublibrary!r}; choose from {tuple(vocabulary)}"
            )
        network = L3ConditionalCNN(
            num_sublibraries=len(vocabulary),
            output_dim=len(state["label_columns"]),
            length=int(state["max_length"]),
        )
        network.load_state_dict(state["model_state"])
        model = _FixedSublibrary(
            network,
            vocabulary[sublibrary],
            state["target_mean"],
            state["target_std"],
        )
        return cls(model, sublibrary=sublibrary, **kwargs)
