"""Adapters for converted tissue-specific DeepSTARR checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from ..architectures import DeepSTARREmbryoNetwork, DeepSTARRMouseNetwork
from ..inference import TorchSequenceModel
from ..resources import weight_path
from .common import encode_acgt_sequences


def _encoder(sequences):
    return encode_acgt_sequences(sequences, exact_length=1001)


class _DeepSTARRTissueModel(TorchSequenceModel):
    architecture: str
    output_kind: str
    checkpoint_file: str
    network_class: type[DeepSTARRMouseNetwork | DeepSTARREmbryoNetwork]

    def __init__(self, model=None, **kwargs):
        super().__init__(
            model or self.network_class(output_kind=self.output_kind),
            _encoder,
            **kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path | None = None,
        weights_dir: str | Path | None = None,
        **kwargs,
    ):
        path = (
            Path(checkpoint)
            if checkpoint
            else weight_path(cls.checkpoint_file, weights_dir)
        )
        if not path.is_file():
            raise FileNotFoundError(f"DeepSTARR checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("architecture") != cls.architecture:
            raise ValueError(
                f"checkpoint architecture is {state.get('architecture')!r}; "
                f"expected {cls.architecture!r}"
            )
        if state.get("output_kind") != cls.output_kind:
            raise ValueError(
                f"checkpoint output kind is {state.get('output_kind')!r}; "
                f"expected {cls.output_kind!r}"
            )
        model = cls.network_class(output_kind=cls.output_kind)
        model.load_state_dict(state["model_state"], strict=True)
        return cls(model=model, **kwargs)


class DeepSTARRMouseHeartActivity(_DeepSTARRTissueModel):
    architecture = "mouse"
    output_kind = "activity"
    checkpoint_file = "checkpoints/deepstarr_mouse_heart_activity_fold01_rep1.pt"
    network_class = DeepSTARRMouseNetwork

    def __init__(self, model=None, **kwargs):
        kwargs.setdefault("task_names", ("heart_activity_probability",))
        super().__init__(model=model, **kwargs)


class DeepSTARRMouseHeartAccessibility(_DeepSTARRTissueModel):
    architecture = "mouse"
    output_kind = "accessibility"
    checkpoint_file = "checkpoints/deepstarr_mouse_heart_accessibility_fold01_rep1.pt"
    network_class = DeepSTARRMouseNetwork

    def __init__(self, model=None, **kwargs):
        kwargs.setdefault("task_names", ("heart_accessibility",))
        super().__init__(model=model, **kwargs)


class DeepSTARREmbryoBrainVNCActivity(_DeepSTARRTissueModel):
    architecture = "embryo"
    output_kind = "activity"
    checkpoint_file = "checkpoints/deepstarr_embryo_brain_vnc_activity_fold01_rep1.pt"
    network_class = DeepSTARREmbryoNetwork

    def __init__(self, model=None, **kwargs):
        kwargs.setdefault("task_names", ("brain_vnc_activity_probability",))
        super().__init__(model=model, **kwargs)
