"""Adapters for converted tissue-specific DeepSTARR checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ..architectures import DeepSTARREmbryoNetwork, DeepSTARRMouseNetwork
from ..inference import TorchSequenceModel
from ..resources import weight_path
from .common import encode_acgt_sequences


def _encoder(sequences):
    return encode_acgt_sequences(sequences, exact_length=1001)


class _DeepSTARRTissueModel(TorchSequenceModel):
    architecture: str
    output_kind: str
    checkpoint_file: str | None = None
    checkpoint_files: tuple[str, ...] = ()
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
        if checkpoint:
            paths = (Path(checkpoint),)
        elif cls.checkpoint_files:
            paths = tuple(
                weight_path(path, weights_dir) for path in cls.checkpoint_files
            )
        elif cls.checkpoint_file:
            paths = (weight_path(cls.checkpoint_file, weights_dir),)
        else:
            raise ValueError(f"{cls.__name__} does not define any checkpoints")

        models = []
        for path in paths:
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
            models.append(model)

        scorer = cls(model=_MeanEnsemble(models), **kwargs)
        scorer.ensemble_size = len(models)
        return scorer


class _MeanEnsemble(nn.Module):
    """Average predictions from the six official folds and replicates."""

    def __init__(self, models) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, inputs):
        return torch.stack([model(inputs) for model in self.models]).mean(dim=0)


def _mouse_checkpoint_files(tissue: str, output_kind: str) -> tuple[str, ...]:
    return tuple(
        f"checkpoints/deepstarr_mouse_{tissue}_{output_kind}_fold{fold:02d}_rep{rep}.pt"
        for fold in range(1, 4)
        for rep in range(1, 3)
    )


class _DeepSTARRMouseModel(_DeepSTARRTissueModel):
    architecture = "mouse"
    network_class = DeepSTARRMouseNetwork
    tissue: str

    def __init__(self, model=None, **kwargs):
        suffix = (
            "activity_probability"
            if self.output_kind == "activity"
            else "accessibility"
        )
        kwargs.setdefault("task_names", (f"{self.tissue}_{suffix}",))
        super().__init__(model=model, **kwargs)


class DeepSTARRMouseHeartActivity(_DeepSTARRMouseModel):
    tissue = "heart"
    output_kind = "activity"
    checkpoint_files = _mouse_checkpoint_files(tissue, output_kind)


class DeepSTARRMouseHeartAccessibility(_DeepSTARRMouseModel):
    tissue = "heart"
    output_kind = "accessibility"
    checkpoint_files = _mouse_checkpoint_files(tissue, output_kind)


class DeepSTARRMouseLimbActivity(_DeepSTARRMouseModel):
    tissue = "limb"
    output_kind = "activity"
    checkpoint_files = _mouse_checkpoint_files(tissue, output_kind)


class DeepSTARRMouseLimbAccessibility(_DeepSTARRMouseModel):
    tissue = "limb"
    output_kind = "accessibility"
    checkpoint_files = _mouse_checkpoint_files(tissue, output_kind)


class DeepSTARRMouseMidbrainActivity(_DeepSTARRMouseModel):
    tissue = "midbrain"
    output_kind = "activity"
    checkpoint_files = _mouse_checkpoint_files(tissue, output_kind)


class DeepSTARRMouseMidbrainAccessibility(_DeepSTARRMouseModel):
    tissue = "midbrain"
    output_kind = "accessibility"
    checkpoint_files = _mouse_checkpoint_files(tissue, output_kind)


class DeepSTARRMouseForebrainActivity(_DeepSTARRMouseModel):
    tissue = "forebrain"
    output_kind = "activity"
    checkpoint_files = _mouse_checkpoint_files(tissue, output_kind)


class DeepSTARRMouseForebrainAccessibility(_DeepSTARRMouseModel):
    tissue = "forebrain"
    output_kind = "accessibility"
    checkpoint_files = _mouse_checkpoint_files(tissue, output_kind)


class DeepSTARRMouseHindbrainActivity(_DeepSTARRMouseModel):
    tissue = "hindbrain"
    output_kind = "activity"
    checkpoint_files = _mouse_checkpoint_files(tissue, output_kind)


class DeepSTARRMouseHindbrainAccessibility(_DeepSTARRMouseModel):
    tissue = "hindbrain"
    output_kind = "accessibility"
    checkpoint_files = _mouse_checkpoint_files(tissue, output_kind)


class DeepSTARRMouseNeuralTubeActivity(_DeepSTARRMouseModel):
    tissue = "neural_tube"
    output_kind = "activity"
    checkpoint_files = _mouse_checkpoint_files(tissue, output_kind)


class DeepSTARRMouseNeuralTubeAccessibility(_DeepSTARRMouseModel):
    tissue = "neural_tube"
    output_kind = "accessibility"
    checkpoint_files = _mouse_checkpoint_files(tissue, output_kind)


class DeepSTARREmbryoBrainVNCActivity(_DeepSTARRTissueModel):
    architecture = "embryo"
    output_kind = "activity"
    checkpoint_file = "checkpoints/deepstarr_embryo_brain_vnc_activity_fold01_rep1.pt"
    network_class = DeepSTARREmbryoNetwork

    def __init__(self, model=None, **kwargs):
        kwargs.setdefault("task_names", ("brain_vnc_activity_probability",))
        super().__init__(model=model, **kwargs)
