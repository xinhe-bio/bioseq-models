"""Lazy registry for the bundled checkpoint-backed activity models."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelInfo:
    name: str
    sequence_length: int
    task_names: tuple[str, ...]
    description: str
    differentiable: bool
    loader_path: str
    aliases: tuple[str, ...] = ()


_MODELS = (
    ModelInfo(
        "optimus5prime",
        50,
        ("mrl",),
        "Optimus 5-Prime mean ribosome load model",
        True,
        "bioseq_models.adapters.optimus5prime:Optimus5Prime",
        ("optimus", "optimus_5prime"),
    ),
    ModelInfo(
        "deepstarr",
        249,
        ("developmental", "housekeeping"),
        "DeepSTARR enhancer activity model",
        True,
        "bioseq_models.adapters.deepstarr:DeepSTARR",
    ),
    ModelInfo(
        "dream_legnet",
        110,
        ("promoter_activity",),
        "DREAM 2022 yeast promoter LegNet model",
        False,
        "bioseq_models.adapters.dream_legnet:DreamLegNet",
        ("dream", "autosome_legnet"),
    ),
    ModelInfo(
        "zhao_ptrc",
        74,
        ("log10_fluorescence_per_od600",),
        "Zhao Ptrc promoter fluorescence CNN",
        True,
        "bioseq_models.adapters.zhao_ptrc:ZhaoPtrc",
        ("ptrc",),
    ),
    ModelInfo(
        "zhou_l3",
        29,
        ("promoter_activity",),
        "Zhou L3 position-aware promoter CNN",
        True,
        "bioseq_models.adapters.zhou_l3:ZhouL3",
        ("l3",),
    ),
)


def _normalize(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


_LOOKUP = {
    _normalize(alias): info for info in _MODELS for alias in (info.name, *info.aliases)
}


def available_models() -> tuple[str, ...]:
    return tuple(info.name for info in _MODELS)


def model_info(name: str) -> ModelInfo:
    key = _normalize(name)
    if key not in _LOOKUP:
        raise KeyError(
            f"unknown model {name!r}; choose from {', '.join(available_models())}"
        )
    return _LOOKUP[key]


def load_model(
    name: str,
    *,
    checkpoint: str | Path | None = None,
    device: Any = "cpu",
    **kwargs: Any,
):
    """Load a model as a scorer accepted by ``bioseq-optimization``."""
    info = model_info(name)
    module_name, class_name = info.loader_path.split(":", maxsplit=1)
    adapter = getattr(import_module(module_name), class_name)
    options = {"device": device, **kwargs}
    if checkpoint is not None:
        options["checkpoint"] = checkpoint
    scorer = adapter.from_checkpoint(**options)
    scorer.model_info = info
    return scorer
