"""Lazy registry for checkpoint-backed activity models."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelInfo:
    name: str
    sequence_length: int | None
    task_names: tuple[str, ...]
    description: str
    differentiable: bool
    loader_path: str
    aliases: tuple[str, ...] = ()
    input_kind: str = "sequence"
    status: str = "reproduced"


_MODELS = (
    ModelInfo(
        "optimus5prime",
        50,
        ("mrl",),
        "Optimus 5-Prime mean ribosome load model",
        True,
        "bioseq_models.adapters.optimus5prime:Optimus5Prime",
        ("optimus", "optimus_5prime"),
        status="paper_architecture",
    ),
    ModelInfo(
        "deepstarr",
        249,
        ("developmental", "housekeeping"),
        "DeepSTARR two-task enhancer activity model",
        True,
        "bioseq_models.adapters.deepstarr:DeepSTARR",
        status="paper_architecture",
    ),
    ModelInfo(
        "dream_legnet",
        110,
        ("promoter_activity",),
        "DREAM 2022 yeast promoter Autosome/LegNet model",
        False,
        "bioseq_models.adapters.dream_legnet:DreamLegNet",
        ("dream", "autosome_legnet"),
        status="paper_architecture",
    ),
    ModelInfo(
        "cetnar_stability",
        None,
        ("decay_rate_k_fit", "stability"),
        "Variable-length mRNA stability CNN-BiLSTM (up to 80 nt)",
        True,
        "bioseq_models.adapters.cetnar:CetnarStability",
        ("cetnar", "cetnar_cnn"),
        status="benchmark_surrogate",
    ),
    ModelInfo(
        "parm_k562",
        None,
        ("log2rpm_k562",),
        "Five-fold exact PARM K562 ensemble; adapters and 600-position padding included",
        True,
        "bioseq_models.adapters.parm:PARMK562",
        ("parm",),
        status="paper_architecture",
    ),
    ModelInfo(
        "agarwal_hepg2",
        230,
        ("log2_rna_dna",),
        "Agarwal lentiMPRA HepG2 ten-fold MPRALegNet ensemble",
        True,
        "bioseq_models.adapters.agarwal:AgarwalHepG2",
        ("mpralegnet_hepg2",),
        status="paper_architecture",
    ),
    ModelInfo(
        "agarwal_k562",
        230,
        ("log2_rna_dna",),
        "Agarwal lentiMPRA K562 ten-fold MPRALegNet ensemble",
        True,
        "bioseq_models.adapters.agarwal:AgarwalK562",
        ("mpralegnet_k562",),
        status="paper_architecture",
    ),
    ModelInfo(
        "agarwal_wtc11",
        230,
        ("log2_rna_dna",),
        "Agarwal lentiMPRA WTC11 ten-fold MPRALegNet ensemble",
        True,
        "bioseq_models.adapters.agarwal:AgarwalWTC11",
        ("mpralegnet_wtc11",),
        status="paper_architecture",
    ),
    ModelInfo(
        "kosuri_composability",
        109,
        ("log10_protein_expression", "log10_rna_expression"),
        "Kosuri promoter/RBS two-branch composability CNN",
        True,
        "bioseq_models.adapters.kosuri:KosuriComposability",
        ("kosuri",),
        input_kind="75nt_promoter+34nt_rbs",
        status="benchmark_surrogate",
    ),
    ModelInfo(
        "zhao_ptrc",
        None,
        ("log10_fluorescence_per_od600",),
        "Zhao Ptrc sequence CNN accepting up to 76 nt",
        True,
        "bioseq_models.adapters.zhao_ptrc:ZhaoPtrc",
        ("ptrc", "zhao_ptrc_cnn"),
        status="benchmark_surrogate",
    ),
    ModelInfo(
        "zhao_ptrc_xgboost",
        None,
        ("log10_fluorescence_per_od600",),
        "Zhao Ptrc positional one-hot XGBoost reproduction accepting up to 76 nt",
        False,
        "bioseq_models.adapters.zhao_ptrc:ZhaoPtrcXGBoost",
        ("ptrc_xgboost",),
        status="functional_reproduction",
    ),
    ModelInfo(
        "zhou_l3_conditional",
        29,
        ("promoter_activity",),
        "Canonical L3 conditional CNN; choose L3_s1, L3_s2, L3_s3, or L3_s4",
        True,
        "bioseq_models.adapters.zhou_l3:ZhouL3Conditional",
        ("l3_conditional",),
        input_kind="sequence+sublibrary",
        status="benchmark_surrogate",
    ),
    ModelInfo(
        "zhou_l3",
        29,
        ("promoter_activity",),
        "L3 sequence-only diagnostic CNN",
        True,
        "bioseq_models.adapters.zhou_l3:ZhouL3",
        ("l3",),
        status="diagnostic_surrogate",
    ),
    ModelInfo(
        "tfbind8_weak",
        8,
        ("binding_score",),
        "Ten-member TFBind8 CNN ensemble trained on the lower 95% pool",
        True,
        "bioseq_models.adapters.tfbind8:TFBind8Weak",
        ("tfbind8",),
        status="weak_surrogate",
    ),
    ModelInfo(
        "utr5_weak",
        50,
        ("mrl",),
        "Five-member 5′UTR WeakSeqCNN ensemble trained on the lower 95% pool",
        True,
        "bioseq_models.adapters.weak:UTR5Weak",
        ("utr5_weakseqcnn",),
        status="weak_surrogate",
    ),
    ModelInfo(
        "dream_weak",
        None,
        ("promoter_activity",),
        "Five-member variable-length DREAM WeakSeqCNN lower-95% ensemble",
        True,
        "bioseq_models.adapters.weak:DreamWeak",
        ("dream_weakseqcnn",),
        status="weak_surrogate",
    ),
)


def _normalize(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


_LOOKUP = {
    _normalize(alias): info for info in _MODELS for alias in (info.name, *info.aliases)
}


def available_models(*, status: str | None = None) -> tuple[str, ...]:
    """List canonical model names, optionally filtered by reproduction status."""
    return tuple(
        info.name for info in _MODELS if status is None or info.status == status
    )


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
    """Load a registered model as a scorer accepted by bioseq-optimization."""
    info = model_info(name)
    module_name, class_name = info.loader_path.split(":", maxsplit=1)
    adapter = getattr(import_module(module_name), class_name)
    options = {"device": device, **kwargs}
    if checkpoint is not None:
        options["checkpoint"] = checkpoint
    scorer = adapter.from_checkpoint(**options)
    scorer.model_info = info
    return scorer
