"""Lazy registry for checkpoint-backed activity models."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from .deepstarr_embryo import checkpoint_files as _embryo_checkpoint_files


@dataclass(frozen=True)
class ModelInfo:
    name: str
    sequence_length: int | None
    task_names: tuple[str, ...]
    description: str
    loader_path: str
    aliases: tuple[str, ...] = ()
    input_kind: str = "sequence"
    status: str = "reproduced"


_DEEPSTARR_MOUSE_TISSUES = (
    ("heart", "Heart"),
    ("limb", "Limb"),
    ("midbrain", "Midbrain"),
    ("forebrain", "Forebrain"),
    ("hindbrain", "Hindbrain"),
    ("neural_tube", "NeuralTube"),
)

_DEEPSTARR_EMBRYO_ACCESSIBILITY_TISSUES = (
    ("brain", "Brain"),
    ("cns", "CNS"),
    ("epidermis", "Epidermis"),
    ("midgut", "Midgut"),
    ("somatic_muscle", "SomaticMuscle"),
    ("visceral_muscle", "VisceralMuscle"),
)

_DEEPSTARR_EMBRYO_ACTIVITY_TISSUES = (
    ("brain_vnc", "BrainVNC"),
    ("muscle", "Muscle"),
    ("epidermis", "Epidermis"),
    ("gut", "Gut"),
    ("brain_specific", "BrainSpecific"),
)


def _deepstarr_mouse_info(tissue: str, class_tissue: str, kind: str) -> ModelInfo:
    canonical = f"deepstarr_mouse_{tissue}_{kind}"
    task = f"{tissue}_{'activity_probability' if kind == 'activity' else kind}"
    aliases: tuple[str, ...] = ()
    if tissue == "heart" and kind == "activity":
        aliases = ("deepstarr_mouse", "deepstarr_mouse_heart")
    elif tissue == "heart" and kind == "accessibility":
        aliases = ("deepstarr_mouse_accessibility",)
    elif tissue == "midbrain" and kind == "activity":
        aliases = ("deepstarr_mouse_cns", "deepstarr_mouse_cns_activity")
    elif tissue == "midbrain" and kind == "accessibility":
        aliases = ("deepstarr_mouse_cns_accessibility",)
    elif tissue == "neural_tube":
        aliases = (f"deepstarr_mouse_neuraltube_{kind}",)
    return ModelInfo(
        canonical,
        1001,
        (task,),
        f"DeepSTARR-Mouse {tissue.replace('_', ' ')} {kind}, six-checkpoint ensemble",
        f"bioseq_models.adapters.deepstarr_tissue:DeepSTARRMouse{class_tissue}{kind.title()}",
        aliases,
        status="converted_official_checkpoint",
    )


def _deepstarr_embryo_info(
    tissue: str, class_tissue: str, kind: str
) -> ModelInfo:
    canonical = f"deepstarr_embryo_{tissue}_{kind}"
    task = f"{tissue}_{'activity_probability' if kind == 'activity' else kind}"
    aliases: tuple[str, ...] = ()
    if tissue == "brain_vnc" and kind == "activity":
        aliases = ("deepstarr_embryo", "deepstarr_embryo_brain_vnc")
    elif tissue == "brain_specific" and kind == "activity":
        aliases = ("deepstarr_embryo_brain_specific",)
    return ModelInfo(
        canonical,
        1001,
        (task,),
        f"DeepSTARR-Embryo {tissue.replace('_', ' ')} {kind}, "
        f"{len(_embryo_checkpoint_files(tissue, kind))}-checkpoint ensemble",
        f"bioseq_models.adapters.deepstarr_tissue:DeepSTARREmbryo{class_tissue}{kind.title()}",
        aliases,
        status="converted_official_checkpoint",
    )


_MODELS = (
    ModelInfo(
        "optimus5prime",
        50,
        ("mrl",),
        "Optimus 5-Prime mean ribosome load model",
        "bioseq_models.adapters.optimus5prime:Optimus5Prime",
        ("optimus", "optimus_5prime"),
        status="paper_architecture",
    ),
    ModelInfo(
        "deepstarr",
        249,
        ("developmental", "housekeeping"),
        "DeepSTARR two-task enhancer activity model",
        "bioseq_models.adapters.deepstarr:DeepSTARR",
        status="paper_architecture",
    ),
    *(
        _deepstarr_mouse_info(tissue, class_tissue, kind)
        for tissue, class_tissue in _DEEPSTARR_MOUSE_TISSUES
        for kind in ("activity", "accessibility")
    ),
    *(
        _deepstarr_embryo_info(tissue, class_tissue, "accessibility")
        for tissue, class_tissue in _DEEPSTARR_EMBRYO_ACCESSIBILITY_TISSUES
    ),
    *(
        _deepstarr_embryo_info(tissue, class_tissue, "activity")
        for tissue, class_tissue in _DEEPSTARR_EMBRYO_ACTIVITY_TISSUES
    ),
    ModelInfo(
        "malinois",
        200,
        ("K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC"),
        "Malinois three-cell-type enhancer activity model",
        "bioseq_models.adapters.malinois:Malinois",
        status="official_checkpoint",
    ),
    ModelInfo(
        "dream_legnet",
        110,
        ("promoter_activity",),
        "DREAM 2022 Autosome/LegNet; 110-nt full or 93-nt variable-region input",
        "bioseq_models.adapters.dream_legnet:DreamLegNet",
        ("dream", "autosome_legnet"),
        status="paper_architecture",
    ),
    ModelInfo(
        "cetnar_stability",
        None,
        ("decay_rate_k_fit",),
        "Variable-length mRNA stability CNN-BiLSTM (up to 80 nt)",
        "bioseq_models.adapters.cetnar:CetnarStability",
        ("cetnar", "cetnar_cnn"),
        status="benchmark_surrogate",
    ),
    ModelInfo(
        "aparent",
        205,
        tuple(f"cleavage_probability_{position}" for position in range(205))
        + ("distal_cleavage_probability",),
        "APARENT 205-position alternative-polyadenylation cleavage distribution",
        "bioseq_models.adapters.aparent:APARENT",
        ("aparent_site_probabilities", "aparent_apa"),
        status="converted_official_checkpoint",
    ),
    ModelInfo(
        "utr_code",
        None,
        ("log1p_rpf_rpkm_x5",),
        "UTR-CODE human context-dependent RPF model for 5'UTR, CDS, or 3'UTR design",
        "bioseq_models.adapters.utr_code:UTRCode",
        ("utrcode", "utr-code"),
        status="project_checkpoint",
    ),
    ModelInfo(
        "deepadapt_heatshock",
        None,
        ("delta_log2_te", "normal_log2_te", "heatshock_log2_te"),
        "DeepAdapt WI-38 senescent heat-shock translation-efficiency model",
        "bioseq_models.adapters.deepadapt:DeepAdaptHeatShock",
        ("deepadapt", "heatshock"),
        status="project_checkpoint",
    ),
    ModelInfo(
        "cds_stability",
        None,
        ("stability_score",),
        "Multi-species CNN-CodonBERT stability model for complete CDS inputs up to 9000 nt",
        "bioseq_models.adapters.cds_stability:CDSStability",
        ("mrna_stability", "codonbert_stability"),
        status="project_checkpoint",
    ),
    ModelInfo(
        "parm_k562",
        None,
        ("log2rpm_k562",),
        "Five-fold exact PARM K562 ensemble; adapters and 600-position padding included",
        "bioseq_models.adapters.parm:PARMK562",
        ("parm",),
        status="paper_architecture",
    ),
    ModelInfo(
        "agarwal_hepg2",
        230,
        ("log2_rna_dna",),
        "Agarwal lentiMPRA HepG2 ten-fold MPRALegNet ensemble",
        "bioseq_models.adapters.agarwal:AgarwalHepG2",
        ("mpralegnet_hepg2",),
        status="paper_architecture",
    ),
    ModelInfo(
        "agarwal_k562",
        230,
        ("log2_rna_dna",),
        "Agarwal lentiMPRA K562 ten-fold MPRALegNet ensemble",
        "bioseq_models.adapters.agarwal:AgarwalK562",
        ("mpralegnet_k562",),
        status="paper_architecture",
    ),
    ModelInfo(
        "agarwal_wtc11",
        230,
        ("log2_rna_dna",),
        "Agarwal lentiMPRA WTC11 ten-fold MPRALegNet ensemble",
        "bioseq_models.adapters.agarwal:AgarwalWTC11",
        ("mpralegnet_wtc11",),
        status="paper_architecture",
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
    """Load a registered sequence model with its bundled checkpoint."""
    info = model_info(name)
    module_name, class_name = info.loader_path.split(":", maxsplit=1)
    adapter = getattr(import_module(module_name), class_name)
    options = {"device": device, **kwargs}
    if checkpoint is not None:
        options["checkpoint"] = checkpoint
    scorer = adapter.from_checkpoint(**options)
    scorer.model_info = info
    return scorer
