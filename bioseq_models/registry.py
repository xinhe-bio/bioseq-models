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
    ModelInfo(
        "deepstarr_mouse_heart_activity",
        1001,
        ("heart_activity_probability",),
        "DeepSTARR-Mouse heart activity, fold 01 replicate 1",
        "bioseq_models.adapters.deepstarr_tissue:DeepSTARRMouseHeartActivity",
        ("deepstarr_mouse", "deepstarr_mouse_heart"),
        status="converted_official_checkpoint",
    ),
    ModelInfo(
        "deepstarr_mouse_heart_accessibility",
        1001,
        ("heart_accessibility",),
        "DeepSTARR-Mouse heart accessibility, fold 01 replicate 1",
        "bioseq_models.adapters.deepstarr_tissue:DeepSTARRMouseHeartAccessibility",
        ("deepstarr_mouse_accessibility",),
        status="converted_official_checkpoint",
    ),
    ModelInfo(
        "deepstarr_embryo_brain_vnc_activity",
        1001,
        ("brain_vnc_activity_probability",),
        "DeepSTARR-Embryo brain/VNC activity, fold 01 replicate 1",
        "bioseq_models.adapters.deepstarr_tissue:DeepSTARREmbryoBrainVNCActivity",
        ("deepstarr_embryo", "deepstarr_embryo_brain_vnc"),
        status="converted_official_checkpoint",
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
        "utr_code",
        None,
        ("log1p_rpf_rpkm_x5",),
        "UTR-CODE human context-dependent RPF model for 5'UTR, CDS, or 3'UTR design",
        "bioseq_models.adapters.utr_code:UTRCode",
        ("utrcode", "utr-code"),
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
