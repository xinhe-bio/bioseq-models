"""Resolve model files locally or download only the requested Hugging Face files."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO_ID = "xinHesysu/bioseq-models"
HF_REVISION = "v0.8.0"

MODEL_FILES = {
    "optimus5prime": ("checkpoints/optimus5prime.pt",),
    "deepstarr": ("checkpoints/deepstarr.pt",),
    **{
        f"deepstarr_mouse_{tissue}_{kind}": tuple(
            f"checkpoints/deepstarr_mouse_{tissue}_{kind}_fold{fold:02d}_rep{rep}.pt"
            for fold in range(1, 4)
            for rep in range(1, 3)
        )
        for tissue in (
            "heart",
            "limb",
            "midbrain",
            "forebrain",
            "hindbrain",
            "neural_tube",
        )
        for kind in ("activity", "accessibility")
    },
    "deepstarr_embryo_brain_vnc_activity": (
        "checkpoints/deepstarr_embryo_brain_vnc_activity_fold01_rep1.pt",
    ),
    "malinois": ("checkpoints/malinois.pt",),
    "dream_legnet": ("checkpoints/dream_legnet.pt",),
    "cetnar_stability": ("checkpoints/cetnar_stability_cnn.pt",),
    "utr_code": ("utr_code/best.epoch", "utr_code/HEK293T.txt"),
    "cds_stability": (
        "cds_stability/best_model.pt",
        "cds_stability/codonbert/config.json",
        "cds_stability/codonbert/pytorch_model.bin",
    ),
    "parm_k562": tuple(f"checkpoints/parm_k562_fold{fold}.pt" for fold in range(5)),
    **{
        f"agarwal_{cell_type}": tuple(
            f"checkpoints/agarwal_{cell_type}_fold{fold}.pt" for fold in range(1, 11)
        )
        for cell_type in ("hepg2", "k562", "wtc11")
    },
}


def _local_weights_dir(path: str | Path | None = None) -> Path | None:
    value = path or os.environ.get("BIOSEQ_MODELS_WEIGHTS_DIR")
    if value is None:
        return None
    directory = Path(value).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"weight directory not found: {directory}")
    return directory


def weight_path(relative_path: str, path: str | Path | None = None) -> Path:
    """Resolve one resource, downloading that file only when it is not local."""
    if directory := _local_weights_dir(path):
        resolved = directory / relative_path
        if not resolved.is_file():
            raise FileNotFoundError(
                f"weight resource {relative_path!r} is missing from {directory}"
            )
        return resolved

    repo_id = os.environ.get("BIOSEQ_MODELS_HF_REPO_ID", HF_REPO_ID)
    revision = os.environ.get("BIOSEQ_MODELS_HF_REVISION", HF_REVISION)
    cache_dir = os.environ.get("BIOSEQ_MODELS_CACHE")
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=relative_path,
            revision=revision,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not download {relative_path!r} from Hugging Face repository "
            f"{repo_id!r} at revision {revision!r}; set "
            "BIOSEQ_MODELS_WEIGHTS_DIR to use local files"
        ) from exc
    return Path(downloaded)


def download_model(
    name: str,
    *,
    cache_dir: str | Path | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
) -> tuple[Path, ...]:
    """Pre-download exactly the files required by one registered model."""
    from .registry import model_info

    canonical = model_info(name).name
    files = MODEL_FILES[canonical]
    selected_repo = repo_id or os.environ.get("BIOSEQ_MODELS_HF_REPO_ID", HF_REPO_ID)
    selected_revision = revision or os.environ.get(
        "BIOSEQ_MODELS_HF_REVISION", HF_REVISION
    )
    selected_cache = cache_dir or os.environ.get("BIOSEQ_MODELS_CACHE")
    return tuple(
        Path(
            hf_hub_download(
                repo_id=selected_repo,
                filename=filename,
                revision=selected_revision,
                cache_dir=selected_cache,
            )
        )
        for filename in files
    )
