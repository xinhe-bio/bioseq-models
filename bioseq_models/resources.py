"""Download and resolve the versioned bioseq-models weight bundle."""

from __future__ import annotations

import os
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


BUNDLE_VERSION = "0.5.0"
BUNDLE_FILENAME = f"bioseq-models-weights-v{BUNDLE_VERSION}.zip"
# Filled after the bundle is deposited. BIOSEQ_MODELS_WEIGHTS_URL can be used
# before then, and remains useful for mirrors.
ZENODO_URL: str | None = None


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _verify(directory: Path) -> None:
    manifest_path = directory / "weights_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("weight bundle does not contain weights_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(
            f"weight bundle version is {manifest.get('bundle_version')!r}; "
            f"expected {BUNDLE_VERSION!r}"
        )
    for relative, expected in manifest.get("files", {}).items():
        path = directory / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"weight resource is missing or incomplete: {relative}")
        if _digest(path) != expected["sha256"]:
            raise ValueError(f"weight resource failed integrity check: {relative}")


def _default_cache_root() -> Path:
    if value := os.environ.get("BIOSEQ_MODELS_CACHE"):
        return Path(value).expanduser()
    if value := os.environ.get("XDG_CACHE_HOME"):
        return Path(value).expanduser() / "bioseq-models"
    return Path.home() / ".cache" / "bioseq-models"


def weights_dir(path: str | Path | None = None, *, download: bool = True) -> Path:
    """Return the extracted weight directory, downloading it when necessary."""
    explicit = path or os.environ.get("BIOSEQ_MODELS_WEIGHTS_DIR")
    if explicit:
        directory = Path(explicit).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError(f"weight directory not found: {directory}")
        return directory

    directory = _default_cache_root() / BUNDLE_VERSION
    if (directory / ".complete").is_file():
        return directory
    if not download:
        raise FileNotFoundError(f"weight bundle is not cached at {directory}")
    return download_weights(cache_dir=directory)


def weight_path(relative_path: str, path: str | Path | None = None) -> Path:
    """Resolve one file from the extracted, versioned weight bundle."""
    directory = weights_dir(path)
    resolved = directory / relative_path
    if not resolved.is_file():
        raise FileNotFoundError(
            f"weight resource {relative_path!r} is missing from {directory}"
        )
    return resolved


def download_weights(
    *,
    url: str | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download and safely extract the complete Zenodo weight bundle."""
    source = url or os.environ.get("BIOSEQ_MODELS_WEIGHTS_URL") or ZENODO_URL
    if not source:
        raise RuntimeError(
            "The Zenodo weight bundle has not been published yet. Set "
            "BIOSEQ_MODELS_WEIGHTS_DIR to an extracted bundle or "
            "BIOSEQ_MODELS_WEIGHTS_URL to the uploaded ZIP URL."
        )

    destination = Path(cache_dir) if cache_dir else _default_cache_root() / BUNDLE_VERSION
    destination = destination.expanduser()
    marker = destination / ".complete"
    if marker.is_file() and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / f".{BUNDLE_FILENAME}.{os.getpid()}.download"
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.extract"
    try:
        with urllib.request.urlopen(source) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        with zipfile.ZipFile(archive) as bundle:
            root = temporary.resolve()
            for member in bundle.infolist():
                target = (temporary / member.filename).resolve()
                if root != target and root not in target.parents:
                    raise ValueError(f"unsafe path in weight bundle: {member.filename}")
            bundle.extractall(temporary)
        _verify(temporary)
        (temporary / ".complete").write_text(BUNDLE_VERSION + "\n")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    finally:
        archive.unlink(missing_ok=True)
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination
