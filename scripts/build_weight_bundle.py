#!/usr/bin/env python3
"""Build the single versioned ZIP uploaded as a Zenodo record file."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-checkpoints", type=Path, required=True)
    parser.add_argument("--utr-code-checkpoint", type=Path, required=True)
    parser.add_argument("--hek293t-profile", type=Path, required=True)
    parser.add_argument("--cds-stability-checkpoint", type=Path, required=True)
    parser.add_argument("--codonbert-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    repository = Path(__file__).resolve().parents[1]
    files: list[tuple[Path, str]] = []
    for source in sorted(args.existing_checkpoints.glob("*.pt")):
        files.append((source, f"checkpoints/{source.name}"))
    files.extend(
        [
            (args.utr_code_checkpoint, "utr_code/best.epoch"),
            (args.hek293t_profile, "utr_code/HEK293T.txt"),
            (args.cds_stability_checkpoint, "cds_stability/best_model.pt"),
            (
                args.codonbert_directory / "config.json",
                "cds_stability/codonbert/config.json",
            ),
            (
                args.codonbert_directory / "pytorch_model.bin",
                "cds_stability/codonbert/pytorch_model.bin",
            ),
            (repository / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
        ]
    )
    missing = [str(source) for source, _ in files if not source.is_file()]
    if missing:
        raise FileNotFoundError("missing weight inputs: " + ", ".join(missing))

    manifest = {
        "format_version": 1,
        "bundle_version": "0.5.0",
        "files": {
            target: {"bytes": source.stat().st_size, "sha256": digest(source)}
            for source, target in files
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        manifest_path = Path(temporary) / "weights_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        with zipfile.ZipFile(args.output, "w", allowZip64=True) as archive:
            for source, target in files:
                archive.write(source, target, compress_type=zipfile.ZIP_STORED)
            archive.write(
                manifest_path,
                "weights_manifest.json",
                compress_type=zipfile.ZIP_DEFLATED,
            )

    sidecar = {
        **manifest,
        "archive": {
            "filename": args.output.name,
            "bytes": args.output.stat().st_size,
            "sha256": digest(args.output),
        },
    }
    sidecar_path = args.output.with_suffix(".manifest.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(sidecar_path)


if __name__ == "__main__":
    main()
