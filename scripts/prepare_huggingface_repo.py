#!/usr/bin/env python3
"""Prepare the file tree uploaded to the bioseq-models Hugging Face repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from bioseq_models.resources import MODEL_FILES  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Directory containing the checkpoints/, utr_code/, and cds_stability/ trees",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = arguments()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    files = sorted({filename for values in MODEL_FILES.values() for filename in values})
    missing = [filename for filename in files if not (args.source / filename).is_file()]
    if missing:
        raise FileNotFoundError("missing weight inputs: " + ", ".join(missing))

    args.output.mkdir(parents=True, exist_ok=True)
    for filename in files:
        link_or_copy(args.source / filename, args.output / filename)
    shutil.copy2(REPOSITORY / "huggingface_repo" / "README.md", args.output / "README.md")
    shutil.copy2(
        REPOSITORY / "THIRD_PARTY_NOTICES.md",
        args.output / "THIRD_PARTY_NOTICES.md",
    )
    version_line = next(
        line
        for line in (REPOSITORY / "pyproject.toml").read_text().splitlines()
        if line.startswith("version = ")
    )
    version = version_line.split("=", maxsplit=1)[1].strip().strip('"')
    manifest = {
        "bundle_version": version,
        "files": {
            filename: {
                "bytes": (args.output / filename).stat().st_size,
                "sha256": hashlib.sha256(
                    (args.output / filename).read_bytes()
                ).hexdigest(),
            }
            for filename in files
        },
    }
    (args.output / "weights_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(args.output)
    print(f"Prepared {len(files)} weight files")


if __name__ == "__main__":
    main()
