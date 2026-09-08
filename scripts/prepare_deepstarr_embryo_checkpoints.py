#!/usr/bin/env python3
"""Convert every official DeepSTARR-Embryo fold and replicate to PyTorch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from convert_deepstarr_tissue_checkpoint import convert_deepstarr_checkpoint

from bioseq_models.deepstarr_embryo import EXCLUDED_CHECKPOINTS

ACCESSIBILITY_TISSUES = {
    "brain": "brain",
    "cns": "CNS",
    "epidermis": "epidermis",
    "midgut": "midgut",
    "somatic_muscle": "somaticmuscle",
    "visceral_muscle": "visceralmuscle",
}
ACTIVITY_TISSUES = {
    "brain_vnc": "brain_VNC",
    "muscle": "muscle",
    "epidermis": "epidermis",
    "gut": "Gut",
    "brain_specific": "brain_spec",
}


def _source_checkpoint(
    source: Path,
    source_tissue: str,
    kind: str,
    fold: int,
    replicate: int,
) -> Path:
    suffix = "_DeepSTARR2" if kind == "accessibility" else ""
    directory_name = (
        f"Results_fold{fold:02d}_{source_tissue}{suffix}_rep{replicate}"
    )
    matches = [
        path
        for path in source.rglob(directory_name)
        if path.is_dir()
        and (path / "Model.json").is_file()
        and (path / "Model.h5").is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {directory_name} checkpoint below {source}; "
            f"found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=("all", "accessibility", "activity"),
        default="all",
    )
    args = parser.parse_args()

    converted = []
    resources = (
        *((tissue, source_tissue, "accessibility") for tissue, source_tissue in ACCESSIBILITY_TISSUES.items()),
        *((tissue, source_tissue, "activity") for tissue, source_tissue in ACTIVITY_TISSUES.items()),
    )
    for tissue, source_tissue, kind in resources:
        if args.kind != "all" and args.kind != kind:
            continue
        for fold in range(1, 11):
            for replicate in range(1, 4):
                if (kind, tissue, fold, replicate) in EXCLUDED_CHECKPOINTS:
                    continue
                source = _source_checkpoint(
                    args.source,
                    source_tissue,
                    kind,
                    fold,
                    replicate,
                )
                output = (
                    args.output
                    / "checkpoints"
                    / (
                        f"deepstarr_embryo_{tissue}_{kind}_"
                        f"fold{fold:02d}_rep{replicate}.pt"
                    )
                )
                convert_deepstarr_checkpoint(
                    "embryo",
                    source / "Model.json",
                    source / "Model.h5",
                    output,
                )
                converted.append(output)
                print(output)

    print(f"Converted {len(converted)} DeepSTARR-Embryo checkpoints")


if __name__ == "__main__":
    main()
