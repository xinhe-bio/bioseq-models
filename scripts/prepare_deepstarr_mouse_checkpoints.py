#!/usr/bin/env python3
"""Convert all official DeepSTARR-Mouse folds and replicates to PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

from convert_deepstarr_tissue_checkpoint import convert_deepstarr_checkpoint

TISSUES = {
    "heart": "heart",
    "limb": "limb",
    "midbrain": "midbrain",
    "forebrain": "forebrain",
    "hindbrain": "hindbrain",
    "neural_tube": "neuralTube",
}
SOURCE_DIRECTORIES = {
    "activity": "enhancer_activity_models",
    "accessibility": "accessibility_models",
}


def _source_checkpoint(
    source: Path,
    tissue: str,
    source_tissue: str,
    kind: str,
    fold: int,
    replicate: int,
) -> Path:
    tissue_directory = source_tissue
    if kind == "activity" and tissue == "midbrain":
        tissue_directory = "midbrain(CNS)"
    root = source / SOURCE_DIRECTORIES[kind] / tissue_directory
    matches = sorted(root.glob(f"results_fold{fold:02d}_*_rep{replicate}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one source checkpoint in {root} for fold {fold:02d}, "
            f"replicate {replicate}; found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    converted = []
    for tissue, source_tissue in TISSUES.items():
        for kind in ("activity", "accessibility"):
            for fold in range(1, 4):
                for replicate in range(1, 3):
                    source = _source_checkpoint(
                        args.source,
                        tissue,
                        source_tissue,
                        kind,
                        fold,
                        replicate,
                    )
                    output = (
                        args.output
                        / "checkpoints"
                        / (
                            f"deepstarr_mouse_{tissue}_{kind}_"
                            f"fold{fold:02d}_rep{replicate}.pt"
                        )
                    )
                    convert_deepstarr_checkpoint(
                        "mouse",
                        source / "Model.json",
                        source / "Model.h5",
                        output,
                    )
                    converted.append(output)
                    print(output)

    print(f"Converted {len(converted)} DeepSTARR-Mouse checkpoints")


if __name__ == "__main__":
    main()
