#!/usr/bin/env python
"""Convert an original Saluki HDF5 checkpoint without TensorFlow."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from bioseq_models.saluki_weights import convert_h5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--species", choices=("human", "mouse"), required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--cross", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.fold < 10 or not 0 <= args.cross < 5:
        parser.error("fold must be 0..9 and cross must be 0..4")
    model, config = convert_h5(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            architecture="saluki_official",
            model_state=model.state_dict(),
            model_config=config,
            species=args.species,
            fold=args.fold,
            cross=args.cross,
            source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            output_units="Species-specific consensus half-life PC1 score; higher predicts stability.",
            reverse_complement=False,
        ),
        args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()
