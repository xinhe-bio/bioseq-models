#!/usr/bin/env python3
"""Extract a bioseq-models checkpoint from the trusted official Malinois file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from bioseq_models.architectures import MalinoisNetwork

EXPECTED_HPARAMETERS = {
    "input_len": 600,
    "conv1_channels": 300,
    "conv1_kernel_size": 19,
    "conv2_channels": 200,
    "conv2_kernel_size": 11,
    "conv3_channels": 200,
    "conv3_kernel_size": 7,
    "n_linear_layers": 1,
    "linear_channels": 1000,
    "n_branched_layers": 3,
    "branched_channels": 140,
    "n_outputs": 3,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    # This file is a trusted artifact linked from the authors' BODA repository.
    checkpoint = torch.load(
        args.official_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("model_module") != "BassetBranched":
        raise ValueError("official checkpoint is not a BassetBranched model")
    hparams = vars(checkpoint["model_hparams"])
    mismatches = {
        key: (hparams.get(key), expected)
        for key, expected in EXPECTED_HPARAMETERS.items()
        if hparams.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"unexpected Malinois hyperparameters: {mismatches}")

    state = checkpoint["model_state_dict"]
    MalinoisNetwork().load_state_dict(state, strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": state,
            "source_timestamp": checkpoint["timestamp"],
            "source_random_tag": checkpoint["random_tag"],
        },
        args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()
