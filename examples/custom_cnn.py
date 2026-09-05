"""Minimal custom PyTorch CNN integration example."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from sequence_optimization import (
    DifferentiableTorchSequenceScorer,
    SequenceSpace,
    genetic_algorithm_optimize,
)


SEQUENCE_LENGTH = 50
BASE_TO_CHANNEL = {base: channel for channel, base in enumerate("ACGT")}


class SimpleActivityCNN(nn.Module):
    """One convolution, global pooling, and one scalar activity output."""

    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv1d(4, 32, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(32, 1)

    def forward(self, sequence_tensor):
        features = torch.relu(self.convolution(sequence_tensor))
        return self.output(self.pool(features).squeeze(-1))


def encode_sequences(sequences) -> torch.Tensor:
    """Convert 50-nt strings to CNN input shaped ``[batch, 4, 50]``."""
    encoded = torch.zeros((len(sequences), 4, SEQUENCE_LENGTH))
    for row, sequence in enumerate(sequences):
        if len(sequence) != SEQUENCE_LENGTH:
            raise ValueError(f"every sequence must contain {SEQUENCE_LENGTH} bases")
        for position, base in enumerate(sequence):
            encoded[row, BASE_TO_CHANNEL[base], position] = 1.0
    return encoded


def wrap_custom_cnn(model: nn.Module, *, device: str = "cpu"):
    """Expose a trained CNN through the common optimization interface."""
    return DifferentiableTorchSequenceScorer(
        model,
        encode_sequences,
        soft_input_transform=lambda probabilities: probabilities.transpose(1, 2),
        task_names=("activity",),
        device=device,
    )


def load_custom_cnn(checkpoint_path: str | Path, *, device: str = "cpu"):
    """Load weights saved either directly or under a ``model_state`` key."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state", checkpoint)
    model = SimpleActivityCNN()
    model.load_state_dict(state_dict)
    return wrap_custom_cnn(model, device=device)


def optimize_with_custom_cnn(checkpoint_path: str | Path, *, device: str = "cpu"):
    """Run a small genetic-algorithm search from four real 5′UTR seeds."""
    scorer = load_custom_cnn(checkpoint_path, device=device)
    initial_sequences = [
        "CCACTCGATTAACATGTTAACAACATACTCGTCCGGCCGATCAGCGGCTA",
        "CAAATCATGTGCAGCCCTGGCGACCGTACTGCGGTACAAGAAAGTATAGT",
        "GTTATACTAGAAGAAACTTGAGATTATGGAGCAGTCCGTCAAGGACGCGA",
        "CTTAGACAAAAACAACGCGCTTTCCAGTATGCGGAGCCTTGACGGTAAAC",
    ]
    return genetic_algorithm_optimize(
        scorer,
        SequenceSpace(length=SEQUENCE_LENGTH),
        initial=initial_sequences,
        num_candidates=4,
        budget=44,
        mutation_rate=0.1,
        elite_fraction=0.5,
        seed=7,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to the trained CNN checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = optimize_with_custom_cnn(args.checkpoint, device=args.device)
    print("sequence\tactivity")
    for sequence, activity in zip(result.sequences, result.activities[:, 0]):
        print(f"{sequence}\t{activity:.6f}")


if __name__ == "__main__":
    main()
