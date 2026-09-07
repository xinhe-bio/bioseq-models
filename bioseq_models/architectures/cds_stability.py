"""CNN component of the multi-species CDS stability predictor."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


CODONS = tuple(
    first + second + third
    for second in "TCAG"
    for first in "TCAG"
    for third in "TCAG"
) + ("NNN",)
CODON_TO_INDEX = {codon: index for index, codon in enumerate(CODONS)}
CODON_COUNT = 3000
MAX_BASES = CODON_COUNT * 3

HYPERPARAMETERS = {
    "dense4": 512,
    "dense6": 32,
    "dilation1": 1,
    "dilation3": 4,
    "dropout1": 0.5,
    "dropout3": 0.9,
    "dropout4": 0.025,
    "kernel_size1": 5,
    "kernel_size3": 30,
    "filters1": 256,
    "filters3": 64,
    "filters_stride1": 3,
    "filters_stride3": 1,
    "pool_size1": 3,
    "pool_size3": 3,
    "stride1": 1,
    "stride3": 1,
}


def _conversion_matrix() -> torch.Tensor:
    base_to_index = {"A": 0, "C": 1, "G": 2, "T": 3}
    eye = np.eye(4, dtype=np.float32)
    rows = [[eye[base_to_index[base]] for base in codon] for codon in CODONS[:-1]]
    rows.append([np.zeros(4, dtype=np.float32) for _ in range(3)])
    return torch.from_numpy(np.asarray(rows, dtype=np.float32).reshape(65, 12))


def encode_cds_sequences(sequences) -> torch.Tensor:
    """Encode validated CDS strings to the original 65 x 3000 codon tensor."""
    encoded = []
    for sequence in sequences:
        indices = [
            CODON_TO_INDEX[sequence[offset : offset + 3]]
            for offset in range(0, len(sequence), 3)
        ]
        indices.extend([CODON_TO_INDEX["NNN"]] * (CODON_COUNT - len(indices)))
        array = np.zeros((65, CODON_COUNT), dtype=np.float32)
        array[np.asarray(indices), np.arange(CODON_COUNT)] = 1.0
        encoded.append(array)
    if not encoded:
        return torch.empty((0, 65, CODON_COUNT), dtype=torch.float32)
    return torch.from_numpy(np.stack(encoded))


class _Transpose(nn.Module):
    def forward(self, value):
        return value.transpose(2, 1)


class CDSStabilityNetwork(nn.Module):
    """Exact CNN-plus-embedding architecture for the selected checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        p = HYPERPARAMETERS
        self.motif_depth = 4
        self.cds_detector_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        4,
                        p["filters1"],
                        p["kernel_size1"],
                        stride=p["filters_stride1"],
                        dilation=p["dilation1"],
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(p["filters1"]),
                )
                for _ in range(self.motif_depth)
            ]
        )
        self.cds_filter = nn.Sequential(
            nn.BatchNorm1d(p["filters1"]),
            nn.Dropout1d(p["dropout1"]),
            nn.MaxPool1d(p["pool_size1"], stride=p["stride1"]),
        )
        self.cds_encoder = nn.Sequential(
            nn.Conv1d(
                p["filters1"],
                p["filters3"],
                p["kernel_size3"],
                stride=p["filters_stride3"],
                dilation=p["dilation3"],
            ),
            nn.ReLU(),
            nn.BatchNorm1d(p["filters3"]),
            nn.MaxPool1d(p["pool_size3"], stride=p["stride3"]),
            _Transpose(),
            nn.Dropout1d(p["dropout3"]),
            nn.Flatten(),
        )
        self.mRNA_layer = nn.Sequential(
            nn.Linear(1, p["dense4"]),
            nn.ReLU(),
            nn.BatchNorm1d(p["dense4"]),
            nn.Dropout(p["dropout4"]),
            nn.Linear(p["dense4"], p["dense6"]),
            nn.ReLU(),
            nn.BatchNorm1d(p["dense6"]),
        )
        # Retained because the tensor exists in the original checkpoint.
        self.feature_layer = nn.Sequential(
            nn.Linear(200, p["dense4"]),
            nn.ReLU(),
            nn.BatchNorm1d(p["dense4"]),
            nn.Dropout(p["dropout4"]),
            nn.Linear(p["dense4"], p["dense6"]),
            nn.ReLU(),
            nn.BatchNorm1d(p["dense6"]),
        )
        self.cds_mlp_pCSCg = nn.Sequential(nn.Linear(184256, p["dense4"]))
        self.prediction_layer_pCSCg = nn.Sequential(
            nn.Linear(1280, p["dense6"]),
            nn.ReLU(),
            nn.BatchNorm1d(p["dense6"]),
            nn.Dropout(p["dropout4"]),
            nn.Linear(p["dense6"], p["dense6"]),
            nn.ReLU(),
            nn.BatchNorm1d(p["dense6"]),
            nn.Dropout(p["dropout4"]),
            nn.Linear(p["dense6"], 1),
        )
        self.attention_cds = nn.Sequential(
            nn.Linear(p["dense6"], p["filters1"] * self.motif_depth), nn.Tanh()
        )
        self.register_buffer("convert_matrix", _conversion_matrix(), persistent=False)

    def _motif_detection(self, sequence, attention):
        results = [detector(sequence) for detector in self.cds_detector_list]
        weights = torch.exp(attention)
        features = torch.sum(torch.stack(results, 3) * weights, 3) / torch.sum(
            weights, 3
        )
        return self.cds_filter(features)

    def forward(self, cds, condition, codonbert_cls):
        length = cds.shape[-1]
        bases = (cds.transpose(2, 1) @ self.convert_matrix).reshape(
            -1, length * 3, 4
        ).transpose(2, 1)
        condition_features = self.mRNA_layer(condition)
        attention = self.attention_cds(condition_features).reshape(
            -1, 256, 1, self.motif_depth
        )
        features = self.cds_encoder(self._motif_detection(bases, attention))
        features = self.cds_mlp_pCSCg(features)
        return self.prediction_layer_pCSCg(
            torch.cat([features, codonbert_cls], dim=1)
        )
