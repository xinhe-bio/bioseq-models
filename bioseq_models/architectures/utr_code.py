"""UTR-CODE network used by the human context-dependent RPF checkpoint."""

from __future__ import annotations

import torch
from torch import nn

HYPERPARAMETERS = {
    "dense4": 512,
    "dense5": 128,
    "dense6": 32,
    "dilation1": 1,
    "dilation3": 4,
    "dropout1": 0.5,
    "dropout3": 0.9,
    "dropout4": 0.025,
    "dropout5": 0.025,
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


class _Transpose(nn.Module):
    def forward(self, value):
        return value.transpose(2, 1)


class UTRCodeNetwork(nn.Module):
    """Exact inference architecture for the selected UTR-CODE checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        p = HYPERPARAMETERS
        self.motif_depth = 4
        self.bias = nn.Parameter(torch.zeros(1))
        self.moving_bias = 0

        self.detectorDict = nn.ModuleDict()
        for region in ("reg", "cds"):
            self.detectorDict[region] = nn.ModuleList(
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

        self.filterDict = nn.ModuleDict()
        self.encoderDict = nn.ModuleDict()
        for region in ("reg", "cds"):
            self.filterDict[region] = nn.Sequential(
                nn.BatchNorm1d(p["filters1"]),
                nn.Dropout1d(p["dropout1"]),
                nn.MaxPool1d(p["pool_size1"], stride=p["stride1"]),
            )
            self.encoderDict[region] = nn.Sequential(
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

        self.utr_mlp = nn.Sequential(
            nn.Linear(self._flatten_size("reg", 5000), p["dense4"]),
            nn.ReLU(),
            nn.BatchNorm1d(p["dense4"]),
            nn.Dropout(p["dropout4"]),
            nn.Linear(p["dense4"], p["dense6"]),
            nn.ReLU(),
            nn.BatchNorm1d(p["dense6"]),
        )
        self.RPF_fc = nn.Sequential(
            nn.Linear(self._flatten_size("cds", 4500), p["dense5"]),
            nn.ReLU(),
            nn.BatchNorm1d(p["dense5"]),
            nn.Dropout(p["dropout5"]),
            nn.Linear(p["dense5"], 1),
        )
        self.attention_utr = nn.Sequential(
            nn.Linear(p["dense6"], p["filters1"] * self.motif_depth), nn.Tanh()
        )
        self.attention_cds = nn.Sequential(
            nn.Linear(p["dense6"] + 1, p["filters1"] * self.motif_depth),
            nn.Tanh(),
        )
        self.mRNA_layer = nn.Sequential(
            nn.Linear(13238, p["dense4"]),
            nn.ReLU(),
            nn.BatchNorm1d(p["dense4"]),
            nn.Dropout(p["dropout4"]),
            nn.Linear(p["dense4"], p["dense6"]),
            nn.ReLU(),
            nn.BatchNorm1d(p["dense6"]),
        )

    def _motif_detection(self, sequence, attention, region):
        results = [detector(sequence) for detector in self.detectorDict[region]]
        weights = torch.exp(attention)
        features = torch.sum(torch.stack(results, 3) * weights, 3) / torch.sum(
            weights, 3
        )
        return self.filterDict[region](features)

    def _flatten_size(self, region: str, length: int) -> int:
        value = torch.zeros(1, 4, length)
        attention = torch.zeros(1, 256, 1, self.motif_depth)
        encoded = self.encoderDict[region](
            self._motif_detection(value, attention, region)
        )
        return encoded.numel()

    def forward(self, regulatory, cds, transcriptome, rna_count):
        transcriptome_features = self.mRNA_layer(transcriptome)
        regulatory_attention = self.attention_utr(transcriptome_features).reshape(
            -1, 256, 1, self.motif_depth
        )
        regulatory_features = self.encoderDict["reg"](
            self._motif_detection(regulatory, regulatory_attention, "reg")
        )
        regulatory_features = self.utr_mlp(regulatory_features)
        cds_attention = self.attention_cds(
            torch.cat([regulatory_features, rna_count], dim=1)
        ).reshape(-1, 256, 1, self.motif_depth)
        cds_features = self.encoderDict["cds"](
            self._motif_detection(cds, cds_attention, "cds")
        )
        translation_component = self.RPF_fc(cds_features)
        rpf = translation_component * rna_count + self.bias
        return torch.cat([rpf, translation_component], dim=1)
