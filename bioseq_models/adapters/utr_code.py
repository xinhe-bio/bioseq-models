"""Design-oriented adapter for the human UTR-CODE RPF model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..architectures import UTRCodeNetwork
from ..inference import _prediction_matrix, normalize_sequences
from ..resources import weight_path
from .common import encode_acgt_sequences


REGION_LIMITS = {"utr5": 1000, "cds": 4500, "utr3": 4000}


def _one_sequence(value: str, name: str, maximum: int) -> str:
    values = normalize_sequences([value])
    sequence = str(values[0])
    if len(sequence) > maximum:
        raise ValueError(f"{name} has length {len(sequence)}, exceeding {maximum} nt")
    return sequence


class UTRCode:
    """Score candidates for one transcript region in a fixed cellular context."""

    task_names = ("log1p_rpf_rpkm_x5",)

    def __init__(
        self,
        model,
        *,
        design_region: str,
        utr5: str | None,
        cds: str | None,
        utr3: str | None,
        transcriptome_profile,
        rna_count: float,
        device="cpu",
        batch_size: int = 256,
    ) -> None:
        region = str(design_region).strip().lower().replace("′", "").replace("'", "")
        region = {"5utr": "utr5", "3utr": "utr3"}.get(region, region)
        if region not in REGION_LIMITS:
            raise ValueError("design_region must be 'utr5', 'cds', or 'utr3'")
        fixed = {"utr5": utr5, "cds": cds, "utr3": utr3}
        if fixed[region] is not None:
            raise ValueError(f"do not supply {region}; it is provided to score()")
        for name, value in fixed.items():
            if name != region and value is None:
                raise ValueError(f"{name} is required when design_region={region!r}")
            if value is not None:
                fixed[name] = _one_sequence(value, name, REGION_LIMITS[name])

        profile = np.asarray(transcriptome_profile, dtype=np.float32).reshape(-1)
        if profile.size != 13238:
            raise ValueError(
                f"transcriptome_profile must contain 13238 values; found {profile.size}"
            )
        if not np.isfinite(profile).all() or np.any(profile < 0):
            raise ValueError("transcriptome_profile must contain finite, non-negative values")
        if not np.isfinite(rna_count) or rna_count < 0:
            raise ValueError("rna_count must be finite and non-negative")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.model = model.to(device).eval()
        self.design_region = region
        self.utr5, self.cds, self.utr3 = fixed["utr5"], fixed["cds"], fixed["utr3"]
        self.transcriptome_profile = np.log1p(profile * 5).astype(np.float32)
        self.rna_count = float(rna_count)
        self.device = device
        self.batch_size = int(batch_size)

    @classmethod
    def from_checkpoint(
        cls,
        *,
        design_region: str,
        utr5: str | None = None,
        cds: str | None = None,
        utr3: str | None = None,
        cell_context: str = "HEK293T",
        transcriptome_profile=None,
        rna_count: float = 6.3,
        checkpoint: str | Path | None = None,
        weights_dir: str | Path | None = None,
        **kwargs,
    ) -> "UTRCode":
        if transcriptome_profile is None:
            if str(cell_context).strip().upper() != "HEK293T":
                raise ValueError(
                    "only the bundled HEK293T context is named; provide "
                    "transcriptome_profile for another context"
                )
            profile_path = weight_path("utr_code/HEK293T.txt", weights_dir)
            transcriptome_profile = np.loadtxt(profile_path, dtype=np.float32)
        path = Path(checkpoint) if checkpoint else weight_path(
            "utr_code/best.epoch", weights_dir
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = UTRCodeNetwork()
        model.load_state_dict(state["model_state_dict"], strict=True)
        return cls(
            model,
            design_region=design_region,
            utr5=utr5,
            cds=cds,
            utr3=utr3,
            transcriptome_profile=transcriptome_profile,
            rna_count=rna_count,
            **kwargs,
        )

    @staticmethod
    def _regulatory(utr5: str, utr3: str) -> str:
        return utr5.rjust(1000, "N") + utr3.ljust(4000, "N")

    def score(self, sequences) -> np.ndarray:
        values = normalize_sequences(sequences)
        if not len(values):
            return np.empty((0, 1), dtype=np.float64)
        limit = REGION_LIMITS[self.design_region]
        for row, sequence in enumerate(values):
            if len(sequence) > limit:
                raise ValueError(
                    f"candidate at row {row} has length {len(sequence)}, "
                    f"exceeding {limit} nt for {self.design_region}"
                )

        predictions = []
        with torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                batch = values[start : start + self.batch_size]
                utr5s = batch if self.design_region == "utr5" else [self.utr5] * len(batch)
                cdss = batch if self.design_region == "cds" else [self.cds] * len(batch)
                utr3s = batch if self.design_region == "utr3" else [self.utr3] * len(batch)
                regulatory = [
                    self._regulatory(str(left), str(right))
                    for left, right in zip(utr5s, utr3s)
                ]
                cds_inputs = [str(value).ljust(4500, "N") for value in cdss]
                count = len(batch)
                output = self.model(
                    encode_acgt_sequences(regulatory, exact_length=5000).to(self.device),
                    encode_acgt_sequences(cds_inputs, exact_length=4500).to(self.device),
                    torch.from_numpy(self.transcriptome_profile)
                    .to(self.device)
                    .reshape(1, -1)
                    .expand(count, -1),
                    torch.full(
                        (count, 1), self.rna_count, dtype=torch.float32, device=self.device
                    ),
                )[:, :1]
                predictions.append(
                    _prediction_matrix(output.cpu().numpy(), count, 1)
                )
        return np.concatenate(predictions)
