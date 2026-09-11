"""Adapter for the WI-38 senescent heat-shock DeepAdapt model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..architectures import DeepAdaptNetwork
from ..inference import _prediction_matrix, normalize_sequences
from ..resources import weight_path
from .common import encode_acgt_sequences

REGION_LIMITS = {"utr5": 1000, "cds": 4500, "utr3": 4000}
TASK_NAMES = ("delta_log2_te", "normal_log2_te", "heatshock_log2_te")
CHECKPOINT_FORMAT = "bioseq_models.deepadapt_heatshock.v1"
CONDITION = "WI38_Senescent_HeatShock_GSE124609"


def _region_name(value: str) -> str:
    region = str(value).strip().lower().replace("′", "").replace("'", "")
    region = {"5utr": "utr5", "3utr": "utr3"}.get(region, region)
    if region not in REGION_LIMITS:
        raise ValueError("design_region must be 'utr5', 'cds', or 'utr3'")
    return region


def _region_values(values, name: str, *, allow_empty: bool) -> np.ndarray:
    sequences = np.asarray(values, dtype=str).reshape(-1)
    sequences = np.char.replace(np.char.upper(sequences), "U", "T")
    maximum = REGION_LIMITS[name]
    for row, sequence in enumerate(sequences):
        invalid = sorted(set(sequence) - set("ACGTN"))
        if invalid:
            raise ValueError(
                f"invalid nucleotide symbols in {name} at row {row}: "
                f"{', '.join(invalid)}"
            )
        if not sequence and not allow_empty:
            raise ValueError(f"{name} at row {row} is empty")
        if len(sequence) > maximum:
            raise ValueError(
                f"{name} at row {row} has length {len(sequence)}, "
                f"exceeding {maximum} nt"
            )
        if name == "cds" and sequence:
            if "N" in sequence:
                raise ValueError(f"cds at row {row} contains N")
            if len(sequence) % 3:
                raise ValueError(f"cds at row {row} length must be divisible by 3")
            if not sequence.startswith("ATG"):
                raise ValueError(f"cds at row {row} must start with ATG")
            if sequence[-3:] not in {"TAA", "TAG", "TGA"}:
                raise ValueError(f"cds at row {row} must end with TAA, TAG, or TGA")
    return sequences


def _overlap_sequences(
    utr5_values: np.ndarray,
    cds_values: np.ndarray,
    utr3_values: np.ndarray,
) -> tuple[list[str], list[str], list[str]]:
    utr5_towers = []
    cds_towers = []
    utr3_towers = []
    for utr5, cds, utr3 in zip(utr5_values, cds_values, utr3_values):
        utr5_towers.append(utr5[-1000:].rjust(1000, "N") + cds[:30].ljust(30, "N"))
        cds_towers.append(
            (utr5[-30:].rjust(30, "N") + cds + utr3[:30]).ljust(4560, "N")
        )
        utr3_towers.append(cds[-30:].rjust(30, "N") + utr3.ljust(4000, "N"))
    return utr5_towers, cds_towers, utr3_towers


class DeepAdaptHeatShock:
    """Predict condition-specific translation efficiency for transcript designs."""

    task_names = TASK_NAMES

    def __init__(
        self,
        model: DeepAdaptNetwork,
        *,
        design_region: str,
        utr5: str | None,
        cds: str | None,
        utr3: str | None,
        environment_normal: torch.Tensor,
        environment_treatment: torch.Tensor,
        species_id: int,
        device="cpu",
        batch_size: int = 64,
    ) -> None:
        region = _region_name(design_region)
        fixed = {"utr5": utr5, "cds": cds, "utr3": utr3}
        if fixed[region] is not None:
            raise ValueError(f"do not supply {region}; it is provided to score()")
        for name, value in fixed.items():
            if name == region:
                continue
            if value is None:
                raise ValueError(f"{name} is required when design_region={region!r}")
            fixed[name] = str(
                _region_values([value], name, allow_empty=name != "cds")[0]
            )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.model = model.to(device).eval()
        self.design_region = region
        self.utr5, self.cds, self.utr3 = fixed["utr5"], fixed["cds"], fixed["utr3"]
        self.environment_normal = environment_normal.to(device)
        self.environment_treatment = environment_treatment.to(device)
        self.species_id = int(species_id)
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
        checkpoint: str | Path | None = None,
        weights_dir: str | Path | None = None,
        **kwargs,
    ) -> DeepAdaptHeatShock:
        path = (
            Path(checkpoint)
            if checkpoint
            else weight_path("deepadapt_heatshock/model.pt", weights_dir)
        )
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported DeepAdapt checkpoint format in {path}")
        if artifact.get("condition") != CONDITION:
            raise ValueError(f"unexpected DeepAdapt condition in {path}")
        model = DeepAdaptNetwork(
            artifact["model_config"],
            artifact["num_celltype_species"],
            total_species=len(artifact["num_celltype_species"]),
        )
        model.load_state_dict(artifact["model_state_dict"], strict=True)
        return cls(
            model,
            design_region=design_region,
            utr5=utr5,
            cds=cds,
            utr3=utr3,
            environment_normal=artifact["environment_normal"],
            environment_treatment=artifact["environment_treatment"],
            species_id=artifact["species_id"],
            **kwargs,
        )

    def _predict_batch(
        self,
        utr5_values: np.ndarray,
        cds_values: np.ndarray,
        utr3_values: np.ndarray,
    ) -> np.ndarray:
        utr5_towers, cds_towers, utr3_towers = _overlap_sequences(
            utr5_values, cds_values, utr3_values
        )
        utr5_tensor = encode_acgt_sequences(utr5_towers, exact_length=1030).to(
            self.device
        )
        cds_tensor = encode_acgt_sequences(cds_towers, exact_length=4560).to(
            self.device
        )
        utr3_tensor = encode_acgt_sequences(utr3_towers, exact_length=4030).to(
            self.device
        )
        sequence_features = self.model.encode_sequences(
            utr5_tensor, utr3_tensor, cds_tensor
        )

        environments = torch.cat(
            [self.environment_normal, self.environment_treatment], dim=0
        )
        sequence_count = len(utr5_values)
        sample_count = environments.shape[0]
        flat_environments = environments.unsqueeze(0).expand(sequence_count, -1, -1, -1)
        flat_environments = flat_environments.reshape(
            sequence_count * sample_count, environments.shape[1], 1
        )
        species = torch.full(
            (sequence_count * sample_count, 1),
            self.species_id,
            dtype=torch.long,
            device=self.device,
        )
        output = self.model.forward_from_sequence_features(
            sequence_features.repeat_interleave(sample_count, dim=0),
            flat_environments,
            species,
        )
        output = output.clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        log2_te = 2.0 * torch.log2(torch.tan(torch.arccos(-output) / 2.0))
        log2_te = log2_te.reshape(sequence_count, sample_count)
        normal_count = self.environment_normal.shape[0]
        normal = log2_te[:, :normal_count].mean(dim=1)
        treatment = log2_te[:, normal_count:].mean(dim=1)
        return torch.stack([treatment - normal, normal, treatment], dim=1).cpu().numpy()

    def score_transcripts(self, utr5, cds, utr3) -> np.ndarray:
        """Score complete transcript-region triples."""
        utr5_values = _region_values(utr5, "utr5", allow_empty=True)
        cds_values = _region_values(cds, "cds", allow_empty=False)
        utr3_values = _region_values(utr3, "utr3", allow_empty=True)
        if not (len(utr5_values) == len(cds_values) == len(utr3_values)):
            raise ValueError("utr5, cds, and utr3 must contain the same number of rows")
        if not len(cds_values):
            return np.empty((0, len(self.task_names)), dtype=np.float64)

        predictions = []
        with torch.inference_mode():
            for start in range(0, len(cds_values), self.batch_size):
                stop = start + self.batch_size
                output = self._predict_batch(
                    utr5_values[start:stop],
                    cds_values[start:stop],
                    utr3_values[start:stop],
                )
                predictions.append(
                    _prediction_matrix(output, len(cds_values[start:stop]), 3)
                )
        return np.concatenate(predictions, axis=0)

    def score(self, sequences) -> np.ndarray:
        values = normalize_sequences(sequences)
        count = len(values)
        if not count:
            return np.empty((0, len(self.task_names)), dtype=np.float64)
        fixed = {"utr5": self.utr5, "cds": self.cds, "utr3": self.utr3}
        regions = {
            name: values if name == self.design_region else [fixed[name]] * count
            for name in REGION_LIMITS
        }
        return self.score_transcripts(regions["utr5"], regions["cds"], regions["utr3"])
