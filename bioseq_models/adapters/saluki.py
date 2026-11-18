"""Annotation-aware Saluki mRNA half-life prediction, using official weights."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..architectures.saluki import SalukiEnsemble, SalukiNetwork
from ..inference import TorchSequenceModel, normalize_sequences, _prediction_matrix
from ..saluki_weights import checkpoint_path, convert_h5

SEQ_LENGTH = 12288


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def annotation_tracks(
    length, *, cds_start, cds_end, exon_lengths=None, splice_sites=None
):
    """Return full-length frame/exon-end channels in the supplied RNA orientation.

    CDS coordinates are zero-based and end-exclusive. As in the released data
    encoder, exon_lengths marks the last base of every exon, including the final
    exon. splice_sites instead supplies raw zero-based channel positions.
    """
    if cds_start is None or cds_end is None:
        raise ValueError(
            "Saluki requires CDS annotations. Supply cds_start and cds_end, or use score_transcripts with annotated records."
        )
    start = _integer(cds_start, "cds_start")
    end = _integer(cds_end, "cds_end")
    if not 0 <= start < end <= length or (end - start) % 3:
        raise ValueError(
            "CDS must lie within the full transcript and contain a whole number of codons"
        )
    if exon_lengths is not None and splice_sites is not None:
        raise ValueError("Supply exon_lengths or splice_sites, not both")
    if exon_lengths is None and splice_sites is None:
        raise ValueError(
            "Saluki requires exon annotations: use exon_lengths=[transcript_length] for a single-exon transcript, or explicit splice_sites."
        )
    tracks = np.zeros((length, 2), dtype=np.float32)
    tracks[start:end:3, 0] = 1
    if exon_lengths is not None:
        lengths = [_integer(v, "exon length") for v in exon_lengths]
        if not lengths or any(v <= 0 for v in lengths) or sum(lengths) != length:
            raise ValueError(
                "Positive exon_lengths in transcript order must sum to the full transcript length"
            )
        positions = np.cumsum(lengths) - 1
    else:
        positions = [_integer(v, "splice position") for v in splice_sites]
        if len(set(positions)) != len(positions) or any(
            v < 0 or v >= length for v in positions
        ):
            raise ValueError(
                "splice_sites must be unique zero-based positions within the full transcript"
            )
    tracks[positions, 1] = 1
    return tracks


def encode_transcript(
    sequence, *, cds_start, cds_end, exon_lengths=None, splice_sites=None
):
    """Encode one complete RNA; right-pad zeros or retain the last 12,288 nt."""
    values = normalize_sequences([sequence])
    sequence = values[0]
    tracks = annotation_tracks(
        len(sequence),
        cds_start=cds_start,
        cds_end=cds_end,
        exon_lengths=exon_lengths,
        splice_sites=splice_sites,
    )
    x = np.zeros((len(sequence), 6), dtype=np.float32)
    for channel, base in enumerate("ACGT"):
        x[:, channel] = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8) == ord(
            base
        )
    x[:, 4:] = tracks
    if len(x) >= SEQ_LENGTH:
        return x[-SEQ_LENGTH:]
    return np.pad(x, ((0, SEQ_LENGTH - len(x)), (0, 0)))


class Saluki(TorchSequenceModel):
    """Human consensus half-life score from a CDS/exon-annotated RNA transcript.

    Default weights are the original human f0_c0 checkpoint. Select additional
    folds/crosses explicitly for an arithmetic-mean ensemble. Outputs are the
    paper's species-specific half-life score, with higher predicting stability.
    """

    species_default = "human"

    def __init__(
        self,
        model,
        *,
        cds_start=None,
        cds_end=None,
        exon_lengths=None,
        splice_sites=None,
        **kwargs,
    ):
        self.annotations = dict(
            cds_start=cds_start,
            cds_end=cds_end,
            exon_lengths=None if exon_lengths is None else tuple(exon_lengths),
            splice_sites=None if splice_sites is None else tuple(splice_sites),
        )
        fixed = (
            cds_start is not None
            and cds_end is not None
            and (exon_lengths is not None or splice_sites is not None)
        )
        kwargs.setdefault("task_names", ("half_life_score",))
        kwargs.setdefault("batch_size", 16)
        if kwargs.get("pad_to_length") is not None:
            raise ValueError(
                "Saluki performs its own annotation-aware 3prime cropping/right padding; omit pad_to_length"
            )
        super().__init__(
            model,
            self._encode,
            soft_encoder=self._encode_soft if fixed else None,
            **kwargs,
        )
        self.species = self.species_default
        self.output_units = (
            "Consensus half-life PC1 score; higher predicts greater stability"
        )
        self.reverse_complement = False

    def _encode(self, sequences):
        return torch.from_numpy(
            np.stack([encode_transcript(s, **self.annotations) for s in sequences])
        ).transpose(1, 2)

    def _encode_soft(self, probabilities):
        if (
            not probabilities.is_floating_point()
            or not torch.isfinite(probabilities).all()
        ):
            raise ValueError(
                "Saluki soft inputs must be finite floating-point probabilities"
            )
        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("Saluki soft nucleotide probabilities must lie in [0,1]")
        sums = probabilities.sum(-1)
        if not torch.all(
            torch.isclose(sums, torch.ones_like(sums), atol=1e-5)
            | torch.isclose(sums, torch.zeros_like(sums), atol=1e-5)
        ):
            raise ValueError("Each position must sum to one, or zero for an unknown N")
        tracks = annotation_tracks(probabilities.shape[1], **self.annotations)
        extra = torch.as_tensor(
            tracks, device=probabilities.device, dtype=probabilities.dtype
        )[None].expand(len(probabilities), -1, -1)
        x = torch.cat([probabilities, extra], dim=-1)[:, -SEQ_LENGTH:]
        x = F.pad(x.transpose(1, 2), (0, max(0, SEQ_LENGTH - x.shape[1])))
        return x.float()

    def score_soft(self, probabilities):
        # cuDNN RNN inference does not retain the state needed for input gradients.
        with torch.backends.cudnn.flags(enabled=False):
            return super().score_soft(probabilities)

    def score_encoded(self, inputs):
        """Score pre-encoded [batch,length,6] arrays, including official TFRecords."""
        x = np.asarray(inputs, dtype=np.float32)
        if x.ndim != 3 or x.shape[2] != 6 or not x.shape[1]:
            raise ValueError("Encoded Saluki inputs must have shape [batch,length,6]")
        if not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
            raise ValueError("Encoded Saluki channels must be finite values in [0,1]")
        if not len(x):
            return np.empty((0, 1), dtype=np.float64)
        x = x[:, -SEQ_LENGTH:]
        if x.shape[1] < SEQ_LENGTH:
            x = np.pad(x, ((0, 0), (0, SEQ_LENGTH - x.shape[1]), (0, 0)))
        predictions = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                batch = (
                    torch.from_numpy(x[start : start + self.batch_size])
                    .transpose(1, 2)
                    .to(self.device)
                )
                values = self.model(batch).detach().cpu().numpy()
                predictions.append(_prediction_matrix(values, len(batch), 1))
        return np.concatenate(predictions)

    def score_transcripts(self, transcripts):
        """Score mappings with sequence, cds_start/end and exon_lengths/splice_sites."""
        records = list(transcripts)
        if not records:
            return np.empty((0, 1), dtype=np.float64)
        predictions = []
        for start in range(0, len(records), self.batch_size):
            rows = []
            for record in records[start : start + self.batch_size]:
                if not isinstance(record, Mapping):
                    raise TypeError(
                        "score_transcripts expects annotated transcript mappings"
                    )
                if not {"sequence", "cds_start", "cds_end"} <= record.keys():
                    raise ValueError(
                        "Each transcript requires sequence, cds_start and cds_end"
                    )
                rows.append(
                    encode_transcript(
                        record["sequence"],
                        cds_start=record["cds_start"],
                        cds_end=record["cds_end"],
                        exon_lengths=record.get("exon_lengths"),
                        splice_sites=record.get("splice_sites"),
                    )
                )
            predictions.append(self.score_encoded(np.stack(rows)))
        return np.concatenate(predictions)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path | None = None,
        weights_dir=None,
        folds=(0,),
        crosses=(0,),
        cache_dir=None,
        **kwargs,
    ):
        paths = []
        if checkpoint is not None:
            path = Path(checkpoint)
            paths = (
                sorted(path.glob(f"saluki_{cls.species_default}_f*_c*.pt"))
                if path.is_dir()
                else [path]
            )
            if not paths:
                raise FileNotFoundError(f"No converted Saluki checkpoints in {path}")
        else:
            folds = tuple(_integer(v, "fold") for v in folds)
            crosses = tuple(_integer(v, "cross") for v in crosses)
            if (
                not folds
                or not crosses
                or len(set(folds)) != len(folds)
                or len(set(crosses)) != len(crosses)
                or any(not 0 <= f < 10 for f in folds)
                or any(not 0 <= c < 5 for c in crosses)
            ):
                raise ValueError("Select unique folds in 0..9 and crosses in 0..4")
            paths = [
                checkpoint_path(
                    cls.species_default,
                    f,
                    c,
                    weights_dir=weights_dir,
                    cache_dir=cache_dir,
                )
                for f in folds
                for c in crosses
            ]
        models = []
        for path in paths:
            if path.suffix.lower() in {".h5", ".hdf5"}:
                model, _ = convert_h5(path)
            else:
                state = torch.load(path, map_location="cpu", weights_only=True)
                if (
                    state.get("architecture") != "saluki_official"
                    or state.get("species") != cls.species_default
                ):
                    raise ValueError(
                        f"Expected a converted official {cls.species_default} Saluki checkpoint: {path}"
                    )
                model = SalukiNetwork(**state["model_config"])
                model.load_state_dict(state["model_state"], strict=True)
            models.append(model)
        scorer = cls(
            models[0] if len(models) == 1 else SalukiEnsemble(models), **kwargs
        )
        scorer.checkpoint_paths = tuple(paths)
        scorer.ensemble_size = len(models)
        return scorer


class SalukiMouse(Saluki):
    """Mouse consensus half-life score, using the released mouse head weights."""

    species_default = "mouse"
