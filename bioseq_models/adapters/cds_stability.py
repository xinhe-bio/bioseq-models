"""Multi-species CDS stability model with an internal CodonBERT encoder."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch

from ..architectures.cds_stability import (
    MAX_BASES,
    CDSStabilityNetwork,
    encode_cds_sequences,
)
from ..inference import _prediction_matrix, normalize_sequences
from ..resources import weight_path

SPECIES = {"human": 1, "mouse": 2, "zebrafish": 3, "fruitfly": 4, "yeast": 5}
CODONBERT_BASES = 3066
_BERT_VOCAB = {
    "".join(codon): index + 5
    for index, codon in enumerate(itertools.product("AUGC", repeat=3))
}


def _require_nonempty_file(path: str | Path, description: str) -> Path:
    """Return *path* only when it resolves to a non-empty regular file."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    if resolved.stat().st_size == 0:
        raise ValueError(f"{description} is empty: {resolved}")
    return resolved


class CDSStability:
    """Predict a normalized stability score from complete CDS candidates."""

    task_names = ("stability_score",)

    def __init__(
        self,
        model,
        codonbert,
        *,
        species: str,
        device="cpu",
        batch_size: int = 4,
    ) -> None:
        key = str(species).strip().lower().replace(" ", "")
        key = {"fruit_fly": "fruitfly", "drosophila": "fruitfly"}.get(key, key)
        if key not in SPECIES:
            raise ValueError(f"species must be one of {', '.join(SPECIES)}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model = model.to(device).eval()
        self.codonbert = codonbert.to(device).eval()
        self.species = key
        self.condition = SPECIES[key]
        self.device = device
        self.batch_size = int(batch_size)

    @classmethod
    def from_checkpoint(
        cls,
        *,
        species: str = "human",
        checkpoint: str | Path | None = None,
        codonbert_model_dir: str | Path | None = None,
        weights_dir: str | Path | None = None,
        **kwargs,
    ) -> CDSStability:
        try:
            from transformers import BertConfig, BertForPreTraining
        except ImportError as exc:
            raise ImportError(
                "CDS stability requires transformers; reinstall bioseq-models"
            ) from exc

        checkpoint_path = _require_nonempty_file(
            Path(checkpoint)
            if checkpoint
            else weight_path("cds_stability/best_model.pt", weights_dir),
            "CDS stability checkpoint",
        )
        if codonbert_model_dir:
            bert_path = Path(codonbert_model_dir)
            bert_config_path = bert_path / "config.json"
            bert_weights_path = bert_path / "pytorch_model.bin"
        else:
            # Resolve both files explicitly. MODEL_FILES is used by
            # download_model(), but load_model() delegates resource resolution
            # to each adapter and does not pre-download sibling files.
            bert_config_path = weight_path(
                "cds_stability/codonbert/config.json", weights_dir
            )
            bert_weights_path = weight_path(
                "cds_stability/codonbert/pytorch_model.bin", weights_dir
            )
            bert_path = bert_weights_path.parent
        bert_config_path = _require_nonempty_file(
            bert_config_path, "CodonBERT config"
        )
        bert_weights_path = _require_nonempty_file(
            bert_weights_path, "CodonBERT checkpoint"
        )
        if bert_config_path.parent != bert_weights_path.parent:
            raise ValueError(
                "CodonBERT config and checkpoint must be in the same directory: "
                f"{bert_config_path.parent} != {bert_weights_path.parent}"
            )
        model = CDSStabilityNetwork()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        # Parse the verified file directly so a missing config can never fall
        # back to BERT's default 30,522-token, 512-position configuration.
        codonbert_config = BertConfig.from_json_file(str(bert_config_path))
        codonbert = BertForPreTraining.from_pretrained(
            bert_path,
            config=codonbert_config,
            local_files_only=True,
        )
        return cls(model, codonbert, species=species, **kwargs)

    @staticmethod
    def _bert_inputs(sequences, device):
        rows = []
        for sequence in sequences:
            rna = sequence[:CODONBERT_BASES].replace("T", "U")
            codons = [rna[offset : offset + 3] for offset in range(0, len(rna), 3)]
            rows.append([2, *(_BERT_VOCAB[codon] for codon in codons), 3])
        width = max(map(len, rows))
        input_ids = torch.zeros((len(rows), width), dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        for row, values in enumerate(rows):
            input_ids[row, : len(values)] = torch.tensor(values, device=device)
            attention_mask[row, : len(values)] = 1
        return input_ids, attention_mask

    def score(self, sequences) -> np.ndarray:
        values = normalize_sequences(sequences)
        if not len(values):
            return np.empty((0, 1), dtype=np.float64)
        for row, sequence in enumerate(values):
            if "N" in sequence:
                raise ValueError(f"CDS candidate at row {row} contains N")
            if len(sequence) % 3:
                raise ValueError(
                    f"CDS candidate at row {row} has length {len(sequence)}; "
                    "length must be divisible by 3"
                )
            if len(sequence) > MAX_BASES:
                raise ValueError(
                    f"CDS candidate at row {row} has length {len(sequence)}, "
                    f"exceeding {MAX_BASES} nt"
                )

        predictions = []
        with torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                batch = values[start : start + self.batch_size]
                input_ids, attention_mask = self._bert_inputs(batch, self.device)
                hidden = self.codonbert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states[-1][:, 0]
                count = len(batch)
                condition = torch.full(
                    (count, 1),
                    float(self.condition),
                    dtype=torch.float32,
                    device=self.device,
                )
                output = self.model(
                    encode_cds_sequences(batch).to(self.device), condition, hidden
                )
                predictions.append(
                    _prediction_matrix(output.cpu().numpy(), count, 1)
                )
        return np.concatenate(predictions)
