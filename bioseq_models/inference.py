"""Small, dependency-local inference interfaces used by bundled models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def normalize_sequences(sequences: Sequence[str] | np.ndarray) -> np.ndarray:
    """Return uppercase DNA notation, converting RNA U to T."""
    values = np.char.replace(
        np.char.upper(np.asarray(sequences, dtype=str).reshape(-1)),
        "U",
        "T",
    )
    for row, sequence in enumerate(values):
        invalid = sorted(set(sequence) - set("ACGTN"))
        if invalid:
            raise ValueError(
                f"invalid nucleotide symbols at row {row}: {', '.join(invalid)}"
            )
        if not sequence:
            raise ValueError(f"sequence at row {row} is empty")
    return values


def _prepare_sequences(
    sequences: Sequence[str] | np.ndarray,
    pad_to_length: int | None,
    padding_side: str,
) -> np.ndarray:
    values = normalize_sequences(sequences)
    if pad_to_length is None:
        return values
    lengths = np.char.str_len(values)
    if np.any(lengths > pad_to_length):
        row = int(np.flatnonzero(lengths > pad_to_length)[0])
        raise ValueError(
            f"sequence at row {row} has length {int(lengths[row])}, "
            f"exceeding model input length {pad_to_length}"
        )
    if padding_side == "left":
        return np.asarray(
            [sequence.rjust(pad_to_length, "N") for sequence in values],
            dtype=str,
        )
    return np.asarray(
        [sequence.ljust(pad_to_length, "N") for sequence in values],
        dtype=str,
    )


def _prediction_matrix(values: Any, rows: int, tasks: int) -> np.ndarray:
    prediction = np.asarray(values, dtype=np.float64)
    if prediction.ndim == 0:
        prediction = prediction.reshape(1, 1)
    elif prediction.ndim == 1:
        prediction = prediction[:, None]
    if prediction.ndim != 2 or prediction.shape != (rows, tasks):
        raise ValueError(
            f"model returned shape {prediction.shape}; expected {(rows, tasks)}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError("model returned non-finite predictions")
    return prediction


class SequenceModel:
    """Common metadata and input handling for sequence prediction models."""

    def __init__(
        self,
        *,
        task_names: Sequence[str],
        pad_to_length: int | None = None,
        padding_side: str = "right",
    ) -> None:
        names = tuple(str(name).strip() for name in task_names)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("task_names must contain unique, non-empty names")
        if pad_to_length is not None and pad_to_length < 1:
            raise ValueError("pad_to_length must be positive or None")
        side = str(padding_side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'")
        self.task_names = names
        self.pad_to_length = pad_to_length
        self.padding_side = side

    def _prepare(self, sequences) -> np.ndarray:
        return _prepare_sequences(
            sequences,
            self.pad_to_length,
            self.padding_side,
        )


class TorchSequenceModel(SequenceModel):
    """Run a frozen PyTorch model on batches of encoded nucleotide strings."""

    def __init__(
        self,
        model,
        encoder: Callable[[np.ndarray], Any],
        *,
        device: Any = "cpu",
        batch_size: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.encoder = encoder
        self.device = device
        self.batch_size = int(batch_size)
        if device is not None and hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "eval"):
            model.eval()

    def score(self, sequences) -> np.ndarray:
        import torch

        values = self._prepare(sequences)
        if not len(values):
            return np.empty((0, len(self.task_names)), dtype=np.float64)
        predictions = []
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                batch = values[start : start + self.batch_size]
                inputs = self.encoder(batch)
                if self.device is not None and hasattr(inputs, "to"):
                    inputs = inputs.to(self.device)
                output = self.model(inputs)
                if isinstance(output, tuple):
                    output = output[0]
                if hasattr(output, "detach"):
                    output = output.detach().cpu().numpy()
                predictions.append(
                    _prediction_matrix(
                        output,
                        len(batch),
                        len(self.task_names),
                    )
                )
        return np.concatenate(predictions, axis=0)
