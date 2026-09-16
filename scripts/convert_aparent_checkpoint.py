#!/usr/bin/env python3
"""Convert the official APARENT Keras HDF5 model to a PyTorch checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from bioseq_models.architectures import APARENTNetwork  # noqa: E402

SOURCE_REVISION = "69ad29791709b48689ff5d9e3a3daefc568de9ce"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_source(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "APARENT conversion requires h5py; install bioseq-models[conversion]"
        ) from exc

    with h5py.File(path, "r") as handle:
        config_value = handle.attrs.get("model_config")
        if config_value is None:
            raise ValueError("source checkpoint has no Keras model_config")
        if isinstance(config_value, bytes):
            config_value = config_value.decode()
        config = json.loads(config_value)
        values: dict[str, np.ndarray] = {}

        def collect(name, item) -> None:
            if isinstance(item, h5py.Dataset):
                values[name] = np.asarray(item)

        handle.visititems(collect)
    return config, values


def _validate_config(config: dict) -> None:
    layers = config.get("config", {}).get("layers", [])
    signatures = [
        (
            layer.get("config", {}).get("name"),
            layer.get("class_name"),
            layer.get("config", {}).get("units"),
            layer.get("config", {}).get("filters"),
        )
        for layer in layers
    ]
    expected = {
        ("conv2d_1", "Conv2D", None, 96),
        ("conv2d_2", "Conv2D", None, 128),
        ("dense_1", "Dense", 512, None),
        ("dense_2", "Dense", 256, None),
        ("dense_3", "Dense", 206, None),
        ("dense_4", "Dense", 1, None),
    }
    missing = expected.difference(signatures)
    if missing:
        raise ValueError(f"unexpected APARENT architecture; missing layers: {missing}")
    input_shapes = {
        tuple(layer.get("config", {}).get("batch_input_shape", ())[1:])
        for layer in layers
        if layer.get("class_name") == "InputLayer"
    }
    if input_shapes != {(205, 4, 1), (13,), (1,)}:
        raise ValueError(f"unexpected APARENT input shapes: {input_shapes}")


def _array(values: dict[str, np.ndarray], layer: str, parameter: str) -> np.ndarray:
    suffix = f"/{layer}/{parameter}:0"
    matches = [value for name, value in values.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"expected one Keras tensor ending in {suffix!r}; found {len(matches)}"
        )
    return matches[0]


def _tensor(values, layer: str, parameter: str) -> torch.Tensor:
    return torch.from_numpy(_array(values, layer, parameter).copy())


def _keras_reference(values, inputs: np.ndarray) -> np.ndarray:
    """Evaluate the source channels-last graph independently with NumPy."""
    kernel_1 = _array(values, "conv2d_1", "kernel")[:, :, 0, :]
    hidden = np.stack(
        [
            np.tensordot(inputs[:, start : start + 8], kernel_1, axes=([1, 2], [0, 1]))
            for start in range(198)
        ],
        axis=1,
    ) + _array(values, "conv2d_1", "bias")
    hidden = np.maximum(hidden, 0)
    hidden = np.maximum(hidden[:, 0::2], hidden[:, 1::2])

    kernel_2 = _array(values, "conv2d_2", "kernel")[:, 0, :, :]
    hidden = np.stack(
        [
            np.tensordot(hidden[:, start : start + 6], kernel_2, axes=([1, 2], [0, 1]))
            for start in range(94)
        ],
        axis=1,
    ) + _array(values, "conv2d_2", "bias")
    hidden = np.maximum(hidden, 0).reshape(len(inputs), -1)
    hidden = np.concatenate(
        [hidden, np.ones((len(inputs), 1), dtype=np.float32)], axis=1
    )
    hidden = np.maximum(
        hidden @ _array(values, "dense_1", "kernel")
        + _array(values, "dense_1", "bias"),
        0,
    )
    hidden = np.maximum(
        hidden @ _array(values, "dense_2", "kernel")
        + _array(values, "dense_2", "bias"),
        0,
    )
    hidden = np.concatenate(
        [hidden, np.zeros((len(inputs), 13), dtype=np.float32)], axis=1
    )
    logits = hidden @ _array(values, "dense_3", "kernel") + _array(
        values, "dense_3", "bias"
    )
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def convert_aparent_checkpoint(source: str | Path, output: str | Path) -> Path:
    """Convert one official APARENT HDF5 file and return the output path."""
    source, output = Path(source), Path(output)
    config, values = _read_source(source)
    _validate_config(config)
    model = APARENTNetwork()
    state = model.state_dict()
    state["conv_1.weight"] = _tensor(values, "conv2d_1", "kernel")[:, :, 0, :].permute(
        2, 1, 0
    )
    state["conv_1.bias"] = _tensor(values, "conv2d_1", "bias")
    state["conv_2.weight"] = _tensor(values, "conv2d_2", "kernel")[:, 0, :, :].permute(
        2, 1, 0
    )
    state["conv_2.bias"] = _tensor(values, "conv2d_2", "bias")
    dense_layers = {
        "dense_1": "dense_1",
        "dense_2": "dense_2",
        "cleavage_head": "dense_3",
        "isoform_head": "dense_4",
    }
    for pytorch_name, keras_name in dense_layers.items():
        state[f"{pytorch_name}.weight"] = _tensor(
            values, keras_name, "kernel"
        ).transpose(0, 1)
        state[f"{pytorch_name}.bias"] = _tensor(values, keras_name, "bias")
    nonfinite = [
        name for name, value in state.items() if not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise ValueError(f"source checkpoint contains non-finite tensors: {nonfinite}")
    model.load_state_dict(state, strict=True)
    model.eval()

    indices = np.arange(2 * 205)[:, None]
    bases = np.arange(4)[None, :]
    encoded = (indices % 4 == bases).astype(np.float32).reshape(2, 205, 4)
    reference = _keras_reference(values, encoded)
    with torch.no_grad():
        converted = model(torch.from_numpy(encoded).transpose(1, 2)).numpy()
    max_error = float(np.max(np.abs(reference - converted)))
    if max_error > 2e-6:
        raise ValueError(
            f"converted APARENT differs from Keras graph by {max_error:.3g}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": "aparent_site_probabilities",
            "source_repository": "https://github.com/johli/aparent",
            "source_revision": SOURCE_REVISION,
            "source_sha256": _sha256(source),
            "conversion_max_abs_error": max_error,
        },
        output,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_h5", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(convert_aparent_checkpoint(args.source_h5, args.output))


if __name__ == "__main__":
    main()
