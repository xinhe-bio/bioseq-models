#!/usr/bin/env python3
"""Convert an official tissue-specific DeepSTARR Keras checkpoint to PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from bioseq_models.architectures import (
    DeepSTARREmbryoNetwork,
    DeepSTARRMouseNetwork,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flatten_layers(model_config: dict[str, Any]) -> list[dict[str, Any]]:
    layers = []
    for layer in model_config.get("config", {}).get("layers", []):
        if layer.get("class_name") in {"Functional", "Model", "Sequential"}:
            layers.extend(_flatten_layers(layer))
        else:
            layers.append(layer)
    return layers


def _layer_names(config: dict[str, Any], class_name: str) -> list[str]:
    return [
        layer["config"]["name"]
        for layer in _flatten_layers(config)
        if layer.get("class_name") == class_name
    ]


def _infer_output_kind(config: dict[str, Any]) -> str:
    activations = [
        layer.get("config", {}).get("activation")
        for layer in _flatten_layers(config)
        if layer.get("class_name") == "Activation"
    ]
    return (
        "activity" if activations and activations[-1] == "sigmoid" else "accessibility"
    )


def _datasets(path: Path) -> dict[tuple[str, str], np.ndarray]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "checkpoint conversion requires h5py; install bioseq-models[conversion]"
        ) from exc

    values = {}
    with h5py.File(path, "r") as handle:

        def collect(name, item) -> None:
            if not isinstance(item, h5py.Dataset):
                return
            parts = name.split("/")
            if len(parts) < 2 or ":" not in parts[-1]:
                return
            key = (parts[-2], parts[-1].split(":", maxsplit=1)[0])
            if key in values:
                raise ValueError(f"ambiguous Keras tensor {key!r} in {path}")
            values[key] = np.asarray(item)

        handle.visititems(collect)
    return values


def _tensor(values, layer: str, parameter: str) -> torch.Tensor:
    key = (layer, parameter)
    if key not in values:
        raise ValueError(f"Keras checkpoint is missing {layer}/{parameter}:0")
    return torch.from_numpy(values[key].copy())


def convert_deepstarr_checkpoint(
    architecture: str,
    keras_json: str | Path,
    keras_weights: str | Path,
    output: str | Path,
) -> Path:
    """Convert one official JSON/HDF5 model pair and return its output path."""
    architecture = str(architecture).strip().lower()
    if architecture not in {"mouse", "embryo"}:
        raise ValueError("architecture must be 'mouse' or 'embryo'")
    keras_json = Path(keras_json)
    keras_weights = Path(keras_weights)
    output = Path(output)
    config = json.loads(keras_json.read_text())
    output_kind = _infer_output_kind(config)
    model_class = {
        "mouse": DeepSTARRMouseNetwork,
        "embryo": DeepSTARREmbryoNetwork,
    }[architecture]
    model = model_class(output_kind=output_kind)
    values = _datasets(keras_weights)

    conv_names = _layer_names(config, "Conv1D")
    batch_norm_names = _layer_names(config, "BatchNormalization")
    dense_names = _layer_names(config, "Dense")
    if (len(conv_names), len(batch_norm_names), len(dense_names)) != (4, 6, 3):
        raise ValueError(
            "unexpected DeepSTARR architecture: expected 4 Conv1D, "
            f"6 BatchNormalization and 3 Dense layers, got "
            f"{len(conv_names)}, {len(batch_norm_names)}, {len(dense_names)}"
        )

    state = model.state_dict()
    for index, layer_name in enumerate(conv_names):
        state[f"convolutions.{index}.weight"] = _tensor(
            values, layer_name, "kernel"
        ).permute(2, 1, 0)
        state[f"convolutions.{index}.bias"] = _tensor(values, layer_name, "bias")

    pytorch_batch_norms = [
        *(f"conv_batch_norms.{index}" for index in range(4)),
        "dense_1_batch_norm",
        "dense_2_batch_norm",
    ]
    parameter_map = {
        "gamma": "weight",
        "beta": "bias",
        "moving_mean": "running_mean",
        "moving_variance": "running_var",
    }
    for pytorch_name, keras_name in zip(pytorch_batch_norms, batch_norm_names):
        for keras_parameter, pytorch_parameter in parameter_map.items():
            state[f"{pytorch_name}.{pytorch_parameter}"] = _tensor(
                values, keras_name, keras_parameter
            )

    for pytorch_name, keras_name in zip(("dense_1", "dense_2", "output"), dense_names):
        state[f"{pytorch_name}.weight"] = _tensor(
            values, keras_name, "kernel"
        ).transpose(0, 1)
        state[f"{pytorch_name}.bias"] = _tensor(values, keras_name, "bias")

    model.load_state_dict(state, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": architecture,
            "output_kind": output_kind,
            "source_json_sha256": _sha256(keras_json),
            "source_weights_sha256": _sha256(keras_weights),
        },
        output,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("architecture", choices=("mouse", "embryo"))
    parser.add_argument("keras_json", type=Path)
    parser.add_argument("keras_weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = convert_deepstarr_checkpoint(
        args.architecture,
        args.keras_json,
        args.keras_weights,
        args.output,
    )
    print(output)


if __name__ == "__main__":
    main()
