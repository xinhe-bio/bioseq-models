#!/usr/bin/env python3
"""Convert the DeepAdapt senescent heat-shock package to one inference artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CONDITION = "WI38_Senescent_HeatShock_GSE124609"
ADAPTER_FILE = "WI38_Senescent_HeatShock_GSE124609__aeacbb18d1.p"
FORMAT = "bioseq_models.deepadapt_heatshock.v1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Extracted deepadapt_inference_package directory",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _state_without_module_prefix(state: dict[str, torch.Tensor]):
    if state and all(name.startswith("module.") for name in state):
        return {name.removeprefix("module."): value for name, value in state.items()}
    return state


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    config = json.loads((args.source / "code/model_config.json").read_text())
    base_path = args.source / "checkpoints/pretrain_epoch23.p"
    adapter_path = args.source / "weights/human/weights" / ADAPTER_FILE
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=True)
    if int(base.get("epoch", -1)) != 23:
        raise ValueError("DeepAdapt base checkpoint is not epoch 23")
    if adapter.get("condition") != CONDITION or adapter.get("species_id") != 0:
        raise ValueError("condition adapter metadata does not match the selected model")
    if adapter.get("base_epoch") != 23:
        raise ValueError("condition adapter uses an unexpected base checkpoint")

    state = _state_without_module_prefix(base["predictor_model_state_dict"])
    state = {name: value.detach().cpu() for name, value in state.items()}
    fine_tuned = adapter["finetuned_state_dict"]
    if set(fine_tuned) != set(adapter["trainable_parameter_names"]):
        raise ValueError("condition adapter tensor inventory is inconsistent")
    for name, value in fine_tuned.items():
        if name not in state or state[name].shape != value.shape:
            raise ValueError(f"condition tensor does not match the base model: {name}")
        state[name] = value.detach().cpu()

    dense4 = int(config[config["training"]["hyper_use"]]["dense4"])
    celltype_counts = []
    for species_id in range(2):
        key = f"te_celltype_aggregator.{species_id}.0.weight"
        if state[key].shape[0] % dense4:
            raise ValueError(f"cannot infer cell-type count from {key}")
        celltype_counts.append(state[key].shape[0] // dense4)

    conditions = json.loads(
        (args.source / "resources/human_condition.json").read_text()
    )
    samples = conditions[CONDITION]
    sample_ids = samples["normal"] + samples["treatment"]
    rna = pd.read_parquet(args.source / "resources/RNA_env.parquet", columns=sample_ids)
    values = rna.loc[:, sample_ids].to_numpy(dtype=np.float32, copy=True).T
    if values.shape != (len(sample_ids), 9474) or not np.isfinite(values).all():
        raise ValueError(f"unexpected RNA environment matrix: {values.shape}")
    environment = torch.from_numpy(np.log1p(values)[..., np.newaxis])
    normal_count = len(samples["normal"])

    artifact = {
        "format": FORMAT,
        "model_name": "deepadapt_heatshock",
        "condition": CONDITION,
        "species": "human",
        "species_id": 0,
        "base_epoch": 23,
        "model_config": config,
        "num_celltype_species": celltype_counts,
        "model_state_dict": state,
        "normal_sample_ids": samples["normal"],
        "treatment_sample_ids": samples["treatment"],
        "environment_normal": environment[:normal_count],
        "environment_treatment": environment[normal_count:],
        "adapter_checkpoint_format": adapter["checkpoint_format"],
        "adapter_best_epoch": int(adapter["best_epoch"]),
        "adapter_best_spearman": float(adapter["best_spearman"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)

    restored = torch.load(args.output, map_location="cpu", weights_only=True)
    if restored["format"] != FORMAT or set(restored["model_state_dict"]) != set(state):
        raise ValueError("written artifact failed validation")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"{args.output} ({args.output.stat().st_size} bytes)")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
