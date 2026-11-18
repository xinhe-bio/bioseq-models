"""Verified Saluki weights: native HDF5 download and TensorFlow-free conversion."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import struct
from urllib.request import Request, urlopen
import zlib

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from .architectures.saluki import SalukiNetwork

SALUKI_HF_REVISION = "0e6bae61a6452b0dc8b356e576da88ba14f1e9db"


def source_manifest():
    return json.loads(
        files("bioseq_models").joinpath("data/saluki_sources.json").read_text()
    )


def _read_range(url, begin, end):
    request = Request(url, headers={"Range": f"bytes={begin}-{end}"})
    with urlopen(request, timeout=45) as response:
        expected = f"bytes {begin}-{end}/"
        if response.status != 206 or not response.headers.get(
            "Content-Range", ""
        ).startswith(expected):
            raise RuntimeError(
                "Saluki source server did not honor the requested byte range"
            )
        data = response.read(end - begin + 2)
    if len(data) != end - begin + 1:
        raise RuntimeError("Incomplete byte range from the Saluki source archive")
    return data


def _archive_member(entry, url):
    offset = entry["offset"]
    header = _read_range(url, offset, offset + 29)
    fields = struct.unpack("<4s5H3L2H", header)
    if fields[0] != b"PK\x03\x04" or fields[3] != entry["method"]:
        raise RuntimeError("Unexpected Saluki ZIP member header")
    begin = offset + 30 + fields[9] + fields[10]
    count = entry["compressed_size"]
    ranges = [
        (a, min(a + 262143, begin + count - 1))
        for a in range(begin, begin + count, 262144)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        blocks = list(pool.map(lambda r: _read_range(url, *r), ranges))
    compressed = b"".join(blocks)
    if entry["method"] == 8:
        return zlib.decompress(compressed, -15)
    if entry["method"] == 0:
        return compressed
    raise ValueError("Unsupported Saluki ZIP compression method")


def native_checkpoint(species="human", fold=0, cross=0, *, cache_dir=None):
    """Fetch one original checkpoint, checking its published ZIP size and CRC."""
    manifest = source_manifest()
    key = f"{species}_f{fold}_c{cross}"
    if key not in manifest["checkpoints"]:
        raise ValueError(f"Unknown Saluki checkpoint: {key}")
    entry = manifest["checkpoints"][key]
    base = (
        Path(
            cache_dir
            or os.environ.get(
                "BIOSEQ_MODELS_CACHE", Path.home() / ".cache/bioseq-models"
            )
        )
        / "saluki"
    )
    path = base / "native" / f"{key}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)

    def valid(data):
        return (
            len(data) == entry["size"]
            and zlib.crc32(data) & 0xFFFFFFFF == entry["crc32"]
            and (
                "sha256" not in entry
                or hashlib.sha256(data).hexdigest() == entry["sha256"]
            )
        )

    if path.is_file():
        if not valid(path.read_bytes()):
            raise ValueError(
                f"Saluki cached native checkpoint failed its integrity check: {path}"
            )
        return path
    data = None
    if entry.get("mirror_url"):
        try:
            with urlopen(entry["mirror_url"], timeout=45) as response:
                candidate = response.read(entry["size"] + 1)
            if valid(candidate):
                data = candidate
        except OSError:
            pass
    if data is None:
        data = _archive_member(entry, manifest["archive_url"])
    if not valid(data):
        raise RuntimeError(
            f"Saluki source checkpoint failed its integrity check: {key}"
        )
    temporary = path.with_suffix(f".{os.getpid()}.partial")
    temporary.write_bytes(data)
    temporary.replace(path)
    return path


def convert_h5(path):
    """Convert only the supported, original Saluki inference graph to PyTorch."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Native Saluki checkpoint conversion needs h5py; install bioseq-models[saluki]"
        ) from exc
    with h5py.File(path) as handle:
        config = json.loads(handle.attrs["model_config"])["config"]
        layers = config["layers"]
        by_class = {}
        for layer in layers:
            by_class.setdefault(layer["class_name"], []).append(layer)
        allowed = {
            "InputLayer",
            "StochasticShift",
            "Conv1D",
            "LayerNormalization",
            "ReLU",
            "Dropout",
            "MaxPooling1D",
            "GRU",
            "BatchNormalization",
            "Dense",
        }
        if set(by_class) - allowed:
            raise ValueError("Unsupported layers in Saluki checkpoint")
        convs = by_class["Conv1D"]
        norms = by_class["LayerNormalization"]
        bns = by_class["BatchNormalization"]
        dense = by_class["Dense"]
        gru = by_class["GRU"][0]["config"]
        shape = by_class["InputLayer"][0]["config"]["batch_input_shape"]
        if (
            len(convs) != 7
            or len(norms) != 7
            or len(bns) != 2
            or len(dense) != 2
            or shape != [None, 12288, 6]
        ):
            raise ValueError("Expected the original six-channel Saluki architecture")
        if (
            not gru["go_backwards"]
            or not gru["reset_after"]
            or gru["recurrent_activation"] != "sigmoid"
            or gru["activation"] != "tanh"
        ):
            raise ValueError("Unsupported Saluki recurrent configuration")
        if any(
            layer["config"]["padding"] != "valid"
            or layer["config"]["strides"] != [1]
            or layer["config"]["activation"] != "linear"
            for layer in convs
        ):
            raise ValueError("Unsupported Saluki convolution configuration")
        if any(
            layer["config"]["pool_size"] != [2]
            or layer["config"]["strides"] != [2]
            or layer["config"]["padding"] != "valid"
            for layer in by_class["MaxPooling1D"]
        ):
            raise ValueError("Unsupported Saluki pooling configuration")
        if not all(
            np.isclose(layer["config"]["epsilon"], norms[0]["config"]["epsilon"])
            for layer in norms
        ) or not np.isclose(bns[0]["config"]["epsilon"], bns[1]["config"]["epsilon"]):
            raise ValueError("Inconsistent Saluki normalization configuration")
        options = dict(
            filters=convs[0]["config"]["filters"],
            kernel_size=convs[0]["config"]["kernel_size"][0],
            num_layers=6,
            ln_epsilon=norms[0]["config"]["epsilon"],
            bn_epsilon=bns[0]["config"]["epsilon"],
            seq_length=12288,
        )
        model = SalukiNetwork(**options)
        root = handle["model_weights"]

        def arrays(name):
            result = {}
            root[name].visititems(
                lambda n, obj: result.__setitem__(
                    n.rsplit("/", 1)[-1].split(":")[0], obj[...]
                )
                if isinstance(obj, h5py.Dataset)
                else None
            )
            return result

        def copy(parameter, value):
            value = np.ascontiguousarray(value)
            if tuple(parameter.shape) != value.shape:
                raise ValueError(
                    f"Saluki parameter shape mismatch: {tuple(parameter.shape)} versus {value.shape}"
                )
            parameter.copy_(torch.from_numpy(value))

        with torch.no_grad():
            for layer, module in zip(
                convs, [model.initial] + [b.conv for b in model.blocks]
            ):
                a = arrays(layer["name"])
                copy(module.weight, a["kernel"].transpose(2, 1, 0))
                if module.bias is not None:
                    copy(module.bias, a["bias"])
            for layer, module in zip(
                norms, [b.norm for b in model.blocks] + [model.final_norm]
            ):
                a = arrays(layer["name"])
                copy(module.weight, a["gamma"])
                copy(module.bias, a["beta"])
            for layer, module in zip(bns, [model.bn1, model.bn2]):
                a = arrays(layer["name"])
                copy(module.weight, a["gamma"])
                copy(module.bias, a["beta"])
                copy(module.running_mean, a["moving_mean"])
                copy(module.running_var, a["moving_variance"])
            for layer, module in zip(dense, [model.dense, model.output]):
                a = arrays(layer["name"])
                copy(module.weight, a["kernel"].T)
                copy(module.bias, a["bias"])
            a = arrays(by_class["GRU"][0]["name"])
            width = options["filters"]
            order = np.r_[width : 2 * width, 0:width, 2 * width : 3 * width]
            copy(model.gru.weight_ih_l0, a["kernel"][:, order].T)
            copy(model.gru.weight_hh_l0, a["recurrent_kernel"][:, order].T)
            copy(model.gru.bias_ih_l0, a["bias"][0, order])
            copy(model.gru.bias_hh_l0, a["bias"][1, order])
    model.eval()
    return model, options


def checkpoint_path(
    species="human", fold=0, cross=0, *, weights_dir=None, cache_dir=None
):
    """Resolve a converted checkpoint, converting a verified original on demand."""
    local = weights_dir or os.environ.get("BIOSEQ_MODELS_WEIGHTS_DIR")
    relative = f"saluki/saluki_{species}_f{fold}_c{cross}.pt"
    if local is not None:
        path = Path(local).expanduser() / relative
        if not path.is_file():
            raise FileNotFoundError(f"Saluki checkpoint not found: {path}")
        return path
    base = Path(
        cache_dir
        or os.environ.get("BIOSEQ_MODELS_CACHE", Path.home() / ".cache/bioseq-models")
    )
    path = base / relative
    if path.is_file():
        return path
    published = source_manifest()["published_checkpoints"].get(relative)
    if published is not None:
        from .resources import HF_REPO_ID

        try:
            downloaded = Path(
                hf_hub_download(
                    repo_id=os.environ.get("BIOSEQ_MODELS_HF_REPO_ID", HF_REPO_ID),
                    revision=os.environ.get(
                        "BIOSEQ_MODELS_HF_REVISION", SALUKI_HF_REVISION
                    ),
                    filename=relative,
                    cache_dir=cache_dir or os.environ.get("BIOSEQ_MODELS_CACHE"),
                )
            )
        except (HfHubHTTPError, LocalEntryNotFoundError):
            pass  # An original HDF5 file may already be cached locally.
        else:
            if hashlib.sha256(downloaded.read_bytes()).hexdigest() != published:
                raise ValueError(
                    f"Published Saluki checkpoint failed its SHA256 check: {relative}"
                )
            return downloaded
    native = native_checkpoint(species, fold, cross, cache_dir=cache_dir)
    model, options = convert_h5(native)
    state = dict(
        architecture="saluki_official",
        model_state=model.state_dict(),
        model_config=options,
        species=species,
        fold=fold,
        cross=cross,
        source_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        output_units="Species-specific consensus half-life score (PC1); higher predicts greater stability.",
        reverse_complement=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.partial")
    torch.save(state, temporary)
    temporary.replace(path)
    return path
