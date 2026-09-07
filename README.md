# bioseq-models

Ready-to-use biological sequence models with bundled checkpoints. Load a model
by name and call `score()` to obtain predictions.

## Installation

```bash
pip install "git+https://github.com/UTR-CODE/bioseq-models.git"
```

## Quick start

```python
from bioseq_models import load_model

model = load_model("optimus5prime", device="cpu")

sequences = [
    "AGGACGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCG",
    "CGTGAAGAGAGATTCCGCCCAGAGTGCAACCTCTACGCCAGGCGTCATCA",
]
prediction = model.score(sequences)

print(prediction)
print(prediction.shape)  # (2, 1)
print(model.task_names)  # ('mrl',)
```

`score()` accepts one sequence or a list of sequences and returns a NumPy array
with shape `(number_of_sequences, number_of_tasks)`. Single-task models return
`(N, 1)` and multi-task models return `(N, T)`.

For example, DeepSTARR has two outputs:

```python
model = load_model("deepstarr", device="cpu")
prediction = model.score(["ACGT" * 62 + "A"])

print(prediction.shape)  # (1, 2)
print(model.task_names)  # ('developmental', 'housekeeping')
```

## Available models

| Model name | Sequence input | Prediction tasks |
|---|---:|---|
| `optimus5prime` | 50 nt | `mrl` |
| `deepstarr` | 249 nt | `developmental`, `housekeeping` |
| `dream_legnet` | 110-nt construct or 93-nt variable region | `promoter_activity` |
| `cetnar_stability` | up to 80 nt | `decay_rate_k_fit` |
| `parm_k562` | up to 586 nt | `log2rpm_k562` |
| `agarwal_hepg2` | 230 nt | `log2_rna_dna` |
| `agarwal_k562` | 230 nt | `log2_rna_dna` |
| `agarwal_wtc11` | 230 nt | `log2_rna_dna` |

The model catalog can also be inspected in Python:

```python
from bioseq_models import available_models, model_info

print(available_models())
print(model_info("deepstarr"))
```

## Sequence input

Sequences are converted to uppercase, and RNA `U` is converted to DNA `T`.
Accepted symbols are `A`, `C`, `G`, `T`, and `N`. An `N` position is encoded as
an all-zero vector.

For a shorter sequence supplied to a fixed-length model, enable left or right
`N` padding:

```python
model = load_model(
    "optimus5prime",
    device="cpu",
    pad_to_length=50,
    padding_side="left",
)
prediction = model.score(["AUGGCU" * 6])
```

Sequences longer than the model input length raise `ValueError` and are not
silently truncated.

## DREAM LegNet variable-region input

By default, `dream_legnet` accepts the complete 110-nt construct and removes
its 17-nt left adapter during preprocessing. Set `variable_region_only=True` to
provide the 93-nt variable region directly:

```python
model = load_model(
    "dream_legnet",
    device="cpu",
    variable_region_only=True,
)
prediction = model.score(["ACGT" * 23 + "A"])
```

## GPU prediction

Pass a CUDA device when loading the model:

```python
model = load_model("deepstarr", device="cuda")
prediction = model.score(["ACGT" * 62 + "A"])
```

Install a PyTorch build compatible with the machine's NVIDIA driver before
using CUDA.

Checkpoint selection and model fidelity are documented in
[`MODEL_PROVENANCE.md`](MODEL_PROVENANCE.md).
