# Biosequence Models

This repository contains reproduced activity-model architectures, trained
weights, encoders, and checkpoint loaders. Optimization algorithms live in the
separate [`bioseq-optimization`](https://github.com/UTR-CODE/bioseq-optimization)
repository.

## Installation

Install directly from GitHub with pip. The optimization package is installed
automatically as a dependency.

```bash
python -m pip install git+https://github.com/UTR-CODE/bioseq-models.git
```

For an editable checkout:

```bash
git clone https://github.com/UTR-CODE/bioseq-models.git
cd bioseq-models
python -m pip install -e .
```

## Included models

| Name | Default biological length | Outputs | Differentiable |
|---|---:|---|---|
| `optimus5prime` | 50 nt | MRL | Yes |
| `deepstarr` | 249 bp | Developmental, housekeeping | Yes |
| `dream_legnet` | 110 bp | Yeast promoter activity | No |
| `zhao_ptrc` | 74 bp | log10 fluorescence/OD600 | Yes |
| `zhou_l3` | 29 bp | Promoter activity | Yes |

```python
from bioseq_models import available_models, load_model

print(available_models())
model = load_model("optimus5prime", device="cpu")
```

## Optimize with the separate algorithm package

The loaded object implements the scorer interface defined by
`bioseq-optimization`, so it can be passed directly to any compatible
optimizer:

```python
from bioseq_models import load_model
from sequence_optimization import SequenceSpace, genetic_algorithm_optimize

model = load_model("optimus5prime", device="cpu")

result = genetic_algorithm_optimize(
    model,
    SequenceSpace(length=50),
    num_candidates=32,
    budget=320,
    seed=7,
)
```

## Variable biological lengths and N padding

Some activity models consume a fixed tensor even though the biological
sequences have different lengths. Select left or right N padding when loading
the model. N is encoded as an all-zero nucleotide vector and is not introduced
into the sequence being optimized.

This example designs 40-nt sequences for the 50-position Optimus input:

```python
from bioseq_models import load_model
from sequence_optimization import SequenceSpace, genetic_algorithm_optimize

model = load_model(
    "optimus5prime",
    pad_to_length=50,
    padding_side="left",
)

result = genetic_algorithm_optimize(
    model,
    SequenceSpace(length=40),
    num_candidates=16,
    budget=160,
    seed=7,
)
```

Use `padding_side="right"` to place N after the biological sequence. RNA U is
converted to T before padding; all optimization candidates remain ACGT.

## Use your own model

[`examples/custom_cnn.py`](examples/custom_cnn.py) shows a minimal one-layer
CNN, checkpoint loading, scorer wrapping, and optimization call.

## Verification

```bash
python -m pytest -q
```
