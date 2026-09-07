# bioseq-models

`bioseq-models` provides ready-to-use biological sequence models with bundled
checkpoints. Its only job is model loading and sequence prediction: pass DNA or
RNA sequences to `score()` and receive a two-dimensional NumPy array.

The package is self-contained and does not depend on a separate workflow or
training package.

## Install

From GitHub:

```bash
pip install "git+ssh://git@github.com/UTR-CODE/bioseq-models.git"
```

For development from a local clone:

```bash
pip install -e ".[dev]"
```

PyTorch wheels must match the NVIDIA driver on GPU machines. Install the
appropriate PyTorch build first if the default wheel is not compatible with
your system. CPU prediction works with the regular installation.

## Basic use

```python
from bioseq_models import load_model

model = load_model("optimus5prime", device="cpu")

sequences = [
    "AGGACGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCG",
    "CGTGAAGAGAGATTCCGCCCAGAGTGCAACCTCTACGCCAGGCGTCATCA",
]
prediction = model.score(sequences)

print(type(prediction))       # <class 'numpy.ndarray'>
print(prediction.shape)       # (2, 1)
print(model.task_names)       # ('mrl',)
print(prediction)
```

The output shape is always `(number_of_sequences, number_of_tasks)`. A
single-task model therefore returns `(N, 1)`, not `(N,)`.

Multi-task models use the same interface:

```python
model = load_model("deepstarr", device="cpu")
prediction = model.score(["ACGT" * 62 + "A"])

print(prediction.shape)       # (1, 2)
print(model.task_names)       # ('developmental', 'housekeeping')
print(prediction[0, 0])       # developmental activity
print(prediction[0, 1])       # housekeeping activity
```

## Sequence handling

Inputs may be a Python string, a list of strings, or a one-dimensional NumPy
array. Sequences are uppercased internally, and `U` is converted to `T`. The
accepted alphabet is `A/C/G/T/N`; `N` is encoded as an all-zero position.

Models with bounded fixed input sizes can pad shorter biological sequences with
`N` on either side:

```python
model = load_model(
    "optimus5prime",
    device="cpu",
    pad_to_length=50,
    padding_side="left",
)
prediction = model.score(["AUGGCU" * 6])
```

Without `pad_to_length`, models such as Optimus require their exact native
length. Models that are intrinsically variable length already configure their
own maximum length and padding behavior.

## Bundled models

| Name | Input | Tasks |
|---|---:|---|
| `optimus5prime` | 50 nt | `mrl` |
| `deepstarr` | 249 nt | `developmental`, `housekeeping` |
| `dream_legnet` | 110 nt full construct | `promoter_activity` |
| `cetnar_stability` | up to 80 nt | `decay_rate_k_fit` |
| `parm_k562` | up to 586 nt before adapters | `log2rpm_k562` |
| `agarwal_hepg2` | 230 nt | `log2_rna_dna` |
| `agarwal_k562` | 230 nt | `log2_rna_dna` |
| `agarwal_wtc11` | 230 nt | `log2_rna_dna` |
| `kosuri_composability` | 75-nt promoter + 34-nt RBS slots | `log10_protein_expression`, `log10_rna_expression` |
| `zhao_ptrc` | up to 76 nt | `log10_fluorescence_per_od600` |
| `zhao_ptrc_xgboost` | up to 76 nt | `log10_fluorescence_per_od600` |
| `zhou_l3_conditional` | 29 nt + sublibrary | `promoter_activity` |
| `zhou_l3` | 29 nt | `promoter_activity` |
| `tfbind8_weak` | 8 nt | `binding_score` |
| `utr5_weak` | 50 nt | `mrl` |
| `dream_weak` | up to 142 nt | `promoter_activity` |

List and inspect models programmatically:

```python
from bioseq_models import available_models, model_info

print(available_models())
print(model_info("deepstarr"))
```

### DREAM LegNet variable region

The default DREAM LegNet interface accepts the complete 110-nt dataset
construct and removes its 17-nt left adapter during preprocessing. To provide
the 93-nt variable region directly:

```python
model = load_model(
    "dream_legnet",
    device="cpu",
    variable_region_only=True,
)
prediction = model.score(["ACGT" * 23 + "A"])
```

### Kosuri promoter/RBS pairs

`score_pairs()` places variable-length promoter and RBS sequences into their
canonical 75-nt and 34-nt slots:

```python
model = load_model("kosuri_composability", device="cpu")
prediction = model.score_pairs(
    promoters=["GGCGCGCCATTAATCATCCGGAATG"],
    rbs_sequences=["ATCTTAATCTAGCGCGGGAGTATTTCATATG"],
)
print(prediction.shape)       # (1, 2)
```

### Zhou L3 condition

Choose one documented L3 source sublibrary when loading the conditional model:

```python
model = load_model(
    "zhou_l3_conditional",
    device="cpu",
    sublibrary="L3_s2",
)
prediction = model.score(["ACGTACGTACGTACGTACGTACGTACGTA"])
```

Valid conditions are `L3_s1`, `L3_s2`, `L3_s3`, and `L3_s4`.

## Reproducibility and testing

Model provenance, checkpoint selection, and fidelity notes are recorded in
[`MODEL_PROVENANCE.md`](MODEL_PROVENANCE.md).

Run the test suite after installation:

```bash
pytest
```

The tests load every bundled model, check exact NumPy output shapes, verify RNA
normalization and padding, and replay retained reference predictions.
