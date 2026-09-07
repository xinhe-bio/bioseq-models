# bioseq-models

Load biological sequence models by name and obtain predictions as NumPy arrays.

## Installation

```bash
pip install "git+https://github.com/UTR-CODE/bioseq-models.git"
```

Only the files required by the selected model are downloaded from Hugging Face
on first use and cached locally. The weights are hosted in the
[`xinHesysu/bioseq-models`](https://huggingface.co/xinHesysu/bioseq-models)
model repository and pinned to the `v0.6.0` revision.

## Basic use

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
```

`score()` returns a NumPy array with shape `(number_of_sequences,
number_of_tasks)`. RNA `U` is converted to DNA `T` internally.

To download a model before running inference:

```python
from bioseq_models import download_model

download_model("optimus5prime")
```

This downloads one Optimus checkpoint, not the complete model collection.

## Malinois

Malinois predicts enhancer activity in three human cell types. Supply the
200-nt candidate enhancer; the adapter adds the official MPRA vector context
expected by the model.

```python
from bioseq_models import load_model

model = load_model("malinois", device="cpu")
prediction = model.score([
    "ACGT" * 50,
    "TGCA" * 50,
])

# Columns: K562_log2FC, HepG2_log2FC, SKNSH_log2FC
print(prediction)
```

## UTR-CODE

UTR-CODE predicts `log1p(RPF_RPKM × 5)` in a fixed transcript and cellular
context. Choose one region to design and supply the other two regions when the
model is loaded.

```python
from bioseq_models import load_model

model = load_model(
    "utr_code",
    design_region="utr3",
    utr5="GGGAGACCGGCGCGGCGGCGGCGGAGGCGGAGGCGG",
    cds="ATGGCCGCCGCCGACCTGAACTTCGAGGAGTAA",
    cell_context="HEK293T",
    rna_count=6.3,
    device="cpu",
)

candidates = [
    "GCTGCTGCTGCTGCTGCTGCTGCTGCTGCT",
    "ACACACACACACACACACACACACACACACA",
]
prediction = model.score(candidates)
```

`design_region` can be `utr5`, `cds`, or `utr3`. The corresponding maximum
lengths are 1000, 4500, and 4000 nt. `rna_count` is the transformed value
`log1p(RNA_RPKM × 5)`. A custom cellular context can be supplied as a
13,238-value raw expression vector with `transcriptome_profile=`.

## CDS stability

The CDS stability model combines a sequence CNN with CodonBERT and returns one
normalized stability score.

```python
from bioseq_models import load_model

model = load_model(
    "cds_stability",
    species="human",
    device="cpu",
)

cds_candidates = [
    "ATGGCCGCCGCCGACCTGAACTTCGAGGAGTAA",
    "ATGGCGGCGGCGGATCTCAACTTTGAAGAGTAA",
]
prediction = model.score(cds_candidates)
```

Supported species are `human`, `mouse`, `zebrafish`, `fruitfly`, and `yeast`.
A CDS must use complete codons and cannot exceed 9000 nt. The output is a
normalized model score, not a half-life measured in hours.

## Available models

| Model | Sequence input | Prediction tasks |
|---|---:|---|
| `optimus5prime` | 50-nt 5′UTR | `mrl` |
| `utr_code` | up to 1000-nt 5′UTR, 4500-nt CDS, or 4000-nt 3′UTR | `log1p_rpf_rpkm_x5` |
| `cds_stability` | complete CDS up to 9000 nt | `stability_score` |
| `deepstarr` | 249 nt | `developmental`, `housekeeping` |
| `malinois` | 200-nt enhancer | `K562_log2FC`, `HepG2_log2FC`, `SKNSH_log2FC` |
| `dream_legnet` | 110-nt construct or 93-nt variable region | `promoter_activity` |
| `cetnar_stability` | up to 80 nt | `decay_rate_k_fit` |
| `parm_k562` | up to 586 nt | `log2rpm_k562` |
| `agarwal_hepg2` | 230 nt | `log2_rna_dna` |
| `agarwal_k562` | 230 nt | `log2_rna_dna` |
| `agarwal_wtc11` | 230 nt | `log2_rna_dna` |

```python
from bioseq_models import available_models, model_info

print(available_models())
print(model_info("deepstarr"))
```

## Input padding

For a shorter sequence supplied to a fixed-length model, select left or right
`N` padding:

```python
model = load_model(
    "optimus5prime",
    pad_to_length=50,
    padding_side="left",
)
prediction = model.score(["AUGGCU" * 6])
```

Sequences longer than the supported model input raise `ValueError` and are not
silently truncated.

Checkpoint selection and model fidelity are documented in
[`MODEL_PROVENANCE.md`](MODEL_PROVENANCE.md).
