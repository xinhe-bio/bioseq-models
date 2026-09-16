# bioseq-models

Pretrained biological sequence models with a common interface for loading
checkpoints and scoring DNA or RNA sequences.

## Installation

```bash
pip install "git+https://github.com/UTR-CODE/bioseq-models.git"
```

Model weights are downloaded on first use from
[`xinHesysu/bioseq-models`](https://huggingface.co/xinHesysu/bioseq-models)
at revision `v0.11.1` and cached locally.

## Quick start

```python
from bioseq_models import load_model

model = load_model("optimus5prime", device="cpu")

sequences = [
    "AGGACGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCG",
    "CGTGAAGAGAGATTCCGCCCAGAGTGCAACCTCTACGCCAGGCGTCATCA",
]
scores = model.score(sequences)

print(scores)
```

`optimus5prime`, `deepstarr`, and the three Agarwal lentiMPRA models also expose
`score_soft(probabilities)` for differentiable sequence design. `probabilities`
has shape `[batch, length, 4]` in A/C/G/T order. Each adapter applies its model's
required layout and channel order; Agarwal converts the canonical input to
A/G/C/T internally.

Use `download_model()` to cache a model before inference:

```python
from bioseq_models import download_model

download_model("optimus5prime")
```

## Available models

### General models

| Model | Input → output |
|---|---|
| `optimus5prime` ([paper](https://doi.org/10.1038/s41587-019-0164-5)) | 50-nt 5′UTR → mean ribosome load |
| `utr_code` | 5′UTR, CDS, or 3′UTR → context-dependent translation score |
| `deepadapt_heatshock` | 5′UTR, CDS, or 3′UTR → WI-38 senescent heat-shock delta and condition-specific log2 TE |
| `cds_stability` | complete CDS → stability score |
| `deepstarr` ([paper](https://doi.org/10.1038/s41588-022-01048-5)) | 249-nt enhancer → developmental and housekeeping activity |
| `malinois` ([paper](https://doi.org/10.1038/s41586-024-08070-z)) | 200-nt enhancer → K562, HepG2, and SK-N-SH activity |
| `dream_legnet` ([paper](https://doi.org/10.1093/bioinformatics/btad457)) | 110-nt construct or 93-nt variable region → promoter activity |
| `cetnar_stability` ([paper](https://doi.org/10.1038/s41467-024-54059-7)) | sequence up to 80 nt → RNA decay rate |
| `aparent` ([paper](https://doi.org/10.1016/j.cell.2019.04.046)) | 205-nt poly(A)-site window → positional cleavage probabilities |
| `parm_k562` ([paper](https://doi.org/10.1038/s41586-025-10093-z)) | sequence up to 586 nt → K562 RNA abundance |
| `agarwal_hepg2`, `agarwal_k562`, `agarwal_wtc11` ([paper](https://doi.org/10.1038/s41586-024-08430-9)) | 230-nt enhancer → lentiMPRA activity |

### DeepSTARR-Mouse

[DeepSTARR-Mouse](https://doi.org/10.1038/s41588-026-02729-1) accepts
1,001-nt sequences. Activity and accessibility models are available for
`heart`, `limb`, `midbrain`, `forebrain`, `hindbrain`, and `neural_tube`.

Model names follow these patterns:

```text
deepstarr_mouse_<tissue>_activity
deepstarr_mouse_<tissue>_accessibility
```

Activity models return tissue-specific activity probabilities. Accessibility
models return predicted chromatin-accessibility signals. Each resource averages
the six official checkpoints from three cross-validation folds and two
replicates.

### DeepSTARR-Embryo

[DeepSTARR-Embryo](https://doi.org/10.1038/s41586-023-06905-9) accepts
1,001-nt sequences from *Drosophila melanogaster*.

| Model pattern | Tissues |
|---|---|
| `deepstarr_embryo_<tissue>_accessibility` | `brain`, `cns`, `epidermis`, `midgut`, `somatic_muscle`, `visceral_muscle` |
| `deepstarr_embryo_<tissue>_activity` | `brain_vnc`, `muscle`, `epidermis`, `gut`, `brain_specific` |

Each resource averages the released checkpoints with finite parameters from ten
genomic folds and three replicates.

Example:

```python
from bioseq_models import load_model

sequence = "ACGT" * 250 + "A"

activity_model = load_model(
    "deepstarr_mouse_heart_activity",
    device="cpu",
)
activity_scores = activity_model.score([sequence])

accessibility_model = load_model(
    "deepstarr_embryo_cns_accessibility",
    device="cpu",
)
accessibility_scores = accessibility_model.score([sequence])
```

### Malinois

Malinois predicts MPRA activity in K562, HepG2, and SK-N-SH cells. Supply the
200-nt candidate enhancer; the model adapter adds the vector context used by the
official 600-nt MPRA construct.

```python
from bioseq_models import load_model

model = load_model("malinois", device="cpu")
scores = model.score([
    "ACGT" * 50,
    "TGCA" * 50,
])

# Output order: K562_log2FC, HepG2_log2FC, SKNSH_log2FC
print(scores)
```

### APARENT

APARENT predicts the cleavage-position distribution around a polyadenylation
site from a 205-nt sequence window:

```python
from bioseq_models import load_model

model = load_model("aparent", device="cpu")
probabilities = model.score(["ACGT" * 51 + "A"])

# Columns 0:205 are cleavage probabilities at sequence positions 0:205.
# Column 205 is the probability of cleavage outside the sequence window.
assert probabilities.shape == (1, 206)
```

The 206 columns form a probability distribution and sum to one. The adapter
uses the sequence-only inference context from the released APARENT model: an
all-zero library indicator and a positive distal-PAS indicator.

### UTR-CODE

UTR-CODE predicts `log1p(RPF_RPKM × 5)` in a fixed transcript and cellular
context. Select one region to score and supply the other two regions when
loading the model.

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
scores = model.score(candidates)
```

`design_region` can be `utr5`, `cds`, or `utr3`, with maximum lengths of
1,000, 4,500, and 4,000 nt, respectively. `rna_count` is
`log1p(RNA_RPKM × 5)`. A custom cellular context can be supplied as a
13,238-value raw expression vector with `transcriptome_profile=`.

### DeepAdapt heat shock

DeepAdapt predicts translation efficiency in normal and heat-shocked senescent
WI-38 cells. Select one transcript region to score and supply the fixed sequence
for the other two regions.

```python
from bioseq_models import load_model

model = load_model(
    "deepadapt_heatshock",
    design_region="utr5",
    cds="ATG" + "GCT" * 30 + "TAA",
    utr3="ACGT" * 20,
    device="cpu",
)

candidates = [
    "GCGTGCGTGCGTGCGTGCGTGCGTGCGTGCGT",
    "AACCAACCAACCAACCAACCAACCAACCAACC",
]
scores = model.score(candidates)

# Columns: delta_log2_te, normal_log2_te, heatshock_log2_te
print(scores)
```

`delta_log2_te` is `heatshock_log2_te - normal_log2_te`; positive values predict
increased translation efficiency after heat shock. The bundled environment uses
two normal and two heat-shock RNA-seq profiles from condition
`WI38_Senescent_HeatShock_GSE124609`.

`design_region` can be `utr5`, `cds`, or `utr3`, with maximum lengths of 1,000,
4,500, and 4,000 nt. CDS inputs must start with `ATG`, end with `TAA`, `TAG`, or
`TGA`, and contain complete codons. Longer sequences raise `ValueError`.

### CDS stability

The CDS stability model combines a sequence CNN with CodonBERT.

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
scores = model.score(cds_candidates)
```

Supported species are `human`, `mouse`, `zebrafish`, `fruitfly`, and `yeast`.
A CDS must contain complete codons and cannot exceed 9,000 nt. The model returns
a normalized stability score rather than a half-life in hours.

## Model discovery

```python
from bioseq_models import available_models, model_info

print(available_models())
print(model_info("deepstarr"))
```

## Sequence handling

Sequences may contain `A`, `C`, `G`, `T`, `U`, or `N`. RNA `U` is treated as
DNA `T`.

For a shorter sequence supplied to a fixed-length model, select left or right
`N` padding:

```python
model = load_model(
    "optimus5prime",
    pad_to_length=50,
    padding_side="left",
)
scores = model.score(["AUGGCU" * 6])
```

Sequences longer than the supported input length raise `ValueError` and are not
truncated.

## Model weights and provenance

The package downloads only the files required by the selected model. Set
`BIOSEQ_MODELS_WEIGHTS_DIR` to use a local weight directory or
`BIOSEQ_MODELS_CACHE` to choose the Hugging Face cache location.

Checkpoint sources, conversions, and numerical validation are documented in
[`MODEL_PROVENANCE.md`](MODEL_PROVENANCE.md). Third-party licenses and notices
are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Development

```bash
git clone https://github.com/UTR-CODE/bioseq-models.git
cd bioseq-models
pip install -e ".[dev]"
pytest
```
