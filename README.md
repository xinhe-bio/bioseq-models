# bioseq-models

Pretrained biological sequence models with a common interface for loading
checkpoints and scoring DNA or RNA sequences.

## Installation

```bash
pip install "git+https://github.com/xinhe-bio/bioseq-models.git"
```

Model weights are downloaded on first use from
[`xinHesysu/bioseq-models`](https://huggingface.co/xinHesysu/bioseq-models)
at revision `v0.11.1` and cached locally.
Saluki loads verified PyTorch weights from Hugging Face and converts additional
original checkpoints on demand. See the [Saluki guide](docs/saluki.md).

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

`optimus5prime`, `deepstarr`, the three MPRALegNet models, and Saluki with
fixed transcript annotations also expose
`score_soft(probabilities)` for differentiable sequence design. `probabilities`
has shape `[batch, length, 4]` in A/C/G/T order. Each adapter applies its model's
required layout and channel order; MPRALegNet converts the canonical input to
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
| `saluki`, `saluki_mouse` ([paper](https://doi.org/10.1186/s13059-022-02811-x)) | full mRNA + CDS/exon annotations → consensus half-life score |
| `deepstarr` ([paper](https://doi.org/10.1038/s41588-022-01048-5)) | 249-nt enhancer → developmental and housekeeping activity |
| `malinois` ([paper](https://doi.org/10.1038/s41586-024-08070-z)) | 200-nt enhancer → K562, HepG2, and SK-N-SH activity |
| `dream_legnet` ([paper](https://doi.org/10.1093/bioinformatics/btad457)) | 110-nt construct or 93-nt variable region → promoter activity |
| `cetnar_stability` ([paper](https://doi.org/10.1038/s41467-024-54059-7)) | sequence up to 80 nt → RNA decay rate |
| `aparent` ([paper](https://doi.org/10.1016/j.cell.2019.04.046)) | 205-nt poly(A)-site window → positional cleavage probabilities |
| `parm_k562` ([paper](https://doi.org/10.1038/s41586-025-10093-z)) | sequence up to 586 nt → K562 RNA abundance |
| `mpralegnet_hepg2`, `mpralegnet_k562`, `mpralegnet_wtc11` ([paper](https://doi.org/10.1038/s41586-024-08430-9)) | 230-nt enhancer → lentiMPRA activity |

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

### MPRALegNet

MPRALegNet predicts lentiMPRA activity from 230-nt sequences. Each cell-type
model averages ten cross-validation checkpoints and returns one log₂ RNA/DNA
activity score per sequence.

```python
from bioseq_models import load_model

sequences = ["ACGT" * 57 + "AC"]
scores = {
    cell: load_model(f"mpralegnet_{cell.lower()}", device="cpu").score(sequences)
    for cell in ("K562", "HepG2", "WTC11")
}
```

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
git clone https://github.com/xinhe-bio/bioseq-models.git
cd bioseq-models
pip install -e ".[dev]"
pytest
```
