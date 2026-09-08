# bioseq-models

Load biological sequence models by name and obtain predictions as NumPy arrays.

## Installation

```bash
pip install "git+https://github.com/UTR-CODE/bioseq-models.git"
```

Only the files required by the selected model are downloaded from Hugging Face
on first use and cached locally. The weights are hosted in the
[`xinHesysu/bioseq-models`](https://huggingface.co/xinHesysu/bioseq-models)
model repository and pinned to the `v0.9.0` revision.

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

## Tissue-specific DeepSTARR

The converted DeepSTARR-Mouse and DeepSTARR-Embryo models accept 1,001-nt
sequences. Activity models return probabilities; accessibility models have
linear outputs.

```python
from bioseq_models import load_model

sequence = "ACGT" * 250 + "A"

heart_activity = load_model("deepstarr_mouse_heart_activity", device="cpu")
heart_probability = heart_activity.score([sequence])

limb_accessibility = load_model(
    "deepstarr_mouse_limb_accessibility",
    device="cpu",
)
limb_accessibility_score = limb_accessibility.score([sequence])

embryo = load_model("deepstarr_embryo", device="cpu")
embryo_activity = embryo.score([sequence])
```

DeepSTARR-Mouse provides separate activity and accessibility resources for
heart, limb, midbrain, forebrain, hindbrain, and neural tube. Each resource
averages the six official checkpoints from three cross-validation folds and two
replicates. `deepstarr_mouse` selects heart activity, while
`deepstarr_mouse_cns` selects midbrain activity. DeepSTARR-Embryo provides six
accessibility resources and five activity resources. Each resource averages all
released checkpoints with finite parameters from ten genomic folds and three
replicates.
`deepstarr_embryo` selects brain/VNC activity.

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

| Model | Sequence input | Prediction tasks | Paper |
|---|---:|---|---|
| `optimus5prime` | 50-nt 5′UTR | `mrl` | [Sample et al. (2019)](https://doi.org/10.1038/s41587-019-0164-5) |
| `utr_code` | up to 1000-nt 5′UTR, 4500-nt CDS, or 4000-nt 3′UTR | `log1p_rpf_rpkm_x5` | |
| `cds_stability` | complete CDS up to 9000 nt | `stability_score` | |
| `deepstarr` | 249 nt | `developmental`, `housekeeping` | [de Almeida et al. (2022)](https://doi.org/10.1038/s41588-022-01048-5) |
| `deepstarr_mouse_heart_activity` | 1,001 nt | `heart_activity_probability` | [Chen et al. (2026)](https://doi.org/10.1038/s41588-026-02729-1) |
| `deepstarr_mouse_heart_accessibility` | 1,001 nt | `heart_accessibility` | [Chen et al. (2026)](https://doi.org/10.1038/s41588-026-02729-1) |
| `deepstarr_mouse_limb_activity` | 1,001 nt | `limb_activity_probability` | [Chen et al. (2026)](https://doi.org/10.1038/s41588-026-02729-1) |
| `deepstarr_mouse_limb_accessibility` | 1,001 nt | `limb_accessibility` | [Chen et al. (2026)](https://doi.org/10.1038/s41588-026-02729-1) |
| `deepstarr_mouse_midbrain_activity` | 1,001 nt | `midbrain_activity_probability` | [Chen et al. (2026)](https://doi.org/10.1038/s41588-026-02729-1) |
| `deepstarr_mouse_midbrain_accessibility` | 1,001 nt | `midbrain_accessibility` | [Chen et al. (2026)](https://doi.org/10.1038/s41588-026-02729-1) |
| `deepstarr_mouse_forebrain_activity` | 1,001 nt | `forebrain_activity_probability` | [Chen et al. (2026)](https://doi.org/10.1038/s41588-026-02729-1) |
| `deepstarr_mouse_forebrain_accessibility` | 1,001 nt | `forebrain_accessibility` | [Chen et al. (2026)](https://doi.org/10.1038/s41588-026-02729-1) |
| `deepstarr_mouse_hindbrain_activity` | 1,001 nt | `hindbrain_activity_probability` | [Chen et al. (2026)](https://doi.org/10.1038/s41588-026-02729-1) |
| `deepstarr_mouse_hindbrain_accessibility` | 1,001 nt | `hindbrain_accessibility` | [Chen et al. (2026)](https://doi.org/10.1038/s41588-026-02729-1) |
| `deepstarr_mouse_neural_tube_activity` | 1,001 nt | `neural_tube_activity_probability` | [Chen et al. (2026)](https://doi.org/10.1038/s41588-026-02729-1) |
| `deepstarr_mouse_neural_tube_accessibility` | 1,001 nt | `neural_tube_accessibility` | [Chen et al. (2026)](https://doi.org/10.1038/s41588-026-02729-1) |
| `deepstarr_embryo_brain_accessibility` | 1,001 nt | `brain_accessibility` | [de Almeida et al. (2023)](https://doi.org/10.1038/s41586-023-06905-9) |
| `deepstarr_embryo_cns_accessibility` | 1,001 nt | `cns_accessibility` | [de Almeida et al. (2023)](https://doi.org/10.1038/s41586-023-06905-9) |
| `deepstarr_embryo_epidermis_accessibility` | 1,001 nt | `epidermis_accessibility` | [de Almeida et al. (2023)](https://doi.org/10.1038/s41586-023-06905-9) |
| `deepstarr_embryo_midgut_accessibility` | 1,001 nt | `midgut_accessibility` | [de Almeida et al. (2023)](https://doi.org/10.1038/s41586-023-06905-9) |
| `deepstarr_embryo_somatic_muscle_accessibility` | 1,001 nt | `somatic_muscle_accessibility` | [de Almeida et al. (2023)](https://doi.org/10.1038/s41586-023-06905-9) |
| `deepstarr_embryo_visceral_muscle_accessibility` | 1,001 nt | `visceral_muscle_accessibility` | [de Almeida et al. (2023)](https://doi.org/10.1038/s41586-023-06905-9) |
| `deepstarr_embryo_brain_vnc_activity` | 1,001 nt | `brain_vnc_activity_probability` | [de Almeida et al. (2023)](https://doi.org/10.1038/s41586-023-06905-9) |
| `deepstarr_embryo_muscle_activity` | 1,001 nt | `muscle_activity_probability` | [de Almeida et al. (2023)](https://doi.org/10.1038/s41586-023-06905-9) |
| `deepstarr_embryo_epidermis_activity` | 1,001 nt | `epidermis_activity_probability` | [de Almeida et al. (2023)](https://doi.org/10.1038/s41586-023-06905-9) |
| `deepstarr_embryo_gut_activity` | 1,001 nt | `gut_activity_probability` | [de Almeida et al. (2023)](https://doi.org/10.1038/s41586-023-06905-9) |
| `deepstarr_embryo_brain_specific_activity` | 1,001 nt | `brain_specific_activity_probability` | [de Almeida et al. (2023)](https://doi.org/10.1038/s41586-023-06905-9) |
| `malinois` | 200-nt enhancer | `K562_log2FC`, `HepG2_log2FC`, `SKNSH_log2FC` | [Gosai et al. (2024)](https://doi.org/10.1038/s41586-024-08070-z) |
| `dream_legnet` | 110-nt construct or 93-nt variable region | `promoter_activity` | [Penzar et al. (2023)](https://doi.org/10.1093/bioinformatics/btad457) |
| `cetnar_stability` | up to 80 nt | `decay_rate_k_fit` | [Cetnar et al. (2024)](https://doi.org/10.1038/s41467-024-54059-7) |
| `parm_k562` | up to 586 nt | `log2rpm_k562` | [Barbadilla-Martínez et al. (2026)](https://doi.org/10.1038/s41586-025-10093-z) |
| `agarwal_hepg2` | 230 nt | `log2_rna_dna` | [Agarwal et al. (2025)](https://doi.org/10.1038/s41586-024-08430-9) |
| `agarwal_k562` | 230 nt | `log2_rna_dna` | [Agarwal et al. (2025)](https://doi.org/10.1038/s41586-024-08430-9) |
| `agarwal_wtc11` | 230 nt | `log2_rna_dna` | [Agarwal et al. (2025)](https://doi.org/10.1038/s41586-024-08430-9) |

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
