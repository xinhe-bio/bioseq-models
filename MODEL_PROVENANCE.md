# Model provenance

The checkpoints were selected from the BioSeqBench reproductions and UTR-CODE
projects. They are distributed as individually downloadable files in the
versioned Hugging Face model repository.
The Python package pins version 0.8.0 to the immutable `v0.8.0` repository tag.

| Model | Bundled checkpoint | Notes |
|---|---|---|
| `optimus5prime` | one selected checkpoint | Published Optimus 5-Prime architecture with BioSeqBench-trained weights. |
| `deepstarr` | one selected two-output checkpoint | Published DeepSTARR architecture with BioSeqBench-trained weights. |
| `deepstarr_mouse_heart_activity` | six official checkpoints | Heart activity probability averaged across three folds and two replicates. |
| `deepstarr_mouse_heart_accessibility` | six official checkpoints | Heart accessibility averaged across three folds and two replicates. |
| `deepstarr_mouse_limb_activity` | six official checkpoints | Limb activity probability averaged across three folds and two replicates. |
| `deepstarr_mouse_limb_accessibility` | six official checkpoints | Limb accessibility averaged across three folds and two replicates. |
| `deepstarr_mouse_midbrain_activity` | six official checkpoints | Midbrain/CNS activity probability averaged across three folds and two replicates. |
| `deepstarr_mouse_midbrain_accessibility` | six official checkpoints | Midbrain accessibility averaged across three folds and two replicates. |
| `deepstarr_mouse_forebrain_activity` | six official checkpoints | Forebrain activity probability averaged across three folds and two replicates. |
| `deepstarr_mouse_forebrain_accessibility` | six official checkpoints | Forebrain accessibility averaged across three folds and two replicates. |
| `deepstarr_mouse_hindbrain_activity` | six official checkpoints | Hindbrain activity probability averaged across three folds and two replicates. |
| `deepstarr_mouse_hindbrain_accessibility` | six official checkpoints | Hindbrain accessibility averaged across three folds and two replicates. |
| `deepstarr_mouse_neural_tube_activity` | six official checkpoints | Neural-tube activity probability averaged across three folds and two replicates. |
| `deepstarr_mouse_neural_tube_accessibility` | six official checkpoints | Neural-tube accessibility averaged across three folds and two replicates. |
| `deepstarr_embryo_brain_vnc_activity` | official fold 01, replicate 1 | Drosophila brain/VNC enhancer-activity Keras checkpoint converted to PyTorch; sigmoid probability output. |
| `malinois` | official `20211113_021200` checkpoint | Official BODA PyTorch architecture and weights. The adapter places each 200-nt sequence between the two 200-nt MPRA vector contexts used by the authors. Outputs retain the training targets' log2 fold-change scale in K562, HepG2, and SK-N-SH order. |
| `dream_legnet` | one validation-selected checkpoint | Reproduced Autosome/LegNet model with the published 150-position preprocessing and forward/reverse-complement average. |
| `cetnar_stability` | best validation CNN-BiLSTM | Sequence model trained to predict `decay_rate_k_fit`. |
| `utr_code` | human 13,238-feature checkpoint, epoch 5 | Joint 5′UTR, CDS, 3′UTR, transcriptome-context and RNA-abundance model. Only the supervised RPF output is exposed. |
| `cds_stability` | selected epoch 27 checkpoint and CodonBERT | Multi-species CDS model. The prediction is on the normalized training-target scale, not in hours. |
| `parm_k562` | five fold checkpoints | Exact PARM CNN for K562. Predictions are averaged across the five folds. |
| `agarwal_hepg2` | ten held-out-fold checkpoints | Four-channel MPRALegNet reproduction with reverse-complement averaging. |
| `agarwal_k562` | ten held-out-fold checkpoints | Same architecture and evaluation protocol with K562 labels. |
| `agarwal_wtc11` | ten held-out-fold checkpoints | Same architecture and evaluation protocol with WTC11 labels. |

“Published architecture” describes the reproduced architecture and training
protocol. The bundled files are BioSeqBench-trained checkpoints unless noted
otherwise; they are not represented as author-released weight files.

## Malinois source

- Paper: https://doi.org/10.1038/s41586-024-08070-z
- Official BODA source: https://github.com/sjgosai/boda2
- Official artifact: https://storage.googleapis.com/tewhey-public-data/CODA_resources/malinois_artifacts__20211113_021200__287348.tar.gz

The distributed `malinois.pt` contains the tensor state dictionary extracted
from the official artifact. Training-only Lightning state and optimizer state
are omitted; the parameter tensors are unchanged.

## Tissue-specific DeepSTARR sources

- DeepSTARR-Mouse paper: https://doi.org/10.1038/s41588-026-02729-1
- DeepSTARR-Mouse source: https://github.com/Shenzhi-Chen/DeepSTARR-Mouse
- DeepSTARR-Mouse weights: https://huggingface.co/Shenzhi-Chen/DeepSTARR-Mouse
- DeepSTARR-Embryo paper: https://doi.org/10.1038/s41586-023-06905-9
- DeepSTARR-Embryo source: https://github.com/bernardo-de-almeida/DeepSTARR_embryo
- DeepSTARR-Embryo weights: https://doi.org/10.5281/zenodo.8011696

The original models use channels-last Keras tensors. Conversion transposes the
Conv1D and dense kernels and restores channels-last feature order immediately
before the flattened dense layers. All 72 DeepSTARR-Mouse checkpoints were
compared with their Keras sources on three deterministic sequences. The maximum
absolute difference across every tissue, task, fold, and replicate was
`3.34e-6`. The existing embryo brain/VNC conversion has a maximum absolute
difference of `1.29e-6`.
The validation used DeepSTARR-Mouse source revision
`64245a665f78c7797aedda308e372633f842adb8` and DeepSTARR-Embryo revision
`d966384e179eb5fba9f96e19c26ffcdabde95ae4`. The reusable conversion command is
provided in `scripts/convert_deepstarr_tissue_checkpoint.py`; the complete
Mouse conversion is prepared by `scripts/prepare_deepstarr_mouse_checkpoints.py`.
