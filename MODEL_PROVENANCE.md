# Model provenance

The checkpoints were selected from the BioSeqBench reproductions and UTR-CODE
projects. They are distributed together in the versioned Zenodo weight bundle.

| Model | Bundled checkpoint | Notes |
|---|---|---|
| `optimus5prime` | one selected checkpoint | Published Optimus 5-Prime architecture with BioSeqBench-trained weights. |
| `deepstarr` | one selected two-output checkpoint | Published DeepSTARR architecture with BioSeqBench-trained weights. |
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
