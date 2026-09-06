# Model provenance and fidelity

This catalog records what was migrated from BioSeqBench and how each public
bundle should be interpreted. Only canonical final checkpoints were selected.
Smoke tests, calibration runs, repeated epochs, and superseded checkpoints were
not copied.

## Publication-oriented reproductions

| Bundle | Checkpoint selection | Fidelity note |
|---|---|---|
| `optimus5prime` | one selected Optimus 5-Prime checkpoint | Published Optimus architecture; BioSeqBench-trained weights. |
| `deepstarr` | one selected two-output checkpoint | Published DeepSTARR architecture; BioSeqBench-trained weights. |
| `dream_legnet` | selected Autosome/LegNet checkpoint | Reproduced DREAM promoter model and validation-selected weights. |
| `parm_k562` | epoch-10 checkpoint from each of 5 published folds | Exact PARM CNN, K562 task, adapters, fold protocol, and five-model average. Inference uses user-selected deterministic left/right padding; training used random placement. |
| `agarwal_hepg2` | 10 practical held-out-fold checkpoints | Four-channel MPRALegNet reproduction with reverse-complement test-time averaging. |
| `agarwal_k562` | 10 practical held-out-fold checkpoints | Same protocol, K562 labels. |
| `agarwal_wtc11` | 10 practical held-out-fold checkpoints | Same protocol, WTC11 labels. |

The Agarwal bundle keeps one validation choice for each of the ten held-out
folds. The complete 90 validation/test-fold combinations are training artifacts,
not 90 distinct biological models, and are therefore not distributed.

## Functional and benchmark models

| Bundle | Checkpoint selection | Fidelity note |
|---|---|---|
| `cetnar_stability` | best validation CNN-BiLSTM | Sequence-only differentiable surrogate. It exposes measured decay and `stability = -decay` so maximization has a direct meaning. |
| `CetnarLightGBM` | 5 timepoint boosters plus 1 decay booster | Paper-style feature cascade. It requires the engineered multi-isoform feature table and is not a raw-sequence model. |
| `kosuri_composability` | best validation two-branch CNN | Sequence-only promoter/RBS surrogate trained on standardized log10 protein and RNA labels. The public adapter restores real log10 units. |
| `zhao_ptrc_xgboost` | validation-selected XGBoost refit on train + validation | Functional reproduction because the exact paper split and hyperparameters were unavailable. |
| `zhao_ptrc` | validation-selected CNN refit on train + validation | Differentiable BioSeqBench comparator on the same log10 target. |
| `zhou_l3_conditional` | best validation conditional CNN | Canonical BioSeqBench surrogate with an explicit `L3_s1`–`L3_s4` condition. |
| `zhou_l3` | best sequence-only CNN | Diagnostic model that intentionally omits the L3 sublibrary condition. |

The original Kosuri evaluation CSV used column names containing `log10` for
standardized values. The bundled adapter fixes that ambiguity: its outputs are
inverse-standardized and are genuine log10 protein/RNA expression values.

## Weak-model ensembles

| Bundle | Checkpoints | Intended use |
|---|---:|---|
| `tfbind8_weak` | 10 | Lower-95% training-pool surrogate used for optimization-benchmark experiments. |
| `utr5_weak` | 5 | Lower-95% WeakSeqCNN ensemble for 5′UTR optimization. |
| `dream_weak` | 5 | Lower-95% variable-length WeakSeqCNN ensemble for yeast promoter optimization. |

These are deliberately weak or restricted-data oracles, not replacements for
the corresponding strong/reference model. Their prediction object includes
between-member standard deviation.

## Migration replay checks

Predictions from the migrated adapters were replayed against the retained
BioSeqBench evaluation artifacts. Cetnar CNN, Zhao CNN/XGBoost, one explicit
Agarwal fold, and all three weak ensembles agree within floating-point CPU/GPU
tolerance. Kosuri agrees after applying the intended inverse standardization
described above. Registry tests additionally load and execute all 16 sequence
interfaces.

## Datasets without a bundled trained model

The migration audit also considered dataset-only, removed, or incomplete-model
targets such as GB1, GFP, OpenVaccine, TFBind10, PARM v1/PHY3, yeast glucose,
and yeast SC-Ura. No canonical trained scorer existed for these at migration
time, so this package does not present them as reproduced models. They can be
added when a completed training protocol and a validated final checkpoint are
available.
