# Biosequence Models

`bioseq-models` packages the activity models reproduced in BioSeqBench: model
architectures, trained checkpoints, sequence encoders, target-scale restoration,
and a common loading API. Sequence-search algorithms remain in the separate
[`bioseq-optimization`](https://github.com/UTR-CODE/bioseq-optimization)
repository. The two packages communicate through the scorer interface defined
by `bioseq-optimization`.

All default checkpoints are included in this package. Calling `load_model()`
does not download a weight file at runtime.

## Installation

Install from GitHub with pip. The optimizer package is installed automatically:

```bash
python -m pip install git+ssh://git@github.com/UTR-CODE/bioseq-models.git
```

For GPU execution, install the PyTorch build matching the machine's NVIDIA
driver first; the command above will reuse it. Otherwise pip may select a newer
CUDA runtime than an older server driver supports. CPU execution is unaffected.

For an editable checkout:

```bash
git clone git@github.com:UTR-CODE/bioseq-models.git
cd bioseq-models
python -m pip install -e .
```

Alternatively, install the minimum dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Included sequence models

The registry contains 16 ready-to-score model interfaces. Multiple checkpoints
belonging to one cross-validation or weak-model ensemble are exposed as one
model, rather than as unrelated epoch files.

| Registry name | Biological input | Outputs | Status | Gradient |
|---|---|---|---|---|
| `optimus5prime` | 50-nt 5′ UTR | MRL | paper architecture | Yes |
| `deepstarr` | 249-bp enhancer | developmental, housekeeping | paper architecture | Yes |
| `dream_legnet` | 110-bp construct or 93-bp variable region | promoter activity | paper architecture | Yes |
| `cetnar_stability` | RNA/DNA, up to 80 nt | decay rate, negative decay (`stability`) | benchmark surrogate | Yes |
| `parm_k562` | up to 586 bp; adapters added internally | K562 log2RPM | paper architecture, 5 folds | Yes |
| `agarwal_hepg2` | 230-bp element | HepG2 log2(RNA/DNA) | paper architecture, 10 folds | Yes |
| `agarwal_k562` | 230-bp element | K562 log2(RNA/DNA) | paper architecture, 10 folds | Yes |
| `agarwal_wtc11` | 230-bp element | WTC11 log2(RNA/DNA) | paper architecture, 10 folds | Yes |
| `kosuri_composability` | promoter up to 75 nt + RBS up to 34 nt | log10 protein, log10 RNA | benchmark surrogate | Yes |
| `zhao_ptrc` | Ptrc promoter, up to 76 nt | log10 fluorescence/OD600 | benchmark surrogate CNN | Yes |
| `zhao_ptrc_xgboost` | Ptrc promoter, up to 76 nt | log10 fluorescence/OD600 | functional paper reproduction | No |
| `zhou_l3_conditional` | 29 bp + L3 sublibrary | promoter activity | canonical benchmark surrogate | Yes |
| `zhou_l3` | 29 bp | promoter activity | sequence-only diagnostic | Yes |
| `tfbind8_weak` | 8 bp | binding score | lower-95% weak ensemble | Yes |
| `utr5_weak` | 50-nt 5′ UTR | MRL | lower-95% weak ensemble | Yes |
| `dream_weak` | yeast promoter, up to 142 bp | promoter activity | lower-95% weak ensemble | Yes |

“Paper architecture” means that the reproduced architecture and training
protocol follow the corresponding publication; the bundled weights are the
BioSeqBench-trained checkpoints, not a claim that the authors released those
exact weight files. See [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md) for the
selection and fidelity of every bundle.

## Load and score sequences

```python
from bioseq_models import available_models, load_model, model_info

print(available_models())
print(model_info("optimus5prime"))

model = load_model("optimus5prime", device="cpu")
sequences = [
    "CCACTCGATTAACATGTTAACAACATACTCGTCCGGCCGATCAGCGGCTA",
    "CAAATCATGTGCAGCCCTGGCGACCGTACTGCGGTACAAGAAAGTATAGT",
]
prediction = model.score(sequences)
print(prediction.scores)
```

`prediction.scores` has one row per sequence. A multi-task model has one column
per output, in the same order as `model.task_names`. Ensembles also return
between-member standard deviation in `prediction.uncertainties`.

For example, DeepSTARR returns two columns:

```python
model = load_model("deepstarr", device="cpu")
sequence = "ACGT" * 62 + "A"
prediction = model.score([sequence])

print(model.task_names)       # ('developmental', 'housekeeping')
print(prediction.scores)      # shape: (1, 2)
```

Aliases such as `optimus`, `dream`, `parm`, `kosuri`, and `tfbind8` are
also accepted. Canonical names are returned by `available_models()`.

## Variable length, RNA, and N padding

Every public sequence interface accepts DNA notation. RNA `U` is converted to
`T` internally. `N` is represented by an all-zero nucleotide vector and is
never emitted as an optimization candidate.

For a fixed-input model, set the tensor length and choose left or right padding:

```python
model = load_model(
    "optimus5prime",
    device="cpu",
    pad_to_length=50,
    padding_side="left",
)

prediction = model.score(["AUGGCUACGUAACCGGCUACGUAACCGGCUACGUAACCGG"])
```

Models whose registry length is shown as “up to” already apply their canonical
maximum length. Their `padding_side` can still be selected where it is
biologically meaningful. Inputs longer than the model maximum raise an error
instead of being silently truncated.

PARM is a special case: it adds the published `CAGTGAT` and `CACGACG` adapters
before deterministic left or right zero/N padding to 600 positions:

```python
model = load_model("parm_k562", padding_side="right")
prediction = model.score(["ACGTCGATGCTAGCTAGGCTA"])
```

## Models with additional inputs

Kosuri is a two-branch model. Use `score_pairs()` so promoter and RBS lengths
remain unambiguous:

```python
model = load_model("kosuri_composability")
prediction = model.score_pairs(
    promoters=["GGCGCGCCATTAATCATCCGGAATG"],
    rbs_sequences=["ATCTTAATCTAGCGCGGGAGTATTTCATATG"],
)

print(model.task_names)
print(prediction.scores)
```

The conditional Zhou L3 model requires a fixed source sublibrary for one
optimization run:

```python
model = load_model("zhou_l3_conditional", sublibrary="L3_s2")
prediction = model.score(["ACGTCGATGCTAGCTAGGCTAACGTACGT"])
```

Valid choices are `L3_s1`, `L3_s2`, `L3_s3`, and `L3_s4`.

## Gradient optimization with DREAM LegNet

The canonical DREAM construct contains a 17-bp left adapter that is discarded
by the published preprocessing. For scoring dataset rows, the default interface
still accepts the complete 110-bp construct. For gradient optimization, use the
93-bp variable-region interface so every optimized position affects activity:

```python
from bioseq_models import load_model
from sequence_optimization import SequenceSpace, fast_seqprop_optimize

model = load_model(
    "dream_legnet",
    variable_region_only=True,
    device="cpu",
)

result = fast_seqprop_optimize(
    model,
    SequenceSpace(length=93),
    num_candidates=4,
    budget=20,
    seed=7,
)
```

The differentiable transform inserts the fixed 57-bp context, changes channel
order from A/C/G/T to A/G/C/T, adds the constant fifth channel, and retains
forward/reverse-complement averaging. Calling `score_soft()` on the default
110-bp interface is also supported, but the first 17 positions correctly have
zero gradient.

## Cetnar paper LightGBM cascade

The Cetnar paper-style model needs the full engineered feature table for five
isoforms, not only a nucleotide sequence, so it is intentionally kept outside
the raw-sequence registry:

```python
import pandas as pd

from bioseq_models.tabular import CetnarLightGBM

features = pd.read_csv("cetnar_features.csv")
model = CetnarLightGBM.from_checkpoint_dir()

timepoint_predictions = model.predict_timepoints(features)
decay_rate = model.predict(features)
```

Use `cetnar_stability` when the available input is only a sequence.

## Optimize with `bioseq-optimization`

Loaded sequence models can be passed directly to any compatible optimizer:

```python
from bioseq_models import load_model
from sequence_optimization import SequenceSpace, genetic_algorithm_optimize

model = load_model("optimus5prime", device="cpu")
result = genetic_algorithm_optimize(
    model,
    SequenceSpace(length=50),
    num_candidates=16,
    budget=160,
    seed=7,
)

print(result.sequences[:5])
print(result.activities[:5])
```

[`examples/custom_cnn.py`](examples/custom_cnn.py) shows how to load a small
custom PyTorch CNN, wrap it in the same scorer interface, and optimize it.

## Verification

```bash
python -m pytest -q
python -m ruff check bioseq_models examples tests
```

The test suite loads every bundled model, checks task shapes and finite outputs,
tests padding and special inputs, validates the Cetnar LightGBM cascade, and
runs a model-to-optimizer integration test.
