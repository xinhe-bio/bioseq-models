# Saluki: annotated mRNA half-life prediction

Saluki is the model of [Agarwal and Kelley, Genome Biology (2022)](https://doi.org/10.1186/s13059-022-02811-x). It predicts a species-specific consensus half-life score derived from PC1 of aggregated half-life measurements. **Higher scores predict greater stability. The score is not a half-life in hours.** Human and mouse scores have separately defined target scales.

The adapter uses PyTorch conversions of the original Keras weights. The published defaults load with the base package. Install the optional HDF5 reader to convert other original checkpoints:

```bash
pip install 'bioseq-models[saluki]'
```

For a local checkout, use `python -m pip install -e '.[saluki]'`.

## Annotated transcript records

```python
from bioseq_models import load_model

model = load_model('saluki', device='cuda')      # human
mouse_model = load_model('saluki_mouse', device='cuda')

transcripts = [{
    'sequence': full_mrna,                     # spliced RNA, 5prime to 3prime
    'cds_start': cds_start,                    # zero-based
    'cds_end': cds_end,                        # end-exclusive
    'exon_lengths': exon_lengths,              # in transcript order; sum = len(full_mrna)
}]
scores = model.score_transcripts(transcripts)  # shape [N, 1]
```

`full_mrna` includes the 5′UTR, CDS and 3′UTR. A/C/G/T or A/C/G/U is accepted. Coordinates describe the full input before cropping; the CDS must contain a whole number of codons. For a synthetic single-exon transcript, explicitly use `exon_lengths=[len(full_mrna)]`. The model was trained on full coding transcripts, so isolated UTR predictions are not a calibrated substitute for full-transcript stability.

The sixth channel follows the released data encoder: it marks the last nucleotide of **every** exon, including the final exon. This terminal mark is retained for compatibility with the original model. Alternatively, `splice_sites=[...]` supplies the exact zero-based positions to set in this raw channel. Supply one annotation form. An explicit empty list yields a zero exon-end channel. Missing CDS or exon annotations raise an actionable error; sequence case is not used to infer a CDS.

## Shared annotations and differentiable design

For sequences with fixed transcript coordinates and the same exon structure:

```python
model = load_model(
    'saluki', device='cuda',
    cds_start=cds_start, cds_end=cds_end,
    exon_lengths=exon_lengths,
)
scores = model.score([full_mrna])

# probabilities is a torch Tensor: [N, full_transcript_length, 4], A/C/G/T order.
predicted, uncertainty = model.score_soft(probabilities)
predicted.sum().backward()
```

`score_soft` keeps the supplied annotation tracks fixed and differentiates through nucleotide probabilities. It is available when annotations are bound at construction. This is appropriate for substitutions or fixed-length design with unchanged CDS/exon boundaries; annotations must be updated for length-changing edits or altered reading frames. Gradient computation uses the differentiable PyTorch GRU path, while ordinary prediction can use cuDNN. No reverse-complement averaging is performed.

## Six-channel encoding

Inputs comprise A/C/G/T, the first base of each CDS codon and the exon-end track, in that order. Complete transcripts shorter than 12,288 nt are left-justified and padded with all-zero channels at the 3′ end. Longer transcripts retain their last 12,288 nt; annotation tracks are cropped together with nucleotides, preserving the original frame. Unknown N nucleotides have zero nucleotide channels.

Already encoded arrays can be passed to `model.score_encoded(array)` with shape `[N,L,6]`. This route supports the original TFRecord channels without reannotating them. It applies the same padding/cropping policy. The PyTorch network's internal layout is `[N,6,12288]`.

## Official weights and ensembles

The default is the released **f0_c0 single model** for the selected species: human `model0_best.h5` or mouse `model1_best.h5`. The human alias `saluki_human` also resolves to `saluki`. To average more official models, specify folds and crosses explicitly:

```python
ensemble = load_model(
    'saluki', device='cuda', folds=range(10), crosses=range(5),
)
print(ensemble.ensemble_size)  # 50
```

The [Saluki weights on Hugging Face](https://huggingface.co/xinHesysu/bioseq-models/tree/0e6bae61a6452b0dc8b356e576da88ba14f1e9db/saluki) include the converted human f0_c0, human f0_c1 and mouse f0_c0 checkpoints. The loader checks their SHA256 values. Other selected folds are converted from the original release on demand. The default native HDF5 files use a pinned mirror whose sizes and CRCs match the author's [released data archive](https://doi.org/10.5281/zenodo.6326409); their SHA256 hashes are pinned as well. Original archive members are fetched using byte ranges. All original files are size/CRC checked, converted locally and cached. Conversion retains normalization epsilons, the backward GRU, gate order, both GRU bias terms and the species-specific output head. Optimizer state and training augmentation are omitted from inference.

```python
from bioseq_models import download_model
paths = download_model('saluki')  # converted default human checkpoint
```

`BIOSEQ_MODELS_CACHE` or `cache_dir=...` changes the cache root. To use local converted files, set `BIOSEQ_MODELS_WEIGHTS_DIR` or `weights_dir=...` with files named `saluki/saluki_human_f0_c0.pt` or `saluki/saluki_mouse_f0_c0.pt`. `checkpoint='/path/to/file.pt'` selects one file directly. An original HDF5 file can also be supplied explicitly, or converted with:

```bash
python scripts/convert_saluki_checkpoint.py model0_best.h5 saluki_human_f0_c0.pt --species human
```

The package source manifest lists all 100 human/mouse fold/cross entries, the released archive DOI/MD5 and pinned default source hashes. The full 50-model ensemble's published performance is not attributed to the default single-model configuration.

## Numerical verification

The conversion was checked manually against the original HDF5 inference graph
on 32 released held-out human transcripts and 32 released mouse transcripts.
Six-channel encodings matched the original record arrays exactly. Maximum
absolute prediction differences were 1.19×10⁻⁶ (human) and 6.56×10⁻⁷ (mouse).
CPU/GPU agreement, hard/soft agreement and finite nonzero input gradients were
also verified. A two-member official f0/c0+c1 human ensemble matched the mean of
its individual members within 7.45×10⁻⁹. These checks establish conversion
agreement; they do not estimate the published ensemble's predictive accuracy.

An executable example accepts annotated JSON records:

```bash
python examples/saluki.py transcripts.json --species human --device cuda --output saluki_scores.csv
```

Source receipts and the manual comparison reports are in
`provenance/saluki/`. No automated software test suite was run.
