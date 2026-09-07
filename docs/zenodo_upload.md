# Publishing the weight bundle

Upload this file as one Zenodo record file:

```text
/mnt/data/home/hexin/bioseq-models-weights-v0.5.0.zip
```

The adjacent `.manifest.json` file records the bundle contents and can also be
attached to the record, but the Python package only downloads the ZIP.

After publishing the record, copy the permanent file-download URL into
`ZENODO_URL` in `bioseq_models/resources.py`. Test the published file with:

```bash
BIOSEQ_MODELS_CACHE=/tmp/bioseq-models-clean-cache \
python -c "from bioseq_models import load_model; print(load_model('optimus5prime').score(['A' * 50]))"
```

To rebuild the same layout for a later release, run:

```bash
python scripts/build_weight_bundle.py \
  --existing-checkpoints /path/to/current/checkpoints \
  --utr-code-checkpoint /path/to/utr-code/best.epoch \
  --hek293t-profile /path/to/HEK293T.txt \
  --cds-stability-checkpoint /path/to/best_model.pt \
  --codonbert-directory /path/to/codonbert/checkpoint \
  --output /path/to/bioseq-models-weights-v0.5.0.zip
```
