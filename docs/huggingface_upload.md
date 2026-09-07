# Publishing weights on Hugging Face

Create a public **Model** repository named `xinHesysu/bioseq-models` and select
`Other` as its license. The repository model card contains the applicable
CodonBERT artifact-license metadata and notice.

Prepare the upload tree:

```bash
python scripts/prepare_huggingface_repo.py \
  --source /mnt/data/home/hexin/bioseq-models-weight-cache/0.6.0 \
  --output /mnt/data/home/hexin/bioseq-models-huggingface-v0.6.0
```

Install the Hugging Face CLI, log in with a write token, and upload the tree:

```bash
pip install -U huggingface_hub
hf auth login
HF_XET_HIGH_PERFORMANCE=1 hf upload \
  xinHesysu/bioseq-models \
  /mnt/data/home/hexin/bioseq-models-huggingface-v0.6.0 \
  . \
  --repo-type model
```

The upload command handles large files automatically and can be run again if
an interrupted upload needs to resume. The package is pinned to the immutable
`v0.6.0` repository tag.
