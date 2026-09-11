---
license: other
license_name: codonbert-artifact-license
license_link: https://github.com/Sanofi-Public/CodonBERT/blob/master/ARTIFACT_LICENSE.md
library_name: bioseq-models
tags:
  - biology
  - genomics
  - dna
  - rna
  - sequence-design
---

# bioseq-models weights

Pretrained weights consumed by the
[`bioseq-models`](https://github.com/UTR-CODE/bioseq-models) Python package.
The package downloads only the files required by the model selected by the
user.

The repository contains weights for Optimus 5-Prime, DeepSTARR, DeepAdapt heat
shock, DeepSTARR-Mouse, DeepSTARR-Embryo, Malinois, DREAM LegNet, Cetnar stability,
PARM K562, Agarwal lentiMPRA, UTR-CODE, and the multi-species CDS stability
model with CodonBERT.

`deepadapt_heatshock/model.pt` contains the WI-38 senescent heat-shock model,
its two normal and two treatment RNA environments, and returns delta, normal,
and treatment log2 translation efficiency through the Python package.

DeepSTARR-Mouse includes separate activity and accessibility models for heart,
limb, midbrain, forebrain, hindbrain, and neural tube. Each package resource
uses all six official checkpoints from three cross-validation folds and two
replicates; requesting one resource downloads only its six checkpoint files.
DeepSTARR-Embryo includes six accessibility resources and five activity
resources. Each resource uses all author-released checkpoints with finite
parameters; only the selected resource is downloaded.

Most checkpoints are project-trained reproductions and are not represented as
author-released weights. See the package's `MODEL_PROVENANCE.md` for details.

## License

The CodonBERT files under `cds_stability/codonbert/` are restricted to academic
research and non-commercial use. See `THIRD_PARTY_NOTICES.md` and the linked
upstream artifact license. The complete repository is distributed under these
restrictive terms so that a single unambiguous license applies to the weight
collection.
