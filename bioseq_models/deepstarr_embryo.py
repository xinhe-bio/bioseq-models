"""Released and usable checkpoint inventory for DeepSTARR-Embryo."""

from __future__ import annotations

UNRELEASED_CHECKPOINTS = {
    ("accessibility", "midgut", 8, 3),
    ("accessibility", "visceral_muscle", 9, 3),
    ("activity", "gut", 8, 3),
    ("activity", "muscle", 9, 3),
}

# These author-released HDF5 files contain non-finite parameter values and also
# produce NaN predictions in the original Keras model.
NONFINITE_CHECKPOINTS = {
    ("accessibility", "brain", 1, 2),
    ("accessibility", "brain", 2, 2),
    ("accessibility", "cns", 2, 1),
    ("accessibility", "cns", 2, 3),
    ("accessibility", "cns", 4, 1),
    ("accessibility", "cns", 5, 3),
    ("accessibility", "cns", 6, 1),
    ("accessibility", "cns", 9, 3),
    ("accessibility", "cns", 10, 1),
    ("accessibility", "cns", 10, 3),
    ("accessibility", "epidermis", 2, 2),
    ("accessibility", "epidermis", 4, 1),
    ("accessibility", "epidermis", 5, 3),
    ("accessibility", "epidermis", 7, 1),
    ("accessibility", "epidermis", 9, 3),
    ("accessibility", "midgut", 1, 1),
    ("accessibility", "midgut", 2, 2),
    ("accessibility", "midgut", 5, 1),
    ("accessibility", "midgut", 6, 2),
    ("accessibility", "midgut", 9, 1),
    ("accessibility", "midgut", 10, 1),
    ("accessibility", "somatic_muscle", 1, 2),
    ("accessibility", "somatic_muscle", 1, 3),
    ("accessibility", "somatic_muscle", 2, 2),
    ("accessibility", "somatic_muscle", 5, 1),
    ("activity", "brain_specific", 1, 2),
    ("activity", "brain_specific", 2, 2),
    ("activity", "brain_specific", 5, 2),
    ("activity", "brain_vnc", 2, 1),
    ("activity", "brain_vnc", 2, 3),
    ("activity", "brain_vnc", 4, 1),
    ("activity", "brain_vnc", 5, 3),
    ("activity", "brain_vnc", 6, 1),
    ("activity", "brain_vnc", 9, 3),
    ("activity", "brain_vnc", 10, 1),
    ("activity", "brain_vnc", 10, 3),
    ("activity", "epidermis", 2, 2),
    ("activity", "epidermis", 4, 1),
    ("activity", "epidermis", 5, 1),
    ("activity", "epidermis", 5, 2),
    ("activity", "epidermis", 5, 3),
    ("activity", "epidermis", 7, 1),
    ("activity", "epidermis", 9, 3),
    ("activity", "gut", 1, 1),
    ("activity", "gut", 2, 2),
    ("activity", "gut", 5, 1),
    ("activity", "gut", 6, 2),
    ("activity", "gut", 9, 1),
    ("activity", "gut", 10, 1),
    ("activity", "muscle", 7, 2),
}

EXCLUDED_CHECKPOINTS = UNRELEASED_CHECKPOINTS | NONFINITE_CHECKPOINTS


def checkpoint_files(tissue: str, output_kind: str) -> tuple[str, ...]:
    """Return every released checkpoint with finite parameters for one task."""
    return tuple(
        f"checkpoints/deepstarr_embryo_{tissue}_{output_kind}_fold{fold:02d}_rep{rep}.pt"
        for fold in range(1, 11)
        for rep in range(1, 4)
        if (output_kind, tissue, fold, rep) not in EXCLUDED_CHECKPOINTS
    )
