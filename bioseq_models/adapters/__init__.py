"""Inference adapters for bundled model checkpoints."""

from .agarwal import AgarwalHepG2, AgarwalK562, AgarwalWTC11
from .cetnar import CetnarStability
from .cds_stability import CDSStability
from .deepstarr import DeepSTARR
from .deepstarr_tissue import (
    DeepSTARREmbryoBrainVNCActivity,
    DeepSTARRMouseHeartAccessibility,
    DeepSTARRMouseHeartActivity,
)
from .dream_legnet import DreamLegNet
from .malinois import Malinois
from .optimus5prime import Optimus5Prime
from .parm import PARMK562
from .utr_code import UTRCode

__all__ = [
    "AgarwalHepG2",
    "AgarwalK562",
    "AgarwalWTC11",
    "CetnarStability",
    "CDSStability",
    "DeepSTARR",
    "DeepSTARREmbryoBrainVNCActivity",
    "DeepSTARRMouseHeartAccessibility",
    "DeepSTARRMouseHeartActivity",
    "DreamLegNet",
    "Malinois",
    "Optimus5Prime",
    "PARMK562",
    "UTRCode",
]
