"""Inference adapters for bundled model checkpoints."""

from .agarwal import AgarwalHepG2, AgarwalK562, AgarwalWTC11
from .cetnar import CetnarStability
from .deepstarr import DeepSTARR
from .dream_legnet import DreamLegNet
from .optimus5prime import Optimus5Prime
from .parm import PARMK562

__all__ = [
    "AgarwalHepG2",
    "AgarwalK562",
    "AgarwalWTC11",
    "CetnarStability",
    "DeepSTARR",
    "DreamLegNet",
    "Optimus5Prime",
    "PARMK562",
]
