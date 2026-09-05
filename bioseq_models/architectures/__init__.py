"""Neural architectures required by the bundled checkpoints."""

from .deepstarr import DeepSTARRPaper
from .dream_legnet import LegNetAutosome
from .optimus5prime import Optimus5Prime
from .zhao_ptrc import PtrcPromoterCNN
from .zhou_l3 import L3PositionAwareCNN

__all__ = [
    "DeepSTARRPaper",
    "LegNetAutosome",
    "Optimus5Prime",
    "PtrcPromoterCNN",
    "L3PositionAwareCNN",
]
