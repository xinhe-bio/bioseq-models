"""Optimization-ready adapters for bundled model checkpoints."""

from .deepstarr import DeepSTARR
from .dream_legnet import DreamLegNet
from .optimus5prime import Optimus5Prime
from .zhao_ptrc import ZhaoPtrc
from .zhou_l3 import ZhouL3

__all__ = ["Optimus5Prime", "DeepSTARR", "DreamLegNet", "ZhaoPtrc", "ZhouL3"]
