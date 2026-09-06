"""Optimization-ready adapters for bundled model checkpoints."""

from .agarwal import AgarwalHepG2, AgarwalK562, AgarwalWTC11
from .cetnar import CetnarStability
from .deepstarr import DeepSTARR
from .dream_legnet import DreamLegNet
from .kosuri import KosuriComposability
from .optimus5prime import Optimus5Prime
from .parm import PARMK562
from .tfbind8 import TFBind8Weak
from .weak import DreamWeak, UTR5Weak
from .zhao_ptrc import ZhaoPtrc, ZhaoPtrcXGBoost
from .zhou_l3 import ZhouL3, ZhouL3Conditional

__all__ = [
    "AgarwalHepG2",
    "AgarwalK562",
    "AgarwalWTC11",
    "CetnarStability",
    "DeepSTARR",
    "DreamLegNet",
    "DreamWeak",
    "KosuriComposability",
    "Optimus5Prime",
    "PARMK562",
    "TFBind8Weak",
    "UTR5Weak",
    "ZhaoPtrc",
    "ZhaoPtrcXGBoost",
    "ZhouL3",
    "ZhouL3Conditional",
]
