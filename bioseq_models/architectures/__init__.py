"""Neural architectures required by the bundled checkpoints."""

from .agarwal import MPRALegNet
from .cetnar import CetnarStabilityCNNBiLSTM
from .deepstarr import DeepSTARRPaper
from .dream_legnet import LegNetAutosome
from .optimus5prime import Optimus5Prime
from .parm import PARMExactCNN

__all__ = [
    "MPRALegNet",
    "CetnarStabilityCNNBiLSTM",
    "DeepSTARRPaper",
    "LegNetAutosome",
    "Optimus5Prime",
    "PARMExactCNN",
]
