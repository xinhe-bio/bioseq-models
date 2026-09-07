"""Neural architectures required by the bundled checkpoints."""

from .agarwal import MPRALegNet
from .cetnar import CetnarStabilityCNNBiLSTM
from .cds_stability import CDSStabilityNetwork
from .deepstarr import DeepSTARRPaper
from .dream_legnet import LegNetAutosome
from .optimus5prime import Optimus5Prime
from .parm import PARMExactCNN
from .utr_code import UTRCodeNetwork

__all__ = [
    "MPRALegNet",
    "CetnarStabilityCNNBiLSTM",
    "CDSStabilityNetwork",
    "DeepSTARRPaper",
    "LegNetAutosome",
    "Optimus5Prime",
    "PARMExactCNN",
    "UTRCodeNetwork",
]
