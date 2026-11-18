"""Neural architectures required by the bundled checkpoints."""

from .aparent import APARENTNetwork
from .agarwal import MPRALegNet
from .cds_stability import CDSStabilityNetwork
from .cetnar import CetnarStabilityCNNBiLSTM
from .deepadapt import DeepAdaptNetwork
from .deepstarr import DeepSTARRPaper
from .deepstarr_tissue import (
    DeepSTARREmbryoNetwork,
    DeepSTARRMouseNetwork,
    DeepSTARRTissueNetwork,
)
from .dream_legnet import LegNetAutosome
from .malinois import MalinoisNetwork
from .optimus5prime import Optimus5Prime
from .parm import PARMExactCNN
from .saluki import SalukiNetwork
from .utr_code import UTRCodeNetwork

__all__ = [
    "APARENTNetwork",
    "CDSStabilityNetwork",
    "CetnarStabilityCNNBiLSTM",
    "DeepAdaptNetwork",
    "DeepSTARREmbryoNetwork",
    "DeepSTARRMouseNetwork",
    "DeepSTARRPaper",
    "DeepSTARRTissueNetwork",
    "LegNetAutosome",
    "MPRALegNet",
    "MalinoisNetwork",
    "Optimus5Prime",
    "PARMExactCNN",
    "SalukiNetwork",
    "UTRCodeNetwork",
]
