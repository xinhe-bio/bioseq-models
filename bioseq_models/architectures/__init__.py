"""Neural architectures required by the bundled checkpoints."""

from .agarwal import MPRALegNet
from .cetnar import CetnarStabilityCNNBiLSTM
from .deepstarr import DeepSTARRPaper
from .dream_legnet import LegNetAutosome
from .kosuri import KosuriTwoBranchCNN
from .optimus5prime import Optimus5Prime
from .parm import PARMExactCNN
from .tfbind8 import TFBindCNN
from .weakseqcnn import WeakSeqCNN
from .zhao_ptrc import PtrcPromoterCNN
from .zhou_l3 import L3ConditionalCNN, L3PositionAwareCNN

__all__ = [
    "MPRALegNet",
    "CetnarStabilityCNNBiLSTM",
    "DeepSTARRPaper",
    "LegNetAutosome",
    "KosuriTwoBranchCNN",
    "Optimus5Prime",
    "PARMExactCNN",
    "TFBindCNN",
    "WeakSeqCNN",
    "PtrcPromoterCNN",
    "L3ConditionalCNN",
    "L3PositionAwareCNN",
]
