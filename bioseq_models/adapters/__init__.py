"""Inference adapters for bundled model checkpoints."""

from .agarwal import AgarwalHepG2, AgarwalK562, AgarwalWTC11
from .cds_stability import CDSStability
from .cetnar import CetnarStability
from .deepadapt import DeepAdaptHeatShock
from .deepstarr import DeepSTARR
from .deepstarr_tissue import (
    DeepSTARREmbryoBrainAccessibility,
    DeepSTARREmbryoBrainSpecificActivity,
    DeepSTARREmbryoBrainVNCActivity,
    DeepSTARREmbryoCNSAccessibility,
    DeepSTARREmbryoEpidermisAccessibility,
    DeepSTARREmbryoEpidermisActivity,
    DeepSTARREmbryoGutActivity,
    DeepSTARREmbryoMidgutAccessibility,
    DeepSTARREmbryoMuscleActivity,
    DeepSTARREmbryoSomaticMuscleAccessibility,
    DeepSTARREmbryoVisceralMuscleAccessibility,
    DeepSTARRMouseForebrainAccessibility,
    DeepSTARRMouseForebrainActivity,
    DeepSTARRMouseHeartAccessibility,
    DeepSTARRMouseHeartActivity,
    DeepSTARRMouseHindbrainAccessibility,
    DeepSTARRMouseHindbrainActivity,
    DeepSTARRMouseLimbAccessibility,
    DeepSTARRMouseLimbActivity,
    DeepSTARRMouseMidbrainAccessibility,
    DeepSTARRMouseMidbrainActivity,
    DeepSTARRMouseNeuralTubeAccessibility,
    DeepSTARRMouseNeuralTubeActivity,
)
from .dream_legnet import DreamLegNet
from .malinois import Malinois
from .optimus5prime import Optimus5Prime
from .parm import PARMK562
from .utr_code import UTRCode

__all__ = [
    "PARMK562",
    "AgarwalHepG2",
    "AgarwalK562",
    "AgarwalWTC11",
    "CDSStability",
    "CetnarStability",
    "DeepAdaptHeatShock",
    "DeepSTARR",
    "DeepSTARREmbryoBrainAccessibility",
    "DeepSTARREmbryoBrainSpecificActivity",
    "DeepSTARREmbryoBrainVNCActivity",
    "DeepSTARREmbryoCNSAccessibility",
    "DeepSTARREmbryoEpidermisAccessibility",
    "DeepSTARREmbryoEpidermisActivity",
    "DeepSTARREmbryoGutActivity",
    "DeepSTARREmbryoMidgutAccessibility",
    "DeepSTARREmbryoMuscleActivity",
    "DeepSTARREmbryoSomaticMuscleAccessibility",
    "DeepSTARREmbryoVisceralMuscleAccessibility",
    "DeepSTARRMouseForebrainAccessibility",
    "DeepSTARRMouseForebrainActivity",
    "DeepSTARRMouseHeartAccessibility",
    "DeepSTARRMouseHeartActivity",
    "DeepSTARRMouseHindbrainAccessibility",
    "DeepSTARRMouseHindbrainActivity",
    "DeepSTARRMouseLimbAccessibility",
    "DeepSTARRMouseLimbActivity",
    "DeepSTARRMouseMidbrainAccessibility",
    "DeepSTARRMouseMidbrainActivity",
    "DeepSTARRMouseNeuralTubeAccessibility",
    "DeepSTARRMouseNeuralTubeActivity",
    "DreamLegNet",
    "Malinois",
    "Optimus5Prime",
    "UTRCode",
]
