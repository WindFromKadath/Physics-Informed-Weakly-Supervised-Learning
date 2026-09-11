"""Reusable PWL algorithm building blocks.

This package contains scenario-independent model, feature-library, and
optimization code.  Scenario adapters live in :mod:`pwl_repro.scenarios`.
"""

from .features import FeatureSpec, PWLFeatureLibrary, get_feature_spec
from .model import PWLRegressor, calibrate_theta
from .optimization import ConsensusResult, consensus_admm

__all__ = [
    "ConsensusResult",
    "FeatureSpec",
    "PWLFeatureLibrary",
    "PWLRegressor",
    "calibrate_theta",
    "consensus_admm",
    "get_feature_spec",
]
