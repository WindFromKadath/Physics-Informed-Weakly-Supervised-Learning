"""Physics-informed weakly-supervised learning reproduction package."""

from .model import PWLRegressor
from .simulation import SimulationData, generate_simulation

__all__ = ["PWLRegressor", "SimulationData", "generate_simulation"]
__version__ = "0.1.0"

