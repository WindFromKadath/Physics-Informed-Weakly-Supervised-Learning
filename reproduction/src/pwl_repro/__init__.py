"""Physics-informed weakly-supervised learning reproduction package."""

# Register the built-in simulation feature library before exposing the model.
from .scenarios import simulation_features as _simulation_features
from .core.model import PWLRegressor
from .scenarios.simulation import SimulationData, generate_simulation

__all__ = [
    "PWLRegressor",
    "SimulationData",
    "generate_simulation",
]
__version__ = "0.1.0"
