"""Backward-compatible alias for :mod:`pwl_repro.scenarios.simulation`."""

import sys as _sys

from .scenarios import simulation as _implementation

_sys.modules[__name__] = _implementation
