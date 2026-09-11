"""Backward-compatible alias for :mod:`pwl_repro.scenarios.case_b`."""

import sys as _sys

from .scenarios import case_b as _implementation

_sys.modules[__name__] = _implementation
