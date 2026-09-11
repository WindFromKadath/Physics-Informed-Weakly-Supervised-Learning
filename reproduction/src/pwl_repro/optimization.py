"""Backward-compatible alias for :mod:`pwl_repro.core.optimization`."""

import sys as _sys

from .core import optimization as _implementation

_sys.modules[__name__] = _implementation
