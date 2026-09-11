"""Backward-compatible alias for :mod:`pwl_repro.core.features`."""

import sys as _sys

from .core import features as _implementation

_sys.modules[__name__] = _implementation
