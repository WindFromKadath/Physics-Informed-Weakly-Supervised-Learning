"""Backward-compatible alias for :mod:`pwl_repro.core.model`.

New code should import from ``pwl_repro.core``.  The module alias preserves
existing imports and monkey-patching behavior used by diagnostic scripts.
"""

import sys as _sys

from .core import model as _implementation

_sys.modules[__name__] = _implementation
