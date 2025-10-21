# ruff: noqa
"""
Re-export of viscv.ops for dotted import support.

This module allows `from visdet.cv.ops import X` to work properly.
"""

from viscv.ops import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from viscv.ops import __all__  # noqa: F401
except ImportError:
    pass
