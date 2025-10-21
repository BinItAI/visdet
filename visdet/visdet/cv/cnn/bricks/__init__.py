# ruff: noqa
"""
Re-export of viscv.cnn.bricks for dotted import support.

This module allows `from visdet.cv.cnn.bricks import X` to work properly.
"""

from viscv.cnn.bricks import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from viscv.cnn.bricks import __all__  # noqa: F401
except ImportError:
    pass
