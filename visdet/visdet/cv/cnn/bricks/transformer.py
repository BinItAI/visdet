# ruff: noqa
"""
Re-export of viscv.cnn.bricks.transformer for dotted import support.

This module allows `from visdet.cv.cnn.bricks.transformer import X` to work properly.
"""

from viscv.cnn.bricks.transformer import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from viscv.cnn.bricks.transformer import __all__  # noqa: F401
except ImportError:
    pass
