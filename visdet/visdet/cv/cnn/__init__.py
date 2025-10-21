# ruff: noqa
"""
Re-export of viscv.cnn for dotted import support.

This module allows `from visdet.cv.cnn import X` to work properly.
"""

from viscv.cnn import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visdet.cv.cnn import __all__  # noqa: F401
except ImportError:
    pass
