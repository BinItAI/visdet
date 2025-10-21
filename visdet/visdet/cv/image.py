# ruff: noqa
"""
Re-export of viscv.image for dotted import support.

This module allows `from visdet.cv.image import X` to work properly.
"""

from viscv.image import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visdet.cv.image import __all__  # noqa: F401
except ImportError:
    pass
