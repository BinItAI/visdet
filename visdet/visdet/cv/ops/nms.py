# ruff: noqa
"""
Re-export of viscv.ops.nms for dotted import support.

This module allows `from visdet.cv.ops.nms import X` to work properly.
"""

from viscv.ops.nms import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from viscv.ops.nms import __all__  # noqa: F401
except ImportError:
    pass
