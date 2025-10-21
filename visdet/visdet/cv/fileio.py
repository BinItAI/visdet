# ruff: noqa
"""
Re-export of viscv.fileio for dotted import support.

This module allows `from visdet.cv.fileio import X` or `import visdet.cv.fileio` to work properly.
"""

from viscv.fileio import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from viscv.fileio import __all__  # noqa: F401
except ImportError:
    pass
