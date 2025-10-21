# ruff: noqa
"""
Re-export of visengine.fileio for dotted import support.

This module allows `from visdet.engine.fileio import X` to work properly.
"""

from visengine.fileio import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visengine.fileio import __all__  # noqa: F401
except ImportError:
    pass
