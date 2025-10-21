# ruff: noqa
"""
Re-export of visengine.visualization for dotted import support.

This module allows `from visdet.engine.visualization import X` to work properly.
"""

from visengine.visualization import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visdet.engine.visualization import __all__  # noqa: F401
except ImportError:
    pass
