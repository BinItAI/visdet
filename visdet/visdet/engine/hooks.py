# ruff: noqa
"""
Re-export of visengine.hooks for dotted import support.

This module allows `from visdet.engine.hooks import X` to work properly.
"""

from visengine.hooks import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visengine.hooks import __all__  # noqa: F401
except ImportError:
    pass
