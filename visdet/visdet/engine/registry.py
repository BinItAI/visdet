# ruff: noqa
"""
Re-export of visengine.registry for dotted import support.

This module allows `from visdet.engine.registry import X` to work properly.
"""

from visengine.registry import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visengine.registry import __all__  # noqa: F401
except ImportError:
    pass
