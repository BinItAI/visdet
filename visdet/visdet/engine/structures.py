# ruff: noqa
"""
Re-export of visengine.structures for dotted import support.

This module allows `from visdet.engine.structures import X` to work properly.
"""

from visengine.structures import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visdet.engine.structures import __all__  # noqa: F401
except ImportError:
    pass
