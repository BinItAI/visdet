# ruff: noqa
"""
Re-export of visengine.logging for dotted import support.

This module allows `from visdet.engine.logging import X` to work properly.
"""

from visengine.logging import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visdet.engine.logging import __all__  # noqa: F401
except ImportError:
    pass
