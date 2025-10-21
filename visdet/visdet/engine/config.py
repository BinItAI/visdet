# ruff: noqa
"""
Re-export of visengine.config for dotted import support.

This module allows `from visdet.engine.config import X` to work properly.
"""

from visengine.config import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visengine.config import __all__  # noqa: F401
except ImportError:
    pass
