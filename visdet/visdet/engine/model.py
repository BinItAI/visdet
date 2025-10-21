# ruff: noqa
"""
Re-export of visengine.model for dotted import support.

This module allows `from visdet.engine.model import X` to work properly.
"""

from visengine.model import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visengine.model import __all__  # noqa: F401
except ImportError:
    pass
