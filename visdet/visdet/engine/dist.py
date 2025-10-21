# ruff: noqa
"""
Re-export of visengine.dist for dotted import support.

This module allows `from visdet.engine.dist import X` to work properly.
"""

from visengine.dist import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visengine.dist import __all__  # noqa: F401
except ImportError:
    pass
