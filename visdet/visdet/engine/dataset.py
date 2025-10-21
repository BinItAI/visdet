# ruff: noqa
"""
Re-export of visengine.dataset for dotted import support.

This module allows `from visdet.engine.dataset import X` to work properly.
"""

from visengine.dataset import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visdet.engine.dataset import __all__  # noqa: F401
except ImportError:
    pass
