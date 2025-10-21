# ruff: noqa
"""
Re-export of visengine.runner for dotted import support.

This module allows `from visdet.engine.runner import X` to work properly.
"""

from visengine.runner import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visdet.engine.runner import __all__  # noqa: F401
except ImportError:
    pass
