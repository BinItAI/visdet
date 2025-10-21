# ruff: noqa
"""
Re-export of visengine.runner.checkpoint for dotted import support.

This module allows `from visdet.engine.runner.checkpoint import X` to work properly.
"""

from visengine.runner.checkpoint import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visengine.runner.checkpoint import __all__  # noqa: F401
except ImportError:
    pass
