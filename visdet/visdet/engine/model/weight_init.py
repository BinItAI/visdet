# ruff: noqa
"""
Re-export of visengine.model.weight_init for dotted import support.

This module allows `from visdet.engine.model.weight_init import X` to work properly.
"""

from visengine.model.weight_init import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visengine.model.weight_init import __all__  # noqa: F401
except ImportError:
    pass
