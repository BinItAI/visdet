# ruff: noqa
"""
Re-export of visengine.infer for dotted import support.

This module allows `from visdet.engine.infer import X` to work properly.
"""

from visengine.infer import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visengine.infer import __all__  # noqa: F401
except ImportError:
    pass
