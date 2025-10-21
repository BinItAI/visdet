# ruff: noqa
"""
Re-export of visengine.utils for dotted import support.

This module allows `from visdet.engine.utils import X` to work properly.
"""

from visengine.utils import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visengine.utils import __all__  # noqa: F401
except ImportError:
    pass
