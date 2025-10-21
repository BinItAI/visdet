# ruff: noqa
"""
Re-export of visengine.evaluator for dotted import support.

This module allows `from visdet.engine.evaluator import X` to work properly.
"""

from visengine.evaluator import *  # noqa: F401, F403

# Preserve the __all__ from upstream if it exists
try:
    from visdet.engine.evaluator import __all__  # noqa: F401
except ImportError:
    pass
