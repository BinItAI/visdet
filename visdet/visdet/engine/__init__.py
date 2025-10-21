# ruff: noqa
# Copyright (c) OpenMMLab. All rights reserved.
"""
Engine utilities for training and inference.

This module includes:
1. visdet-specific hooks (from . import hooks)
2. Re-exports of visengine functionality for visdet.engine access

Usage:
    # New preferred way
    from visdet import engine
    from visdet.engine import Config, Runner

    # Legacy way (still works)
    import visengine
    from visengine import Config, Runner

All visengine functionality is re-exported here for backwards compatibility
and namespace consistency within the visdet package.
"""

from . import hooks

# Re-export all visengine functionality
from visengine import *  # noqa: F401, F403

# Keep hooks in __all__ for visdet-specific functionality
__all__ = ["hooks"]
