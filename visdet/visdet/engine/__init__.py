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

# Re-export all visengine functionality
from visdet.engine import *  # noqa: F401, F403

# NOTE: We don't eagerly import hooks here to avoid circular imports.
# The hooks package can still be accessed via explicit import:
#   from visdet.engine import hooks
# or:
#   from visdet.engine.hooks import DetVisualizationHook
#
# Uncomment this if needed (but it causes circular imports with visengine):
# from . import hooks
# __all__ = ["hooks"]
