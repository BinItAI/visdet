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

    # Legacy way (still works but discouraged inside visdet)
    import visengine
    from visengine import Config, Runner

All visengine functionality is re-exported here for backwards compatibility
and namespace consistency within the visdet package.
"""

# 1. Re-export all top-level symbols from the original `visengine` library.
from visengine import *  # noqa: F401, F403

# 2. Explicitly import submodules using relative paths to make them accessible
#    under the `visdet.engine` namespace (e.g., `visdet.engine.runner`).
from . import (
    config,
    dataset,
    dist,
    evaluator,
    fileio,
    infer,
    logging,
    model,
    registry,
    runner,
    structures,
    utils,
    visualization,
)

# NOTE: We don't eagerly import `hooks` here to avoid the circular import
# issue identified during development. It remains accessible via direct import:
# `from visdet.engine import hooks` or `from visdet.engine.hooks import ...`

# 3. Construct `__all__` to control `from visdet.engine import *` behavior.
try:
    from visengine import __all__ as visengine_all
except ImportError:
    visengine_all = []

# Expose re-exported symbols and all submodules except 'hooks'.
__all__ = list(visengine_all) + [
    "config",
    "dataset",
    "dist",
    "evaluator",
    "fileio",
    "infer",
    "logging",
    "model",
    "registry",
    "runner",
    "structures",
    "utils",
    "visualization",
]
