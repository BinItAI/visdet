# ruff: noqa
# Copyright (c) OpenMMLab. All rights reserved.
"""
Engine utilities for training and inference.

This module includes:
1. visdet-specific hooks
2. Training infrastructure and model management

Usage:
    from visdet import engine
    from visdet.engine import Config, Runner
    from visdet.engine import hooks
"""

# Import submodules to make them accessible under the `visdet.engine` namespace
# (e.g., `visdet.engine.runner`)
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
)  # noqa: F401

# NOTE: We don't eagerly import `hooks` here to avoid the circular import
# issue identified during development. It remains accessible via direct import:
# `from visdet.engine import hooks` or `from visdet.engine.hooks import ...`

# Export all submodules.
__all__ = [
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
