# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for visdet.core.bbox.assigners."""

from visdet.models.task_modules.assigners import (
    AssignResult,
    BaseAssigner,
    MaxIoUAssigner,
)
from visdet.models.task_modules.builder import build_assigner

__all__ = [
    "AssignResult",
    "BaseAssigner",
    "MaxIoUAssigner",
    "build_assigner",
]
