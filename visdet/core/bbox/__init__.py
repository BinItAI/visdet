# Copyright (c) OpenMMLab. All rights reserved.
"""Bbox utilities - backward compatibility for visdet.core.bbox namespace."""

# Import real implementations
# Assigners, samplers, and coders are in task_modules
from visdet.models.task_modules.builder import (
    build_assigner,
    build_bbox_coder,
    build_sampler,
)
from visdet.structures.bbox.transforms import bbox2roi, roi2bbox

__all__ = [
    "bbox2roi",
    "roi2bbox",
    "build_assigner",
    "build_sampler",
    "build_bbox_coder",
]
