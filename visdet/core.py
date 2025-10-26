# Copyright (c) OpenMMLab. All rights reserved.
"""
Backward compatibility shim for visdet.core module.

This module provides imports from the old visdet.core namespace mapped to their
new locations in the flattened package structure.

Maps old imports to new locations:
  - visdet.core.bbox.* -> visdet.structures.bbox.*
  - visdet.core.mask.* -> visdet.structures.mask.*
  - visdet.core.hook.* -> visdet.engine.hooks.*
  - visdet.core evaluators -> visdet.evaluation.*
"""

# Assigner and sampler builders
from visdet.models.task_modules.builder import (
    build_assigner,
    build_bbox_coder,
    build_sampler,
)

# Bbox transforms
from visdet.structures.bbox.transforms import bbox2roi, roi2bbox

# Mask utilities
from visdet.structures.mask import BitmapMasks, PolygonMasks

__all__ = [
    "build_assigner",
    "build_bbox_coder",
    "build_sampler",
    "BitmapMasks",
    "PolygonMasks",
    "bbox2roi",
    "roi2bbox",
]
