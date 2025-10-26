# Copyright (c) OpenMMLab. All rights reserved.
"""
Backward compatibility shim for visdet.core module.

This package provides imports from the old visdet.core namespace mapped to their
new locations in the flattened package structure.

Old structure:
  - visdet.core.anchor.* -> visdet.models.task_modules.prior_generators.*
  - visdet.core.bbox.* -> visdet.structures.bbox.* + task_modules.*
  - visdet.core.mask.* -> visdet.structures.mask.*
  - visdet.core.hook.* -> visdet.engine.hooks.*
  - visdet.core.evaluation.* -> visdet.evaluation.*
  - visdet.core.post_processing.* -> various locations
  - visdet.core.utils.* -> various locations
  - visdet.core.optimizers.* -> visdet.engine.optim.*
"""

# Re-export commonly used top-level imports
from visdet.models.task_modules.builder import (
    build_assigner,
    build_bbox_coder,
    build_sampler,
)
from visdet.structures.bbox import distance2bbox
from visdet.structures.bbox.transforms import bbox2roi, roi2bbox
from visdet.structures.mask import BitmapMasks, PolygonMasks

__all__ = [
    "build_assigner",
    "build_bbox_coder",
    "build_sampler",
    "BitmapMasks",
    "PolygonMasks",
    "bbox2roi",
    "roi2bbox",
    "distance2bbox",
]
