# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for visdet.core.bbox submodule."""

# Bbox transforms from structures
# Builders from task_modules
from visdet.models.task_modules.builder import (
    build_assigner,
    build_bbox_coder,
    build_sampler,
)
from visdet.structures.bbox import (
    bbox2corner,
    bbox2distance,
    bbox2result,
    bbox2roi,
    bbox_cxcywh_to_xyxy,
    bbox_flip,
    bbox_mapping,
    bbox_mapping_back,
    bbox_overlaps,
    bbox_project,
    bbox_rescale,
    bbox_xyxy_to_cxcywh,
    corner2bbox,
    distance2bbox,
    find_inside_bboxes,
    roi2bbox,
)

__all__ = [
    # Transforms
    "bbox2corner",
    "bbox2distance",
    "bbox2result",
    "bbox2roi",
    "bbox_cxcywh_to_xyxy",
    "bbox_flip",
    "bbox_mapping",
    "bbox_mapping_back",
    "bbox_overlaps",
    "bbox_project",
    "bbox_rescale",
    "bbox_xyxy_to_cxcywh",
    "corner2bbox",
    "distance2bbox",
    "find_inside_bboxes",
    "roi2bbox",
    # Builders
    "build_assigner",
    "build_bbox_coder",
    "build_sampler",
]
