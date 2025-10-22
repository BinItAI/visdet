# Copyright (c) OpenMMLab. All rights reserved.
"""
================================================================================
WARNING: THIS IS A TEMPORARY COMPATIBILITY SHIM.

This module is designed to bridge the gap between MMDetection v2-style tests
and the MMDetection v3-based structure of `visdet`. It allows old imports
like `from visdet.core import X` to function.

DO NOT USE THESE IMPORTS IN NEW CODE.

The long-term goal is to refactor all tests to use the new import paths
and remove this file entirely.
================================================================================
"""

import warnings

warnings.warn(
    "`visdet.core` is a deprecated compatibility module. Please update "
    "imports to point to their new locations in the v3 architecture. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# This shim explicitly re-exports common components from their new locations.
# If you encounter an ImportError, it likely means you need to:
# 1. Port the corresponding module from MMDetection v3 into your `visdet` codebase.
# 2. Add the new import redirection to this file.

# --- from visdet.core import visualization ---
# Check if visualization exists in the package
try:
    from visdet import visualization
except ImportError:
    try:
        from visdet.engine import visualization
    except ImportError:
        visualization = None

# --- from visdet.core import BitmapMasks, PolygonMasks ---
try:
    from visdet.structures.mask import BitmapMasks, PolygonMasks
except ImportError as e:
    warnings.warn(
        f"Could not import mask structures from `visdet.structures.mask`. "
        f"Please ensure this module has been ported from MMDetection v3. Original error: {e}",
        ImportWarning,
    )
    BitmapMasks, PolygonMasks = None, None

# --- from visdet.core.bbox ---
try:
    from visdet.models.task_modules.assigners import AssignResult, BaseAssigner
    from visdet.models.task_modules.coders import BaseBBoxCoder
    from visdet.models.task_modules.samplers import BaseSampler, SamplingResult
except ImportError as e:
    warnings.warn(
        f"Could not import bbox structures from `visdet.models.task_modules`. "
        f"Please ensure these modules have been ported from MMDetection v3. Original error: {e}",
        ImportWarning,
    )
    AssignResult, BaseAssigner, BaseSampler, SamplingResult, BaseBBoxCoder = (None,) * 5

# --- from visdet.core.evaluation ---
try:
    from visdet.evaluation.functional import eval_map, eval_recalls
except ImportError as e:
    warnings.warn(
        f"Could not import evaluation functions from `visdet.evaluation`. "
        f"Please ensure this module has been ported from MMDetection v3. Original error: {e}",
        ImportWarning,
    )
    eval_map, eval_recalls = None, None

# --- from visdet.core.post_processing ---
try:
    from visdet.models.layers import multiclass_nms
except ImportError:
    try:
        from visdet.models.task_modules.nms import multiclass_nms
    except ImportError as e:
        warnings.warn(
            f"Could not import multiclass_nms. Original error: {e}",
            ImportWarning,
        )
        multiclass_nms = None

# Additional common imports that tests might use
try:
    from visdet.structures.bbox import bbox2roi, bbox_overlaps, distance2bbox
except ImportError as e:
    warnings.warn(
        f"Could not import bbox utilities from `visdet.structures.bbox`. Original error: {e}",
        ImportWarning,
    )
    bbox2roi, bbox_overlaps, distance2bbox = None, None, None

# Define __all__ to control `from visdet.core import *` and for introspection
__all__ = [
    "visualization",
    "BitmapMasks",
    "PolygonMasks",
    "AssignResult",
    "BaseAssigner",
    "BaseSampler",
    "SamplingResult",
    "BaseBBoxCoder",
    "eval_map",
    "eval_recalls",
    "multiclass_nms",
    "bbox2roi",
    "bbox_overlaps",
    "distance2bbox",
]
