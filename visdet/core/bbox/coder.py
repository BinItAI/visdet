# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for visdet.core.bbox.coder."""

from visdet.models.task_modules.builder import build_bbox_coder
from visdet.models.task_modules.coders import DeltaXYWHBBoxCoder


# Stubs for coders not in minimal build
class DistancePointBBoxCoder:
    """Stub for DistancePointBBoxCoder - not in minimal build."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DistancePointBBoxCoder is not available in the minimal visdet build. "
            "This is used for anchor-free detectors, not Mask R-CNN."
        )


class TBLRBBoxCoder:
    """Stub for TBLRBBoxCoder - not in minimal build."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TBLRBBoxCoder is not available in the minimal visdet build.")


__all__ = [
    "DeltaXYWHBBoxCoder",
    "DistancePointBBoxCoder",
    "TBLRBBoxCoder",
    "build_bbox_coder",
]
