# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for visdet.core.anchor submodule."""

# Re-export from parent module
from visdet.models.task_modules.builder import build_prior_generator
from visdet.models.task_modules.prior_generators import (
    AnchorGenerator,
    anchor_inside_flags,
)

build_anchor_generator = build_prior_generator


class MlvlPointGenerator:
    """Stub for MlvlPointGenerator - not in minimal Mask R-CNN build."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MlvlPointGenerator is not available in the minimal visdet build. "
            "This is used for anchor-free detectors (FCOS, etc.), not Mask R-CNN."
        )


class SSDAnchorGenerator:
    """Stub for SSDAnchorGenerator - not in minimal build."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SSDAnchorGenerator is not available in the minimal visdet build.")


class YOLOAnchorGenerator:
    """Stub for YOLOAnchorGenerator - not in minimal build."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("YOLOAnchorGenerator is not available in the minimal visdet build.")


__all__ = [
    "AnchorGenerator",
    "MlvlPointGenerator",
    "SSDAnchorGenerator",
    "YOLOAnchorGenerator",
    "anchor_inside_flags",
    "build_prior_generator",
    "build_anchor_generator",
]
