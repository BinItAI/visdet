# Copyright (c) OpenMMLab. All rights reserved.
"""Placeholder dense heads temporarily blocking unsupported models."""

from visdet.models.dense_heads.base_dense_head import BaseDenseHead
from visdet.registry import MODELS


class _UnavailableHead(BaseDenseHead):
    """Base class for placeholder heads."""

    missing_impl: str = "unknown"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} ({self.missing_impl}) is not implemented in visdet yet.")


@MODELS.register_module()
class FSAFHead(_UnavailableHead):
    """Placeholder for Feature Selective Anchor-Free head."""

    missing_impl = "FSAF"


@MODELS.register_module()
class SSDHead(_UnavailableHead):
    """Placeholder for SSD head."""

    missing_impl = "SSD"


@MODELS.register_module()
class YOLOV3Head(_UnavailableHead):
    """Placeholder for YOLOv3 head."""

    missing_impl = "YOLOv3"
