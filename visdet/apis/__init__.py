# ruff: noqa
# Copyright (c) OpenMMLab. All rights reserved.
# Import models to ensure they are registered before any API function is called
from visdet import models as _  # noqa: F401

from .det_inferencer import DetInferencer
from .inference import inference_detector, init_detector

__all__ = [
    "DetInferencer",
    "inference_detector",
    "init_detector",
]
