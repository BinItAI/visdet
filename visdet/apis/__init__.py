# ruff: noqa
# Copyright (c) OpenMMLab. All rights reserved.
# Import models to ensure they are registered before any API function is called
from visdet import models as _  # noqa: F401

from visdet.apis.det_inferencer import DetInferencer
from visdet.apis.inference import async_inference_detector, inference_detector, init_detector

__all__ = [
    "DetInferencer",
    "async_inference_detector",
    "inference_detector",
    "init_detector",
]
